import gc
import os
import sys
import time

import apex
from apex import amp
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from lib.configs import get_cfg_defaults
from lib.data.endocv_dataset import EDDDataset
from lib.modeling.loss import BinaryDiceLoss, BinaryIoULoss, \
    SigmoidFocalLoss
from lib.modeling.semantic_seg import build_sem_seg_model
from lib.solver import make_optimizer, WarmupCyclicalLR
from lib.utils import setup_determinism, setup_logger
from args import parse_args
from tools import distil_train_loop, train_loop, valid_model, test_model, \
                  copy_model, moving_average, bn_update


def make_dataloader(cfg, mode):

    def _test_collate_fn(batch):
        imgs, img_ids, orig_sizes = zip(*batch)
        return torch.stack(imgs), img_ids, orig_sizes

    dataset = EDDDataset(cfg=cfg, mode=mode)
    if cfg.DEBUG:
        dataset = Subset(dataset,
                         np.random.choice(np.arange(len(dataset)), 50))
    shuffle = True if mode == "train" else False
    dataloader = DataLoader(dataset, cfg.TRAIN.BATCH_SIZE,
                            pin_memory=False, shuffle=shuffle,
                            drop_last=False,
                            collate_fn=_test_collate_fn if mode == "test" else None,
                            num_workers=cfg.SYSTEM.NUM_WORKERS)
    return dataloader


def main(args, cfg):
    # Set logger
    logger = setup_logger(
        args.mode,
        cfg.DIRS.LOGS,
        0,
        filename=f"{cfg.EXP}.txt")

    # Declare variables
    best_metric = 0.
    start_cycle = 0
    start_epoch = 0

    # Define model
    if args.mode == "distil":
        student_model = build_sem_seg_model(cfg)
        teacher_model = build_sem_seg_model(cfg)
        optimizer = make_optimizer(cfg, student_model)
    else:
        model = build_sem_seg_model(cfg)
        if args.mode == "swa":
            swa_model = build_sem_seg_model(cfg)
        if cfg.DATA.AUGMENT == "augmix":
            from timm.models import convert_splitbn_model

            model = convert_splitbn_model(model, 3)
            swa_model = convert_splitbn_model(model, 3)
        optimizer = make_optimizer(cfg, model)

    # Define loss
    loss_name = cfg.LOSS.NAME
    if loss_name == "bce":
        train_criterion = nn.BCEWithLogitsLoss()
    elif loss_name == "focal":
        train_criterion = SigmoidFocalLoss(1.25, 0.25)
    elif loss_name == "dice":
        train_criterion = BinaryDiceLoss()
    elif loss_name == "iou":
        train_criterion = BinaryIoULoss()

    # CUDA & Mixed Precision
    if cfg.SYSTEM.CUDA:
        if args.mode == "distil":
            student_model = student_model.cuda()
            teacher_model = teacher_model.cuda()
        elif args.mode == "swa":
            model = model.cuda()
            swa_model = swa_model.cuda()
        else:
            model = model.cuda()
        train_criterion = train_criterion.cuda()


    if cfg.SYSTEM.FP16:
        bn_fp32 = True if cfg.SYSTEM.OPT_L == "O2" else None
        if args.mode == "distil":
            [student_model, teacher_model], optimizer = amp.initialize(models=[student_model, teacher_model],
                                                                       optimizers=optimizer,
                                                                       opt_level=cfg.SYSTEM.OPT_L,
                                                                       keep_batchnorm_fp32=bn_fp32)
        if args.mode == "swa":
            [model, swa_model], optimizer = amp.initialize(models=[model, swa_model], optimizers=optimizer,
                                              opt_level=cfg.SYSTEM.OPT_L,
                                              keep_batchnorm_fp32=bn_fp32)
        else:
            model, optimizer = amp.initialize(models=model, optimizers=optimizer,
                                              opt_level=cfg.SYSTEM.OPT_L,
                                              keep_batchnorm_fp32=bn_fp32)

    # Load checkpoint
    if args.load != "":
        if os.path.isfile(args.load):
            logger.info(f"=> loading checkpoint {args.load}")
            ckpt = torch.load(args.load, "cpu")
            model.load_state_dict(ckpt.pop('state_dict'))
            if args.swa:
                swa_model.load_state_dict(model.state_dict())
            if not args.finetune:
                logger.info("resuming optimizer ...")
                optimizer.load_state_dict(ckpt.pop('optimizer'))
                if args.mode == "cycle":
                    start_cycle = ckpt["cycle"]
                start_epoch, best_metric = ckpt['epoch'], ckpt['best_metric']
            logger.info(
                f"=> loaded checkpoint '{args.load}' (epoch {ckpt['epoch']}, best_metric: {ckpt['best_metric']})")
            if args.mode == "swa":
                ckpt = torch.load(args.load, "cpu")
                swa_model.load_state_dict(ckpt.pop('state_dict'))
        else:
            logger.info(f"=> no checkpoint found at '{args.load}'")

    if cfg.SYSTEM.MULTI_GPU:
        model = nn.DataParallel(model)

    # Load and split data
    train_loader = make_dataloader(cfg, "train")
    valid_loader = make_dataloader(cfg, "valid")

    scheduler = WarmupCyclicalLR("cos", cfg.OPT.BASE_LR, cfg.TRAIN.EPOCHS,
                                 iters_per_epoch=len(train_loader) // cfg.OPT.GD_STEPS,
                                 warmup_epochs=cfg.OPT.WARMUP_EPOCHS)

    if args.mode == "train":
        for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
            train_loop(logger.info, cfg, model,
                       train_loader, train_criterion, optimizer,
                       scheduler, epoch)
            best_metric = valid_model(logger.info, cfg, model,
                                      valid_loader, optimizer,
                                      epoch, None,
                                      best_metric, True)
    elif args.mode == "cycle":
        for cycle in range(start_cycle, cfg.TRAIN.NUM_CYCLES):
            for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
                train_loop(logger.info, cfg, model,
                           train_loader, train_criterion, optimizer,
                           scheduler, epoch)
                best_metric = valid_model(logger.info, cfg, model,
                                          valid_loader, optimizer,
                                          epoch, cycle,
                                          best_metric, True)
            # reset scheduler for new cycle
            scheduler = WarmupCyclicalLR("cos", cfg.OPT.BASE_LR, cfg.TRAIN.EPOCHS,
                                         iters_per_epoch=len(train_loader) // cfg.OPT.GD_STEPS,
                                         warmup_epochs=cfg.OPT.WARMUP_EPOCHS)
    elif args.mode == "distil":
        for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
            distil_train_loop(logger.info, cfg, student_model, teacher_model,
                              train_loader, train_criterion, optimizer,
                              scheduler, epoch)
            best_metric = valid_model(logger.info, cfg, student_model,
                                      valid_loader, optimizer,
                                      epoch, None,
                                      best_metric, True)
    elif args.mode == "swa":
        for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
            train_loop(logger.info, cfg, model,
                       train_loader, train_criterion, optimizer,
                       scheduler, epoch)
            best_metric = valid_model(logger.info, cfg, model,
                                      valid_loader, optimizer,
                                      epoch, None,
                                      best_metric, True)
            if (epoch+1) == cfg.OPT.SWA.START:
                copy_model(swa_model, model)
                swa_n = 0
            if ((epoch+1) >= cfg.OPT.SWA.START) and ((epoch+1) % cfg.OPT.SWA.FREQ == 0):
                moving_average(swa_model, model, 1.0 / (swa_n + 1))
                swa_n += 1
                bn_update(train_loader, swa_model)
                best_metric = valid_model(logger.info, cfg, swa_model,
                                          valid_loader, optimizer,
                                          epoch, None,
                                          best_metric, True)

    elif args.mode == "valid":
        valid_model(logger.info, cfg, model,
                    valid_loader, optimizer, start_epoch)
    else:
        test_loader = make_dataloader(cfg, "test")
        predictions = test_model(logger.info, cfg, model, test_loader)


if __name__ == "__main__":
    args = parse_args()
    cfg = get_cfg_defaults()

    if args.config != "":
        cfg.merge_from_file(args.config)
    if args.mode != "train":
        cfg.merge_from_list(['INFER.TTA', args.tta])
    if args.debug:
        opts = ["DEBUG", True, "TRAIN.EPOCHS", 2]
        cfg.merge_from_list(opts)
    # cfg.freeze()
    # make dirs
    for _dir in ["WEIGHTS", "OUTPUTS", "LOGS"]:
        if not os.path.isdir(cfg.DIRS[_dir]):
            os.makedirs(cfg.DIRS[_dir])
    # seed, run
    main(args, cfg)
