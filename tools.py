import apex
from apex import amp
import numpy as np
import os
import tifffile as tiff
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from lib.data import cutmix_data, mixup_data
from lib.loss import binary_dice_loss, binary_dice_metric, \
    binary_iou_loss, binary_iou_metric, fbeta_loss, SigmoidFocalLoss
from lib.utils import AverageMeter, save_checkpoint


def freeze_batchnorm(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            module.weight.requires_grad = False
            module.bias.requires_grad = False


def train_loop(_print, cfg, model, train_loader,
               criterion, optimizer, scheduler, epoch):
    _print(f"\nEpoch {epoch + 1}")
    losses = AverageMeter()
    model.train()
    tbar = tqdm(train_loader)

    use_cutmix = cfg.DATA.CUTMIX
    use_mixup = cfg.DATA.MIXUP
    fl = SigmoidFocalLoss(gamma=2., alpha=0.25)
    # bce = torch.nn.BCEWithLogitsLoss()
    # for i, (image, mask) in enumerate(tbar):
    for i, (image, mask, label) in enumerate(tbar):
        image = image.cuda()
        mask = mask.cuda()
        label = label.cuda()
        # mixup/ cutmix
        if use_mixup:
            image, mask = mixup_data(image, mask, alpha=cfg.DATA.CM_ALPHA)
        elif use_cutmix:
            image, mask = cutmix_data(image, mask, alpha=cfg.DATA.CM_ALPHA)

        # compute loss
        # output = model(image)
        # loss = criterion(output, mask)
        output, cls_output = model(image)
        # aux_loss = bce(cls_output, label)
        aux_loss = fl(cls_output, label)
        loss = criterion(output, mask) + aux_loss * 0.4

        # gradient accumulation
        loss = loss / cfg.OPT.GD_STEPS
        if cfg.SYSTEM.FP16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        # lr scheduler and optim. step
        if (i + 1) % cfg.OPT.GD_STEPS == 0:
            scheduler(optimizer, i, epoch)
            optimizer.step()
            optimizer.zero_grad()
        # record loss
        losses.update(loss.item() * cfg.OPT.GD_STEPS, image.size(0))
        tbar.set_description("Train loss: %.5f, learning rate: %.6f" % (
            losses.avg, optimizer.param_groups[-1]['lr']))

    _print("Train loss: %.5f, learning rate: %.6f" %
           (losses.avg, optimizer.param_groups[-1]['lr']))


def valid_model(_print, cfg, model, valid_loader,
                optimizer, epoch, cycle=None,
                best_metric=None, checkpoint=False):
    tta = cfg.INFER.TTA
    threshold = cfg.INFER.THRESHOLD
    # switch to evaluate mode
    model.eval()
    freeze_batchnorm(model)

    # valid_dice = []
    # valid_iou = []
    valid_output = []
    valid_mask = []
    valid_cls_output = []
    valid_label = []
    tbar = tqdm(valid_loader)

    with torch.no_grad():
        for i, (image, mask, label) in enumerate(tbar):
            image = image.cuda()
            mask = mask.cuda()
            if tta:
                output, cls_output = model(image)
                tta_output, tta_cls_output = model(image.flip(3))
                output = (output + tta_output.flip(3)) / 2.
                cls_output = (cls_output + tta_cls_output) / 2.
            else:
                output, cls_output = model(image)

            cls_output = torch.sigmoid(cls_output)
            valid_cls_output.append(cls_output.cpu().numpy())
            valid_label.append(label.numpy())
            valid_output.append(output.cpu()); valid_mask.append(mask.cpu())
            # batch_dice = binary_dice_metric(output, mask,
            #     threshold).cpu()
            # valid_dice.append(batch_dice)
            # batch_iou = binary_iou_metric(output, mask,
            #     threshold).cpu()
            # valid_iou.append(batch_iou)

    valid_cls_output = np.concatenate(valid_cls_output, 0)
    valid_label = np.concatenate(valid_label, 0)
    np.save(os.path.join(cfg.DIRS.OUTPUTS, f'{cfg.EXP}_cls.npy'),
        valid_cls_output)
    np.save(os.path.join(cfg.DIRS.OUTPUTS, f'label_{cfg.DATA.FOLD}.npy'),
        valid_label)
    cls_threshold = search_threshold(valid_cls_output, valid_label)
    valid_cls_output = valid_cls_output > np.expand_dims(
        np.array(cls_threshold), 0)
    valid_f1 = [f1_score(valid_label[:, i], valid_cls_output[:, i])
        for i in range(5)]
    macro_f1 = np.average(valid_f1)

    valid_output = torch.cat(valid_output, 0); valid_mask = torch.cat(valid_mask, 0)
    # torch.save(valid_output,
    #     os.path.join(cfg.DIRS.OUTPUTS, f'{cfg.EXP}.pth'))
    torch.save(valid_mask,
        os.path.join(cfg.DIRS.OUTPUTS, f'mask_{cfg.DATA.FOLD}.pth'))
    # valid_dice = torch.cat(valid_dice, 0).mean(0).numpy()
    # valid_iou = torch.cat(valid_iou, 0).mean(0).numpy()
    valid_cls_output = torch.from_numpy(valid_cls_output)
    valid_cls_output_mask = torch.stack([
        valid_cls_output[:, i, ...] > th for i, th in enumerate(cls_threshold)], 1)
    valid_cls_output_mask = valid_cls_output_mask.unsqueeze(-1).unsqueeze(-1).float()
    valid_output *= valid_cls_output_mask
    torch.save(valid_output,
        os.path.join(cfg.DIRS.OUTPUTS, f'{cfg.EXP}.pth'))
    valid_dice = binary_dice_metric(valid_output, valid_mask, threshold).mean(0).numpy()
    valid_iou = binary_iou_metric(valid_output, valid_mask, threshold).mean(0).numpy()
    mean_dice = np.average(valid_dice)
    mean_iou = np.average(valid_iou)
    final_score = mean_iou
    log_info = "Mean Dice: %.8f - BE: %.8f - suspicious: %.8f - HGD: %.8f - cancer: %.8f - polyp: %.8f\n"
    log_info += "Mean IoU: %.8f - BE: %.8f - suspicious: %.8f - HGD: %.8f - cancer: %.8f - polyp: %.8f\n"
    log_info += "Macro F1: %.8f - BE: %.8f - suspicious: %.8f - HGD: %.8f - cancer: %.8f - polyp: %.8f"
    _print(log_info % (mean_dice, valid_dice[0], valid_dice[1],
        valid_dice[2], valid_dice[3], valid_dice[4], mean_iou,
        valid_iou[0], valid_iou[1], valid_iou[2], valid_iou[3], valid_iou[4],
        macro_f1, valid_f1[0], valid_f1[1], valid_f1[2], valid_f1[3], valid_f1[4]))

    # checkpoint
    if checkpoint:
        is_best = final_score > best_metric
        best_metric = max(final_score, best_metric)
        save_dict = {"epoch": epoch + 1,
                     "arch": cfg.EXP,
                     "state_dict": model.state_dict(),
                     "best_metric": best_metric,
                     "optimizer": optimizer.state_dict()}
        if cycle is not None:
            save_dict["cycle"] = cycle
            save_filename = f"{cfg.EXP}_cycle{cycle}.pth"
        else:
            save_filename = f"{cfg.EXP}.pth"
        save_checkpoint(save_dict, is_best,
                        root=cfg.DIRS.WEIGHTS, filename=save_filename)
        return best_metric


def test_model(_print, cfg, model, test_loader):
    output_dir = cfg.DIRS.OUTPUTS
    tta = cfg.INFER.TTA
    threshold = cfg.INFER.THRESHOLD
    cls_threshold = [0.4, 0.2, 0.2, 0.2, 0.4]
    # switch to evaluate mode
    model.eval()
    freeze_batchnorm(model)

    test_output = []
    test_cls_output = []
    test_orig_size = []
    test_img_id = []
    tbar = tqdm(test_loader)
    with torch.no_grad():
        for i, (image, img_id, orig_size) in enumerate(tbar):
            image = image.cuda()
            if tta:
                output, cls_output = model(image)
                tta_output, tta_cls_output = model(image.flip(3))
                output = (output + tta_output.flip(3)) / 2.
                cls_output = (cls_output + tta_cls_output) / 2.
            else:
                output, cls_output = model(image)

            test_output.append(output.cpu())
            test_cls_output.append(cls_output.cpu())
            test_orig_size.extend(orig_size)
            test_img_id.extend(img_id)

    test_output = torch.cat(test_output, 0)
    test_cls_output = torch.cat(test_cls_output, 0)
    # search thresholds using validation
    valid_cls_output = np.load(os.path.join(
        cfg.DIRS.OUTPUTS, f'{cfg.EXP}_cls.npy'))
    valid_label = np.load(os.path.join(
        cfg.DIRS.OUTPUTS, f'label_{cfg.DATA.FOLD}.npy'))
    cls_threshold = search_threshold(valid_cls_output, valid_label)
    print(cls_threshold)
    # filter masks
    test_cls_output_mask = torch.stack([
        test_cls_output[:, i, ...] > th for i, th in enumerate(cls_threshold)], 1)
    test_cls_output_mask = test_cls_output_mask.unsqueeze(-1).unsqueeze(-1).float()
    test_output *= test_cls_output_mask

    torch.save(test_output,
        os.path.join(cfg.DIRS.OUTPUTS, f'{cfg.EXP}_test.pth'))
    torch.save(test_cls_output,
        os.path.join(cfg.DIRS.OUTPUTS, f'{cfg.EXP}_cls_test.pth'))
    torch.save(test_img_id,
        os.path.join(cfg.DIRS.OUTPUTS, 'test_img_ids.pth'))
    torch.save(test_orig_size,
        os.path.join(cfg.DIRS.OUTPUTS, 'test_sizes.pth'))
    return None


def search_threshold(inputs, targets,
        grid_thresholds=np.linspace(0.1, 0.6, 100),
        metric_func=f1_score):
    num_classes = inputs.shape[1]
    best_thresholds = []
    for i in range(num_classes):
        class_inp = inputs[:, i]
        class_tar = targets[:, i]
        grid_scores = []
        for thresh in grid_thresholds:
            grid_scores.append(metric_func(class_tar, class_inp > thresh))
        best_t = grid_thresholds[np.argmax(grid_scores)]
        best_score = np.max(grid_scores)
        best_thresholds.append(best_t)
    return best_thresholds


def distil_train_loop(_print, cfg, student_model, teacher_model,
                      train_loader, criterion, optimizer, scheduler, epoch):
    # TODO
    # _print(f"\nEpoch {epoch + 1}")

    # losses = AverageMeter()
    # student_model.train()
    # teacher_model.eval()
    # tbar = tqdm(train_loader)
    pass


def compute_jsd_loss(logits_clean, logits_aug1, logits_aug2, lamb=12.):
    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean,
                                        dim=1), F.softmax(logits_aug1,
                                                          dim=1), F.softmax(logits_aug2,
                                                                            dim=1)
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1.).log()
    jsd = lamb * (F.kl_div(p_mixture, p_clean, reduction="batchmean") +
                  F.kl_div(p_mixture, p_aug1, reduction="batchmean") +
                  F.kl_div(p_mixture, p_aug2, reduction="batchmean")) / 3.
    return jsd


def compute_distil_loss(student_logit, teacher_logit, temp=4.):
    student_prob = F.softmax(student_logit / temp, dim=-1)
    teacher_prob = F.softmax(teacher_logit / temp, dim=-1).log()
    loss = F.kl_div(teacher_prob, student_prob, reduction="batchmean")
    return loss


def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha

def copy_model(net1, net2):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= 0
        param1.data += param2.data

def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.
        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    tbar = tqdm(loader)
    # for i, (input, _) in enumerate(tbar):
    for i, (input, _, _) in enumerate(tbar):
        input = input.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input)
        b = input_var.data.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        model(input_var)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))
