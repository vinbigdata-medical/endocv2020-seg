from collections import OrderedDict
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import gc
import numpy as np
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations import pytorch


class EDDDataset(Dataset):
    def __init__(self, cfg, mode="train"):
        super(EDDDataset, self).__init__()
        self.data_dir = cfg.DIRS.DATA
        self.img_size = (cfg.DATA.IMG_SIZE, cfg.DATA.IMG_SIZE)
        self.classes = [
            "BE",
            "suspicious",
            "HGD",
            "cancer",
            "polyp"
        ]
        if mode != "test":
            self.df = pd.read_csv(os.path.join(
                self.data_dir, f"{mode}_fold{cfg.DATA.FOLD}.csv"))
        else:
            img_ids = os.listdir(os.path.join(
                self.data_dir, "edd2020"))
            self.df = pd.DataFrame(img_ids, columns=['img'])

        if mode == "train":
            # oversampling HGD/cancer
            hgd_cancer = self.df[(self.df['HGD']==1)|(self.df['cancer']==1)]
            self.df = pd.concat([self.df, hgd_cancer], 0)
            self.aug = A.Compose([
                A.OneOf([
                    A.HorizontalFlip(p=0.8),
                    A.VerticalFlip(p=0.8)]),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=30,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.8),
                A.GridDistortion(
                    distort_limit=0.2,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.8),
                A.OneOf([
                    A.IAAAdditiveGaussianNoise(p=0.8),
                    A.GaussNoise(p=0.8)]),
                A.OneOf([
                    A.MedianBlur(blur_limit=3, p=0.8),
                    A.Blur(blur_limit=3, p=0.8)]),
                A.Normalize(),
                A.pytorch.ToTensorV2()
            ])
        else:
            self.aug = A.Compose([
                A.Normalize(),
                A.pytorch.ToTensorV2()
            ])
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df['img'].values[idx]

        if self.mode == "test":
            img = Image.open(os.path.join(
                self.data_dir, "edd2020", img_id))
        else:
            img = Image.open(os.path.join(
                self.data_dir, "originalImages", img_id))

        img = np.asarray(img, dtype=np.uint8)

        if self.mode != "test":
            img_label = self.df[self.classes].values[idx]
            mask = np.zeros((*img.shape[:-1], len(self.classes)))
            for i, cl in enumerate(self.classes):
                if img_label[i] == 1:
                    mask_id = "masks/" + img_id.replace(".jpg", f"_{cl}.tif")
                    mask[..., i] = np.asarray(Image.open(
                        os.path.join(self.data_dir, mask_id)).convert('L'),
                        dtype=np.uint8)

            mask = np.where(mask > 0, 255, 0)
            img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)

            data = self.aug(image=img, mask=mask)
            img, mask = data['image'], data['mask']
            mask = mask.permute(2, 0, 1).div(255.).float()
            # print(img_id, img_label, torch.unique(mask))
            # return img, mask
            return img, mask, torch.FloatTensor(img_label)
        else:
            orig_size = img.shape[:-1]
            img = cv2.resize(img, self.img_size,
                interpolation=cv2.INTER_LINEAR)
            img = self.aug(image=img)['image']
            return img, img_id, orig_size