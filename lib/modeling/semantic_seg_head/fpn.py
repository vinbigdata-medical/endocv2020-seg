import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from cvcore.modeling.backbone.fpn import get_act, get_norm
from .build import SEM_SEG_HEAD_REGISTRY

__all__ = ["FPNHead"]


@SEM_SEG_HEAD_REGISTRY.register()
class FPNHead(nn.Module):
    """
    A semantic segmentation head described in detail in the Panoptic Feature Pyramid Networks paper
    (https://arxiv.org/abs/1901.02446). It takes FPN features as input and merges information from
    all levels of the FPN into single output.
    """

    def __init__(self, cfg, in_feature_strides, in_feature_channels):
        super(FPNHead, self).__init__()

        self.in_features = cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        feature_strides = {f: in_feature_strides[f] for f in self.in_features}
        feature_channels = {f: in_feature_channels[f] for f in self.in_features}
        num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        conv_dims = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        self.common_stride = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
        norm = cfg.MODEL.SEM_SEG_HEAD.NORM
        act = cfg.MODEL.SEM_SEG_HEAD.ACT
        fuse_type = cfg.MODEL.SEM_SEG_HEAD.FUSE_TYPE

        use_bias = norm == ""
        self.scale_heads = []
        for in_feature in self.in_features:
            head_ops = []
            head_length = max(
                1, int(np.log2(feature_strides[in_feature]) - np.log2(self.common_stride))
            )
            for k in range(head_length):
                conv = nn.Conv2d(
                    feature_channels[in_feature] if k == 0 else conv_dims, conv_dims,
                    kernel_size=3, stride=1, padding=1, bias=use_bias)
                norm_layer = get_norm(norm, conv_dims)
                act_layer = get_act(act)
                head_ops.extend([conv, norm_layer, act_layer])
                if feature_strides[in_feature] != self.common_stride:
                    head_ops.append(
                        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
            self.scale_heads.append(nn.Sequential(*head_ops))
            self.add_module(in_feature, self.scale_heads[-1])

            if fuse_type == "cat":
                predictor_in_chans = conv_dims * len(self.in_features)
            else:
                predictor_in_chans = conv_dims
            self.predictor = nn.Conv2d(predictor_in_chans, num_classes,
                kernel_size=1, stride=1, padding=0)

            self._fuse_type = fuse_type

    def forward(self, features):
        x = []
        for i, f in enumerate(self.in_features):
            x.append(self.scale_heads[i](features[f]))
        if self._fuse_type == "cat":
            x = torch.cat(x, 1)
        elif self._fuse_type == "sum":
            x = torch.stack(x, 0).sum(0)
        x = self.predictor(x)
        x = F.interpolate(x, scale_factor=self.common_stride, mode="bilinear", align_corners=False)
        return x