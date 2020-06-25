import torch
import torch.nn as nn
import torch.nn.functional as F

from .build import SEM_SEG_HEAD_REGISTRY
from cvcore.modeling.backbone.fpn import get_act, get_norm


__all__ = ["UNetDecoder", "UNetHead"]


class UNetDecoder(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels,
                 norm_layer, act_layer):
        super(UNetDecoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels,
                kernel_size=3, padding=1, bias=False),
            norm_layer,
            act_layer)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels,
                kernel_size=3, padding=1, bias=False),
            norm_layer,
            act_layer)

    def forward(self, x, skip=None):
        if skip is not None:
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


@SEM_SEG_HEAD_REGISTRY.register()
class UNetHead(nn.Module):
    def __init__(self, cfg, in_feature_strides, in_feature_channels):
        super(UNetHead, self).__init__()

        self.in_features = cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES[::-1]
        feature_strides = {f: in_feature_strides[f] for f in self.in_features}
        feature_channels = {f: in_feature_channels[f] for f in self.in_features}
        num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        conv_dims = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        self.common_stride = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
        norm = cfg.MODEL.SEM_SEG_HEAD.NORM
        act = cfg.MODEL.SEM_SEG_HEAD.ACT

        skip_channels = [0] + [feature_channels[f] for f in self.in_features[1:]]
        out_channels = [conv_dims * 2 ** i for i in range(len(self.in_features))][::-1]
        in_channels = [feature_channels[self.in_features[0]]] + out_channels[:-1]
        blocks = []
        for in_feature, in_ch, skip_ch, out_ch in zip(self.in_features,
            in_channels, skip_channels, out_channels):
            blocks.append(UNetDecoder(in_ch, skip_ch, out_ch,
                get_norm(norm, out_ch), get_act(act)))
            self.add_module(in_feature, blocks[-1])
        self.blocks = blocks
        self.predictor = nn.Conv2d(out_channels[-1], num_classes,
            kernel_size=1, stride=1, padding=0)

    def forward(self, features):
        features = [features[f] for f in self.in_features]
        x = self.blocks[0](features[0])
        for skip, block in zip(features[1:], self.blocks[1:]):
            x = block(x, skip)
        x = self.predictor(x)
        x = F.interpolate(x, scale_factor=self.common_stride, mode="bilinear", align_corners=False)
        return x