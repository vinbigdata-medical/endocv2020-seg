import torch.nn as nn
from cvcore.modeling.backbone import build_backbone, FPN, BiFPN
from cvcore.modeling.semantic_seg_head import build_sem_seg_head
from cvcore.utils import Registry


SEM_SEG_MODEL_REGISTRY = Registry("SEM_SEG_MODELS")
"""
Registry for semantic segmentation models.
"""


def build_sem_seg_model(cfg):
    name = cfg.MODEL.NAME
    return SEM_SEG_MODEL_REGISTRY.get(name)(cfg)


@SEM_SEG_MODEL_REGISTRY.register()
class FPNSegmentor(nn.Module):

    def __init__(self, cfg):
        super(FPNSegmentor, self).__init__()

        self.backbone = build_backbone(cfg)
        self.fpn = FPN(cfg,
            self.backbone._out_feature_strides, self.backbone._out_feature_channels)
        self.sem_seg_head = build_sem_seg_head(cfg,
            self.fpn._out_feature_strides, self.fpn._out_feature_channels)

    def forward(self, x):
        x = self.backbone(x)
        x_linear = x["linear"]
        x = self.fpn(x)
        x = self.sem_seg_head(x)
        return x, x_linear


@SEM_SEG_MODEL_REGISTRY.register()
class BiFPNSegmentor(nn.Module):
    def __init__(self, cfg):
        super(BiFPNSegmentor, self).__init__()

        self.backbone = build_backbone(cfg)
        self.fpn = BiFPN(cfg,
            self.backbone._out_feature_strides, self.backbone._out_feature_channels)
        self.sem_seg_head = build_sem_seg_head(cfg,
            self.fpn._out_feature_strides, self.fpn._out_feature_channels)

    def forward(self, x):
        x = self.backbone(x)
        x_linear = x["linear"]
        x = self.fpn(x)
        x = self.sem_seg_head(x)
        return x, x_linear


@SEM_SEG_MODEL_REGISTRY.register()
class UNetSegmentor(nn.Module):

    def __init__(self, cfg):
        super(UNetSegmentor, self).__init__()

        self.backbone = build_backbone(cfg)
        self.sem_seg_head = build_sem_seg_head(cfg,
            self.backbone._out_feature_strides, self.backbone._out_feature_channels)

    def forward(self, x):
        x = self.backbone(x)
        x_linear = x["linear"]
        x = self.sem_seg_head(x)
        return x, x_linear