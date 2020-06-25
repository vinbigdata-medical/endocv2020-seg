from cvcore.utils import Registry


SEM_SEG_HEAD_REGISTRY = Registry("SEM_SEG_HEADS")
"""
Registry for semantic segmentation heads, which make semantic segmentation predictions
from feature maps.
"""


def build_sem_seg_head(cfg, in_feature_strides, in_feature_channels):
    name = cfg.MODEL.SEM_SEG_HEAD.NAME
    return SEM_SEG_HEAD_REGISTRY.get(name)(cfg, in_feature_strides, in_feature_channels)