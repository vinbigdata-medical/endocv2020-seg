from collections import OrderedDict
import timm
from timm.models.layers import SelectAdaptivePool2d
import torch
import torch.nn as nn
import torch.nn.functional as F

from .build import BACKBONE_REGISTRY


__all__ = ["EfficientNet", "build_efficientnet_backbone"]


class EfficientNet(nn.Module):
    """
    EfficientNet B0-B8.

    Args:
    """
    def __init__(self, model_name, pretrained, input_channels,
            pool_type, num_classes, out_features, drop_rate, drop_connect_rate):
        super(EfficientNet, self).__init__()

        backbone = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            in_chans=input_channels,
            drop_connect_rate=drop_connect_rate)

        stem_modules = [
            ('conv_stem', backbone.conv_stem),
            ('bn1', backbone.bn1),
            ('act1', backbone.act1)
        ]
        self.stem = nn.Sequential(OrderedDict(stem_modules))

        current_stride = backbone.conv_stem.stride[0]
        self._out_feature_strides = {"stem": current_stride}
        self._out_feature_channels = {"stem": backbone.bn1.num_features}

        self.blocks_and_names = []
        for i in range(7):
            block = backbone.blocks[i]
            name = f"block{i}"
            self.add_module(name, block)
            self.blocks_and_names.append((block, name))
            self._out_feature_strides[name] = current_stride = int(
                current_stride * block[0].conv_dw.stride[0])
            if i == 0:
                self._out_feature_channels[name] = block[-1].bn2.num_features
            else:
                self._out_feature_channels[name] = block[-1].bn3.num_features

        head_modules = [
            ('conv_head', backbone.conv_head),
            ('bn2', backbone.bn2),
            ('act2', backbone.act2)
        ]
        self.head = nn.Sequential(OrderedDict(head_modules))
        current_stride *= backbone.conv_head.stride[0]
        self._out_feature_strides["head"] = current_stride
        self._out_feature_channels["head"] = backbone.conv_head.out_channels

        if num_classes is not None:
            self.global_pool = SelectAdaptivePool2d(pool_type=pool_type)
            self.num_features = backbone.num_features * self.global_pool.feat_mult()
            self.linear = nn.Linear(self.num_features, num_classes)

        del backbone
        self.drop_rate = drop_rate
        self.num_classes = num_classes
        self._out_features = out_features

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for block, name in self.blocks_and_names:
            x = block(x)
            if name in self._out_features:
                outputs[name] = x
        x = self.head(x)
        if "head" in self._out_features:
            outputs["head"] = x
        if self.num_classes is not None:
            x = self.global_pool(x)
            x = torch.flatten(x, 1)
            if self.drop_rate > 0.:
                x = F.dropout(x, p=self.drop_rate, training=self.training)
            x = self.linear(x)
            if "linear" in self._out_features:
                outputs["linear"] = x
        return outputs


@BACKBONE_REGISTRY.register()
def build_efficientnet_backbone(cfg):
    """
    Create an EfficientNet instance from config.

    Returns:
        EfficientNet: a :class:`EfficientNet` instance.
    """
    model_name = cfg.MODEL.BACKBONE.NAME
    pretrained = cfg.MODEL.BACKBONE.PRETRAINED
    input_channels = cfg.DATA.IN_CHANS
    pool_type = cfg.MODEL.BACKBONE.POOL_TYPE
    # num_classes = None
    num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
    out_features = cfg.MODEL.BACKBONE.OUT_FEATURES
    drop_rate = cfg.MODEL.BACKBONE.DROPOUT
    drop_connect_rate = cfg.MODEL.BACKBONE.DROP_CONNECT

    return EfficientNet(model_name, pretrained, input_channels,
        pool_type, num_classes, out_features, drop_rate, drop_connect_rate)