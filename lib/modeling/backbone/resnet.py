from collections import OrderedDict
import numpy as np
from timm import create_model
from timm.models.layers import SelectAdaptivePool2d
import torch
import torch.nn as nn
import torch.nn.functional as F

from .build import BACKBONE_REGISTRY


__all__ = ["ResNet", "build_resnet_backbone"]


class ResNet(nn.Module):
    """
    ResNet, ResNeXt, SENet.

    Args:
    """
    def __init__(self, model_name, pretrained, input_channels,
            pool_type, num_classes, out_features, drop_rate, output_stride):
        super(ResNet, self).__init__()

        backbone = create_model(
            model_name=model_name,
            pretrained=pretrained,
            output_stride=output_stride)
        if input_channels > 3:
            if hasattr(backbone, "layer0"):
                old_conv = backbone.layer0.conv1
            else:
                old_conv = backbone.conv1
            new_conv = nn.Conv2d(input_channels, old_conv.out_channels,
                old_conv.kernel_size, old_conv.stride, old_conv.padding,
                old_conv.dilation, old_conv.groups, old_conv.bias)
            new_conv.weight.data[:, :3, ...] = old_conv.weight.data
            if hasattr(backbone, "layer0"):
                backbone.layer0.conv1 = new_conv
            else:
                backbone.conv1 = new_conv

        if hasattr(backbone, "layer0"):
            self.stem = backbone.layer0
        else:
            layer0_modules = [
                ('conv1', backbone.conv1),
                ('bn1', backbone.bn1),
                ('relu', backbone.act1),
                ('maxpool', backbone.maxpool)
            ]
            self.stem = nn.Sequential(OrderedDict(layer0_modules))

        current_stride = 4 # = stride 2 conv -> stride 2 max pool
        self._out_feature_strides = {"stem": current_stride}
        self._out_feature_channels = {"stem": 64}

        self.stages_and_names = []
        for i in range(4):
            stage = getattr(backbone, f"layer{i + 1}")
            name = f"res{i + 2}"
            self.add_module(name, stage)
            self.stages_and_names.append((stage, name))
            self._out_feature_strides[name] = current_stride = int(
                current_stride * np.prod([l.stride for l in stage]))
            try:
                self._out_feature_channels[name] = stage[-1].bn3.num_features
            except:
                self._out_feature_channels[name] = stage[-1].conv2.attn.fc_select.out_channels

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
        for stage, name in self.stages_and_names:
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
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
def build_resnet_backbone(cfg):
    """
    Create a ResNet instance from config.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """

    model_name = cfg.MODEL.BACKBONE.NAME
    pretrained = cfg.MODEL.BACKBONE.PRETRAINED
    input_channels = cfg.DATA.IN_CHANS
    pool_type = cfg.MODEL.BACKBONE.POOL_TYPE
    # num_classes = None
    num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
    out_features = cfg.MODEL.BACKBONE.OUT_FEATURES
    drop_rate = cfg.MODEL.BACKBONE.DROPOUT
    output_stride = cfg.MODEL.BACKBONE.OUTPUT_STRIDE

    return ResNet(model_name, pretrained, input_channels,
        pool_type, num_classes, out_features, drop_rate, output_stride)