import math
import torch
import torch.nn.functional as F
import torch.nn as nn

from timm.models.efficientnet_blocks import DepthwiseSeparableConv
from cvcore.modeling.backbone.utils import get_act, get_norm


__all__ = ["FPN", "BiFPN"]


class FPN(nn.Module):
    """
    This module implements Feature Pyramid Network.
    It creates pyramid features built on top of some input feature maps.
    """
    def __init__(self, cfg, in_feature_strides, in_feature_channels):
        super(FPN, self).__init__()

        self.in_features = cfg.MODEL.FPN.IN_FEATURES
        out_channels = cfg.MODEL.FPN.OUT_CHANNELS
        norm = cfg.MODEL.FPN.NORM
        act = cfg.MODEL.FPN.ACT
        self._fuse_type = cfg.MODEL.FPN.FUSE_TYPE

        # Feature map strides and channels from the bottom up network (e.g. ResNet)
        in_strides = [in_feature_strides[f] for f in self.in_features]
        in_channels = [in_feature_channels[f] for f in self.in_features]

        lateral_convs = []
        output_convs = []
        use_bias = norm == ""

        for idx, in_channels in enumerate(in_channels):
            lateral_norm = get_norm(norm, out_channels)
            output_norm = get_norm(norm, out_channels)

            lateral_act = get_act(act)
            output_act = get_act(act)

            lateral_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, bias=use_bias)
            output_conv = nn.Conv2d(
                out_channels, out_channels,
                kernel_size=3, stride=1, padding=1, bias=use_bias)

            fpn_lateral = nn.Sequential(lateral_conv, lateral_norm, lateral_act)
            fpn_output = nn.Sequential(output_conv, output_norm, output_act)

            self.add_module(
                f"fpn_lateral_{self.in_features[idx]}", fpn_lateral)
            self.add_module(
                f"fpn_output_{self.in_features[idx]}", fpn_output)
            lateral_convs.append(fpn_lateral)
            output_convs.append(fpn_output)

        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]
        self._out_feature_strides = {f: s
            for f, s in zip(self.in_features, in_strides)}

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {f: out_channels for f in self._out_features}

    def forward(self, x):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.

        Returns:
            dict[str->Tensor]: mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        x = [x[f] for f in self.in_features[::-1]]
        prev_features = self.lateral_convs[0](x[0])
        results = []
        results.append(self.output_convs[0](prev_features))

        for features, lateral_conv, output_conv in zip(
            x[1:], self.lateral_convs[1:], self.output_convs[1:]):
            lateral_features = lateral_conv(features)
            if prev_features.shape[-1] != lateral_features.shape[-1]:
                top_down_features = F.interpolate(prev_features, scale_factor=2, mode="nearest")
            else:
                top_down_features = prev_features
            prev_features = lateral_features + top_down_features
            if self._fuse_type == "avg":
                prev_features /= 2
            results.insert(0, output_conv(prev_features))

        return dict(zip(self._out_features, results))


class FastNormalizedFusion(nn.Module):
    def __init__(self, num_weights=2, eps=1e-4):
        super(FastNormalizedFusion, self).__init__()
        self.num_weights = num_weights
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(self.num_weights))

    def __repr__(self):
        main_str = self._get_name() + '('
        main_str += f'num_weights={self.num_weights}'
        main_str += f', eps={self.eps}'
        main_str += ')'
        return main_str

    def forward(self, features):
        weights = F.relu(self.weights)
        features = torch.stack(features, dim=0)
        features = (features * weights.view(
            weights.shape[0], 1, 1, 1, 1)).sum(0)
        features = features / (weights.sum() + self.eps)
        return features


class BiFPN(nn.Module):
    def __init__(self, cfg, in_feature_strides, in_feature_channels):
        super(BiFPN, self).__init__()

        self.in_features = cfg.MODEL.FPN.IN_FEATURES
        out_channels = cfg.MODEL.FPN.OUT_CHANNELS
        norm = cfg.MODEL.FPN.NORM
        act = cfg.MODEL.FPN.ACT
        self._fuse_type = cfg.MODEL.FPN.FUSE_TYPE

        # Feature map strides and channels from the bottom up network (e.g. ResNet)
        in_strides = [in_feature_strides[f] for f in self.in_features]
        in_channels = [in_feature_channels[f] for f in self.in_features]

        lateral_convs = []
        lateral_fuse_convs = []
        output_convs = []
        use_bias = norm == ""

        for idx, in_channel in enumerate(in_channels):
            lateral_norm = get_norm(norm, out_channels)
            lateral_act = get_act(act)
            lateral_conv = nn.Conv2d(
                in_channel, out_channels, kernel_size=1, bias=use_bias)
            fpn_lateral = nn.Sequential(lateral_conv, lateral_norm, lateral_act)
            self.add_module(
                f"fpn_lateral_{self.in_features[idx]}", fpn_lateral)
            lateral_convs.append(fpn_lateral)

            if idx != len(in_channels) - 1:
                fpn_lateral_fuse = nn.Sequential(
                    FastNormalizedFusion(num_weights=2),
                    DepthwiseSeparableConv(out_channels, out_channels)
                )
                self.add_module(
                    f"fpn_lateral_fuse_{self.in_features[idx]}", fpn_lateral_fuse)
                lateral_fuse_convs.append(fpn_lateral_fuse)

            if idx != 0:
                num_fmaps = 2 if idx == len(in_channels) - 1 else 3
                fpn_output = nn.Sequential(
                    FastNormalizedFusion(num_weights=num_fmaps),
                    DepthwiseSeparableConv(out_channels, out_channels)
                )
                self.add_module(f"fpn_output_{self.in_features[idx]}" \
                    .replace("res", ""), fpn_output)
                output_convs.append(fpn_output)

        self.lateral_convs = lateral_convs[::-1]
        self.lateral_fuse_convs = lateral_fuse_convs[::-1]
        self.output_convs = output_convs
        self._out_feature_strides = {f: s
            for f, s in zip(self.in_features, in_strides)}

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {f: out_channels for f in self._out_features}

    def forward(self, x):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.

        Returns:
            dict[str->Tensor]: mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        x = [x[f] for f in self.in_features[::-1]]
        # top-down path
        top_down = []
        in_list = []
        prev_features = self.lateral_convs[0](x[0])
        in_list.append(prev_features)
        top_down.append(prev_features)
        for features, lateral_conv, lateral_fuse_conv in zip(
            x[1:], self.lateral_convs[1:], self.lateral_fuse_convs):
            in_features = lateral_conv(features)
            in_list.insert(0, in_features)
            if prev_features.shape[-1] != in_features.shape[-1]:
                top_down_features = F.interpolate(prev_features, scale_factor=2, mode="nearest")
            else:
                top_down_features = prev_features
            prev_features = lateral_fuse_conv([in_features, top_down_features])
            top_down.insert(0, prev_features)
        # bottom-up path
        bottom_up = []
        bottom_up.append(prev_features)
        for in_features, top_down_features, output_conv in zip(
            in_list[1:-1], top_down[1:-1], self.output_convs[:-1]):
            if prev_features.shape[-1] != in_features.shape[-1]:
                out_features = F.interpolate(prev_features, scale_factor=0.5, mode="nearest")
            else:
                out_features = prev_features
            prev_features = output_conv([in_features, top_down_features, out_features])
            bottom_up.append(prev_features)
        bottom_up.append(self.output_convs[-1]([in_list[-1], top_down[-1]]))

        return dict(zip(self._out_features, bottom_up))