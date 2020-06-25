from .build import build_backbone, BACKBONE_REGISTRY

from .efficientnet import EfficientNet, build_efficientnet_backbone
from .resnet import ResNet, build_resnet_backbone
from .fpn import FPN, BiFPN