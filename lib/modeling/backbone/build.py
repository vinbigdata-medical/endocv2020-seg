from cvcore.utils import Registry


BACKBONE_REGISTRY = Registry("BACKBONE")
BACKBONE_REGISTRY.__doc__ = """
Registry for backbones, which extract feature maps from images

The registered object must be a callable that accepts one argument:

1. A :class:`yacs.config.CfgNode`

It must returns an instance of :class:`nn.Module`.
"""


def build_backbone(cfg):
    """
    Build a backbone from `cfg.MODEL.BACKBONE.NAME`.

    Returns:
        an instance of :class:`nn.Module`
    """
    backbone_name = cfg.MODEL.BACKBONE.FUNC_NAME
    backbone = BACKBONE_REGISTRY.get(backbone_name)(cfg)
    return backbone
