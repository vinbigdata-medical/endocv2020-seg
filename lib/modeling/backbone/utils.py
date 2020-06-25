import torch.nn as nn


def get_norm(norm, out_channels):
    """
    Args:
        norm (str or callable):
    Returns:
        nn.Module or None: normalization layer
    """
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": nn.BatchNorm2d,
            "GN": lambda channels: nn.GroupNorm(32, channels)
        }[norm]
    return norm(out_channels)


def get_act(act, inplace=True):
    """
    Args:
        act (str or callable):
    Returns:
        nn.Module or None: activation layer
    """
    act = {
        "relu": nn.ReLU,
        "leaky_relu": lambda inplace: nn.LeakyReLU(0.01, inplace)
    }[act]
    return act(inplace)