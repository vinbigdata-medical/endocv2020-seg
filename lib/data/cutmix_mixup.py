import numpy as np
import torch


def mixup_data(input, target=None, alpha=0.4):
    """
    Return mixed input, and/or pairs of target, lambda.
    """
    bsize, _, _, _ = input.shape
    shuffled_idx = torch.randperm(bsize).cuda()
    if target is None:
        lamb = np.random.beta(alpha + 1., alpha)
    else:
        lamb = np.random.beta(alpha, alpha)
    input = lamb * input + (1 - lamb) * input[shuffled_idx]
    if target is None:
        return input
    else:
        # Classification
        # return input, target, target[shuffled_idx], lamb
        # Segmentation
        target = lamb * target + (1 - lamb) * target[shuffled_idx]
        return input, target


def cutmix_data(input, target=None, alpha=1.0):
    """
    Return cut-mixed input, and/or pairs of target, lambda.
    """
    bsize, _, h, w = input.shape
    shuffled_idx = torch.randperm(bsize).cuda()

    input_s = input[shuffled_idx]
    if target is None:
        lamb = np.random.beta(alpha + 1., alpha)
    else:
        lamb = np.random.beta(alpha, alpha)
    rx = np.random.randint(w)
    ry = np.random.randint(h)
    cut_ratio = np.sqrt(1. - lamb)
    rw = np.int(cut_ratio * w)
    rh = np.int(cut_ratio * h)

    x1 = np.clip(rx - rw // 2, 0, w)
    x2 = np.clip(rx + rw // 2, 0, w)
    y1 = np.clip(ry - rh // 2, 0, h)
    y2 = np.clip(ry + rh // 2, 0, h)

    input[:, :, x1:x2, y1:y2] = input_s[:, :, x1:x2, y1:y2]
    if target is None:
        return input
    else:
        # Classification
        # lamb = 1 - ((x2 - x1) * (y2 - y1) / (h * w)) # adjust lambda to exactly match pixel ratio
        # return input, target, target[shuffled_idx], lamb
        # Segmentation
        target_s = target[shuffled_idx]
        target[:, :, x1:x2, y1:y2] = target_s[:, :, x1:x2, y1:y2]
        return input, target


def mixup_criterion(criterion, output, target, target_s, lamb):
    """
    Compute mixed loss
    """
    return lamb * criterion(output, target) + (1 - lamb) * criterion(output, target_s)