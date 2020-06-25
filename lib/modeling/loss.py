import torch
import torch.nn as nn
import torch.nn.functional as F


class KnowledgeDistillationLoss(nn.Module):
    """
    Reference: https://nervanasystems.github.io/distiller/knowledge_distillation.html.

    Args:
        temperature (float): Temperature value used when calculating soft targets and logits.
    """
    def __init__(self, temperature=4.):
        super(KnowledgeDistillationLoss, self).__init__()
        self.temperature = temperature

    def forward(self, student_logit, teacher_logit):
        student_prob = F.softmax(student_logit, dim=-1)
        teacher_prob = F.softmax(teacher_prob, dim=-1).log()
        loss = F.kl_div(teacher_prob, student_prob, reduction="batchmean")
        return loss


class JSDCrossEntropyLoss(nn.Module):
    """ Jensen-Shannon divergence + Cross-entropy loss
    """
    def __init__(self, num_splits=3, alpha=12, clean_target_loss=nn.CrossEntropyLoss()):
        super(JSDCrossEntropyLoss, self).__init__()
        self.num_splits = num_splits
        self.alpha = alpha
        self.cross_entropy_loss = clean_target_loss

    def forward(self, logit, target):
        split_size = logit.shape[0] // self.num_splits
        assert split_size * self.num_splits == logit.shape[0]
        logits_split = torch.split(logit, split_size)
        # Cross-entropy is only computed on clean images
        loss = self.cross_entropy_loss(logits_split[0], target[:split_size])
        probs = [F.softmax(logits, dim=1) for logits in logits_split]
        logp_mixture = torch.clamp(torch.stack(probs).mean(0), 1e-7, 1.).log()
        loss += self.alpha * sum([F.kl_div(
            logp_mixture, p_split, reduction="batchmean") for p_split in probs]) / len(probs)
        return loss


class SigmoidFocalLoss(nn.Module):
    """
    Compute focal loss from
    `'Focal Loss for Dense Object Detection' (https://arxiv.org/pdf/1708.02002.pdf)`.

    Args:
        gamma (float): (default=2.).
        alpha (float): (default=0.25).
    """
    def __init__(self, gamma=2., alpha=0.25):
        super(SigmoidFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logit, target):
        ce_loss = F.binary_cross_entropy_with_logits(
            logit, target, reduction='none')
        p_t = torch.exp(-ce_loss)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        modulate = ((1 - p_t) ** self.gamma)
        loss = alpha_t * ce_loss * modulate
        foreground_idxs = target > 0. # TODO: add ignore classes idxs
        num_foreground = foreground_idxs.sum()
        return loss.sum() / max(1, num_foreground)


class SoftmaxFocalLoss(nn.Module):
    """
    Compute the softmax version of focal loss.
    Loss value is normalized by sum of modulating factors.

    Args:
        gamma (float): (default=2.).
    """
    def __init__(self, gamma=2.):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, logit, target):
        ce_loss = F.cross_entropy(
            logit, target, reduction="none")
        p_t = torch.exp(-ce_loss)
        modulate = ((1 - p_t) ** self.gamma)
        loss = modulate * ce_loss / modulate.sum()
        return loss.sum()


class LabelSmoothingCrossEntropyLoss(nn.Module):
    """
    Negative log-likelihood loss with label smoothing.

    Args:
        smoothing (float): label smoothing factor (default=0.1).
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, logit, target):
        logprobs = F.log_softmax(logit, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def fbeta_loss(logit, target, beta=1.):
    prob = torch.sigmoid(logit)
    tp = (prob * target).sum((-1, -2))
    fp = (prob * (1 - target)).sum((-1, -2))
    fn = ((1 - prob) * target).sum((-1, -2))
    beta_sq = beta ** 2
    fbeta = ((1 + beta_sq) * tp) / ((1 + beta_sq) * tp + beta_sq * fn + fp)
    return (1 - fbeta).mean()


def binary_dice_loss(logit, target):
    prob = torch.sigmoid(logit)
    intersection = (prob * target).sum((-1, -2))
    dice = (2. * intersection) / (prob.sum((-1, -2)) + target.sum((-1, -2)) + 1e-6)
    return (1. - dice).mean()


def binary_dice_metric(logit, target, threshold=0.5):
    if isinstance(threshold, (tuple, list)):
        pred = torch.stack([
            torch.sigmoid(logit[:, i, ...]) > th for i, th in enumerate(threshold)
        ], 1)
        pred = pred.float()
    else:
        pred = (torch.sigmoid(logit) > threshold).float()

    intersection = (pred * target).sum((-1, -2))
    denominator = pred.sum((-1, -2)) + target.sum((-1, -2))
    dice = (2. * intersection) / (denominator + 1e-6)
    # print(dice)
    dice = torch.where(denominator == 0.,
        torch.FloatTensor([1.]).to(denominator.device), dice)
    # print(dice)
    return dice


class BinaryDiceLoss(nn.Module):
    def forward(self, logit, target):
        return binary_dice_loss(logit, target)


def binary_iou_metric(logit, target, threshold=0.5):
    if isinstance(threshold, (tuple, list)):
        pred = torch.stack([
            torch.sigmoid(logit[:, i, ...]) > th for i, th in enumerate(threshold)
        ], 1)
        pred = pred.float()
    else:
        pred = (torch.sigmoid(logit) > threshold).float()

    intersection = (pred * target).sum((-1, -2))
    denominator = pred.sum((-1, -2)) + target.sum((-1, -2))
    union = denominator - intersection + 1e-6
    iou = intersection / union
    iou = torch.where(denominator == 0.,
        torch.FloatTensor([1.]).to(denominator.device), iou)
    return iou


def binary_iou_loss(logit, target):
    prob = torch.sigmoid(logit)
    intersection = (prob * target).sum((-1, -2))
    union = prob.sum((-1, -2)) + target.sum((-1, -2)) - intersection + 1e-6
    iou = intersection / union
    return (1. - iou).mean()


class BinaryIoULoss(nn.Module):
    def forward(self, logit, target):
        return binary_iou_loss(logit, target)