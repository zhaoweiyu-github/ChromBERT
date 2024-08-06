import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss


class FocalLoss(nn.Module):
    """Multi-class Focal loss implementation.
    Args:
        gamma (float): The larger the gamma, the smaller
            the loss weight of easier samples.
        alpha (float): Weighting factor in range (0,1) to balance
            positive vs negative examples or -1 for ignore. Default: ``-1``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
            ``'none'``: No reduction will be applied to the output.
            ``'mean'``: The output will be averaged.
            ``'sum'``: The output will be summed. Default: ``'none'``.
    """

    def __init__(self, gamma = 2, alpha = -1, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        loss = sigmoid_focal_loss(inputs=input, targets=target, gamma=self.gamma, alpha=self.alpha, reduction=self.reduction)
        return loss
    
class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss(reduction = 'none')

    def forward(self, y_logit, y_true):
        loss = self.loss(y_logit, y_true)
        loss = torch.sqrt(loss.mean())
        return loss
    
class ZeroInflationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logit, target):
        zero_prob_logit, reg_value = logit
        changes_target = (target != 0).float() 
        zero_mask = (target == 0).float()  
        reg_mask = (target != 0).float()
        zero_loss = F.binary_cross_entropy_with_logits(zero_prob_logit, changes_target,reduction='none')
        zero_loss = (zero_loss * zero_mask).sum() / (zero_mask.sum() + 1e-10)
        mae_loss = torch.abs(reg_value - target) * reg_mask
        mae_loss = mae_loss.sum() / (reg_mask.sum() + 1e-10)
        total_loss = zero_loss + mae_loss
        return total_loss