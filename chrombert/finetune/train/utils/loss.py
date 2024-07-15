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
        zero_target_indices = torch.nonzero(target == 0).squeeze()
        if zero_target_indices.numel() == 0:
            zero_loss = torch.tensor(0.0, device=zero_prob_logit.device)
            zero_target = torch.tensor([], device=target.device)
        else:
            zero_target = target[zero_target_indices].squeeze() 
            zero_prob_logit = zero_prob_logit[zero_target_indices].squeeze()
            zero_loss = F.binary_cross_entropy_with_logits(zero_prob_logit, zero_target) 
        reg_target_indices = torch.nonzero(target).squeeze()
        reg_target = target[reg_target_indices].squeeze()
        reg_value = reg_value[reg_target_indices].squeeze()
        mae_loss = torch.abs(reg_value - reg_target).sum() / (reg_target.numel() + 1e-10)
        return zero_loss + mae_loss