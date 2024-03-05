import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Multi-class Focal loss implementation.
    Args:
        gamma (float): The larger the gamma, the smaller
            the loss weight of easier samples.
        weight (float): A manual rescaling weight given to each
            class.
        ignore_index (int): Specifies a target value that is ignored
            and does not contribute to the input gradient.
    """

    def __init__(self, gamma = 2, vocab_size = 10, vocab_shift = 5, alpha = torch.tensor([1,1,1,1,1]), ignore_index = -100):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        
        alphas = torch.ones(vocab_size)
        alphas[vocab_shift:] = torch.tensor(alpha)
        self.weight = alphas / alphas.sum() * len(alphas)
        self.weight.requires_grad = False

    def __repr__(self):
        tmpstr = f"{self.__class__.__name__ }:\n\tgamma:{self.gamma}\n\talpha:{self.weight}\n\tignore_index:{self.ignore_index}"
        return tmpstr

    def forward(self, input, target):
        logit = F.log_softmax(input, dim = 1)
        pt = torch.exp(logit)
        logit = (1 - pt) ** self.gamma * logit
        logit = logit * self.weight.to(target.device)

        loss = F.nll_loss(
                logit, target, ignore_index = self.ignore_index
            )

        return loss
