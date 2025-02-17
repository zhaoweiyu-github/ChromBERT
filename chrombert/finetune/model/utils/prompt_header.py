import torch
from torch import nn
from .residual_block import ResidualBlock

class PromptHeader(nn.Module):
    def __init__(self, n_parts = 3,dropout = 0.1):
        super().__init__()
        self.fcs = nn.Sequential(
            ResidualBlock(n_parts * 768, n_parts * 768, dropout = dropout),
            ResidualBlock(n_parts * 768, 768, dropout = dropout),
            ResidualBlock(768, 768, dropout = dropout),
            ResidualBlock(768, 64, dropout = dropout),
            nn.Linear(64, 1),
        )
    def forward(self, *args):
        for arg in args:
            assert isinstance(arg, torch.Tensor)

        full_emb = torch.cat(args, dim = -1)
        logit = self.fcs(full_emb).squeeze(-1)
        assert len(logit.shape) == 1
        return logit   
        
        
class AdapterExternalEmb(nn.Module):
    def __init__(self, prompt_dim_external, dropout = 0.1):
        super().__init__()
        dim1 = prompt_dim_external
        dim2 = 768
        dropout = dropout
        self.fc1 = ResidualBlock(dim1, dim2, dropout = dropout)
        self.fc2 = ResidualBlock(dim2, dim2, dropout = dropout)
    
    def forward(self, x):
        # x = x.bfloat16()
        x = x.to(self.fc1.fc1.weight.dtype)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
        
class Pooling(nn.Module):
    def __init__(self, operation):
        super().__init__()
        
        if operation in ["mean", "max"]:
            self.operation = operation
        else:
            raise ValueError(f"operation must be one of ['mean', 'max'], but got {operation}")

    def forward(self, x):
        if self.operation == "mean":
            return torch.mean(x, dim=1)
        elif self.operation == "max":
            # torch.max returns both values and indices, we only need the values
            return torch.max(x, dim=1).values

class PromptsEmb(nn.Module):
    def __init__(self):
        super().__init__()
        self.pooling = Pooling('mean')
    def forward(self,x,prompts):
        prompts = prompts.unsqueeze(2)
        emb_sum = x.mul(prompts).sum(dim=1)
        emb_count = prompts.sum(dim=1)
        emb = emb_sum/emb_count
        return emb