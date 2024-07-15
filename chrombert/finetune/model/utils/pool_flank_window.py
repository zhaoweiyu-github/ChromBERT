import torch 
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint

class PoolFlankWindow(nn.Module):
    """
    When using multiple flanking windows for pooling
    """
    def  __init__(self, flank_region_num=9,pretrain_model=None,parallel_embedding=False,gradient_checkpoint=False):
        super().__init__()
        self.flank_region_num = flank_region_num
        self.pretrain_model = pretrain_model      
        self.parallel_embedding = parallel_embedding
        self.gradient_checkpoint = gradient_checkpoint 
        
    def forward(self,x,position_ids):   
        batch_size = x.shape[0]
        seq_len = x.shape[-1]
        x = x.float()
        x.requires_grad = True
        if not self.parallel_embedding:
            embeddings = []
            if self.gradient_checkpoint == 0:
                for i in range(self.flank_region_num):
                    x_i = x[:,i,:].clone()
                    position_ids_i = position_ids[:,i,:].clone() 
                    x_i = self.pretrain_model(x_i, position_ids_i)
                    embeddings.append(x_i)
                x = torch.stack(embeddings, dim = 1)
            elif self.gradient_checkpoint == 1:
                all_embeddings = torch.zeros((batch_size, self.flank_region_num, seq_len, 768), device=x.device)
                for i in range(self.flank_region_num):
                    all_embeddings[:, i, :, :] = checkpoint(
                        self.pretrain_model, x[:, i, :], position_ids[:, i, :]
                    )
                x = all_embeddings
        else: 
            x = rearrange(x, 'b n l -> (b n) l')
            position_ids = rearrange(position_ids, 'b n l -> (b n) l') 
            x = self.pretrain_model(x, position_ids) 
            x = rearrange(x, '(b n) l h -> b n l h', b = batch_size)

        x = rearrange(x, 'b n l h -> b l n h', b = batch_size)
        x = torch.max(x, dim=-2).values #[b,l,h]
        return x        

