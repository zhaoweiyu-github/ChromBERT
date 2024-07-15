from torch import nn
from .residual_block import ResidualBlock
from .emb_manager import CistromeEmbeddingManager


class GeneralHeader(nn.Module):
    """
    general
    """

    def __init__(self, hidden_dim, dim_output, mtx_mask, ignore=False,ignore_index=None,medium_dim = 256):
        """
        :param hidden: output size of BERT model
        :param dim_output: number of class
        """
        super().__init__()
        self.interface = CistromeEmbeddingManager(mtx_mask = mtx_mask,
                                                  ignore = ignore,
                                                ignore_index = ignore_index)
        self.conv = nn.Conv2d(1, 1, (1, hidden_dim))
        self.activation = nn.ReLU()
        self.res1 = ResidualBlock(in_features=self.interface.normalized_mtx_mask.shape[1], out_features=1024)
        self.res2 = ResidualBlock(in_features=1024, out_features=hidden_dim)
        self.res3 = ResidualBlock(in_features=hidden_dim, out_features=medium_dim)
        self.fc = nn.Linear(in_features=medium_dim, out_features=dim_output, bias=True)


    def forward(self, x, return_emb = False):
        x = self.interface(x) # [batch_size, factors, hidden]
        x = x.permute(0, 2, 1) # [batch_size, hidden, factors]
        x = self.res1(x) # [batch_size, hidden, 1024]
        x = self.res2(x) # [batch_size, hidden, 768]
        x = x[:,None,:,:]  # [batch_size, 1, hidden, 768]
        x = self.conv(x)  # [batch_size, 1, hidden, 1]
        x = self.activation(x) 
        x = x.view(x.shape[0],-1) # [batch_size, hidden] 
        if return_emb:
            return x

        x = self.res3(x)
        x = self.fc(x)

        return x