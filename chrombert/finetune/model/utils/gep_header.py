from torch import nn
from .residual_block import ResidualBlock
from .emb_manager import CistromeEmbeddingManager

class GepHeader(nn.Module):
    """
    predicting gene expression changes
    """

    def __init__(self, hidden_dim, dim_output, mtx_mask,ignore=False,ignore_index=None,dropout=0.1,medium_dim = 256):
        """
        :param hidden: output size of BERT model
        :param dim_output: number of class
        """
        super().__init__()
        self.interface = CistromeEmbeddingManager(mtx_mask = mtx_mask,
                                        ignore=ignore,
                                        ignore_index=ignore_index)
        self.conv = nn.Conv2d(1, 1, (1, hidden_dim))
        self.activation = nn.ReLU()
        self.res1 = ResidualBlock(in_features=self.interface.normalized_mtx_mask.shape[1], out_features=1024,dropout=dropout)
        self.res2 = ResidualBlock(in_features=1024, out_features=hidden_dim,dropout=dropout)
        self.res3 = ResidualBlock(in_features=hidden_dim, out_features=medium_dim,dropout=dropout)
        self.zero_inflation = nn.Sequential(
			nn.Linear(in_features=medium_dim, out_features=1),
		)
        self.regression  = nn.Linear(in_features=medium_dim, out_features=1, bias=True)

    def forward(self, x, **kwargs):
        x = self.interface(x)
        x = x.permute(0, 2, 1)
        x = self.res1(x)
        x = self.res2(x)

        x = x[:,None,:,:] 
        x = self.conv(x) 
        x = self.activation(x)
        x = x.view(x.shape[0],-1) 
        x = self.res3(x)
        zero_prob_logit = self.zero_inflation(x)
        reg_value = self.regression(x)
        return zero_prob_logit, reg_value