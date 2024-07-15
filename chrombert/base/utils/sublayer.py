import torch.nn as nn
from .layer_norm import LayerNorm


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer, index=None):
        "Apply residual connection to any sublayer with the same size."
        y = sublayer(self.norm(x))
        if index is not None:
            y1 = y[index]
            o = x + self.dropout(y1), y[-1] # pick attention_weights
        else:
            o = x + self.dropout(y)
        return o
