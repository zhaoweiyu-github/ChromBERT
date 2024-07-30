import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import flash_attn

if flash_attn.__version__.split(".")[0] == "1":
    from flash_attn.flash_attention import FlashAttention
    print("flash attention version 2 is not installed, using version 1 instead")
    flash_attention_version = 1
else:
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
    flash_attention_version = 2


from functools import partial
from einops import rearrange
from .feed_forward import PositionwiseFeedForward
from .sublayer import SublayerConnection


class SelfAttentionFlashMHA(nn.Module):

    def __init__(self, config) -> None:
        assert config.flash_batch_first
        factory_kwargs = {'device': config.flash_device, 'dtype': config.dtype}
        super().__init__()
        self.embed_dim = config.hidden_dim
        self.causal = config.flash_causal
        self.num_heads = config.num_attention_heads
        assert self.embed_dim % self.num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // self.num_heads
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"

        self.Wqkv = nn.Linear(self.embed_dim, 3 *self.embed_dim, bias=config.flash_bias, **factory_kwargs)
        if flash_attention_version == 1:
            f = FlashAttention(attention_dropout=config.dropout)
            self.inner_attn = f
        else:
            self.inner_attn = partial(flash_attn_qkvpacked_func, dropout_p = config.dropout, causal=self.causal)

        self.dtype = config.dtype

    def forward(self, x, key_padding_mask=None, need_weights=False, attn_weight = False):
        """x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        key_padding_mask: bool tensor of shape (batch, seqlen)
        """
        x = x.to(self.dtype)
        qkv = self.Wqkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.num_heads)

        if flash_attention_version == 2:
            context = self.inner_attn(qkv)
        else:
            context,_ = self.inner_attn(qkv, key_padding_mask=key_padding_mask,
                                                need_weights=need_weights, causal=self.causal)
        if attn_weight :
            with torch.no_grad():
                qkvhp = qkv.permute(0,2,3,1,4)
                q,k,v = qkvhp[:,0,:,:,:], qkvhp[:,1,:,:,:], qkvhp[:,2,:,:,:]
                attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1))/ math.sqrt(q.shape[-1]),dim = -1).detach().cpu()
                # attn = attn.sum(axis = -2).squeeze(0) # key dim sum
        else:
            attn = None

        return (rearrange(context, 'b s h d -> b s (h d)'), attn) if attn_weight else rearrange(context, 'b s h d -> b s (h d)')


class EncoderTransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """
    def __init__(self, config):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = SelfAttentionFlashMHA(config)
        self.feed_forward = PositionwiseFeedForward(d_model=config.hidden_dim, d_ff=config.feed_forward_dim, dropout=config.dropout)
        self.input_sublayer = SublayerConnection(size=config.hidden_dim, dropout=config.dropout)
        self.output_sublayer = SublayerConnection(size=config.hidden_dim, dropout=config.dropout)
        self.dropout = nn.Dropout(p=config.dropout)

        self.dtype = config.dtype

    def forward(self, x, mask, attn_weight = False):
        x = x.to(self.dtype)
        if attn_weight :
                x, out_attn = self.input_sublayer(x, lambda x: self.attention.forward(x, mask, need_weights=False, attn_weight=attn_weight), index = 0) # get context and attention_score
        else:
            x = self.input_sublayer(x, lambda x: self.attention.forward(x, mask))

        x = self.output_sublayer(x, self.feed_forward)

        if attn_weight:
            out = x, out_attn
        else:
            out = x
        # return out 
        return out 





    

