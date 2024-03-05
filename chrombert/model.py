import torch.nn as nn
import lightning.pytorch as pl
from .utils import BERTEmbedding
from .utils import EncoderTransformerBlock, DecoderTransformerBlock
from .utils import DECODER

from typing import List

class ChromBERT(nn.Module):
    """
    ChromBERT model : Epigenetic Network Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, config):
        """
        :param config.vocab_size: vocab_size of total words
        :param config.hidden_dim: BERT model hidden size
        :param config.n_layers: numbers of Transformer blocks(layers)
        :param config.num_attention_heads: number of attention heads
        :param config.dropout_prob: dropout rate
        :param config.dna_embedding: whether contain DNA embedding in the embedding layer

        """

        super().__init__()
        self.hidden = config.hidden_dim
        self.n_layers = config.num_layers
        self.attn_heads = config.num_attention_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = config.hidden_dim * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(config)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [EncoderTransformerBlock(config) for _ in range(self.n_layers)])


    def forward(self, x, position_ids, key_padding_mask = None, attn_weight = False, attn_layer = None):
        # attention masking for padded token
        x = self.embedding(x, position_ids)

        if attn_layer == -1:
            attn = []
        # running over multiple transformer blocks
        for i,transformer in enumerate(self.transformer_blocks):
            if attn_weight: 
                if attn_layer == -1:
                    x, attn_score = transformer.forward(x, key_padding_mask, attn_weight = True, )
                    attn.append(attn_score)
                elif i == attn_layer:
                    x, attn = transformer.forward(x, key_padding_mask, attn_weight = attn_weight, )
                    # attn.append(attn)
                else:
                    x = transformer.forward(x, key_padding_mask, attn_weight = False,)
            else:
                x = transformer.forward(x, key_padding_mask)
      
        # return outs
        return (x, attn) if attn_weight else x

class ChromBERTLM(nn.Module):
    """
    BERT Language Model
    Masked Language Model
    """

    def __init__(self, config):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.config = config
        self.decoder_header = config.decoder_header
        self.bert = ChromBERT(config)
        self.decoder = DECODER(config)

        self.seqstart = 0

    def forward(self, x, position_ids, key_padding_mask = None, attn_weight = False, attn_layer = None, **kwargs):
        
        if attn_weight:
            x, attn_scores = self.bert(x, position_ids, key_padding_mask,  attn_weight = attn_weight, attn_layer = attn_layer)
        else: 
            x = self.bert(x, position_ids, key_padding_mask)

        x = x[:,self.seqstart:,:] # (batch_size, seq_len, hidden_dim)
        y = self.decoder(x, key_padding_mask = key_padding_mask, **kwargs)

        if attn_weight: 
            out = y, attn_scores
        else:
            out = y
        return out

