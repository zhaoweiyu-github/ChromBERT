import torch.nn as nn
import torch


class TokenEmbedding(nn.Embedding):
    def __init__(self, config):
        super().__init__(config.vocab_size, config.hidden_dim, config.token_id_pad)


class PositionalEmbeddingTrainable(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pe = nn.Embedding(config.n_datasets, config.hidden_dim)
        self.d_model = config.hidden_dim
        self.n_datasets = config.n_datasets

    def forward(self, x):
        return self.pe(x)

class PositionalEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.pe_mode == "train":
            self.pe = PositionalEmbeddingTrainable(config)
        else: 
            raise ValueError(f"only support train mode for positional embedding! {config.pe_mode} is not supported!")

    def forward(self, x):
        return self.pe(x)


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, config):
        """
        :param config.vocab_size: total vocab size
        :param config.hidden_dim: embedding size of token embedding
        :param config.dropout_prob: dropout rate 
        :param config.pe_mode: train or word2vec, for positional embedding choice
        :param config.pkl_embeding: path to pkl file of word2vec embedding
        :param config.dtype
        """
        super().__init__()
        self.token = TokenEmbedding(config)
        self.position = PositionalEmbedding(config)

        self.dropout = nn.Dropout(p=config.dropout_prob)
        self.embed_size = config.hidden_dim
        self.dtype = config.dtype
        self.config = config

    def forward(self, sequence, position_ids):
        sequence = sequence.long()
        x = self.token(sequence) + self.position(position_ids)
        return self.dropout(x).to(self.dtype)

