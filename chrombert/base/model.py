import torch 
import torch.nn as nn
import lightning.pytorch as pl
from .utils import BERTEmbedding
from .utils import EncoderTransformerBlock
class ChromBERT(nn.Module):
    def __init__(self, config):
        """
        ChromBERT model : Epigenetic Network Bidirectional Encoder Representations from Transformers.
        Args: 
            config (:obj:`ChromBERTConfig`): configuration of the model. 
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

    def load_ckpt(self, ckpt_path):
        ck = torch.load(ckpt_path, map_location=torch.device('cpu'))
        self.load_state_dict(ck)
        return None

    def freeze(self, trainable = 2):
        '''
        freeze parameters of the model, except for the last free layers. 
        if trainable = 0, freeze all layers. if trainable < 0, freeze none. 
        Noting: if trainable >=0, embedding layer will be frozen. 
        '''
        assert isinstance(trainable, int), 'trainable should be an integer'
        if trainable >= 0:
            for name, parameter in self.named_parameters():
                parameter.requires_grad = False

            total_layers = len(self.transformer_blocks)
            for i in range(total_layers - trainable, total_layers):
                for name, parameter in self.transformer_blocks[i].named_parameters():
                    parameter.requires_grad = True


        if trainable < 0:
            for name, parameter in self.named_parameters():
                parameter.requires_grad = True

        return None

    def display_trainable_parameters(self, verbose = True):
        '''
        display the number of trainable parameters in the model
        '''
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        o = {"total_params": total_params, "trainable_params": trainable_params}
        print(o)
        if verbose:
            for name, parameter in self.named_parameters():
                if parameter.requires_grad:
                    print(name, ": trainable")
                else:
                    print(name, ": frozen")
        return o 
