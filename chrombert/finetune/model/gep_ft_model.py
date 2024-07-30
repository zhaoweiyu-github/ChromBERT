import os,sys
import torch
import torch.nn as nn
from chrombert import ChromBERTConfig
from .utils import PoolFlankWindow,GepHeader
from .basic_model import BasicModel
from .utils import ChromBERTEmbedding
class ChromBERTGEP(BasicModel):
    """
    Finetune pre-trained ChromBERT to perform gene expression prediction. 
    Attrs:
        pretrain_config: pretrain configuration
        finetune_config: finetune configuration
        pretrain_model: basic chrombert model
        pool_flank_window:[PoolFlankWindow]: layers to pool the flank window
        ft_header:[GepHeader]: supervised header 
    """

    def create_layers(self):
        """add a supervised header to fine-tune model"""

        pretrain_model = self.pretrain_config.init_model()
        self.flank_region_num = int(self.finetune_config.gep_flank_window) * 2 + 1 
        self.pool_flank_window = PoolFlankWindow(
                flank_region_num = self.flank_region_num,
                pretrain_model = pretrain_model,
                parallel_embedding = self.finetune_config.gep_parallel_embedding,
                gradient_checkpoint=self.finetune_config.gep_gradient_checkpoint
            )

        self.ft_header = GepHeader(
            self.pretrain_config.hidden_dim, 
            self.finetune_config.dim_output, 
            self.finetune_config.mtx_mask, 
            self.finetune_config.ignore,
            self.finetune_config.ignore_index,
            self.finetune_config.dropout
            )
        return None

    def forward(self,batch):
        input_ids = batch["input_ids"]
        position_ids = batch["position_ids"]
        pool_out= self.pool_flank_window.forward(
            input_ids, position_ids)
        header_out = self.ft_header(pool_out)
        return header_out
    
    
    def get_embedding_manager(self, **kwargs):
        '''
        get a embedding manager for the pretrain model.
        params:
            kwargs: additional parameters for EmbManager
        '''
        pretrain_model = self.get_pretrain()
        finetune_config = self.finetune_config.clone()
        finetune_config.update(**kwargs)
        PoolFlankWindow_model = PoolFlankWindow(
            flank_region_num = int(finetune_config.gep_flank_window) * 2 + 1, 
            pretrain_model = pretrain_model,
            parallel_embedding = finetune_config.gep_parallel_embedding,
            gradient_checkpoint = finetune_config.gep_gradient_checkpoint
        )
        model_emb = ChromBERTEmbedding(PoolFlankWindow_model, finetune_config.mtx_mask, finetune_config.ignore, finetune_config.ignore_index)
        return model_emb 
