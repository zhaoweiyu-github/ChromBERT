import os,sys
import torch
import torch.nn as nn
import lightning.pytorch as pl
from chrombert import ChromBERTConfig
from .utils import GeneralHeader
from .basic_model import BasicModel

class ChromBERTGeneral(BasicModel):
    '''
    Attrs:
        pretrain_config: pretrain configuration
        finetune_config: finetune configuration
        pretrain_model: basic chrombert model
        ft_header: supervised header (GeneralHeader)
    '''

    def create_layers(self):
        """
        add a supervised header to fine-tune model.
        """
        self.pretrain_model = self.pretrain_config.init_model()

        self.ft_header = GeneralHeader(
            self.pretrain_config.hidden_dim, 
            self.finetune_config.dim_output, 
            self.finetune_config.mtx_mask, 
            self.finetune_config.ignore,
            self.finetune_config.ignore_index,
            )
        return None

    def forward(self, batch):
        input_ids = batch["input_ids"]
        position_ids = batch["position_ids"]
        chrombert_out= self.pretrain_model.forward(
            input_ids.long(), position_ids)
        header_out = self.ft_header(chrombert_out)
        return header_out 
