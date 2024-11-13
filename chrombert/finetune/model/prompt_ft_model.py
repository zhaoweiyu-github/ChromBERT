import os,sys
import torch
import torch.nn as nn
import lightning.pytorch as pl

from chrombert import ChromBERTConfig
from .utils import PromptHeader,PromptsEmb,AdapterExternalEmb
from .basic_model import BasicModel

class ChromBERTPrompt(BasicModel):
    '''
    Fine-tuning a pre-trained ChromBERT model using enhanced prompt, for TFBS prediction.
    
    pretrain_config: ChromBERTConfig object
    finetune_config: FinetuneConfig

    The model will be initialized using the following steps:
        self.pretrain_config = pretrain_config
        self.finetune_config = finetune_config
        self.create_layers() 
    '''

    def create_layers(self):
        """add a supervised header to fine-tune model"""
        self.pretrain_model = self.pretrain_config.init_model()

        if self.finetune_config.prompt_kind == 'expression':
            self.adapter_cell_emb = AdapterExternalEmb(
                prompt_dim_external = self.finetune_config.prompt_dim_external,
                dropout=self.finetune_config.dropout
            )

        self.gather_emb = PromptsEmb() # for gather regulator and cell prompt

        self.ft_header = PromptHeader(n_parts = self.finetune_config.n_prompt_parts + 1,
                                      dropout=self.finetune_config.dropout)
        return None 

    def forward(self,batch):

        emb_cell,emb_regulator,emb_all = self.get_emb_parts(batch, dtype = self.ft_header.fcs[0].fc1.weight.dtype)
        header_out = self.ft_header(emb_cell,emb_regulator,emb_all)
        
        return header_out 

    def get_emb_parts(self,batch, dtype =torch.bfloat16):  
        '''
        Gather the necessary inputs for forwarding, handling cached embedding or forwarding directly.
        '''  

        if 'emb_cell' not in batch.keys() or 'emb_regulator' not in batch.keys():
            input_ids = batch["input_ids"]
            position_ids = batch["position_ids"]
            chrombert_out= self.pretrain_model.forward(
            input_ids.long(), position_ids
            )
        
        if 'emb_cell' in batch.keys():
            emb_cell = batch["emb_cell"]
        else:
            prompts_cell = batch["prompts_cell"]
            emb_cell =  self.gather_emb(chrombert_out,prompts_cell)
            
        if 'emb_regulator' in batch.keys():
            emb_regulator = batch["emb_regulator"]
            emb_all = batch["emb_all"]
        else:
            prompts_all = batch["prompts_all"]
            prompts_regulator = batch["prompts_regulator"]
            emb_regulator =  self.gather_emb(chrombert_out,prompts_regulator)
            emb_all =  self.gather_emb(chrombert_out,prompts_all)
        
        if self.finetune_config.prompt_kind == 'expression':
            emb_cell = self.adapter_cell_emb(emb_cell)
        
        return emb_cell.to(dtype), emb_regulator.to(dtype), emb_all.to(dtype)
