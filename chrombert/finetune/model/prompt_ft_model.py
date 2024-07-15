import os,sys
import torch
import torch.nn as nn
import lightning.pytorch as pl

from chrombert import ChromBERTConfig
from .utils import PromptHeader,PromptsEmb,AdapterExternalEmb
from .basic_model import BasicModel

class ChromBERTPrompt(BasicModel):

    def create_layers(self):
        """add a supervised header to fine-tune model"""
        self.pretrain_model = self.pretrain_config.init_model()

        if self.finetune_config.prompt_kind == 'expression':
            self.adapter_cell_emb = AdapterExternalEmb(
                prompt_dim_external = self.finetune_config.prompt_dim_external,
                dropout=self.finetune_config.dropout
            )

        self.gather_emb = PromptsEmb()
        self.ft_header = PromptHeader(n_parts = self.finetune_config.n_prompt_parts + 1)
        return None 

    def forward(self,batch):

        emb_cell,emb_regulator,emb_all = self.get_emb_parts(batch)
        header_out = self.ft_header(emb_cell,emb_regulator,emb_all)
        
        return header_out 

    def get_emb_parts(self,batch):    
        input_ids = batch["input_ids"]
        position_ids = batch["position_ids"]
        if 'emb_cell' not in batch.keys() or 'emb_regulator' not in batch.keys():
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
        
        return emb_cell, emb_regulator, emb_all
