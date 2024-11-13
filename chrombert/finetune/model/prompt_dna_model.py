import os
import torch
from torch import nn

from chrombert.base import ChromBERTConfig
from .utils.dnabert2 import DNABERT2Interface
from .utils.prompt_header import AdapterExternalEmb, PromptHeader
from .utils.general_header import GeneralHeader 
from .basic_model import BasicModel

class ChromBERTPromptDNA(BasicModel):
    '''
    Fine-tuning a pre-trained ChromBERT model using DNA-enhanced prompt.
    
    pretrain_config: ChromBERTConfig object
    finetune_config: FinetuneConfig

    The model will be initialized using the following steps:
        self.pretrain_config = pretrain_config
        self.finetune_config = finetune_config
        self.create_layers() 
    '''

    NECESSARY_KEYS = ["input_ids", "position_ids","seq_raw", "seq_alt"]

    def create_layers(self):
        """add a supervised header to fine-tune model"""
        assert self.finetune_config.prompt_kind in ["dna", "sequence"], "prompt_kind must be dna or sequence here!"
        assert self.finetune_config.prompt_dim_external == 768, "prompt_dim_external must be 768 here, only DNABERT2 supported now!"

        self.pretrain_model = self.pretrain_config.init_model()
        self.dnabert2 = DNABERT2Interface(self.finetune_config.dnabert2_ckpt, pooling="mean")
        self.adapter_dna_emb = AdapterExternalEmb(self.finetune_config.prompt_dim_external, 
                                                  dropout = self.finetune_config.dropout)

        self.adapter_chrombert = GeneralHeader(
            self.pretrain_config.hidden_dim, 
            self.finetune_config.dim_output, 
            self.finetune_config.mtx_mask, 
            self.finetune_config.ignore,
            self.finetune_config.ignore_index,
            self.finetune_config.dropout
            )
        self.head_output = PromptHeader(n_parts = 2,dropout=self.finetune_config.dropout)
        return None 
    
    def valid_batch(self, batch):
        for key in self.NECESSARY_KEYS:
            assert key in batch, f"{key} not in batch"
        return None 

    def forward(self, batch):
        self.valid_batch(batch)

        dna_embed_alt = self.dnabert2(batch["seq_alt"])["embedding_dna"]
        dna_emb = self.adapter_dna_emb(dna_embed_alt)

        chrom_embedding = self.pretrain_model(
            batch["input_ids"], batch["position_ids"]
        )
        chrom_embedding = self.adapter_chrombert(chrom_embedding, return_emb = True) # (batch_size, 768)

        logit = self.head_output(dna_emb, chrom_embedding)
        return logit 

    @DeprecationWarning
    def get_factor_emb(self, batch):
        self.valid_batch(batch)
        chrom_embedding = self.pretrain_model(
            batch["input_ids"], batch["position_ids"]
        )
        emb_factor = self.adapter_chrombert.interface(chrom_embedding)
        return emb_factor

class ChromBERTPromptSequence(ChromBERTPromptDNA):
    
    NECESSARY_KEYS = ["input_ids", "position_ids","sequence"]
    
    def create_layers(self):
        super().create_layers()
        
    def forward(self, batch):
        super().valid_batch(batch)

        dna_embed_alt = self.dnabert2(batch["sequence"])["embedding_dna"]
        dna_emb = self.adapter_dna_emb(dna_embed_alt)

        chrom_embedding = self.pretrain_model(
            batch["input_ids"], batch["position_ids"]
        )
        chrom_embedding = self.adapter_chrombert(chrom_embedding, return_emb = True) # (batch_size, 768)

        logit = self.head_output(dna_emb, chrom_embedding)
        return logit

    @DeprecationWarning
    def get_factor_emb(self, batch):
        self.valid_batch(batch)
        chrom_embedding = self.pretrain_model(
            batch["input_ids"], batch["position_ids"]
        )
        emb_factor = self.adapter_chrombert.interface(chrom_embedding)
        return emb_factor
        