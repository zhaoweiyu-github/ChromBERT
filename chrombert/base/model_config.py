import os
import json
import torch

from dataclasses import dataclass, field, asdict, fields
from typing import Dict, Any, Union, Optional, List
from .model import ChromBERT


@dataclass
class ChromBERTConfig:
    genome: str = 'hg38'
    dropout: float = 0.1
    dtype_str: str = field(default='bfloat16', repr=False) 
    ckpt: str = None



    def __post_init__(self):
        """
        ChromBERTConfig, the configuration of ChromBERT model. It is able to instantiate a ChromBERT model through from the pretrained model.
        """ 
        assert self.genome in ['hg38', 'mm10'], f"genome should be hg38 for human, or mm10 for mouse, but got {self.genome}"
        print(f"use organisim {self.genome}; max sequence length is {self.n_datasets - 1}")

    @property
    def n_datasets(self):
        if self.genome == 'hg38':
            return 6392
        elif self.genome == 'mm10':
            return 5616
       
    @property
    def dtype(self):
        return getattr(torch, self.dtype_str)

    @property
    def vocab_size(self):
        return 10 

    @property
    def vocab_size_shift(self):
        return 5

    @property
    def hidden_dim(self):
        return 768

    @property
    def num_layers(self):
        return 8

    @property
    def feed_forward_dim(self):
        return 3072
    
    @property
    def num_attention_heads(self):
        return 8

    @property
    def token_id_pad(self):
        return 0

    @property
    def pe_mode(self):
        return 'train'


    @property
    def flash_bias(self):
        return True

    @property
    def flash_batch_first(self):
        return True

    @property
    def flash_causal(self):
        return False

    @property
    def flash_device(self):
        return None

    def save(self, config_file: str):
        values = asdict(self)
        with open(config_file, 'w') as f:
            json.dump(values, f, indent=4)
    
    def __repr__(self):
        values = asdict(self)
        return f"ChromBERTConfig({values})"
    
    def __str__(self):
        values = asdict(self)
        return json.dumps(values, indent=4)


    @classmethod
    def load(cls, config: Union[str, Dict[str, Any], "ChromBERTConfig",None]=None, **kwargs: Any):
        if config == None:
            config_dict = {}
        elif isinstance(config, str):
            with open(config, 'r') as f:
                config_dict = json.load(f)
        elif isinstance(config, Dict):
            config_dict = config
        elif isinstance(config, ChromBERTConfig):
            config_dict = asdict(config)
        else:
            raise TypeError(f"config must be a str, Dict, or ChromBERTConfig, but got {type(config)}")
        
        config_dict.update(kwargs)

        return cls(**config_dict)

    def init_model(self, ckpt=None):
        '''
        Instantiate the model using the configuration.
        '''
        model = ChromBERT(self)
        if ckpt is None:
            ckpt = self.ckpt
        if ckpt is None:
            print(f"Warning: no ckpt provided, use random initialization!")
        elif os.path.exists(ckpt):
            model.load_ckpt(ckpt)
        else:
            print(f"Warning: ckpt {ckpt} not exists, use random initialization!")
        return model

    @classmethod
    def get_ckpt_type(cls, ckpt):
        assert isinstance(ckpt, str)
        ckpt = torch.load(ckpt, map_location='cpu')
        if "state" in ckpt:
            return "finetune"
        else:
            return "pretrain"

        

        

