import os
import torch
import json
from copy import deepcopy
from typing import Optional, Union, Any, Dict,Tuple
from dataclasses import dataclass, field, asdict
import numpy as np
from chrombert.base import ChromBERTConfig

@dataclass
class ChromBERTFTConfig:
    genome: str = field(default='hg38', metadata={"help": "hg38 for human, and mm10 for mouse"})
    task: str = field(default='general', metadata={"help": "task of the model"})
    dim_output: int = field(default=1, metadata={"help": "dimension of output"})
    mtx_mask: str = field(default = None, metadata = {"help": "mask matrix for chrombert"})
    dropout: float = field(default = 0.1, metadata = {"help": "dropout rate"})
    pretrain_ckpt: str = field(default = None, metadata= {"help": "loading pretrain checkpoint"})
    finetune_ckpt: str = field(default = None, metadata= {"help": "loading finetune checkpoint"})
    
    ignore: bool = False
    ignore_index: Tuple[Optional[np.ndarray], Optional[np.ndarray]] = field(default = (None, None), metadata = {"help": "ignore index for regulators and gsmids. See tutorials for detail. "})

    gep_flank_window: int = field(default = 4, metadata = {"help": "the number of flank region"})
    gep_parallel_embedding: bool = field(default=False, metadata = {"help": "whether use parallel embedding which need more GPU memrory"})
    gep_gradient_checkpoint: bool = field(default=False, metadata = {"help": "whether use gradient_checkpoint which no more GPU memrory"})
    gep_zero_inflation: bool = field(default=True, metadata = {"help": "whether use zero inflation header, if false, use general header"})
    
    prompt_kind: str = field(default='cistrome', metadata={"help": "prompt data class"})
    prompt_dim_external: int = field(default = 512, metadata = {"help": "dimension of external data. use 512 for scgpt, 768 for dnabert2"})

    dnabert2_ckpt: str = field(default = None, metadata = {"help": "loading dnabert2 checkpoint"})

    def __post_init__(self):
        self.validation()
    
    def to_dict(self):
        state = {}
        for k, v in self.__dataclass_fields__.items():
            state[k] = deepcopy(getattr(self, k))
        return state

    def __iter__(self):
        for name, value in self.to_dict().items():
            yield name, value

    @classmethod
    def load(cls, config: Union[str, Dict[str, Any], "ChromBERTFTConfig", None] = None, **kwargs: Any):
        if config is None:
            config_dict = {}
        elif isinstance(config, str):
            with open(config, 'r') as f:
                config_dict = json.load(f)
        elif isinstance(config, Dict):
            config_dict = deepcopy(config)
        elif isinstance(config, ChromBERTFTConfig):
            config_dict = config.to_dict()
        else:
            raise TypeError(f"config must be a str, Dict, or ChromBERTFTConfig, but got {type(config)}")
        
        config_dict.update(kwargs)

        config = cls(**config_dict)
        config.validation()
        return config

    def clone(self):
        return ChromBERTFTConfig.load(self.to_dict())
    
    def validation(self):
        assert self.genome in ['hg38', 'mm10'], f"genome must be one of ['hg38', 'mm10'], but got {self.genome}"
        task_avaliable = ['general','gep','prompt']
        assert self.task in task_avaliable, f"task must be one of {task_avaliable}, but got {self.task}"
        prompt_kind_avaliable = ['cistrome','expression','dna','sequence','cctp_sequence']
        assert self.prompt_kind in prompt_kind_avaliable, f"header must be one of {prompt_kind_avaliable}, but got {self.prompt_kind}"
        if self.prompt_kind not in ['cistrome','expression']:
            assert self.mtx_mask is not None, "mtx_mask must be specified"
        return None
    
    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Warning: '{key}' is not a valid field name in ChromBERTFTConfig")
        return None 
    
    @property
    def n_prompt_parts(self):
        if self.prompt_kind == 'cistrome':
            return 2
        elif self.prompt_kind == 'expression':
            return 2
        elif self.prompt_kind == 'cctp_sequence':
            return 2
        elif self.prompt_kind in ['dna', 'sequence']:
            return 1
        else:
            raise ValueError(f"prompt_kind must be one of ['cistrome', 'expression', 'dna', 'sequence', 'cctp_sequence'], but got {self.prompt_kind}")

    def __str__(self):
        s = self.to_dict()
        s = json.dumps(s, indent=4)
        return s

    def __repr__(self):
        return f"ChromBERTFTConfig:\n{str(self)}"
    
    def init_model(self, **kwargs):
        '''
        Instantiate the model using the configuration. 
        '''
        pretrain_ckpt = self.pretrain_ckpt if kwargs.get('pretrain_ckpt') is None else kwargs.get('pretrain_ckpt')
        if pretrain_ckpt is None:
            print("Warning: pretrain_ckpt is not specified in fine-tune model initiation")
        finetune_config = self.clone()
        finetune_config.update(**kwargs)
        
        pretrain_config = ChromBERTConfig.load(ckpt=pretrain_ckpt, genome=finetune_config.genome, dropout=finetune_config.dropout)

        if finetune_config.task == 'gep':
            from .gep_ft_model import ChromBERTGEP
            model =  ChromBERTGEP(pretrain_config, finetune_config)
    
        elif finetune_config.task == 'prompt':
            if finetune_config.prompt_kind in ['dna', 'sequence', 'cctp_sequence']:
                assert finetune_config.dnabert2_ckpt is not None, "dnabert2_ckpt must be specified for prompt_kind=dna or sequence"
                if finetune_config.dnabert2_ckpt is not None:
                    assert isinstance(finetune_config.dnabert2_ckpt, str)
                    if not os.path.exists(finetune_config.dnabert2_ckpt):
                        print(f"Warning: {finetune_config.dnabert2_ckpt} does not exist! Try to use huggingface cached...")
                if finetune_config.prompt_kind == 'dna':
                    from .prompt_dna_model import ChromBERTPromptDNA
                    model =  ChromBERTPromptDNA(pretrain_config, finetune_config)
                elif finetune_config.prompt_kind == 'sequence':
                    from .prompt_dna_model import ChromBERTPromptSequence
                    model = ChromBERTPromptSequence(pretrain_config, finetune_config)
                elif finetune_config.prompt_kind == 'cctp_sequence':
                    from .prompt_dna_model import ChromBERTPromptCCTPSequence
                    model = ChromBERTPromptCCTPSequence(pretrain_config, finetune_config)
            else:
                from .prompt_ft_model import ChromBERTPrompt
                model = ChromBERTPrompt(pretrain_config, finetune_config)
        else:
            from .general_ft_model import ChromBERTGeneral
            model =  ChromBERTGeneral(pretrain_config, finetune_config)
            
        # finetune_ckpt = self.finetune_ckpt if kwargs.get('finetune_ckpt') is None else kwargs.get('finetune_ckpt')
        if finetune_config.finetune_ckpt is not None:
            model.load_ckpt(finetune_config.finetune_ckpt)
        return model
        
    @classmethod
    def get_ckpt_type(cls, ckpt):
        assert isinstance(ckpt, str)
        ckpt = torch.load(ckpt, map_location='cpu')
        if "state_dict" in ckpt:
            return "finetune"
        else:
            return "pretrain"




def get_preset_model_config(preset: str = "default", basedir: str = os.path.expanduser("~/.cache/chrombert/data"),**kwargs):
    '''
    A method to get the preset dataset configuration.
    Args:
        preset: str, the predefined preset name of the dataset, or a user-defined file in JSON format. 
        basedir: str, the basedir of the used files. Default is "~/.cache/chrombert/data".
        kwargs: dict, the additional arguments to update the preset. See chrombert.ChromBERTFTConfig for more details.
    '''
    basedir = os.path.abspath(basedir)
    assert os.path.exists(basedir), f"{basedir=} does not exist"
    if not os.path.exists(preset):
        list_presets_available = os.listdir(os.path.join(os.path.dirname(__file__), "presets"))
        list_presets_available = [x.split(".")[0] for x in list_presets_available]

        if preset not in list_presets_available:
            raise ValueError(f"preset must be one of {list_presets_available}, but got {preset}")

        preset_file = os.path.join(os.path.dirname(__file__), "presets", f"{preset}.json")
    else:
        preset_file = preset
    with open(preset_file, 'r') as f:
        config = json.load(f)
    config.update(kwargs)
    for key, value in config.items():
        if key in ["mtx_mask", "pretrain_ckpt", "finetune_ckpt"]:
            # print(key, value)
            print(f"update path: {key} = {value}")
            if value is not None:
                if os.path.abspath(value) != value:
                    config[key] = os.path.join(basedir, value)
                    assert os.path.exists(config[key]), f"{key}={config[key]} does not exist"

    dc = ChromBERTFTConfig(**config)
    return dc
    
