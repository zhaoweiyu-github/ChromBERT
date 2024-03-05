from dataclasses import dataclass, field, asdict, fields
from typing import Dict, Any, Union, Optional, List
import json
import torch

@dataclass
class ChromBERTConfig:
    vocab_size: int = 10
    vocab_size_shift: int = 5
    vocab_size_multitask: Optional[List[int]] = field(default_factory=lambda: [5])
    hidden_dim: int = 768
    num_layers: int = 4
    feed_forward_dim: int = 3072
    num_attention_heads: int = 8
    dropout_prob: float = 0.1
    token_id_pad: int = 0
    n_datasets: int = 15759
    dtype: str = 'bfloat16'

    pe_mode: str = 'train' # or "word2vec"

    flash_bias: bool = True
    flash_batch_first: bool = True
    flash_causal: bool = False
    flash_device: Optional[torch.device] = None

    loss_function: str = 'FocalLoss'
    fl_gamma: float = 2 # if loss function is FocalLoss
    fl_alpha: List[float] = field(default_factory= lambda : [1,1,1,1,1])

    learning_rate: float = 1e-4
    Adam_beta1: float = 0.9
    Adam_beta2: float = 0.999
    weight_decay: float = 1e-2
    scheduler: str = 'OneCycle' # or 'Cosine'
    OneCycle_warmup_steps: Union[float, int] = 0.1


    """
    :param vocab_size: vocab size excluding the special token (e.g. 'cls')
    :param hidden_dim: embedding size of token embedding
    :param dropout_prob: dropout rate
    :param token_id_pad: index for the special token 'pad'
    :param n_datasets: number of datasets, for trainable positional embedding
    :param pe_mode: train or word2vec, for positional embedding choice
    :param pkl_embeding: path to pkl file of word2vec embedding
    :param did_to_gsmid: dict of dataset id to gsm id, for word2vec positional embedding
    :param pe_d_in: dimension of word2vec embedding
    :param dna_embedding: whether contain DNA embedding in the embedding layer
    """ 

    def __post_init__(self):
        self.validation()
        self.dtype = getattr(torch, self.dtype)
        # self.vocab_size = self.vocab_size + self.vocab_size_shift
        if self.vocab_size_multitask and len(self.vocab_size_multitask) > 1:
            self.vocab_size_multitask = [v + 5 for v in self.vocab_size_multitask]
            self.decoder_header = 'multitask'
        else:
            self.decoder_header = 'single'
        if self.decoder_header == 'multitask':
            self.weighted_loss = False
            assert self.loss_function != 'PenaltyFocalLoss', "loss_function can not be PenaltyFocalLoss if decoder_header is multitask"

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
    
    def validation(self):
        valid_loss_functions = {'FocalLoss', 'CrossEntropyLoss'}
        valid_pe_mode = {'train'}
        valid_scheduler = {'OneCycle'}
        if isinstance(self.OneCycle_warmup_steps, float) and not (0 <= self.OneCycle_warmup_steps < 1):
            raise ValueError("If transformers_warmup_steps is a float, it must be in the range [0, 1)")
        if self.loss_function not in valid_loss_functions:
            raise ValueError(f'parameter_to_check must be one of {valid_loss_functions}, '
                             f'but got {self.loss_function}')
        if self.pe_mode not in valid_pe_mode:
            raise ValueError(f'parameter_to_check must be one of {valid_pe_mode}, '
                             f'but got {self.pe_mode}')
        if self.scheduler not in valid_scheduler:
            raise ValueError(f'schedular must be one of {valid_scheduler}, '
                             f'but got {self.scheduler}')

        if self.loss_function == 'FocalLoss':
            if self.fl_gamma is None:
                raise ValueError(f'loss_function is FocalLoss, fl_gamma must be provided.')
            if len(self.fl_alpha) != self.vocab_size - self.vocab_size_shift:
                raise ValueError(f'loss_function is FocalLoss, fl_alpha must be provided with length of vocab_size - vocab_size_shift. Basic alpha is 1.')


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


