import os
import torch
import json
from copy import deepcopy
from typing import Optional, Union, Any, Dict,Tuple
from dataclasses import dataclass, field, asdict
import numpy as np
from chrombert.base import ChromBERTConfig

@dataclass
class TrainConfig:
    kind: str = field(default='classification', metadata={"help": "kind of the model"})
    loss: str = field(default='bce', metadata={"help": "loss function"})
    tag: str = field(default='default', metadata={"help": "tag of the trainer, used for grouping logged results"})

    adam_beta1: float = field(default=0.9, metadata={"help": "Adam beta1"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Adam beta2"})
    weight_decay: float = field(default=0.01, metadata={"help": "weight decay"})

    lr: float = field(default=1e-4, metadata={"help": "learning rate"})
    warmup_ratio: float = field(default=0.1, metadata={"help": "warmup ratio"})
    max_epochs: int = field(default=10, metadata={"help": "number of epochs"})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "gradient accumulation steps"})

    batch_size: int = field(default=4, metadata={"help": "batch size"})
    num_workers: int = field(default=4, metadata={"help": "number of workers"})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "gradient accumulation steps"})
    limit_validation_batch: int = field(default=50, metadata={"help":'number of batches to use for each validation'})
    validation_check_interval: int = field(default=50, metadata={"help":'validation check interval'})
    checkpoint_metric: str = field(default='loss', metadata={"help": "checkpoint metric"})


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
    def load(cls, config: Union[str, Dict[str, Any], "TrainConfig", None] = None, **kwargs: Any):
        if config is None:
            config_dict = {}
        elif isinstance(config, str):
            with open(config, 'r') as f:
                config_dict = json.load(f)
        elif isinstance(config, Dict):
            config_dict = deepcopy(config)
        elif isinstance(config, TrainConfig):
            config_dict = config.to_dict()
        else:
            raise TypeError(f"config must be a str, Dict, or TrainConfig, but got {type(config)}")
        
        config_dict.update(kwargs)

        config = cls(**config_dict)
        config.validation()
        return config

    def clone(self):
        return TrainConfig.load(self.to_dict())
    
    def validation(self):
        assert self.kind in ['classification', 'regression', 'zero_inflation'], f"{self.kind=} must be one of ['classification', 'regression', 'zero_inflation']"

        if self.kind == 'classification':
            assert self.loss in ['bce', 'focal'], f"{self.loss=} must be one of ['bce', 'focal']"
        elif self.kind == 'regression':
            assert self.loss in ['mae', 'mse', 'rmse'], f"{self.loss=} must be one of ['mae', 'mse', 'rmse']"
        else:
            assert self.loss in ['zero_inflation'], f"{self.loss=} must be one of ['zero_inflation']"
    
        return None
    
    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Warning: '{key}' is not a valid field name in DatasetConfig")
        return None 
    
    def init_pl_module(self, model, **kwargs):
        # raise NotImplementedError("init_model method must be implemented in the subclass")
        train_config = self.clone()
        train_config.update(**kwargs)
        if train_config.kind == "classification":
            from . import ClassificationPLModule as T
        elif train_config.kind == "regression":
            from . import RegressionPLModule as T 
        elif train_config.kind == "zero_inflation":
            from . import ZeroInflationPLModule as T 
        else:
            raise(ValueError("Not supported kind!"))

        trainer = T(model, train_config)

        return trainer
        


