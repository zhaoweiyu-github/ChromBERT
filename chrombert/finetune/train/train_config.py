import os
import torch
import json
from copy import deepcopy
from typing import Optional, Union, Any, Dict,Tuple
from dataclasses import dataclass, field, asdict
import numpy as np

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


    accumulate_grad_batches: int = field(default=1, metadata={"help": "gradient accumulation steps"})
    limit_val_batches: int = field(default=64, metadata={"help":'number of batches to use for each validation'})
    val_check_interval: int = field(default=64, metadata={"help":'validation check interval'})
    checkpoint_metric: str = field(default='bce', metadata={"help": "checkpoint metric"})
    checkpoint_mode: str = field(default='min', metadata={"help": "checkpoint mode"})


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

        pl_module = T(model, train_config)

        return pl_module

    def init_trainer(self, name="chrombert-ft", **kwargs):
        ''' 
        a simple wrapper for PyTorch Lightning Trainer. For advanced usage, please use PyTorch Lightning Trainer directly.
        '''
        import lightning.pytorch as pl

        # trainer = Trainer(**kwargs)
        params = {
            "max_epochs": self.max_epochs,
            "accumulate_grad_batches": self.accumulate_grad_batches,
            "limit_val_batches": self.limit_val_batches,
            "val_check_interval": self.val_check_interval,
        }
        params.update(kwargs)
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor=f"{self.tag}_validation/{self.checkpoint_metric}",
            mode= self.checkpoint_mode,
            save_top_k=kwargs.get("save_top_k", 1), 
            save_last=True,
            filename='{epoch}-{step}',
            verbose=True,
        )
        params.pop("save_top_k", None)
        trainer = pl.Trainer(
            logger = pl.loggers.TensorBoardLogger(save_dir=os.path.join(os.getcwd(),"lightning_logs"),name = name),
            callbacks = [checkpoint_callback, pl.callbacks.LearningRateMonitor()],
            **params
        )
        return trainer
        


