import os,sys
import torch
import torch.nn as nn
import torchmetrics as tm
from torch.optim import AdamW
import lightning.pytorch as pl
from transformers import get_linear_schedule_with_warmup
from abc import ABC, abstractmethod

from .utils.loss import FocalLoss,RMSELoss,ZeroInflationLoss
from .utils.logger import LogTensorValues


class BasicPLModule(pl.LightningModule, ABC):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config 
        self.log_values = LogTensorValues()
        self.configure_loss_and_metrics()
    
    def forward(self, x):
        return self.model(x)

    @abstractmethod
    def configure_loss_and_metrics(self):
        pass

    @abstractmethod
    def logging_temp_states(self, logit, batch):
        pass

    @abstractmethod
    def process_metrics_validation_end(self):
        pass
    
    def calculate_metrics(self, logits, labels, mode = 'train',pbar=False):
        logits, labels = self.clean_inputs(logits, labels)
            # only calculate loss in training mode
        metrics = {name: func.to(labels.device)(logits, labels) for name, func in self.loss_funcs.items()}
        loss = metrics[self.config.loss]
 
        state = {f"{self.config.tag}_{mode}/{name}": value for name, value in metrics.items()}
        for name, value in state.items():
            self.log(name, value,  sync_dist = True, on_step = True, on_epoch = True, prog_bar = pbar)
        return loss, metrics

    def clean_inputs(self, logits, labels, *args, **kwargs):
        return logits.view(-1), labels.view(-1).float()

    def training_step(self, batch, batch_idx):
        logits = self.model(batch)
        loss, mertrics = self.calculate_metrics(logits, batch["label"], mode='train')
        return loss

    def on_validation_epoch_start(self):
        self.logger_values = LogTensorValues() 

    def validation_step(self, batch, batch_idx):
        logits = self.model(batch)
        loss, mertrics = self.calculate_metrics(logits, batch["label"], mode='validation')
        self.logging_temp_states(logits, batch)
        return loss
    
    def test_step(self, batch, batch_idx):
        return self.forward(batch)
    
    
    def on_validation_epoch_end(self):
        self.process_metrics_validation_end()
        self.logger_values = None
        self.trainer.datamodule.setup('val')

        return None 

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), 
                          lr = self.config.lr, 
                          betas = (self.config.adam_beta1, self.config.adam_beta2), 
                          weight_decay = self.config.weight_decay)
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(self.config.warmup_ratio * total_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = warmup_steps, num_training_steps = total_steps)
        lr_scheduler_config = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "frequency": 1,
                "interval": "step",
                "monitor": "train_loss"
            },
        }
        return lr_scheduler_config


    def freeze(self, trainable = 2):
        self.model.freeze_pretrain(trainable)

    def save_ckpt(self, ckpt_path):
        torch.save({"state_dict":self.model.state_dict()}, ckpt_path)
        return None