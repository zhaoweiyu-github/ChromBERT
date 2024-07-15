import os,sys
import torch
import torch.nn as nn
import torchmetrics as tm
from torch.optim import AdamW
import lightning.pytorch as pl
from transformers import get_linear_schedule_with_warmup
from abc import ABC, abstractmethod

from .utils.loss import FocalLoss,RMSELoss,ZeroInflationLoss
from .basic_pl_module import BasicPLModule
class ClassificationPLModule(BasicPLModule):
    def configure_loss_and_metrics(self):
        self.loss_funcs = {
            'bce': nn.BCEWithLogitsLoss(),
            'focal': FocalLoss(),
        }

        self.metric_funcs = {
            'precision': tm.Precision(task="binary"),
            'recall': tm.Recall(task="binary"),
            'mcc': tm.MatthewsCorrCoef(task="binary"),
            'f1': tm.F1Score(task="binary"),
            'auroc': tm.AUROC(task="binary"),
            'auprc': tm.AveragePrecision(task="binary"),
            'acc': tm.Accuracy(task="binary"),
        }
        return None 

    def logging_temp_states(self, logits, batch):
        self.logger_values.log("logits", logits)
        self.logger_values.log("label", batch["label"])
        return None 

    def process_metrics_validation_end(self):
        logits = self.logger_values.get_values("logits")
        labels = self.logger_values.get_values("label")
        logits, labels = self.clean_inputs(logits, labels)

        probs = torch.sigmoid(logits)
        metrics_loss = {name: func.to(labels.device)(logits, labels) for name, func in self.loss_funcs.items()}
        metrics_metrics = {name: func.to(labels.device)(logits, labels.long()) for name, func in self.metric_funcs.items()}

        metrics = {}
        metrics.update(metrics_loss)
        metrics.update(metrics_metrics)

        metrics["mean_logit"] = torch.mean(logits)
        metrics["median_logit"] = torch.median(logits)
        metrics["mean_prob"] = torch.mean(probs)
        metrics["median_prob"] = torch.median(probs)

        state = {f"{self.config.tag}_validation/{name}": value for name, value in metrics.items()}
        for name, value in state.items():
            self.log(name, value,  sync_dist = False, on_step = False, on_epoch = True, prog_bar = True)

        for k, func in self.metric_funcs.items():
            func.reset()
        return None 
        

class RegressionPLModule(BasicPLModule):
    def configure_loss_and_metrics(self):
        self.loss_funcs = {
            'mae': nn.L1Loss(),
            'mse': nn.MSELoss(),
            'rmse': RMSELoss(),
        }

        self.metric_funcs = {
            'r2': tm.R2Score(),
            'pcc': tm.PearsonCorrCoef(),
            'scc': tm.SpearmanCorrCoef(),
        }
        return None 

    def logging_temp_states(self, logits, batch):
        self.logger_values.log("logits", logits)
        self.logger_values.log("label", batch["label"])
        return None 

    def process_metrics_validation_end(self):
        logits = self.logger_values.get_values("logits")
        labels = self.logger_values.get_values("label")
        logits, labels = self.clean_inputs(logits, labels)

        metrics_loss = {name: func.to(labels.device)(logits, labels) for name, func in self.loss_funcs.items()}
        metrics_metrics = {name: func.to(labels.device)(logits, labels) for name, func in self.metric_funcs.items()}
        metrics_metrics.update(metrics_loss)
        metrics = metrics_metrics
        metrics["mean"] = torch.mean(logits)
        metrics["median"] = torch.median(logits)

        state = {f"{self.config.tag}_validation/{name}": value for name, value in metrics.items()}
        for name, value in state.items():
            self.log(name, value,  sync_dist = False, on_step = False, on_epoch = True, prog_bar = True)

        for k, func in self.metric_funcs.items():
            func.reset()
        return None


class ZeroInflationPLModule(BasicPLModule):

    def clean_inputs(self,logits, labels):
        probs, regs = logits
        return (probs.view(-1), regs.view(-1)), labels.view(-1)
    
    def configure_loss_and_metrics(self):
        self.loss_funcs = {
            'zero_inflation': ZeroInflationLoss(),
        }


        self.metrics_for_prob = {
            
        }
        self.loss_for_reg = {
            'mae': nn.L1Loss(),
            'mse': nn.MSELoss(),
            'rmse': RMSELoss(),
        }
        self.metrics_for_reg = {
            'r2': tm.R2Score(),
            'pcc': tm.PearsonCorrCoef(),
            'scc': tm.SpearmanCorrCoef(),
        }
        return None 

    def logging_temp_states(self, logits, batch):
        prob, reg_value = logits
        self.logger_values.log("zero_prob_logit", prob)
        self.logger_values.log("reg_value", reg_value)
        self.logger_values.log("label", batch["label"])
        return None 

    def process_metrics_validation_end(self):
        zero_prob_logit = self.logger_values.get_values("zero_prob_logit")
        reg_value = self.logger_values.get_values("reg_value")
        labels = self.logger_values.get_values("label")

        metrics_loss = {name: func.to(labels.device)((zero_prob_logit, reg_value), labels) for name, func in self.loss_funcs.items()}
        # metrics_for_probs = {func(zero_prob_logit, labels > 0) for name, func in self.metrics_for_prob.items()}
        loss_for_reg =  {name: func.to(labels.device)(reg_value, labels) for name, func in self.loss_for_reg.items()}
        metrics_for_reg =  {name: func.to(labels.device)(reg_value, labels) for name, func in self.metrics_for_reg.items()}

        metrics = {}
        metrics.update(metrics_loss)
        # metrics.update(metrics_for_probs)
        metrics.update(loss_for_reg)
        metrics.update(metrics_for_reg)

        state = {f"{self.config.tag}_validation/{name}": value for name, value in metrics.items()}
        for name, value in state.items():
            self.log(name, value,  sync_dist = False, on_step = False, on_epoch = True, prog_bar = False)

        for k, func in self.metrics_for_reg.items():
            func.reset()
        return None
