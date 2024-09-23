import numpy as np
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Subset
from typing import Optional
from .dataset_config import DatasetConfig
from .multi_flankwindow_dataset import MultiFlankwindowDataset
from .general_dataset import GeneralDataset
from .prompt_dataset import PromptDataset

class LitChromBERTFTDataModule(LightningDataModule):
    '''
    For training with pytorch lightning. 
    '''
    def __init__(self, config=None, train_params={}, val_params={}, test_params={}, **params):
        '''
        LitENBERTDataModule is a class that defines the configuration of the dataset.
        Args:
            config: DatasetConfig. 
            {train|val|test}_params: specific params to modify config for train|val|test respetively.
            
        '''
        if isinstance(config, str):
            config = DatasetConfig(config, **params)

        assert isinstance(config, DatasetConfig), f"config must be a DatasetConfig object, but got {type(config)}"
        self.basic_config = type(config)(config, **params)
        self.train_config = type(config)(config=self.basic_config, **train_params)
        self.val_config = type(config)(config=self.basic_config, **val_params)
        self.test_config = type(config)(config=self.basic_config, **test_params)

        assert self.train_config.kind == self.val_config.kind == self.test_config.kind

        super().__init__()
        self.num_train_epochs = None
        self.num_val_epochs = None
        self.num_test_epochs = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.has_train = train_params != {}
        self.has_val = val_params != {}
        self.has_test = test_params != {}

    def setup(self, stage: Optional[str] = None):

        if stage == "fit" or stage is None:
            if self.has_train:
                self.train_dataset = self.train_config.init_dataset()
                self.num_train_epochs = len(self.train_dataset) // self.train_config.batch_size
            
            if self.has_val:
                self.val_dataset = self.val_config.init_dataset()
                self.num_val_epochs = len(self.val_dataset) // self.val_config.batch_size
                indices = list(range(len(self.val_dataset)))
                np.random.shuffle(indices)
                self.shuffled_val_dataset = Subset(self.val_dataset, indices[:len(self.val_dataset)])

            if self.has_test:
                self.test_dataset =  self.test_config.init_dataset()
                self.num_test_epochs = len(self.test_dataset) // self.test_config.batch_size

        elif stage == "val":
            if self.has_val:
                indices = list(range(len(self.val_dataset)))
                np.random.shuffle(indices)
                self.shuffled_val_dataset = Subset(self.val_dataset, indices[:len(self.val_dataset)])

    def train_dataloader(self):
        if self.train_dataset:
            dl = DataLoader(self.train_dataset, batch_size=self.train_config.batch_size, shuffle=True, num_workers=self.train_config.num_workers)
            self.num_train_epochs = len(dl) # force shuffling
            return dl
        return None
    
    def val_dataloader(self):
        if self.val_dataset:
            dl = DataLoader(self.shuffled_val_dataset, batch_size=self.val_config.batch_size, shuffle=True, num_workers=self.val_config.num_workers) # specifically for validation, because the foreced shuffing
            self.num_val_epochs = len(dl)
            return dl
        return None
    
    def test_dataloader(self):
        if self.test_dataset:
            dl = DataLoader(self.test_dataset, batch_size=self.test_config.batch_size, shuffle=False, num_workers=self.test_config.num_workers) # forced no shuffling
            self.num_test_epochs = len(dl)
            return dl
        return None
