import os
import numpy as np
import pandas as pd

import h5py
import psutil

import torch
from torch.utils.data import Dataset
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from dataclasses import dataclass, field, fields
from typing import List, Optional, Union, IO, Any, Dict
import json

print("Warning: do you really want to use this dataset? It's suggested to use dataset from fine-tune part")
@dataclass
class DatasetConfig:
    
    hdf5_file: Union[str, IO]
    seed: int = 1024

    resample_row_degree: Union[float,int] = 1 # if == 1, no change; if > 1, upsample high occupancy regions to the given times; if < 1, downsample low occupancy regions to the given ratio
    resample_row_threshold: float = 0.05 # determine which regions are repeated
    resample_col_degree: float = 1 # how many samples to save; default 1, if need to resample col(gsmids), set to < 1
    vocab_shift: int = 5 # vocab_shift of vocab ids
    vocab_levels: int = 5 # number of vocab levels
    mask_ratio: float = 0.15 # ratio of masked tokens

    mask_baseline: int = 10
    mask_prob: float = 0.8 # prob of mask
    sub_prob: float = 0.1 # prob of sub
    token_id_cls: int = 1 # id of cls token
    prepend_cls: Optional[bool] = True # prepend cls token
    token_id_pad: int = 0 # id of pad token
    max_length: Optional[int] = None # length of padded sequence
    token_id_mask: int = 2 # id of mask token
    batch_size: int = 8
    position_id_cls: int = 0
    position_id_pad: int = 0
    deterministic: bool = False # whether to set random seed

    ignore_factor: bool = False
    ignore_gsmids: Union[str, np.ndarray, None] = None

    dataset_class: str = "MaskDataset"


    def __init__(self, config: Union[str, Dict[str, Any], "DatasetConfig"] = None, **kwargs: Any):
        '''
    DatasetConfig is a dataclass that defines the configuration of the dataset.
        hdf5_file: Union[str, IO]
        seed: 1024

        resample_row_degree: Union[float,int] = 1 # if == 1, no change; if > 1, upsample high occupancy regions to the given times; if < 1, downsample high occupancy regions to the given ratio
        resample_row_threshold: float = 0.01 # determine which regions are repeated
        resample_col_degree: float = 1 # how many samples to save; default 1, if need to resample col(gsmids), set to < 1
        vocab_shift: int = 5 # vocab_shift of vocab ids
        vocab_levels: int = 5 # number of vocab levels
        mask_ratio: float = 0.15 # ratio of masked tokens
        mask_prob: float = 0.8 # prob of mask
        sub_prob: float = 0.1 # prob of sub
        token_id_cls: int = 1 # id of cls token
        prepend_cls: Optional[str] = True # prepend cls token
        token_id_pad: int = 0 # id of pad token
        max_length: Optional[int] = None # length of padded sequence
        token_id_mask: int = 2 # id of mask token
        batch_size: int = 8
        position_id_cls: int = 0
        position_id_pad: int = 0
        deterministic: bool = False # whether to set random seed
    '''
        self.hdf5_file = None

        self.load(config, kwargs)

        self.update(**kwargs)

        self.normal_vocab_ids = []
        self.special_vocab_ids = []
        self.__init__vocabs__()

        self.validate()
            
    def validate(self):

        dataset_classes = ["MaskDataset" , "SampleDataset"]
        if self.dataset_class not in dataset_classes:
            raise(ValueError(f"dataset_class must be one of {dataset_classes}"))


    def load(self,config, kwargs = None):
        if config is None :
            if "hdf5_file" not in kwargs:
                raise(TypeError("hdf5_file file must be provided"))
            else:
                self.update(**kwargs)
        
        if isinstance(config, str):  
            with open(config, 'r') as f:
                config = json.load(f)
            if "hdf5_file" in config:
                self.hdf5_file = config["hdf5_file"]  
                
        if isinstance(config, Union[dict, DatasetConfig]):  
            for key, value in config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    raise(AttributeError(f"Warning: '{key}' is not a valid field name in DatasetConfig"))
    
    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise(AttributeError(f"Warning: '{key}' is not a valid field name in DatasetConfig"))


    def save(self, config_file: str):
        values = self.__dict__()
        with open(config_file, 'w') as f:
            json.dump(values, f, indent=4)

    def __repr__(self):
        values = self.__dict__()
        return f"DatasetConfig({values})"
    
    def __str__(self):
        values = self.__dict__()
        return json.dumps(values, indent=4)
    
    def __dict__(self):
        return {f.name: getattr(self, f.name) for f in fields(self)}
    
    def items(self):
        return self.__dict__().items()
    
    def __init__vocabs__(self):
        self.normal_vocab_ids = list(range(self.vocab_shift,self.vocab_levels + self.vocab_shift))
        self.special_vocab_ids = list(range(self.vocab_shift))
        return None
    
    def clone(self):
        return DatasetConfig(config = self)


class BasicDataset(Dataset):
    def __init__(self, mode= "train", config = None, **params: Any):
        self.mode = mode
        self.config = DatasetConfig(config, **params)
       
        if isinstance(self.config.hdf5_file, str):
            self.dataset = None
        else:
            assert isinstance(self.config.hdf5_file, h5py.File)  
            self.config.hdf5_file = self.config.hdf5_file.filename 
            self.dataset = self.config.hdf5_file[f"/{self.mode}"] 
            
        with h5py.File(self.config.hdf5_file, 'r') as f:
            self.len = f[f'/{self.mode}/signal'].attrs['shape'][0]
            self.gsmids = [i.decode() for i in f[f'/{self.mode}/GSMID'][:] ]
            self.gsmid_to_did = {gsmid: i for i, gsmid in enumerate(self.gsmids)}
            self.did_to_gsmid = {i: gsmid for i, gsmid in enumerate(self.gsmids)}
            self.len_gsmids = len(self.gsmids)
            self.gsmid_index = np.arange(self.len_gsmids) + 1
            self.region = f[f'/{self.mode}/region'][:]
            self.ratios = f[f'/{self.mode}/ratio'][:]
            self.high_ratios = np.sum(self.ratios[:, 2:], axis=1) # sum of high occupancy regions

                      
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if self.dataset is None:
            self.dataset = h5py.File(self.config.hdf5_file, 'r')[f"/{self.mode}"]
        data = torch.from_numpy(self.dataset['signal'][index, :])   
        gsmid = torch.from_numpy(self.gsmid_index)
        region = torch.from_numpy(self.region[index, :])
        return {"input_ids": data, "position_ids": gsmid, "region": region, "sequence":torch.tensor([])}


    @staticmethod
    def print_memory_usage():
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        print('Memory used:', mem_info.rss / (1024 * 1024), 'MB')  # rss is the resident set size, i.e., physical memory used

        
    @property
    def gsm_num(self):
        return len(self.gsmids)
       

class RowResampleDataset(BasicDataset):
    def __init__(self, mode = "train", config = None, **params: Any):

        super().__init__(mode, config, **params)

        if self.config.deterministic:
            np.random.seed(self.config.seed + 1)

        if self.config.resample_row_degree == 1:
            self.index_dict = np.arange(self.len)
            self.resample_len = self.len
        else:
            sample_bool = self.high_ratios > self.config.resample_row_threshold
            index_array = np.arange(self.len)
            if self.config.resample_row_degree > 1: 
                repeated_indices = np.repeat(index_array, np.where(sample_bool, self.config.resample_row_num, 1))

                self.index_dict = repeated_indices
                self.resample_len = len(repeated_indices)
            else:
                prob = np.random.rand(self.len)
                reserved_rows = (prob < self.config.resample_row_degree ) | sample_bool 
                self.index_dict = index_array[reserved_rows]
                self.resample_len = np.sum(reserved_rows)

        self.ratios = self.ratios[self.index_dict, :]
        self.high_ratios = self.high_ratios[self.index_dict]

        
    def __len__(self):
        return self.resample_len
    
    def __getitem__(self, index):
        raw_index = self.index_dict[index]
        item = super().__getitem__(raw_index)
        return item


class ColResampleDataset(RowResampleDataset):

    def __getitem__(self, index): 
        
        item = super().__getitem__(index)
        
        if self.config.deterministic:
            np.random.seed(self.config.seed + index)
        if self.config.resample_col_degree == 1:
            return item
        else:

            data, data_index, region = item["input_ids"], item["position_ids"], item["region"]

            data_num = self.data_num
            del_num = self.len_gsmids - data_num   
            del_index = np.random.choice(np.arange(self.len_gsmids), size=del_num, replace=False)  
            mask = np.in1d(data_index, del_index + 1, invert=True) # del_index start 0，but gsmid_index start 1
            enh_index = data_index[mask]
            enh_data = data[mask]
            
            item["input_ids"]   = enh_data
            item["position_ids"] = enh_index

            return item   
        
    @property
    def data_num(self):
        if self.config.resample_col_degree == 1:
            col_len = self.len_gsmids

        else:
            if self.config.resample_col_degree < 1:
                col_len = int(self.config.resample_col_degree * self.len_gsmids)
            else:
                col_len = self.config.resample_col_degree

        return col_len 


class MaskDataset(ColResampleDataset):
    def __init__(self, mode = "train", config = None, **params: Any):
        
        super().__init__(mode, config, **params)
        
        if self.config.max_length is None:
            self.max_length = self.data_num + 1
        else:
            self.max_length = self.config.max_length
        
    def pad(self, x:torch.Tensor):
        x = x.to(torch.long)
        shape = x.shape
        assert len(shape) == 1
        if self.config.prepend_cls:
            x = torch.cat([torch.tensor([self.config.token_id_cls]), x])
        length_remain = self.max_length - shape[0] -1 
        if length_remain > 0:
            x = torch.cat([x, torch.tensor([self.config.token_id_pad] * length_remain)])
        else:
            x = x[:self.max_length]
        
        mask_pad = x != self.config.token_id_pad

        return x, mask_pad
    
    
    @property
    def vocab_size(self):

        return self.config.vocab_shift + self.config.vocab_levels

    def mask(self,x:torch.Tensor):
        shape = x.shape
        assert len(shape) == 1
        mask_attn = torch.zeros(shape, dtype=torch.bool)
        mask_attn_p = torch.rand(shape)
        for k in self.config.normal_vocab_ids:
            t1 = mask_attn_p < self.config.mask_ratio 
            t2 = x == k
            t = t1 & t2
            mask_attn = mask_attn | t
        
        baselevel_mask = self.config.mask_baseline
        t = baselevel_mask -  mask_attn.to(int).sum().item()
        if t > 0:
            sites_sup = np.random.randint(0, mask_attn.size(), int(t))
            mask_attn[sites_sup] = True  


        mask_attn = mask_attn & (x != self.config.token_id_pad)
        mask_attn = mask_attn & (x != self.config.token_id_cls)

        prob_replace = torch.rand(shape)
        mask_replace = prob_replace < self.config.mask_prob
        mask_replace = mask_replace & mask_attn
        x[mask_replace] = self.config.token_id_mask

        mask_sub = prob_replace < self.config.sub_prob + self.config.mask_prob
        mask_sub = mask_sub & mask_attn & ~mask_replace
        x[mask_sub] = torch.tensor(
            np.random.choice(self.config.normal_vocab_ids, mask_sub.sum().item())
        )


        return x, mask_attn # mask_attn: 1 means been masked
    
    
    def get_position_ids(self, x:torch.Tensor, sample_index) -> torch.Tensor:
        x = x.clone()
        x[x == self.config.token_id_pad] = self.config.position_id_pad
        x[x == self.config.token_id_cls] = self.config.position_id_cls

        p = (x != self.config.position_id_pad) & (x != self.config.position_id_cls)
        x[p] = sample_index
        return x
    
    
    def __getitem__(self, index):
        item = super().__getitem__(index)
        x, sample_index = item["input_ids"], item["position_ids"]
        x = x + self.config.vocab_shift
        x, mask_pad = self.pad(x)
        x_m, mask_loss = self.mask(x.clone())
        position_ids = self.get_position_ids(x_m, sample_index)

        item["input_ids"] = x_m 
        item["position_ids"] = position_ids
        item["mask_for_loss"] = mask_loss
        item["mask_for_pad"] = mask_pad
        item["labels"] = x
        return item

class SampleDataset(MaskDataset):

    def __init__(self, mode="train", config=None, **params: Any):
        super().__init__(mode, config, **params)
        
        self.config = config
        self.ignore_gsmids_index = np.array([])
        if config.ignore_factor:
            if isinstance(config.ignore_gsmids, str):
                ignore_gsmids = pd.read_csv(config.ignore_gsmids, header = None).iloc[:,0].values
                print("ignore_gsmids files be readed")
            elif isinstance(config.ignore_gsmids, np.ndarray):
                ignore_gsmids = config.ignore_gsmids
            else:
                raise AttributeError('When ignore_factor is set, ignore_gsmids should be set correctly.')
            
            self.ignore_gsmids_index = np.array([self.gsmid_to_did[key] for key in ignore_gsmids])


    def __getitem__(self, index):
        item = super().__getitem__(index)
        if self.config.ignore_factor:
            data, data_index, region, mask_for_loss, mask_for_pad, labels = item["input_ids"], item["position_ids"], item["region"], item["mask_for_loss"], item["mask_for_pad"], item["labels"]
            mask = np.in1d(data_index, self.ignore_gsmids_index + 1, invert = True) # ignore_gsmids_index start from 0，but gsmid_index start from 1
            filter_index = data_index[mask]
            filter_data = data[mask]
            filter_mask_for_loss = mask_for_loss[mask]
            filter_mask_for_pad = mask_for_pad[mask]
            filter_labels = labels[mask]
            return {"input_ids": filter_data, "position_ids": filter_index, "region": region, "mask_for_loss": filter_mask_for_loss, "mask_for_pad": filter_mask_for_pad, "labels": filter_labels}
        else:
            return item


class LitChromBERTDataModule(LightningDataModule):
    def __init__(self, config = None, train_params = {},val_params = dict(), test_params = dict(), **params):
        '''
        LitChromBERTDataModule is a class that defines the configuration of the dataset.
        config: Union[str, Dict[str, Any], "DatasetConfig"] = None
        train_params: dict = {}
        val_params: dict = dict()
        test_params: dict = dict()

        '''
        self.basic_config = DatasetConfig(config, **params)
        self.train_config = DatasetConfig(config=self.basic_config, **train_params)
        self.val_config = DatasetConfig(config=self.basic_config, **val_params)
        self.test_config = DatasetConfig(config=self.basic_config, **test_params)

        assert self.train_config.dataset_class == self.val_config.dataset_class == self.test_config.dataset_class
        self.dataset_class = eval(self.basic_config.dataset_class)

        self.current_train_config = self.train_config.clone()
        self.current_val_config = self.val_config.clone()
        self.current_test_config = self.test_config.clone()

        super().__init__()
        self.num_train_epochs = None
        self.train_dataset = None
        self.val_dataset = None

        


    def setup(self, stage: Optional[str] = None):

        self.train_dataset = self.dataset_class(mode = "train", config = self.current_train_config)
        self.val_dataset = self.dataset_class(mode = "vali", config = self.current_val_config)
        self.test_dataset = self.dataset_class(mode = "test", config = self.current_test_config)
        self.num_train_epochs = len(self.train_dataset) // self.train_config.batch_size
        self.num_val_epochs = len(self.val_dataset) // self.val_config.batch_size
        self.num_test_epochs = len(self.test_dataset) // self.test_config.batch_size



    def train_dataloader(self):
        dl = DataLoader(self.train_dataset, batch_size=self.train_config.batch_size, shuffle=True, num_workers=0)
        self.num_train_epochs = len(dl)
        return dl
    
    def val_dataloader(self):
        dl = DataLoader(self.val_dataset, batch_size=self.val_config.batch_size, shuffle=True, num_workers=0)
        self.num_val_epochs = len(dl)
        return dl
    
    def test_dataloader(self):
        dl = DataLoader(self.test_dataset, batch_size=self.test_config.batch_size, shuffle=False, num_workers=0)
        self.num_test_epochs = len(dl)
        return dl
    
