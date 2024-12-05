import os
import h5py
import psutil
import torch
import numpy as np
import pandas as pd
import pickle
import json
from torch.utils.data import Dataset
from typing import Any
from .dataset_config import DatasetConfig
from functools import lru_cache


'''
This file implements dataset classes for processing the reference HDF5 dataset. Direct usage of these classes is not recommended. 
'''
    
class BasicDataset(Dataset):
    '''
    Basic dataset class for ChromBERT. Implementation for fetching data from reference cistrome dataset. 

    '''
    def __init__(self, config = None, **params: Any):
        self.config = DatasetConfig(config, **params)
       
        if isinstance(self.config.hdf5_file, str):
            self.dataset = None
        else:
            assert isinstance(self.config.hdf5_file, h5py.File), f"hdf5_file must be a h5py.File object, but got {self.config.hdf5_file}"
            self.config.hdf5_file = self.config.hdf5_file.filename 
            self.dataset = self.config.hdf5_file[f"/full"] 
            
        with h5py.File(self.config.hdf5_file, 'r') as f:
            self.len = f[f'/full/signal'].attrs['shape'][0]
            self.gsmids = [i.decode().lower() for i in f[f'/full/GSMID'][:]]
            self.gsmid_to_did = {gsmid: i for i, gsmid in enumerate(self.gsmids)}
            self.did_to_gsmid = {i: gsmid for i, gsmid in enumerate(self.gsmids)}
            self.len_gsmids = len(self.gsmids)
            self.gsmid_index = np.arange(self.len_gsmids) + 1
            self.region = f[f'/full/region'][:]
                      
    def __len__(self):
        return self.len

    def get_position_ids(self, x:torch.Tensor, sample_index) -> torch.Tensor:
        x = x.clone()
        x[x == self.config.token_id_pad] = self.config.position_id_pad
        x = x.to(torch.long)
        x[(x != self.config.position_id_pad)] = sample_index
        return x

    def __getitem__(self, index):
        if self.dataset is None:
            self.dataset = h5py.File(self.config.hdf5_file, 'r')[f"/full"]
        data = torch.from_numpy(self.dataset['signal'][index, :])   
        gsmid = torch.from_numpy(self.gsmid_index)
        region = torch.from_numpy(self.region[index, :])
        data = data + self.config.vocab_shift # shift token, from 0-4 level to 5-9 level, allow using of special tokens like cls. 
        position_ids = self.get_position_ids(data, gsmid)
        return {"input_ids": data, "position_ids": position_ids, "region": region, "build_region_index": index}

    @staticmethod
    def print_memory_usage():
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        print('Memory used:', mem_info.rss / (1024 * 1024), 'MB')  # rss is the resident set size, i.e., physical memory used
        
    @property
    def gsm_num(self):
        return len(self.gsmids)
    
class PerturbationDataset(BasicDataset):
    '''
    Dataset class for perturbation some GSMIDs or regulators.
    ''' 
    def __init__(self, config=None, **params: Any):
        super().__init__(config, **params)
        self.config = config
        if config.perturbation:
            assert self.config.meta_file is not None
            assert os.path.exists(self.config.meta_file)
            with open(self.config.meta_file, 'r') as f:
                self.meta_file = json.load(f)
                self.regulators = sorted(self.meta_file['regulator'])
            self.perturbation_value = self.config.perturbation_value
            
        
    def __getitem__(self, index):
        item = super().__getitem__(index)
        if self.config.perturbation and self.config.perturbation_object is not None and self.config.perturbation_object != "none":
            self.perturb_gsmids_index = self.process_perturbation_target(self.config.perturbation_object)
            perturb_mask = np.in1d(item["position_ids"], self.perturb_gsmids_index + 1) 
            item['input_ids'][perturb_mask] = self.perturbation_value + self.config.vocab_shift
        return item
        
    @lru_cache(maxsize=32)
    def process_perturbation_target(self, perturbation_object):
        perturbation_objects = perturbation_object.lower().split(";")
        # remove 'none'
        perturbation_objects = [i for i in perturbation_objects if i != 'none']

        perturbation_gsmids = list(sorted(set(perturbation_objects) & set(self.gsmids)))
        perturbation_regulators = list(sorted(set(perturbation_objects) & set(self.regulators)))
        assert len(perturbation_gsmids) + len(perturbation_regulators) == len(set(perturbation_objects)), f"Detect unknown perturbation objects: {set(perturbation_objects) - set(perturbation_gsmids) - set(perturbation_regulators)}"

        for regulator in perturbation_regulators:
            perturbation_gsmids.extend(self.meta_file[regulator].split(";"))
        
        perturbation_gsmids = list(set(perturbation_gsmids))
        perturb_gsmids_index = np.array([self.gsmid_to_did[key] for key in perturbation_gsmids])  # index in the all gsmid
        return perturb_gsmids_index

                
class IgnoreDataset(PerturbationDataset):
    '''
    Dataset class for ignoring some GSMIDs or regulators.
    '''

    def __init__(self, config=None, **params: Any):
        super().__init__(config, **params)
        
        self.config = config
        if config.ignore:
            assert self.config.meta_file is not None
            assert os.path.exists(self.config.meta_file)
            with open(self.config.meta_file, 'r') as f:
                self.meta_file = json.load(f)
                self.regulators = sorted(self.meta_file['regulator'])

    def __getitem__(self, index):
        item = super().__getitem__(index)
        if self.config.ignore and self.config.ignore_object is not None:
            self.ignore_gsmid_index, self.ignore_regulator_index = self.process_ignore_object(self.config.ignore_object)
            self.ignore_index = (self.ignore_gsmid_index, self.ignore_regulator_index)
            data, data_index = item["input_ids"], item["position_ids"]
            mask = np.in1d(data_index, self.ignore_gsmid_index + 1, invert = True) # ignore_gsmid_index start from 0，but gsmid_index start from 1
            item['position_ids'] = data_index[mask]
            item['input_ids'] = data[mask]
            item['ignore_index'] = (self.ignore_gsmid_index,self.ignore_regulator_index)

        return item
    
    @property
    def gsm_num(self):
        if self.config.ignore and self.config.ignore_object is not None:
            return len(self.gsmid_index) - len(self.ignore_gsmid_index)
        else:
            return len(self.gsmid_index)
    
    @lru_cache(maxsize=32)
    def process_ignore_object(self, ignore_object):
        ignore_objects = ignore_object.lower().split(";")
        ignore_gsmids = list(sorted(set(ignore_objects) & set(self.gsmids)))
        ignore_regulators = list(sorted(set(ignore_objects) & set(self.regulators)))
        assert len(ignore_gsmids) + len(ignore_regulators) == len(set(ignore_objects)), f"Detect unknown ignore objects: {set(ignore_objects) - set(ignore_gsmids) - set(ignore_regulators)}"
        
        list_regulators=[]
        for gsmid in ignore_gsmids:
            new_regulator = self.meta_file[gsmid].split(";")[-1]
            list_regulators.append(new_regulator)
        list_regulators = sorted(list(set(list_regulators)))


        for regulator in list_regulators:
            if len(set(self.meta_file[regulator].split(';')) - set(ignore_gsmids)) == 0:
                ignore_regulators.append(regulator)
                
        for regulator in ignore_regulators:
            ignore_gsmids.extend(self.meta_file[regulator].split(';'))

        ignore_gsmids_index = np.array([self.gsmid_to_did[key] for key in set(ignore_gsmids)])  # index in the all gsmid
        ignore_regulator_index = np.array([self.regulators.index(ignore_regulator.lower()) for ignore_regulator in ignore_regulators])
        
        return ignore_gsmids_index, ignore_regulator_index

            