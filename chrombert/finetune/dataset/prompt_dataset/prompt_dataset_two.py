import os
import h5py
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from ..basic_dataset import BasicDataset
from .prompt_dataset_single import PromptDatasetForDNASequence
from .interface_manager import RegulatorInterfaceManager, CelltypeInterfaceManager

'''
This file implements classes for the prompt-enhanced dataset used for TFBS prediction. 
Direct usage is not recommended; please use through PromptDataset or DatasetConfig instead.
'''


class SupervisedForH5():
    '''
    For hdf5 format supervised file processing. 
    input: h5 format supervised_file, which contains cell, regulator, label, build_region_index.  
        regulator is optional if provived in config, but must match the length of cell if provided. 
        label is optional, with shape of (num_regions, num of cell-regulator). 
        See tutorials for detail format. 
    return: cell, regulator, label, build_region_index
    '''
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.load_data(config.supervised_file)

    def load_data(self, supervised_file=None):
        with h5py.File(supervised_file, 'r') as hdf:
            assert 'regions' in hdf.keys(), "regions key is missing in h5 file"
            self.h5_regions = hdf['regions'][:]

            if self.config.prompt_celltype:
                self.prompt_celltype = [self.config.prompt_celltype]
            elif 'cell' in hdf.keys():
                self.prompt_celltype = [item.decode('utf-8') for item in hdf['cell'][:]]
            else:
                raise ValueError('prompt of cell type needs to be set')

            if self.config.prompt_regulator:
                self.prompt_regulator = [self.config.prompt_regulator] * len(self.prompt_celltype)
            elif 'regulator' in hdf.keys():
                assert len(hdf['regulator'][:]) == len(self.prompt_celltype), "Celltype and regulator lengths do not match"
                self.prompt_regulator = [item.decode('utf-8') for item in hdf['regulator'][:]]
            else:
                raise ValueError('prompt regulator needs to be set')

            assert len(self.prompt_celltype) == len(self.prompt_regulator), "Celltype and regulator lengths do not match"
            self.supervised_indices = self.h5_regions[:, 3]
            self.supervised_indices_len = self.h5_regions.shape[0] * len(self.prompt_celltype)

            self.supervised_labels = hdf['label'][:] > 0 if 'label' in hdf.keys() else None

    def __len__(self):
        return self.supervised_indices_len

    def __getitem__(self, index):
        index_row = index % len(self.h5_regions)
        index_col = index // len(self.h5_regions)
        return {
            'build_region_index': self.supervised_indices[index_row],
            'cell': self.prompt_celltype[index_col],
            'regulator': self.prompt_regulator[index_col],
            'label': self.supervised_labels[index_row, index_col] if self.supervised_labels is not None else None
        }    
            
class SupervisedForTable():     
    '''
    For table format supervised file processing. 

    input: supervised_file
    return: cell, regulator, label, build_region_index
    '''
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.load_data(config.supervised_file)
    def load_data(self, supervised_file=None):
        if supervised_file.endswith(".csv"):
            df_supervised = pd.read_csv(supervised_file)
        elif supervised_file.endswith(".tsv"):
            df_supervised = pd.read_csv(supervised_file, sep = "\t")
        elif supervised_file.endswith(".feather"):
            df_supervised = pd.read_feather(supervised_file)
        else:
            raise(ValueError(f"supervised_file must be h5, csv, tsv or feather file!"))
        neccessary_columns = ["chrom","start","end","build_region_index"]
        for column in neccessary_columns:
            if column not in df_supervised.columns:
                raise(ValueError(f"{column} not in supervised_file! it must contain headers: {neccessary_columns}"))

        if self.config.prompt_celltype is not None:
            self.prompt_celltype =[self.config.prompt_celltype]*len(df_supervised)
            
        elif "cell" in df_supervised.columns:
            self.prompt_celltype = df_supervised["cell"].values
        else:
            raise(ValueError(f'prompt cell need to set'))
        
        if self.config.prompt_regulator is not None:
            self.prompt_regulator = [self.config.prompt_regulator]*len(df_supervised)
        elif "regulator" in df_supervised.columns:
            self.prompt_regulator = df_supervised["regulator"].values
        else:
            raise(ValueError(f'prompt regulator need to set'))
        
        assert len(self.prompt_celltype) == len(self.prompt_regulator), "Celltype and regulator lengths do not match"
        
        self.supervised_indices = df_supervised["build_region_index"].values
        self.supervised_indices_len = len(self.supervised_indices)
        self.supervised_labels = df_supervised['label'].values if 'label' in df_supervised.columns else None
        
    def __len__(self):
        return self.supervised_indices_len
    
    def __getitem__(self, index):
        return {
            'build_region_index': self.supervised_indices[index],
            'cell': self.prompt_celltype[index],
            'regulator': self.prompt_regulator[index],
            'label': self.supervised_labels[index] if self.supervised_labels is not None else None
        } 
          

class PromptDatasetForCCTP(BasicDataset):
    def __init__(self,config):
        super().__init__(config)
        self.config = config
        self.prompt_map = {i:j for i,j in self.gsmid_to_did.items()} #self.gsmid_to_did from BasicDataset
        self.seq_len = self.gsm_num
        self.cell_interface = CelltypeInterfaceManager(config,self.prompt_map)
        self.regulator_interface = RegulatorInterfaceManager(config,self.prompt_map)

        self.supervised_file = config.supervised_file
        if self.supervised_file.endswith("h5"):
            self.sv_dataset = SupervisedForH5(config)
        else:
            self.sv_dataset = SupervisedForTable(config)
            
    def __len__(self):
        return len(self.sv_dataset)
    
    def __getitem__(self,index):
        sv_item = self.sv_dataset[index]
        cell = sv_item['cell']
        regulator = sv_item['regulator']
        label = sv_item['label']
        build_region_index = sv_item['build_region_index']

        celltype_item = self.cell_interface.get_prompt_item(build_region_index,cell,self.seq_len)    
        regulator_item = self.regulator_interface.get_prompt_item(build_region_index, regulator, self.seq_len)
        if self.config.prompt_regulator_cache_file is None or self.config.prompt_celltype_cache_file is None:
            item = super().__getitem__(build_region_index)
        else:
            item = {"build_region_index": build_region_index}
        if label is not None:
            item['label'] = label
        else:
            del sv_item['label']
        item.update(celltype_item)
        item.update(regulator_item)
        item.update(sv_item)
        return item
