
import pandas as pd
from typing import Any
from .basic_dataset import IgnoreDataset
import numpy as np
class GeneralDataset(IgnoreDataset):
    '''
    Dataset class for general purposes. 
    '''

    def __init__(self,config = None, **params: Any):
        '''
        It's recommend to instantiate the class using DatasetConfig.init(). 
        params:
            config: DatasetConfig. supervised_file must be provided. 

        '''
        super().__init__(config, **params)
        self.config = config
        self.supervised(config.supervised_file)
        self.__getitem__(0) # make sure initiation 

    def supervised(self, supervised_file = None):
        '''
        process supervised file to obtain necessary information
        '''
        assert isinstance(supervised_file, str)
        if supervised_file.endswith('.csv'):
            df_supervised = pd.read_csv(supervised_file, header = 0) # csv format, [chrom, start, end, build_region_index, label, other meta datas]
        elif supervised_file.endswith('.tsv'):
            df_supervised = pd.read_csv(supervised_file, header = 0,sep='\t') # tsv format, [chrom, start, end, build_region_index, label, other meta datas]
        elif supervised_file.endswith('.feather'):
            df_supervised = pd.read_feather(supervised_file)
        else: 
            raise(ValueError(f"supervised_file must be csv, tsv or feather file!"))
        
        self.supervised_indices = df_supervised["build_region_index"]
        self.supervised_indices_len = len(self.supervised_indices)
        
        neccessary_columns = ["chrom","start","end","build_region_index"]
        for column in neccessary_columns:
            if column not in df_supervised.columns:
                raise(ValueError(f"{column} not in supervised_file! it must contain headers: {neccessary_columns}"))
            
        self.optional_columns(df_supervised)   

    def optional_columns(self,df):
        if 'label' not in df.columns:  ### only "chrom","start","end","build_region_index" columns and to predict
            self.supervised_labels = None
            print(f"Your supervised_file does not contain the 'label' column. Please verify whether ground truth column ('label') is required. If it is not needed, you may disregard this message.")
        else:
            self.supervised_labels = df['label'].values
        
        if self.config.perturbation:
            if self.config.perturbation_object is not None:
                self.perturbation_object = [self.config.perturbation_object] * (self.supervised_indices_len)
                print("use perturbation_object in dataset config which high priority than supervised_file")
            elif "perturbation_object" in df.columns:
                self.perturbation_object = df['perturbation_object'].fillna("none").values
                print("use perturbation_object in supervised_file")                
            else:
                raise AttributeError("When perturbation is set, perturbation_object should be set correctly. you can provided 'perturbation_object' column in your supervised_file or you can set perturbation_object in dataset config")
            
        if "ignore_object" in df.columns:
            self.ignore_object = df['ignore_object'].unique().tolist()
            assert(len(self.ignore_object)==1)
            if self.config.ignore_object is None:
                self.config.ignore_object = self.ignore_object[0]

        else:
            self.ignore_object = None
            

    def __len__(self):
        return self.supervised_indices_len

    def __getitem__(self, index):
        basic_index = self.supervised_indices[index]
        
        if self.config.perturbation: 
            self.config.perturbation_object = self.perturbation_object[index]
          
        if self.config.ignore and self.config.ignore_object is None:
            raise AttributeError("When ignore is set, ignore_object should be set correctly. you can provided 'ignore_object' column in your supervised_file or you can set ignore_object in dataset config")
        
        item = super().__getitem__(basic_index)
        
        if self.supervised_labels is not None:
            item['label'] = self.supervised_labels[index]
        
        return item
    
