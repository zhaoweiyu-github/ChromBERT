import numpy as np
import pandas as pd
import torch
from typing import Any
from .basic_dataset import IgnoreDataset


class MultiFlankwindowDataset(IgnoreDataset):
    '''
    Dataset class for process multi-flank-window dataset. Supervised file is required.
    '''
    def __init__(self, 
                 config = None, 
                 **params: Any):
        '''
        It's recommend to instantiate the class using DatasetConfig.init(). 
        params:
            config: DatasetConfig. supervised_file must be provided. 
        '''
        super().__init__(config)
        self.flank_window = config.flank_window
        self.max_region_idx = self.len - 1 # self.len from BasicDataset, mean the maximum region idx
        self.supervised(config.supervised_file)

    def supervised(self, supervised_file = None):
        assert isinstance(supervised_file, str)
        if supervised_file.endswith('.csv'):
            df_supervised = pd.read_csv(supervised_file, header = 0) # csv format, [chrom, start, end, build_region_index, label,other_meta]
        elif supervised_file.endswith('.tsv'):
            df_supervised = pd.read_csv(supervised_file, header = 0,sep='\t') # tsv format, [chrom, start, end, build_region_index, label,other_meta]
        elif supervised_file.endswith('.feather'):
            df_supervised = pd.read_feather(supervised_file)
        else: 
            raise(ValueError(f"supervised_file must be csv, tsv, feather file!"))
        neccessary_columns = ["chrom","start","end","build_region_index"]
        for column in neccessary_columns:
            if column not in df_supervised.columns:
                raise(ValueError(f"{column} not in supervised_file! it must contain headers: {neccessary_columns}"))
        self.supervised_indices = df_supervised["build_region_index"]
        self.supervised_indices_len = len(self.supervised_indices)
        
        ### labels
        if 'label' not in df_supervised.columns:  ### only "chrom","start","end","build_region_index" columns and to predict
            self.supervised_labels = None
            print(f"Your file '{supervised_file}' does not contain the 'label' column. Please verify whether the true ground truth ('label') is required. If it is not needed, you may disregard this message.")
        else:
            self.supervised_labels = df_supervised['label'].values
              
          
        
    def __len__(self):
        return self.supervised_indices_len
    
    def _get_item_from_super(self, idx):
        return super().__getitem__(idx)
    
    def __getitem__(self, index):

        label = torch.tensor(self.supervised_labels[index])
        
        region_id = int(self.supervised_indices[index])
        flank_region_id = np.arange(region_id - self.flank_window, region_id + self.flank_window + 1)
        flank_region_id[flank_region_id < 0] = 0
        flank_region_id[flank_region_id > self.max_region_idx] = self.max_region_idx
        input_ids = torch.stack([self._get_item_from_super(id)['input_ids'] for id in flank_region_id])
        position_ids = torch.stack([self._get_item_from_super(id)['position_ids'] for id in flank_region_id])
        center_region = self._get_item_from_super(region_id)['region']
        center_build_region_index = self._get_item_from_super(region_id)['build_region_index']
        return {
            'label': label,
            'flank_region_id': flank_region_id,
            'input_ids': input_ids, 
            'position_ids': position_ids,
            'center_region': center_region,
            'center_build_region_index': center_build_region_index
        }