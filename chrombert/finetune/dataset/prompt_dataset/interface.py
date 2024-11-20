import os
import h5py
import pickle
import json 
import torch
import numpy as np
import pandas as pd
from functools import lru_cache
from torch.utils.data import Dataset
from pyfaidx import Fasta

class RegulatorEmbInterface():
    '''
    return regulator embedding
    '''
    def __init__(self, h5emb, cache = False, cache_limit = 3):
        super().__init__()
        self.h5emb = h5emb
        self.cache = cache
        assert os.path.exists(self.h5emb)
        with h5py.File(self.h5emb, 'r') as h5f:
            if "region" in h5f.keys():
                k = "region"
            else:
                assert "regions" in h5f.keys(), "`region` or `regions` must be in h5 file"
                k = "regions"
            self.build_region_index_for_emb = h5f[k][:, 3]
            self.dict_region_to_index = {region: i  for i, region in enumerate(h5f[k][:,3])}
            self.regulators = list(regulator for regulator in h5f["emb"].keys())
        self.emb_handler = h5py.File(self.h5emb, 'r')
        if self.cache:
            self.emb_all = torch.from_numpy(self.emb_handler[f"/all"][:])
        self.emb_cache = {}
        self.cache_limit = cache_limit
        

    def __len__(self):
        return len(self.build_region_index_for_emb)

    def valid_regulator(self, regulator):
        return regulator in self.regulators

    def get_emb(self, build_region_index, regulator):
        regulator = regulator.lower()
        index=self.dict_region_to_index[build_region_index]
        assert index is not None
        item={}

        if not self.valid_regulator(regulator):
            raise ValueError(f'{regulator} not in embedding flie')
        elif not self.cache:
            item["emb_regulator"] = self.emb_handler[f"/emb/{regulator}"][index,:]
        elif regulator in self.emb_cache.keys():
            item["emb_regulator"] = self.emb_cache[regulator][index,:]
        elif len(self.emb_cache) < self.cache_limit - 1:
            self.emb_cache[regulator] = self.emb_handler[f"/emb/{regulator}"][:]
            item["emb_regulator"] = self.emb_cache[regulator][index,:]
        else:
            raise ValueError(f'Cache limit reached! Much regulator embedding in cache! If you want to cache more regulator embedding, please set cache_limit larger!')

        if self.cache:
            item["emb_all"] = self.emb_all[index,:]
        else:
            item["emb_all"] = self.emb_handler[f"/all"][index,:]
        return item

class CistromeCellEmbInterface():
    '''
    return cistrome cell embedding
    '''
    def __init__(self, h5emb):
        super().__init__()
        self.h5emb = h5emb
        assert os.path.exists(self.h5emb)
        with h5py.File(self.h5emb, 'r') as h5f:
            if "region" in h5f.keys():
                k = "region"
            else:
                assert "regions" in h5f.keys(), "`region` or `regions` must be in h5 file"
                k = "regions"
            self.build_region_index_for_emb = h5f[k][:, 3]
            self.dict_region_to_index = {region: i  for i, region in enumerate(h5f[k][:,3])}
            self.cistrome_cells = list(cistrome_cell for cistrome_cell in h5f["emb"].keys())
        self.emb_handler = h5py.File(self.h5emb, 'r')

    def __len__(self):
        return len(self.build_region_index_for_emb)

    def valid_cell(self, cistrome_cell):
        return cistrome_cell in self.cistrome_cells

    def get_emb(self, build_region_index, cistrome_cell):
        cistrome=cistrome_cell.split(':')[0].lower()
        cell=cistrome_cell.split(':')[-1].lower().replace('-','').replace('_','').replace(' ','')
        # cistrome_cell = f'{cistrome}:{cell}'
        cistrome_cell = f'{cell}'
        index=self.dict_region_to_index[build_region_index]
        assert index is not None
        item={}
        if self.valid_cell(cistrome_cell):
            item["emb_cell"] = self.emb_handler[f"/emb/{cistrome_cell}"][index,:]
        else:
            raise ValueError(f'{cistrome_cell} not in embedding flie')
        return item

class ExpCellEmbInterface():
    '''
    return expression cell embedding
    '''
    def __init__(self, prompt_exp_cache_file):
        self.emb_file = prompt_exp_cache_file
        assert os.path.exists(self.emb_file)
        if self.emb_file.endswith('.pkl') or self.emb_file.endswith('.pl'):
            with open(self.emb_file, 'rb') as f:
                self.emb_dict = pickle.load(f)
        elif self.emb_file.endswith('.json'):
            with open(self.emb_file, 'r') as f:
                self.emb_dict = json.load(f)
        else:
            raise ValueError(f"emb_file must be pkl or json file!")
        self.cells = list(cell for cell in self.emb_dict.keys())
        
    def valid_cell(self, cell):
        return cell in self.cells

    def get_emb(self, cell):
        re_cell = cell.lower().replace('-','').replace('_','').replace(' ','')
        item={}
        if self.valid_cell(cell):
            item["emb_cell"] = self.emb_dict[cell]
        elif self.valid_cell(re_cell):  
           item["emb_cell"] = self.emb_dict[re_cell]
        else:
            raise ValueError(f'{cell} not in embedding flie')
        return item

class PromptsCistromInterface():
    '''
    parse prompts
    '''
    def __init__(self,meta_file,prompt_map):
        super().__init__()
        self.meta_file = meta_file
        self.prompt_map = prompt_map
        assert os.path.exists(self.meta_file)
        with open(self.meta_file,'r') as f:
            self.meta_dict = json.load(f)
            
    @lru_cache(maxsize=None)
    def regulator_parse_prompts(self, prompt,seq_len):
        item={}
        '''
        parse regulator prompts
        '''
        parse_tensor = torch.zeros(seq_len, dtype=torch.int64)
        item['prompts_all'] = torch.ones(seq_len, dtype=torch.int64)
        if not prompt.startswith("gsm") and not prompt.startswith("enc"): # factor
            seg = self.meta_dict[prompt.lower()].split(';')
        else:
            seg = prompt if isinstance(prompt, list) else prompt.split(';')
        oids = []
        for s in seg:
            oids.append(self.prompt_map[s])
        parse_tensor[oids] = 1
        item['prompts_regulator'] = parse_tensor
        return item
    
    @lru_cache(maxsize=None)
    def cistrome_celltype_parse_prompts(self, cistrome_cell,seq_len):
        '''
        parse cistorme cell prompts
        '''
        item={}
        parse_tensor = torch.zeros(seq_len, dtype=torch.int64)
        if not cistrome_cell.startswith("gsm") and not cistrome_cell.startswith("enc"): # example:dnase:k562
            cistrome=cistrome_cell.split(':')[0].lower()
            cell=cistrome_cell.split(':')[-1].lower().replace('-','').replace('_','').replace(' ','')
            cistrome_cell = f'{cistrome}:{cell}'
            seg = self.meta_dict[cistrome_cell].split(';')
        else:
            seg = cistrome_cell if isinstance(cistrome_cell, list) else cistrome_cell.split(';')
        oids = []
        for s in seg:
            oids.append(self.prompt_map[s])
        parse_tensor[oids] = 1
        item['prompts_cell'] = parse_tensor
        return item
    

        
class FastaInterface():
    def __init__(self, fasta_file):
        self.fasta_file = fasta_file

    def __getitem__(self, coord):
        chrom, start, end = coord
        fasta = Fasta(self.fasta_file)
        if isinstance(chrom, str):
            if chrom.startswith('chr'):
                seq =  fasta[chrom][start:end].seq.upper()
            else:
                raise(TypeError(f"chrom in str format 'chr' prefix"))
        else:
            if chrom == 24:
                seq =  fasta[f'chrX'][start:end].seq.upper()
            elif chrom == 25:
                seq =  fasta[f'chrY'][start:end].seq.upper()
            else:
                seq = fasta[f'chr{chrom}'][start:end].seq.upper()
        fasta.close()
        return seq
        
