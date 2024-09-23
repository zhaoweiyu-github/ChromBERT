import os 
import pandas as pd 

from .interface import FastaInterface
from .interface_manager import RegulatorInterfaceManager
from ..basic_dataset import BasicDataset

'''
This file implements classes for the prompt-enhanced dataset used for DNA variation. 
Direct usage is not recommended; please use through PromptDataset or DatasetConfig instead.
'''

class PromptDatasetForDNA(BasicDataset):
    def __init__(self,config):
        super().__init__(config)
        self.config = config
        assert isinstance(config.fasta_file, str)
        assert os.path.exists(config.fasta_file), f"fasta file {config.fasta_file=} does not exist"
        self.fasta_interface = FastaInterface(config.fasta_file)
        self.supervised_file = config.supervised_file
        self.supervised(self.supervised_file)

    def supervised(self, supervised_file = None):
        assert isinstance(supervised_file, str) or isinstance(supervised_file, pd.DataFrame)

        if isinstance(supervised_file, pd.DataFrame):
            df_supervised = supervised_file.copy().reset_index(drop=True)
        elif supervised_file.endswith(".csv"):
            df_supervised = pd.read_csv(supervised_file, header = 0) # csv format, [chrom, start, end, build_region_index, label, pos_alt, base_ref, base_ref, metadata]
        elif supervised_file.endswith(".tsv"):
            df_supervised = pd.read_csv(supervised_file, sep="\t", header = 0)
        else:
            raise ValueError(f"suffix of supervised_file {supervised_file} should be csv or tsv")

        self.df_supervised = df_supervised

        self.supervised_indices = df_supervised["build_region_index"]
        self.supervised_indices_len = len(self.supervised_indices)
        self.pos_alt = df_supervised["pos"].values - df_supervised["start"].values -1
        self.base_ref = df_supervised["base_ref"].values
        self.base_alt = df_supervised["base_alt"].values
        self.variant_id = df_supervised["variant_id"].values

        if "sv_label" in df_supervised.columns:
            self.supervised_labels = df_supervised['sv_label'].values
        elif "label" in df_supervised.columns:
            self.supervised_labels = df_supervised['label'].values
        else:
            self.supervised_labels = [None] * len(df_supervised)


    def get_mutant(self, seq, loci, alt):
        seq = list(seq)
        seq[loci] = alt
        return "".join(seq)

    def __len__(self):
        return self.supervised_indices_len

    def __getitem__(self, index):
        basic_index = self.supervised_indices[index]
        fw = 500 

        item = super().__getitem__(basic_index)
        item['label'] = self.supervised_labels[index]
        
        pos_alt = self.pos_alt[index] 
        region = item["region"]
        coord = [region[0].item(), region[1].item() + pos_alt - fw, region[1].item() + pos_alt + fw]
        seq_raw = self.fasta_interface[coord]
        item['seq_raw'] = seq_raw

        variant_id = self.variant_id[index]
        if self.base_ref[index] == "N" : 
            assert item["label"] == 0 
        else: 
            assert seq_raw[fw] == self.base_ref[index], f"{seq_raw[fw]=} != {self.base_ref[index]=} at {seq_raw[fw-3:fw+3]} of {variant_id=}, {item['region']}, {fw=}"
        seq_alt = self.get_mutant(seq_raw, fw, self.base_alt[index])
        item['seq_alt'] = seq_alt

        # if self.metadata is not None:
        #     item['metadata'] = self.metadata[index,:]

        return item
