import os
import pandas as pd 
from torch.utils.data import Dataset
from .prompt_dataset_two import PromptDatasetForCCTP
from .prompt_dataset_single import PromptDatasetForDNA, PromptDatasetForDNASequence

class PromptDataset(Dataset):
    def __init__(self, config):
        '''
        It's recommend to instantiate the class using DatasetConfig.init(). 
        params:
            config: DatasetConfig. supervised_file must be provided. 
        '''
        super().__init__()
        self.config = config
        if isinstance(self.config.supervised_file, str):
            assert os.path.exists(self.config.supervised_file)
        else:
            assert isinstance(self.config.supervised_file, pd.DataFrame) and self.config.prompt_kind == "dna", "only dna prompt support DataFrame as supervised_file"

        list_available_prompt_kind = ["dna", "cistrome", "expression", "sequence"]
        assert self.config.prompt_kind in list_available_prompt_kind, f"prompt_kind must be one of {list_available_prompt_kind}"
        if self.config.prompt_kind == "dna":
            self.dataset = PromptDatasetForDNA(config)
        elif self.config.prompt_kind in ["cistrome", "expression"]:
            self.dataset = PromptDatasetForCCTP(config)
        elif self.config.prompt_kind == "sequence":
            self.dataset = PromptDatasetForDNASequence(config)
        else:
            raise AttributeError(f"Warning: '{self.config.prompt_kind}' is not a valid prompt cell")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]
    def __getattr__(self, name):
        """
        Delegate attribute and method access to the dataset object if it's not an attribute of this Proxy class.
        """
        return getattr(self.dataset, name)


    