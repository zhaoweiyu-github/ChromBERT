import os
import torch
import pandas as pd
from torch import nn
class CistromeEmbeddingManager(nn.Module):
    def __init__(self, mtx_mask, ignore = False,ignore_index = None) -> None:
        super().__init__()
        assert mtx_mask is not None, "mtx_mask must be specified"
        assert isinstance(mtx_mask, str), "mtx_mask must be a path to a mtx_mask" 
        assert os.path.exists(mtx_mask), f"{mtx_mask} does not exist"
        self.mtx_mask_df = pd.read_csv(mtx_mask, sep='\t', index_col=0)
        self.mtx_mask = torch.tensor(self.mtx_mask_df.values) # (datasets, factors)
        self.gsmid_names = self.mtx_mask_df.index.tolist()
        self.regulator_names = self.mtx_mask_df.columns.tolist()

        if ignore:
            ignore_gsmid_index = ignore_index[0]
            ignore_regulator_index = ignore_index[1]
            print(f"Ignoring {len(ignore_gsmid_index)} cistromes and {len(ignore_regulator_index)} regulators")
            rows_to_keep = torch.tensor([i not in ignore_gsmid_index for i in range(self.mtx_mask.shape[0])])
            cols_to_keep = torch.tensor([j not in ignore_regulator_index for j in range(self.mtx_mask.shape[1])])
            self.mtx_mask=self.mtx_mask[rows_to_keep][:, cols_to_keep]
            self.mtx_mask_df = self.mtx_mask_df.iloc[rows_to_keep.numpy(), cols_to_keep.numpy()]
            self.gsmid_names = self.mtx_mask_df.index.tolist()
            self.regulator_names = self.mtx_mask_df.columns.tolist()

        factor_num = (self.mtx_mask != 0).sum(dim=0)
        self.normalization_factors = factor_num.clamp(min=1)
        self.normalized_mtx_mask = (self.mtx_mask / self.normalization_factors)
        return None 

    def forward(self, x):
        # x: [batch_size, datasets, hidden]
        self.normalized_mtx_mask = self.normalized_mtx_mask.to(x.device)
        self.normalized_mtx_mask = self.normalized_mtx_mask.to(x.dtype)
        x = x.transpose(1, 2)
        x = torch.matmul(x, self.normalized_mtx_mask)
        x = x.transpose(1, 2) # [batch_size, factors, hidden]

        return x


    def get_cistrome_embedding(self, x, gsmid):
        assert gsmid in self.gsmid_names, f"{gsmid} not found in GSMID names"
        index = self.gsmid_names.index(gsmid)
        return x[:, index, :]

    def get_regulator_embedding(self, x, regulator):
        # x: [batch_size, datasets, hidden]
        assert regulator in self.regulator_names, f"{regulator} not found in regulator names"
        index = self.regulator_names.index(regulator)
        x = x.transpose(1, 2)
        self.normalized_mtx_mask = self.normalized_mtx_mask.to(x.device)
        extracted_x = torch.matmul(x, self.normalized_mtx_mask[:, index:index+1])
        return extracted_x.transpose(1, 2)

    def get_region_embedding(self, x):
        return x.mean(dim=1)


class ChromBERTEmbedding(nn.Module):
    def __init__(self, pretrain_model, mtx_mask, ignore = False,ignore_index = None) -> None:
        super().__init__()
        self.pretrain_model = pretrain_model
        self.CistromeEmbeddingManager = CistromeEmbeddingManager(mtx_mask, ignore = ignore,ignore_index = ignore_index)
        self.__hidden_cistrome = None
        self.__hidden_regulator = None
        self.__training = pretrain_model.training
        self.list_regulator = self.CistromeEmbeddingManager.regulator_names
        self.list_cistrome = self.CistromeEmbeddingManager.gsmid_names


    def forward(self, batch):
        with torch.no_grad():
            self.pretrain_model.eval()
            x = self.pretrain_model(batch["input_ids"], batch["position_ids"])
            self.__hidden_cistrome = x
            emb = self.CistromeEmbeddingManager(x)
            self.__hidden_regulator = emb
            if self.__training:
                self.pretrain_model.train()
        return emb

    def get_hidden_state(self):
        return self.__hidden_state

    def get_cistrome_embedding(self, gsmid):
        gsmid = gsmid.lower()
        return self.CistromeEmbeddingManager.get_cistrome_embedding(self.__hidden_cistrome, gsmid)

    def get_regulator_embedding(self, regulator):
        regulator = regulator.lower()
        # return self.CistromeEmbeddingManager.get_regulator_embedding(self.__hidden_state, regulator)
        assert regulator in self.list_regulator, f"{regulator} not found in regulator names"
        index = self.list_regulator.index(regulator)
        return self.__hidden_regulator[:, index, :]

    def get_region_embedding(self):
        return self.CistromeEmbeddingManager.get_region_embedding(self.__hidden_cistrome)

    