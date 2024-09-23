import os 
import torch 
from torch import nn 
from abc import abstractmethod, ABC
from chrombert import ChromBERT
from .utils import ChromBERTEmbedding
from .utils import PoolFlankWindow

class BasicModel(nn.Module, ABC):
    '''
    An abstract class for fine-tuning ChromBERT, which should not be instantiated directly. 
    '''
    def __init__(self, pretrain_config, finetune_config):
        '''
        pretrain_config: ChromBERTConfig object
        finetune_config: FinetuneConfig

        The model will be initialized using the following steps:
            self.pretrain_config = pretrain_config
            self.finetune_config = finetune_config
            self.create_layers() 
        '''
        super().__init__()
        self.pretrain_config = pretrain_config
        self.finetune_config = finetune_config
        self.create_layers()
        return None

    @abstractmethod
    def create_layers(self):
        '''
        add a supervised header to the model
        '''
        raise NotImplementedError

    def load_ckpt(self, ckpt = None):
        if ckpt is not None:
            assert os.path.exists(ckpt), f"Checkpoint file does not exist: {ckpt}"
        else:
            print("No checkpoint file specified, load from finetune_config.finetune_ckpt")
            if self.finetune_config.finetune_ckpt is not None:
                ckpt = self.finetune_config.finetune_ckpt
                assert os.path.exists(ckpt), f"Checkpoint file does not exist: {ckpt}"
            else:
                raise ValueError(f"{ckpt} is not specified!")
        print(f"Loading checkpoint from {ckpt}")

        old_state = self.state_dict()
        new_state = torch.load(ckpt)

        if "state_dict" in new_state:
            new_state = new_state["state_dict"]

        # check whether ckpt from pl module, which has prefix "model."
        num = len([key for key in new_state.keys() if key.startswith("model.")])
        if num/len(new_state) > 0.9:
            new_state = {k[6:]: v for k, v in new_state.items() if k.startswith("model.")}
            print("Loading from pl module, remove prefix 'model.'")
        
        num = len(new_state)
        new_state = {k: v for k, v in new_state.items() if k in old_state} # only load the keys that are in the model
        print(f"Loaded {len(new_state)}/{num} parameters")
        old_state.update(new_state)
        self.load_state_dict(old_state)
        return None 

    def display_trainable_parameters(self, verbose = True):
        '''
        display the number of trainable parameters in the model
        '''
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        o = {"total_params": total_params, "trainable_params": trainable_params}
        print(o)
        if verbose:
            for name, parameter in self.named_parameters():
                if parameter.requires_grad:
                    print(name, ": trainable")
                else:
                    print(name, ": frozen")
        return o 

    def get_pretrain(self):
        '''
        get the pretrain part of the model
        '''
        if hasattr(self, "pretrain_model"):
            assert isinstance(self.pretrain_model, ChromBERT)
            pretrain_model = self.pretrain_model
        else:
            if self.finetune_config.task == "gep":
                pretrain_model = self.pool_flank_window.pretrain_model
                assert isinstance(pretrain_model, ChromBERT)
            else:
                raise ValueError("pretrain_model is not specified! Please specify the pretrain_model attribute in the model, or overwrite this method.")
        return pretrain_model


    def freeze_pretrain(self, trainable = 2):
        '''
        Freeze the model's parameters, allowing fine-tuning of specific transformer blocks.
        For trainable = N layers:
        - If `N = 0`, all transformer blocks are frozen.
        - If `N > 0`, only the last N transformer blocks are trainable and all other blocks are frozen.
        '''
        pretrain_model = self.get_pretrain()
        pretrain_model.freeze(trainable)
        return self
    
    def save_pretrain(self, save_path):
        '''
        save the pretrained part of the model to enable loading it later.
        '''
        pretrain_model = self.get_pretrain()
        state_dict = pretrain_model.state_dict()
        torch.save(state_dict, save_path)
        return state_dict

    def get_embedding_manager(self, **kwargs):
        '''
        get a embedding manager for the pretrain model.
        params:
            kwargs: additional parameters for EmbManager
        '''
        pretrain_model = self.get_pretrain()
        finetune_config = self.finetune_config.clone()
        finetune_config.update(**kwargs)
        model_emb = ChromBERTEmbedding(pretrain_model, finetune_config.mtx_mask, finetune_config.ignore, finetune_config.ignore_index)
        return model_emb

    def save_ckpt(self, save_path):
        '''
        save the model checkpoint
        '''
        state_dict = self.state_dict()
        torch.save(state_dict, save_path)
        return None 
