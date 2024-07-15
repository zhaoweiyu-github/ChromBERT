import os 
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModel

os.environ["TOKENIZERS_PARALLELISM"] = "true"
class DNABERT2Interface(nn.Module):
    """DNA-BERT2 model from hugging face"""
    
    def __init__(self, dnabert2_checkpoint, pooling = 'mean'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(dnabert2_checkpoint, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(dnabert2_checkpoint, trust_remote_code=True)
        self.pooling = pooling
        self.embedding_dim = 768 # the output embedding dimension of DNA-BERT2

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        return None

    def forward(self, dna):
        # dna = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"
        inputs = self.tokenizer(dna, padding = True, truncation = True, return_tensors = 'pt')["input_ids"]
        hidden_states = self.model(inputs.to(self.model.device))[0] # [batch_size, dna_length, embedding_dim]

        if self.pooling == 'mean':
            # embedding with mean pooling
            embedding_dna = torch.mean(hidden_states, dim=1) # [batch_size, embedding_dim]
        elif self.pooling == 'max':
            embedding_dna = torch.max(hidden_states, dim=1)[0] # [batch_size, embedding_dim]
        elif self.pooling == 'cls':
            embedding_dna = hidden_states[:, 0, :]
        else:
            raise(ValueError("Pooling method not supported"))
        
        return {"embedding_dna": embedding_dna, "dna_states": hidden_states}
    