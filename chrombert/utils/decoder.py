import torch 
from torch import nn
class DECODER(nn.Module):
    """
    BERT decoder: mlp
    """
    def __init__(self, config):
        super().__init__()
        
        if config.decoder_header == 'single':
            self.task_headers = MaskedLanguageModel(config.hidden_dim, config.vocab_size)
        elif config.decoder_header == 'multitask':
            self.task_headers = nn.ModuleList([MaskedLanguageModel(config.hidden_dim, v) for v in config.vocab_size_multitask])
            
    def forward(self, x, key_padding_mask = None, **kwargs):
        # If self.task_headers is a single MaskedLanguageModel instance

        if isinstance(self.task_headers, MaskedLanguageModel):
            out = self.task_headers(x)
        # If self.task_headers is a nn.ModuleList of MaskedLanguageModel instances
        elif isinstance(self.task_headers, nn.ModuleList):
            out =  [task_header(x) for task_header in self.task_headers]
        else:
            raise ValueError('self.task_headers must be either a MaskedLanguageModel instance or a nn.ModuleList of MaskedLanguageModel instances')
        return out

class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden_dim, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        return self.linear(self.activation(self.dense(x))) 
