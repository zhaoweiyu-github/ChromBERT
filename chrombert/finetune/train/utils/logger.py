import torch 

class LogTensorValues:
    def __init__(self):
        self.log_data = {}

    def log(self, key, values):
        if key not in self.log_data:
            self.log_data[key] = []
        self.log_data[key].append(values.reshape(-1).squeeze())

    def get_values(self, key):
        if key not in self.log_data:
            return None
        return torch.cat(self.log_data[key])
