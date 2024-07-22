import torch 

class LogTensorValues:
    def __init__(self):
        self.log_data = {}

    def log(self, key, values):
        if key not in self.log_data:
            self.log_data[key] = []
        v = values.reshape(-1)
        # make sure v is not a zero-dim tensor
        assert v.dim() > 0, f"LogTensorValues: {key}={v} has zero-dim tensor"
        self.log_data[key].append(v)

    def get_values(self, key):
        if key not in self.log_data:
            return None
        return torch.cat(self.log_data[key])
