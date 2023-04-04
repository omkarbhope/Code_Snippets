import torch
import torch.nn as nn

class cpnCascade(nn.Module):
    def __init__(self, cpnModel):
        super(cpnCascade, self).__init__()
        self.model = cpnModel

    def forward (self, x):
        res = self.model(x)
        return res