import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN_Layer(nn.Module):
    """
    Simple Linear layer with dropout.
    """
    def __init__(self, in_features, out_features):
        super(GCN_Layer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

        self.act = nn.ReLU()

    def forward(self, x, adj):
        hidden = self.linear(x)
        support = hidden
        output = self.dic(support)

        return hidden
