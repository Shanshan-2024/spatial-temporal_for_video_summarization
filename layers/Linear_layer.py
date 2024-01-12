import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear(nn.Module):
    '''
    带有dropout的简单线性层
    '''
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in, out_features)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.linear.forward(x)
        return out

class decoder(nn.Module):

    def __init__(self, in_features):
        super(decoder, self).__init__()
        self.linear1 = nn.Linear(in_f, 2)
        self.linear2 = nn.Linear(2, 1)
        self.act = nn.Softmax()

    def forward(self, x):

        h = self.p.forward(x)
        h = F.dropout(h, 0.3)
        out = self.l.forward(h)
        return out, h
