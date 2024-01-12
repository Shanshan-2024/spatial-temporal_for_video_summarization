import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transf_layer import Encoder
from models.STGCN2 import STGCN2
from layers.Linear_layer import decoder

class Transf_STGCN2(nn.Module):
    def __init__(self, num_features, args):
        '''
        :param num_features: 1024
        :param args:
        '''
        super(Transf_STGCN2, self).__init__()
        self.num_features = num_features
        self.STGCN2 = STGN2C(num_features, args)
        self.decode = decoder(256)
    def forward(self, X, A_hat, num_nodes):
        '''
        :param X: [1, 10, 24, 1024]
        :param A_hat: [10, 10]
        :param num_nodes: 10
        :return:
        '''
        T = self.STGCN2(X, A_hat, num_nodes)
        output, h = self.decode(T)
        return output, h


