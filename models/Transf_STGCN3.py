import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transf_layer import Encoder
from models.STGCN3 import *
from layers.Linear_layer import decoder
from models.LTT import temporal_transformer


class Transf_STGCN3(nn.Module):
    def __init__(self, num_features, args):
        '''
        :param num_features: 1024
        :param args:
        '''
        super(Transf_STGCN3, self).__init__()
        self.num_features = num_features
        self.STGCN3 = STGCN3(num_features, args)

    def forward(self, X, A_hat, num_nodes):
        '''
        :param X: [1, 10, 24, 1024]
        :param A_hat: [10, 10]
        :param num_nodes: 10
        :return:
        '''
        T = self.STGCN3(X, A_hat, num_nodes)

        output, h = self.decode(T)

        return output, h


