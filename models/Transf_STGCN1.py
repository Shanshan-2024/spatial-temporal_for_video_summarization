import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.Transf_layer import Encoder
from models.Transf__Net1 import Transf_Net1
from utils.STGCN1 import STGCN
from models.Linear_layer import decoder
from models.graph_transformer_net import GraphTransformerNet

class Transf_STGCN(nn.Module):
    def __init__(self, num_features, args):
        '''
        :param num_features: 1024
        :param args:
        '''
        super(Transf_STGCN, self).__init__()
        self.num_features = num_features
        # self.transf = Transf_Net1()
        self.graphtransformer = GraphTransformerNet(args.g_hidden_dim, args.g_n_heads, args.g_out_dim, args.g_dropout, args.g_n_layers)
        self.STGCN = STGCN(num_features, args)
        self.linear = nn.Linear(self.num_features, 512)
        self.linear1 = nn.Linear(self.num_features, 128)

    def forward(self, X, A_hat, lapla_adj, num_nodes):
        '''
        :param X: [1, 10, 24, 1024]
        :param A_hat: [10, 10]
        :param num_nodes: 10
        :return:
        '''
        T_X = X.reshape(1, -1, self.num_features)
        # T_X = self.linear(T_X)
        # H = self.transf(T_X)
        # H = self.linear1(H)
        H = H.reshape(1, 10, -1, H.shape[-1])
        T = self.STGCN(H, A_hat, num_nodes)
        Z = torch.cat((H, T), dim=-1)
        # Z = H+T
        Z = Z.squeeze(0)
        output, h = self.decode(Z)

        return output, h


