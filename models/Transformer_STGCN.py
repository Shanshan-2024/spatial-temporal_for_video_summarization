import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.TimeConv import STGCN
from layers.Transf_layer import Encoder
from torch.autograd import Variable

# class TSTGCN(nn.Module):
#     def __init__(self):
#         super(TSTGCN, self).__init__()
#         # For positional encoding
#         num_timescales = 1024 // 2
#         max_timescale = 10000.0
#         min_timescale = 1.0
#         log_timescale_increment = (
#                 math.log(float(max_timescale) / float(min_timescale)) /
#                 max(num_timescales - 1, 1))
#         inv_timescales = min_timescale * torch.exp(
#             torch.arange(num_timescales, dtype=torch.float32) *
#             -log_timescale_increment)
#         self.register_buffer('inv_timescales', inv_timescales)
#
#         #编码器
#         # self.encode = Encoder(hidden_size=512, filter_size=1, dropout_rate=0.1, n_layers=6)
#         self.encode = Encoder(hidden_size=1024, filter_size=1, dropout_rate=0.1, n_layers=6)
#         self.hidden_size = 1024
#
#     # def forward(self, X, A_hat):
#     #     #X[1, 10, 24, 1024]
#     #     #首先需要把特征和邻接矩阵传过来,只需要transformer的编码部分
#     #     #第一步进行位置编码
#     #         #输入：X，得到与X同一个维度的PE
#     #         #X与PE相加
#     #     #第二步将进行位置编码后的特征向量输入多层encoder
#     #         #对每一层encoder
#     #             #自注意力，前馈，归一化，残差
#     #     #得到encoder编码后的特征，输入STGCN
#     #     #如果是多层的话将STGCN放在transformer的编码层里
#     #     X = torch.randn(1, 332, 1024)
#     #     PE = self.get_position_encoding(X)
#     #     #PE [1, 10, 1024]
#     #     X = X + PE
#     #     X = X.squeeze(0)
#     #     #如何获得mask
#     #     # i_mask = self.create_pad_mask(X, None)
#     #     #X [1, 332, 1024]
#     #     output = self.encode(X)
#     #     return output
#
#     def forward(self, X):
#         #X[1, 10, 24, 1024]
#         #首先需要把特征和邻接矩阵传过来,只需要transformer的编码部分
#         #第一步进行位置编码
#             #输入：X，得到与X同一个维度的PE
#             #X与PE相加
#         #第二步将进行位置编码后的特征向量输入多层encoder
#             #对每一层encoder
#                 #自注意力，前馈，归一化，残差
#         #得到encoder编码后的特征，输入STGCN
#         #如果是多层的话将STGCN放在transformer的编码层里
#         #X [236, 1024] -> [1, 236, 1024] -> + PE -> [1, 236, 1024] -> [236, 1024] -> encode -> [236, 1024]
#         X = X.unsqueeze(0)
#         PE = self.get_position_encoding(X)
#         #PE [1, 10, 1024]
#         X = X + PE
#         X = X.squeeze(0)
#         #如何获得mask
#         # i_mask = self.create_pad_mask(X, None)
#         #X [1, 332, 1024]
#         output = self.encode(X)
#         return output
#
#     def get_position_encoding(self, x):
#         max_length = x.size()[1]
#         position = torch.arange(max_length, dtype=torch.float32,
#                                 device=x.device)
#         scaled_time = position.unsqueeze(1) * self.inv_timescales.unsqueeze(0)
#         signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)],
#                            dim=1)
#         signal = F.pad(signal, (0, 0, 0, self.hidden_size % 2))
#         signal = signal.view(1, max_length, self.hidden_size)
#         return signal
#
#     def create_pad_mask(self, t, pad):
#         return (t == pad).unsqueeze(-2)

#main_Hierarchical_Tenporal_Transformer

class TSTGCN(nn.Module):
    def __init__(self):
        super(TSTGCN, self).__init__()
        # For positional encoding
        num_timescales = (1024 // 2 )
        max_timescale = 10000.0
        min_timescale = 1.0
        log_timescale_increment = (
                math.log(float(max_timescale) / float(min_timescale)) /
                max(num_timescales - 1, 1))
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32) *
            -log_timescale_increment)
        self.register_buffer('inv_timescales', inv_timescales)

        #编码器
        # self.encode = Encoder(hidden_size=512, filter_size=1, dropout_rate=0.1, n_layers=6)
        self.encode = Encoder(hidden_size=1024, filter_size=1, n_layers=6)

    # def forward(self, X, A_hat):
    #     #X[1, 10, 24, 1024]
    #     #首先需要把特征和邻接矩阵传过来,只需要transformer的编码部分
    #     #第一步进行位置编码
    #         #输入：X，得到与X同一个维度的PE
    #         #X与PE相加
    #     #第二步将进行位置编码后的特征向量输入多层encoder
    #         #对每一层encoder
    #             #自注意力，前馈，归一化，残差
    #     #得到encoder编码后的特征，输入STGCN
    #     #如果是多层的话将STGCN放在transformer的编码层里
    #     X = torch.randn(1, 332, 1024)
    #     PE = self.get_position_encoding(X)
    #     #PE [1, 10, 1024]
    #     X = X + PE
    #     X = X.squeeze(0)
    #     #如何获得mask
    #     # i_mask = self.create_pad_mask(X, None)
    #     #X [1, 332, 1024]
    #     output = self.encode(X)
    #     return output

    def forward(self, X):
        #X[1, 10, 24, 1024]
        #首先需要把特征和邻接矩阵传过来,只需要transformer的编码部分
        #第一步进行位置编码
            #输入：X，得到与X同一个维度的PE
            #X与PE相加
        #第二步将进行位置编码后的特征向量输入多层encoder
            #对每一层encoder
                #自注意力，前馈，归一化，残差
        #得到encoder编码后的特征，输入STGCN
        #如果是多层的话将STGCN放在transformer的编码层里
        #X [236, 1024] -> [1, 236, 1024] -> + PE -> [1, 236, 1024] -> [236, 1024] -> encode -> [236, 1024]
        X = X.unsqueeze(0)
        # X = torch.randn(10, 24, 1024)
        PE = self.get_position_encoding(X)
        #PE [1, 10, 1024]
        X = X + PE
        X = X.squeeze(0)
        #如何获得mask
        # i_mask = self.create_pad_mask(X, None)
        #X [1, 332, 1024]
        output = self.encode(X)
        # output = output.reshape(output.shape[0], -1, 1024)
        # output = output.transpose(0, 1)
        return output

    def get_position_encoding(self, x):
        max_length = x.size()[1]
        position = torch.arange(max_length, dtype=torch.float32,
                                device=x.device)
        scaled_time = position.unsqueeze(1) * self.inv_timescales.unsqueeze(0)
        signal = signal
        signal = F.pad(signal, (0, 0, 0, self.hidden_size % 2))
        signal = signal.view(1, max_length, self.hidden_size)
        return signal

    def create_pad_mask(self, t, pad):
        return (t == pad).unsqueeze(-2)
