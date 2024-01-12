import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transf_layer import Encoder

class Transf(nn.Module):
    def __init__(self):
        super(Transf, self).__init__()

        #for position encoding
        num_timescales = (1024 // 2 )
        max_timescale = 10000.0
        min_timescale = 1.0
        log_timescale_increment = (
                math.log(float(max_timescale) / float(min_timescale)) /
                max(num_timescales - 1, 1))
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32)
        self.register_buffer('inv_timescales', inv_timescales)

        #编码器
        self.encode = Encoder(hidden_size=1024, dropout_rate=0.1, n_layers=6)
        self.hidden_size = 1024

    def forward(self, X):
        X = X.unsqueeze(0)
        PE = self.get_position_encoding(X)
        X = X.squeeze(0)
        output = self.encode(X)
        return output

    def get_position_encoding(self):
        max_length = x.size()[1]
        position = torch.arange(max_length, dtype=torch.float32,
                                device=x.device)
        scaled_time = position.unsqueeze(1) * self.inv_timescales.unsqueeze(0)
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)],
                           dim=1)
        signal = F.pad(signal, (0, 0, 0, self.hidden_size % 2))
        signal = signal.view(1, max_length, self.hidden_size)
        return signal