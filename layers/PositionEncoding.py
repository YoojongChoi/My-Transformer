import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, max_length, d_model):
        super(PositionalEncoding, self).__init__()
        positional_encoding = torch.zeros(max_length, d_model)
        pos = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div = torch.pow(10_000, -1*torch.arange(0, d_model, 2) / d_model)

        positional_encoding[:, 0::2] = torch.sin(pos * div) # 짝수 위치
        positional_encoding[:, 1::2] = torch.cos(pos * div) # 홀수 위치

        positional_encoding = positional_encoding.unsqueeze(0)
        self.register_buffer('positional_encoding', positional_encoding)  # 디코더 쪽에서도 사용하려고

    def forward(self, x):
        return x + self.positional_encoding[:, :x.size(1)]

