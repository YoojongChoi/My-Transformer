import torch
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()

        self.w_q = nn.Linear(d_model, d_model, bias = False)
        self.w_k = nn.Linear(d_model, d_model, bias = False)
        self.w_v = nn.Linear(d_model, d_model, bias = False)

        self.n_head = n_head
        self.d_k = d_model // n_head

        self.projection = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        # softmax( q * k^T/sqrt(d_k) ) * v
        scores = torch.matmul(q, k.transpose(2, 3)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = torch.softmax(scores, dim=-1)
        return torch.matmul(attention, v)

    def forward(self, input_q, input_k, input_v, mask=None):
        q = self.w_q(input_q)
        k = self.w_k(input_k)
        v = self.w_v(input_v)

        # Head 쪼개기
        q = q.view(input_q.size(0), -1, self.n_head, self.d_k).transpose(1, 2)
        k = k.view(input_k.size(0), -1, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(input_v.size(0), -1, self.n_head, self.d_k).transpose(1, 2)

        per_head = self.scaled_dot_product_attention(q, k, v, mask)
        concatenated = per_head.transpose(1, 2).contiguous().view(input_q.size(0), -1, self.n_head * self.d_k)
        return self.projection(concatenated)
