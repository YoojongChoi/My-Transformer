from torch import nn
from layers.MHA import MultiHeadAttention
from layers.FFN import FeedForwardNetwork

class Encoder(nn.Module):
    def __init__(self, d_model, n_head, d_ff, drop_prob):
        super(Encoder, self).__init__()
        self.mha = MultiHeadAttention(d_model, n_head)
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)

        self.drop_prob = drop_prob

    def forward(self, x, mask=None):
        x_masked = self.mha(x, x, x, mask)  # self attention
        x = self.norm1(x + nn.Dropout(self.drop_prob)(x_masked)) # add&norm

        x_ffn = self.ffn(x) # ffn
        x = self.norm2(x + nn.Dropout(self.drop_prob)(x_ffn)) # add&norm
        return x