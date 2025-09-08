from torch import nn
from layers.MHA import MultiHeadAttention
from layers.FFN import FeedForwardNetwork

class Decoder(nn.Module):
    def __init__(self, d_model, n_head, d_ff, drop_prob):
        super(Decoder, self).__init__()
        self.mha = MultiHeadAttention(d_model, n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.drop_prob = drop_prob

    def forward(self, x, y, enc_mask, dec_mask):
        y_masked = self.mha(y, y, y, dec_mask)  # masked self attention
        y = self.norm1(y + nn.Dropout(self.drop_prob)(y_masked))  # add&norm

        y_x_masked = self.mha(y, x, x, enc_mask)  # multi-head attention
        y = self.norm2(y + nn.Dropout(self.drop_prob)(y_x_masked))  # add&norm

        y_ffn = self.ffn(y) # ffn
        y = self.norm3(y + nn.Dropout(self.drop_prob)(y_ffn)) # add&norm

        return y
