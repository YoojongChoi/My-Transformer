import torch
from torch import nn
from layers.PositionEncoding import PositionalEncoding
from layers.Encoder import Encoder
from layers.Decoder import Decoder
from utils import device

class Transformer(nn.Module):
    def __init__(self, vocab_size):
        super(Transformer, self).__init__()
        # 기본 세팅
        self.n_layers = 6
        self.d_model = 512
        self.n_head = 8
        self.d_ff = 2048
        self.drop_prob = 0.1
        self.max_length = 64
        self.vocab_size = vocab_size
        self.causal_mask = torch.tril(torch.ones(self.max_length-1, self.max_length-1)).to(device).bool() # 여기 device

        # architecture
        self.enc_emb = nn.Embedding(vocab_size, self.d_model) # 입력 임베딩
        self.dec_emb = nn.Embedding(vocab_size, self.d_model) # 출력 임베딩
        self.p_enc = PositionalEncoding(self.max_length, self.d_model)  # 위치 인코딩
        self.encoders = nn.ModuleList([Encoder(self.d_model, self.n_head, self.d_ff, self.drop_prob) for _ in range(self.n_layers)])  # 인코더
        self.decoders = nn.ModuleList([Decoder(self.d_model, self.n_head, self.d_ff, self.drop_prob) for _ in range(self.n_layers)])
        self.linear = nn.Linear(self.d_model, self.vocab_size)

    def forward(self, x, y):  # x = src, y = tgt[:,:-1]
        enc_mask = (x != 0).unsqueeze(1).unsqueeze(2) # [32, 1, 1, 64]
        dec_mask = (y != 0).unsqueeze(1).unsqueeze(3) # [32, 1, 63, 1]
        dec_mask = dec_mask & self.causal_mask  # 디코더의 경우 자신과 자신보다 앞선 위치의 토큰만 참고 가능하도록

        # start!
        x = self.enc_emb(x) # 입력 임베딩
        x = self.p_enc(x) # 임베딩 + 위치 인코딩
        x = nn.Dropout(self.drop_prob)(x)

        for encoder in self.encoders: # 위치_인코딩+입력_임베딩 -> self attention -> add&norm-> FFN -> add&norm
            x = encoder(x, enc_mask)

        y = self.dec_emb(y) # 출력 임베딩
        y = self.p_enc(y)   # 임베딩 + 위치 인코딩
        y = nn.Dropout(self.drop_prob)(y)

        for decoder in self.decoders: # 위치_인코딩+입력_임베딩 -> masked attention -> add&norm -> 입력&출력 attention -> FFN -> add&norm
            y = decoder(x, y, enc_mask, dec_mask)

        y = self.linear(y)  # vocab 길이만큼 늘리기위함

        return y
