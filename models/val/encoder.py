import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .attention import PositionWiseFeedForward



class EncoderLayer(nn.Module):
    def __init__(self, d_model=256, d_k=32, d_v=32, h=8, d_ff=2048, dropout=.1):
        super(EncoderLayer, self).__init__()
        self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout)

        self.dropout1=nn.Dropout(dropout)
        self.lnorm1=nn.LayerNorm(d_model)

        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)


    def forward(self, input, co_input, attn_map):
        #MHA+AddNorm
        self_att = self.self_att(input, co_input, co_input, attn_map)
        self_att = self.lnorm1(input + self.dropout1(self_att))

        # FFN+AddNorm
        ff = self.pwff(self_att)

        return ff