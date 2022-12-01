import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .DSA_attention import DSAMultiHeadAttention
from .attention import PositionWiseFeedForward



class DecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_k=32, d_v=32, h=8, d_ff=2048, dropout=.1):
        super(DecoderLayer, self).__init__()
        self.self_att = DSAMultiHeadAttention(d_model, d_k, d_v, h, dropout)
        self.enc_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout)

        self.dropout1=nn.Dropout(dropout)
        self.lnorm1=nn.LayerNorm(d_model)

        self.dropout2=nn.Dropout(dropout)
        self.lnorm2=nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)


    def forward(self, input, enc_output, input_map, enc_map):
        #MHA+AddNorm

        
        self_att = self.self_att(input, input, input, input_map)
        self_att = self.lnorm1(input + self.dropout1(self_att))

        # MHA+AddNorm
        enc_att = self.enc_att(self_att, enc_output, enc_output, enc_map)
        enc_att = self.lnorm2(self_att + self.dropout2(enc_att))

        # FFN+AddNorm
        ff = self.pwff(enc_att)

        return ff