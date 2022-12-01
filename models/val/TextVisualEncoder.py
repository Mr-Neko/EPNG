import torch
from torch import nn
from torch.nn import functional as F
from .decoder import DecoderLayer
import numpy as np




class TextVisualEncoder(nn.Module):
    def __init__(self, N_dec, d_model=768, d_k=96, d_v=96, h=8, d_ff=2048, dropout=.1):
        super(TextVisualEncoder, self).__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout) for _ in range(N_dec)])

        self.N = N_dec

    def forward(self, input, enc_input, input_map, enc_map):

        out = input
        for l in self.layers:
            out = l(out, enc_input, input_map, enc_map)

        return out