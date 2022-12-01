import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from .decoder import DecoderLayer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




class MaskDecoder(nn.Module):
    def __init__(self, N_dec, d_model=256, d_k=32, d_v=32, h=8, d_ff=2048, dropout=.1):
        super(MaskDecoder, self).__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout) for _ in range(N_dec)])

        self.mask_activate = nn.Sigmoid()
        self.N = N_dec
        self.h = h

    def _map(self, feature, kernel):
        
        b = feature.shape[0]
        h, w = feature.shape[2], feature.shape[3]
        n = kernel.shape[1]
        mask = torch.bmm(kernel, feature.view(b, -1, h*w))

        assert mask.shape==(b, n, h*w)
        return self.mask_activate(mask).view(b, n, h, w)

    def forward(self, inputs, encoder_output, gt_mask, mask):
        # input (b, n, c)
        # encoder_output  b, c, h, w
        # mask b, n, h, w

        out = inputs

        b = inputs.shape[0]
        n = inputs.shape[1]
        c = encoder_output.shape[1]
        h = encoder_output.shape[2]
        w = encoder_output.shape[-1]
        

        for i, l in enumerate(self.layers):
            
            attn_map = (mask.view(b, n, -1).unsqueeze(dim=1).expand(b, self.h, n, h*w)) < 0.3
            out = l(out, encoder_output.view(b, -1, h*w).permute(0, 2, 1), gt_mask, attn_map)
            mask = self._map(encoder_output, out)

        return out, mask
