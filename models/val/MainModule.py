import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from .MaskDecoder import MaskDecoder
from .TextVisualEncoder import TextVisualEncoder
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class MainModule(nn.Module):
    def __init__(self, mask_layer, decoder_layer) -> None:
        super(MainModule, self).__init__()

        self.vs_backbone = resnet_fpn_backbone('resnet101', pretrained=True, trainable_layers=5)

        self.encoder = TextVisualEncoder(3, d_model=768)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.text_linear = nn.Linear(768, 768)
        self.activate = nn.ReLU6()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    def forward(self, image, text):
        
        b, n = text.shape[0], text.shape[1]

        image_layer = self.vs_backbone(image)

        h, w = image_layer['2'].shape[2], image_layer['2'].shape[3]

        
        image_layer = torch.cat([nn.AdaptiveAvgPool2d((h, w))(image_layer['1']), image_layer['2'], self.upsample2(image_layer['3'])], dim=1)

        text = self.activate(self.text_linear(text))
        

        image_layer = self.encoder(image_layer.view(b, -1, h*w).permute(0, 2, 1), text, None, None)

        image_layer = self.upsample4(image_layer.permute(0, 2, 1).view(b, 768, h, w))

        mask = torch.bmm(text, image_layer.view(b, -1, 16*h*w))

        mask = nn.Sigmoid()(mask.view(b, n, 4*h, 4*w))
        mask = self.upsample4(mask)

        mask = torch.clamp(mask, 0, 1)
        return mask, image_layer, text




