import torch
import torch.nn as nn
import torch.nn.functional as F
from mmedit.models.registry import BACKBONES
from segmentation_models_pytorch import DeepLabV3Plus


@BACKBONES.register_module()
class BGMV2DeepLabV3Plus(nn.Module):
    def __init__(self, encoder_name="resnet50", in_channels=3, base_out_channel=32):
        super(BGMV2DeepLabV3Plus, self).__init__()
        self.model = DeepLabV3Plus(encoder_name=encoder_name, in_channels=in_channels, classes=base_out_channel)
        self.refine_block1 = nn.Sequential(
            nn.Conv2d(base_out_channel + 3, base_out_channel + 3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_out_channel + 3, 16, kernel_size=1),
        )
        self.refine_block2 = nn.Sequential(
            nn.Conv2d(16 + 3, 16 + 3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16 + 3, 1, kernel_size=1),
        )

    def forward(self, x):
        x_512 = F.interpolate(x, (512, 512), mode='bilinear', align_corners=True)
        base_out = self.model(x_512)
        input_input1 = torch.concat([x_512, base_out], dim=1)
        refine_out1 = self.refine_block1(input_input1)
        refine_out1 = F.interpolate(refine_out1, (800, 800), mode='bilinear', align_corners=True)
        input_input2 = torch.concat([x, refine_out1], dim=1)
        out = self.refine_block2(input_input2)
        out = out.clamp_(0., 1.)
        return out

    def init_weights(self, pretrained=None):
        pass
