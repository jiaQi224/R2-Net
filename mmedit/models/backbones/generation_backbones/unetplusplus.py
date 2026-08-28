# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.runner import load_checkpoint

from mmedit.models.common import (UnetSkipConnectionBlock,
                                  generation_init_weights)
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger
import torch.nn as nn
import torch.nn.functional as F
import torch
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import SegmentationHead


@BACKBONES.register_module()
class UnetPlusPlus(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, mutil_att=False, **kwargs):
        super(UnetPlusPlus, self).__init__(**kwargs)

        self.mutil_att = mutil_att
        if self.mutil_att:
            self.model = smp.UnetPlusPlus(encoder_name="resnet50", in_channels=in_channels, classes=32)
            self.head1 = SegmentationHead(32, 32)
            self.head2 = SegmentationHead(32, 32)
            self.merge = MixStructureBlock(1, 64)
            self.head = SegmentationHead(64, 3)
        else:
            self.model = smp.UnetPlusPlus(encoder_name="resnet50", in_channels=in_channels, classes=64)
            self.head = SegmentationHead(64, out_channels)

    def forward(self, x):
        out = self.model(x)
        if self.mutil_att:
            head1_out = self.head1(out)
            head2_out = self.head2(out)
            merged = self.merge(torch.concat([head1_out, head2_out], dim=1))
            out = self.head(merged)
        else:
            out = self.head(out)
        return out

    def init_weights(self, pretrained):
        pass


class MixStructureBlock(nn.Module):
    def __init__(self, net_depth, dim, kernel_size=3, gate_act=nn.Sigmoid):
        super().__init__()

        self.norm1 = nn.InstanceNorm2d(dim)
        self.norm2 = nn.InstanceNorm2d(dim)

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, padding_mode='reflect')
        self.conv3_19 = nn.Conv2d(dim, dim, kernel_size=7, padding=9, groups=dim, dilation=3, padding_mode='reflect')
        self.conv3_13 = nn.Conv2d(dim, dim, kernel_size=5, padding=6, groups=dim, dilation=3, padding_mode='reflect')
        self.conv3_7 = nn.Conv2d(dim, dim, kernel_size=3, padding=3, groups=dim, dilation=3, padding_mode='reflect')

        # Simple Channel Attention
        self.Wv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=3 // 2, groups=dim, padding_mode='reflect')
        )
        self.Wg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )
        # Channel Attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        # Pixel Attention
        self.pa = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1, padding=0, bias=True),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.mlp = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 4, 1),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim * 4, dim, 1)
        )
        self.mlp2 = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 4, 1),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim * 4, dim, 1)
        )

    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.cat([self.conv3_19(x), self.conv3_13(x), self.conv3_7(x)], dim=1)
        x = self.mlp(x)
        x = identity + x

        identity = x
        x = self.norm2(x)
        x = torch.cat([self.Wv(x) * self.Wg(x), self.ca(x) * x, self.pa(x) * x], dim=1)
        x = self.mlp2(x)
        x = identity + x
        return x