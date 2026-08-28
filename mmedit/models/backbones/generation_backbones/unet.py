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


class UnetBackbone(nn.Module):
    def __init__(self, in_channels=3, channel_list=[64, 128, 256, 512, 1024], g_block=False, **kwargs):
        super(UnetBackbone, self).__init__(**kwargs)
        self.inc = InConv(in_channels, channel_list[0])
        self.down1 = Down(channel_list[0], channel_list[1], g_block=g_block)
        self.down2 = Down(channel_list[1], channel_list[2], g_block=g_block)
        self.down3 = Down(channel_list[2], channel_list[3], g_block=g_block)
        self.down4 = Down(channel_list[3], channel_list[4], g_block=g_block)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        return [x1, x2, x3, x4, x5]


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, g_block=False):
        super(Down, self).__init__()
        if g_block:
            self.down_conv = nn.Sequential(
                nn.MaxPool2d(2),
                GBlock(in_ch, out_ch)
            )
        else:
            self.down_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_ch, out_ch)
            )

    def forward(self, x):
        x = self.down_conv(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class ConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, gate_act=nn.Sigmoid):
        super().__init__()
        self.kernel_size = kernel_size

        self.Wv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1),
            nn.Conv2d(out_dim, out_dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=out_dim,
                      padding_mode='reflect')
        )

        self.Wg = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1),
            gate_act() if gate_act in [nn.Sigmoid, nn.Tanh] else gate_act(inplace=True)
        )


    def forward(self, X):
        out = self.Wv(X) * self.Wg(X)

        return out


class GBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d,
                 gate_act=nn.Sigmoid):
        super().__init__()
        self.norm = norm_layer(in_dim)
        self.conv = conv_layer(in_dim, in_dim, kernel_size, gate_act)
        self.proj = nn.Conv2d(in_dim, out_dim, 1)

    def forward(self, x):
        identity = x
        x = self.norm(x)
        x = self.conv(x)
        x = identity + x
        x = self.proj(x)
        return x


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UnetHead(nn.Module):
    def __init__(self, out_channels=3, decoder_channel=[1024, 512, 256, 128, 64],g_block=False, **kwargs):
        super(UnetHead, self).__init__(**kwargs)

        self.up1 = Up(decoder_channel[0], decoder_channel[1], g_block=g_block)
        self.up2 = Up(decoder_channel[1] * 2, decoder_channel[2], g_block=g_block)
        self.up3 = Up(decoder_channel[2] * 2, decoder_channel[3], g_block=g_block)
        self.up4 = Up(decoder_channel[3] * 2, decoder_channel[4], g_block=g_block)

    def forward(self, inputs):
        out = self.up1(inputs[4], inputs[3])
        out = self.up2(out, inputs[2])
        out = self.up3(out, inputs[1])
        out = self.up4(out, inputs[0])
        return out


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True, g_block=False):
        # 定义了self.up的方法
        super(Up, self).__init__()
        if g_block:
            self.conv = GBlock(in_ch, out_ch)
        else:
            self.conv = DoubleConv(in_ch, out_ch)
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)  # // 除以的结果向下取整

    def forward(self, x1, x2):  # x2是左侧的输出，x1是上一大层来的输出
        x1 = self.conv(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)  # 将两个tensor拼接在一起 dim=1：在通道数（C）上进行拼接
        return x


@BACKBONES.register_module()
class UnetOrig(nn.Module):
    def __init__(self, in_channels=3,
                 out_channels=64,
                 g_block=False,
                 **kwargs):
        super(UnetOrig, self).__init__(**kwargs)
        self.backbone = UnetBackbone(in_channels=in_channels, g_block=g_block)
        self.head = UnetHead(out_channels=out_channels, g_block=g_block)
        self.final = nn.Conv2d(128, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        feature_list = self.backbone(x)
        out = self.head(feature_list)
        out = self.final(out)
        return out + x

    def init_weights(self, pretrained):
        pass
