# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch
from ..sr_backbones import ops
from mmcv.runner import load_checkpoint
import torch.nn.functional as F
from mmedit.models.common import (PixelShufflePack, ResidualBlockNoBN,
                                  default_init_weights, make_layer)
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger
from collections import OrderedDict
# import ops as MPNCOV
from .ops import *
# import ops
#

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.c1 = BasicBlock(channel, channel // reduction, 1, 1, 0)
        self.c2 = BasicBlockSig(channel // reduction, channel, 1, 1, 0)

    def forward(self, x):
        y = self.avg_pool(x)
        y1 = self.c1(y)
        y2 = self.c2(y1)
        return x * y2
# 核选择模块
class KernelAttention(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, kernels=[3, 5, 7, 9], reduction=16, group=1, L=32):
        super().__init__()
        self.first = nn.Conv2d(in_channel, 64, kernel_size=3, padding=1, padding_mode='reflect', stride=1)
        channel = out_channel
        self.d = max(L, channel // reduction)
        self.convs = nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(out_channel, out_channel, kernel_size=k, padding=k // 2, groups=group)),
                    ('relu', nn.ReLU())
                ]))
            )
        self.fc = nn.Linear(channel, self.d)
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d, out_channel))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.first(x)
        bs, c, _, _ = x.size()
        conv_outs = []
        ### split
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats = torch.stack(conv_outs, 0)  # k,bs,channel,h,w

        ### fuse
        U = sum(conv_outs)  # bs,c,h,w

        ### reduction channel
        S = U.mean(-1).mean(-1)  # bs,c
        Z = self.fc(S)  # bs,d

        ### calculate attention weight
        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights.append(weight.view(bs, c, 1, 1))  # bs,channel
        attention_weughts = torch.stack(weights, 0)  # k,bs,channel,1,1
        attention_weughts = self.softmax(attention_weughts)  # k,bs,channel,1,1

        ### fuse
        V = (attention_weughts * feats).sum(0)
        return V


# elan模块
class ElanBlcok(nn.Module):
    def __init__(self, ch_in, ch_out, flg=False):

        super(ElanBlcok, self).__init__()
        # 卷积类型一
        self.conv1 = Bconv(ch_in, ch_out, k=1, s=1)
        # 卷积类型二
        self.conv2 = Bconv(ch_out, ch_out, k=3, s=1)

        # cat之后的卷积
        if flg:
            self.conv3 = Bconv(4 * ch_in, ch_in, k=1, s=1)
        else:
            self.conv3 = Bconv(4 * ch_in, ch_out, k=1, s=1)
        self.ca = CALayer(channel=64)
        self.conv4 = Bconv(ch_out, ch_out, k=3, s=1)


    def forward(self, x):
        resudial = x
        # 分支一输出
        output1 = self.conv1(x)
        # 分支二输出
        output2_1 = self.conv1(x)
        output2_2 = self.conv2(output2_1)
        output2_3 = self.conv2(output2_2)
        output2_4 = self.conv2(output2_3)
        output2_5 = self.conv2(output2_4)
        output_cat = torch.cat((output1, output2_1, output2_3, output2_5), dim=1)
        output = self.conv3(output_cat)
        output = self.ca(output)
        output = self.conv4(output)
        return output + resudial


class Bconv(nn.Module):
    def __init__(self, ch_in, ch_out, k, s):
        '''
        :param ch_in: 输入通道数
        :param ch_out: 输出通道数
        :param k: 卷积核尺寸
        :param s: 步长
        :return:
        '''
        super(Bconv, self).__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, k, s, padding=k // 2)
        self.act = nn.SiLU()

    def forward(self, x):
        '''
        :param x: 输入
        :return:
        '''
        return self.act(self.conv(x))


# 非对称卷积核
class ACkernel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ACkernel, self).__init__()
        self.conv1x3 = nn.Conv2d(in_channels, out_channels, (1, 3), 1, (0, 1))
        self.conv3x1 = nn.Conv2d(in_channels, out_channels, (3, 1), 1, (1, 0))
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, (3, 3), 1, (1, 1))

    def forward(self, x):
        conv3x1 = self.conv3x1(x)
        conv1x3 = self.conv1x3(x)
        conv3x3 = self.conv3x3(x)
        return conv3x1 + conv1x3 + conv3x3


class AcElanBlock(nn.Module):
    def __init__(self, inChannals, outChannals):
        """初始化残差模块"""
        super(AcElanBlock, self).__init__()
        self.conv1 = nn.Conv2d(inChannals, outChannals, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outChannals)
        self.conv2 = ElanBlcok(outChannals, outChannals)
        self.bn2 = nn.BatchNorm2d(outChannals)
        self.conv3 = nn.Conv2d(inChannals, outChannals, kernel_size=1, bias=False)
        self.relu = nn.PReLU()

    def forward(self, x):
        """前向传播过程"""
        resudial = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)

        out += resudial
        out = self.relu(out)
        return out


class AcBlock(nn.Module):
    def __init__(self, inChannals, outChannals):
        """初始化残差模块"""
        super(AcBlock, self).__init__()
        self.conv1 = nn.Conv2d(inChannals, outChannals, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outChannals)
        self.conv2 = ACkernel(outChannals, outChannals)
        self.bn2 = nn.BatchNorm2d(outChannals)
        self.conv3 = nn.Conv2d(inChannals, outChannals, kernel_size=1, bias=False)
        self.relu = nn.PReLU()

    def forward(self, x):
        """前向传播过程"""
        resudial = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)

        out += resudial
        out = self.relu(out)
        return out


class ResBlock(nn.Module):
    """残差模块"""

    def __init__(self, inChannals, outChannals):
        """初始化残差模块"""
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inChannals, outChannals, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outChannals)
        self.conv2 = nn.Conv2d(outChannals, outChannals, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outChannals)
        self.conv3 = nn.Conv2d(outChannals, outChannals, kernel_size=1, bias=False)
        self.relu = nn.PReLU()

    def forward(self, x):
        """前向传播过程"""
        resudial = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)

        out += resudial
        out = self.relu(out)
        return out


@BACKBONES.register_module()
class SRResNetStage2(nn.Module):
    """SRResNet模型(4x)"""

    def __init__(self, in_channel=3, acblock=False, elan=False, ka=False):
        """初始化模型配置"""
        super(SRResNetStage2, self).__init__()
        # 卷积模块1
        self.ka = ka
        if ka:
            self.conv1 = KernelAttention(in_channel, 64)
        else:
            self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=9, padding=4, padding_mode='reflect', stride=1)
        self.relu = nn.PReLU()
        # 残差模块
        if acblock and elan:
            self.resBlock = self._makeLayer_(AcElanBlock, 64, 64, 16)
        elif acblock and not elan:
            self.resBlock = self._makeLayer_(AcBlock, 64, 64, 16)
        elif elan and not acblock:
            self.resBlock = self._makeLayer_(ElanBlcok, 64, 64, 16)
        else:
            self.resBlock = self._makeLayer_(ResBlock, 64, 64, 16)
        # 卷积模块2
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.PReLU()

        # 子像素卷积
        self.convPos1 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=2, padding_mode='reflect')
        self.pixelShuffler1 = nn.PixelShuffle(2)
        self.reluPos1 = nn.PReLU()

        self.convPos2 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.pixelShuffler2 = nn.PixelShuffle(2)
        self.reluPos2 = nn.PReLU()

        self.finConv = nn.Conv2d(64, 3, kernel_size=9, stride=1)

    def _makeLayer_(self, block, inChannals, outChannals, blocks):
        """构建残差层"""
        layers = []
        layers.append(block(inChannals, outChannals))

        for i in range(1, blocks):
            layers.append(block(outChannals, outChannals))

        return nn.Sequential(*layers)

    def forward(self, x):
        """前向传播过程"""
        x = self.conv1(x)
        if self.ka:
            x = self.relu(x)
        residual = x

        out = self.resBlock(x)

        out = self.conv2(out)
        out = self.bn2(out)
        out += residual

        out = self.convPos1(out)
        out = self.pixelShuffler1(out)
        out = self.reluPos1(out)

        out = self.convPos2(out)
        out = self.pixelShuffler2(out)
        out = self.reluPos2(out)
        out = self.finConv(out)

        return out

    def init_weights(self, pretrained):
        pass


@BACKBONES.register_module()
class SRResNetStage1(nn.Module):
    def __init__(self, acblock=False, elan=False,):
        """初始化模型配置"""
        super(SRResNetStage1, self).__init__()
        # 卷积模块1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1)
        self.relu = nn.PReLU()
        # 残差模块
        if acblock and elan:
            self.resBlock = self._makeLayer_(AcElanBlock, 64, 64, 16)
        elif acblock and not elan:
            self.resBlock = self._makeLayer_(AcBlock, 64, 64, 16)
        elif elan and not acblock:
            self.resBlock = self._makeLayer_(ElanBlcok, 64, 64, 16)
        else:
            self.resBlock = self._makeLayer_(ResBlock, 64, 64, 16)
        # 卷积模块2
        self.finConv = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def _makeLayer_(self, block, inChannals, outChannals, blocks):
        """构建残差层"""
        layers = []
        layers.append(block(inChannals, outChannals))
        for i in range(1, blocks):
            layers.append(block(outChannals, outChannals))
        return nn.Sequential(*layers)

    def forward(self, x):
        """前向传播过程"""
        residual = x
        x = self.conv1(x)
        x = self.relu(x)
        out = self.resBlock(x)
        out = self.finConv(out)
        out = residual + out
        return out

    def init_weights(self, pretrained):
        pass


# class SOCA(nn.Module):
#     def __init__(self, channel, reduction=8):
#         super(SOCA, self).__init__()
#         # global average pooling: feature --> point
#         # self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         # self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.max_pool = nn.MaxPool2d(kernel_size=2)
#
#         # feature channel downscale and upscale --> channel weight
#         self.conv_du = nn.Sequential(
#             nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
#             nn.Sigmoid()
#             # nn.BatchNorm2d(channel)
#         )
#
#     def forward(self, x):
#         batch_size, C, h, w = x.shape  # x: NxCxHxW
#         N = int(h * w)
#         min_h = min(h, w)
#         h1 = 1000
#         w1 = 1000
#         if h < h1 and w < w1:
#             x_sub = x
#         elif h < h1 and w > w1:
#             # H = (h - h1) // 2
#             W = (w - w1) // 2
#             x_sub = x[:, :, :, W:(W + w1)]
#         elif w < w1 and h > h1:
#             H = (h - h1) // 2
#             # W = (w - w1) // 2
#             x_sub = x[:, :, H:H + h1, :]
#         else:
#             H = (h - h1) // 2
#             W = (w - w1) // 2
#             x_sub = x[:, :, H:(H + h1), W:(W + w1)]
#         # subsample
#         # subsample_scale = 2
#         # subsample = nn.Upsample(size=(h // subsample_scale, w // subsample_scale), mode='nearest')
#         # x_sub = subsample(x)
#         # max_pool = nn.MaxPool2d(kernel_size=2)
#         # max_pool = nn.AvgPool2d(kernel_size=2)
#         # x_sub = self.max_pool(x)
#         ##
#         ## MPN-COV
#         cov_mat = CovpoolLayer(x_sub) # Global Covariance pooling layer
#         cov_mat_sqrt = SqrtmLayer(cov_mat,5) # Matrix square root layer( including pre-norm,Newton-Schulz iter. and post-com. with 5 iteration)
#         ##
#         cov_mat_sum = torch.mean(cov_mat_sqrt,1)
#         cov_mat_sum = cov_mat_sum.view(batch_size,C,1,1)
#         # y_ave = self.avg_pool(x)
#         # y_max = self.max_pool(x)
#         y_cov = self.conv_du(cov_mat_sum)
#         # y_max = self.conv_du(y_max)
#         # y = y_ave + y_max
#         # expand y to C*H*W
#         # expand_y = y.expand(-1,-1,h,w)
#         return y_cov*x