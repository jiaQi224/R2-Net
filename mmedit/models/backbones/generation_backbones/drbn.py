# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797
import torch
import torch.nn as nn

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feats))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feats))

                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feats))

            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)



def make_model(args, parent=False):
    return DRBN(args)


class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c * G, G))
        self.convs = nn.Sequential(*convs)

        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        feat1 = self.convs(x)
        feat2 = self.LFF(feat1) + x
        return feat2

from mmedit.models.registry import BACKBONES

@BACKBONES.register_module()
class DRBN(nn.Module):
    def __init__(self):
        super(DRBN, self).__init__()

        self.recur1 = DRBN_BU()
        self.recur2 = DRBN_BU()
        self.recur3 = DRBN_BU()
        self.recur4 = DRBN_BU()

    def forward(self, x_input):
        x = x_input

        res_g1_s1, res_g1_s2, res_g1_s4, feat_g1_s1, feat_g1_s2, feat_g1_s4 = self.recur1(
            [0, torch.cat((x, x), 1), 0, 0, 0, 0, 0, 0])
        res_g2_s1, res_g2_s2, res_g2_s4, feat_g2_s1, feat_g2_s2, feat_g2_s4 = self.recur2(
            [1, torch.cat((res_g1_s1, x), 1), res_g1_s1, res_g1_s2, res_g1_s4, feat_g1_s1, feat_g1_s2, feat_g1_s4])
        res_g3_s1, res_g3_s2, res_g3_s4, feat_g3_s1, feat_g3_s2, feat_g3_s4 = self.recur3(
            [1, torch.cat((res_g2_s1, x), 1), res_g2_s1, res_g2_s2, res_g2_s4, feat_g2_s1, feat_g2_s2, feat_g2_s4])
        res_g4_s1, res_g4_s2, res_g4_s4, feat_g4_s1, feat_g4_s2, feat_g4_s4 = self.recur4(
            [1, torch.cat((res_g3_s1, x), 1), res_g3_s1, res_g3_s2, res_g3_s4, feat_g3_s1, feat_g3_s2, feat_g3_s4])
        return res_g4_s1

    def init_weights(self, pretrained=None):
        pass


class DRBN_BU(nn.Module):
    def __init__(self):
        super(DRBN_BU, self).__init__()

        G0 = 16
        kSize = 3
        self.D = 6
        G = 8
        C = 4

        self.SFENet1 = nn.Conv2d(3 * 2, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)

        self.RDBs = nn.ModuleList()

        self.RDBs.append(
            RDB(growRate0=G0, growRate=G, nConvLayers=C)
        )
        self.RDBs.append(
            RDB(growRate0=G0, growRate=G, nConvLayers=C)
        )
        self.RDBs.append(
            RDB(growRate0=2 * G0, growRate=2 * G, nConvLayers=C)
        )
        self.RDBs.append(
            RDB(growRate0=2 * G0, growRate=2 * G, nConvLayers=C)
        )
        self.RDBs.append(
            RDB(growRate0=G0, growRate=G, nConvLayers=C)
        )
        self.RDBs.append(
            RDB(growRate0=G0, growRate=G, nConvLayers=C)
        )

        self.UPNet = nn.Sequential(*[
            nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.Conv2d(G0, 3, kSize, padding=(kSize - 1) // 2, stride=1)
        ])

        self.UPNet2 = nn.Sequential(*[
            nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.Conv2d(G0, 3, kSize, padding=(kSize - 1) // 2, stride=1)
        ])

        self.UPNet4 = nn.Sequential(*[
            nn.Conv2d(G0 * 2, G0, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.Conv2d(G0, 3, kSize, padding=(kSize - 1) // 2, stride=1)
        ])

        self.Down1 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=2)
        self.Down2 = nn.Conv2d(G0, G0 * 2, kSize, padding=(kSize - 1) // 2, stride=2)

        self.Up1 = nn.ConvTranspose2d(G0, G0, kSize + 1, stride=2, padding=1)
        self.Up2 = nn.ConvTranspose2d(G0 * 2, G0, kSize + 1, stride=2, padding=1)

        self.Relu = nn.ReLU()
        self.Img_up = nn.Upsample(scale_factor=2, mode='bilinear')

    def part_forward(self, x):
        #
        # Stage 1
        #
        flag = x[0]
        input_x = x[1]

        prev_s1 = x[2]
        prev_s2 = x[3]
        prev_s4 = x[4]

        prev_feat_s1 = x[5]
        prev_feat_s2 = x[6]
        prev_feat_s4 = x[7]

        f_first = self.Relu(self.SFENet1(input_x))
        f_s1 = self.Relu(self.SFENet2(f_first))
        f_s2 = self.Down1(self.RDBs[0](f_s1))
        f_s4 = self.Down2(self.RDBs[1](f_s2))

        if flag == 0:
            f_s4 = f_s4 + self.RDBs[3](self.RDBs[2](f_s4))
            f_s2 = f_s2 + self.RDBs[4](self.Up2(f_s4))
            f_s1 = f_s1 + self.RDBs[5](self.Up1(f_s2)) + f_first
        else:
            f_s4 = f_s4 + self.RDBs[3](self.RDBs[2](f_s4)) + prev_feat_s4
            f_s2 = f_s2 + self.RDBs[4](self.Up2(f_s4)) + prev_feat_s2
            f_s1 = f_s1 + self.RDBs[5](self.Up1(f_s2)) + f_first + prev_feat_s1

        res4 = self.UPNet4(f_s4)
        res2 = self.UPNet2(f_s2) + self.Img_up(res4)
        res1 = self.UPNet(f_s1) + self.Img_up(res2)

        return res1, res2, res4, f_s1, f_s2, f_s4

    def forward(self, x_input):
        x = x_input

        res1, res2, res4, f_s1, f_s2, f_s4 = self.part_forward(x)

        return res1, res2, res4, f_s1, f_s2, f_s4