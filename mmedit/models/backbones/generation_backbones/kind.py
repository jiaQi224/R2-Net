import torch
import torch.nn as nn
import torch.nn.functional as F
from mmedit.models.registry import BACKBONES


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MSIA(nn.Module):
    def __init__(self, filters, activation='lrelu'):
        super().__init__()
        # Down 1
        self.conv_bn_relu_1 = Conv_BN_Relu(filters, activation)
        # Down 2
        self.down_2 = MaxPooling2D(2, 2)
        self.conv_bn_relu_2 = Conv_BN_Relu(filters, activation)
        self.deconv_2 = ConvTranspose2D(filters, filters)
        # Down 4
        self.down_4 = MaxPooling2D(2, 2)
        self.conv_bn_relu_4 = Conv_BN_Relu(filters, activation, kernel=1)
        self.deconv_4_1 = ConvTranspose2D(filters, filters)
        self.deconv_4_2 = ConvTranspose2D(filters, filters)
        # output
        self.out = Conv2D(filters*4, filters)

    def forward(self, R, I_att):
        R_att = R * I_att
        # Down 1
        msia_1 = self.conv_bn_relu_1(R_att)
        # Down 2
        down_2 = self.down_2(R_att)
        conv_bn_relu_2 = self.conv_bn_relu_2(down_2)
        msia_2 = self.deconv_2(conv_bn_relu_2)
        # Down 4
        down_4 = self.down_4(down_2)
        conv_bn_relu_4 = self.conv_bn_relu_4(down_4)
        deconv_4 = self.deconv_4_1(conv_bn_relu_4)
        msia_4 = self.deconv_4_2(deconv_4)
        # concat
        concat = torch.cat([R, msia_1, msia_2, msia_4], dim=1)
        out = self.out(concat)
        return out


class Conv_BN_Relu(nn.Module):
    def __init__(self, channels, activation='lrelu', kernel=3):
        super().__init__()
        self.ActivationLayer = nn.LeakyReLU(inplace=True)
        if activation == 'relu':
            self.ActivationLayer = nn.ReLU(inplace=True)
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel, padding=kernel//2),
            nn.BatchNorm2d(channels, momentum=0.99),  # 原论文用的tf.layer的默认参数
            self.ActivationLayer,
        )

    def forward(self, x):
        return self.conv_bn_relu(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, activation='lrelu'):
        super().__init__()
        self.doubleconv = nn.Sequential(
            Conv2D(in_channels, out_channels, activation),
            Conv2D(out_channels,out_channels, activation)
        )

    def forward(self, x):
        return self.doubleconv(x)

class ResConv(nn.Module):
    def __init__(self, in_channels, out_channels, activation='lrelu'):
        super().__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        if activation == 'relu':
            self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=0.8)
        self.cbam = CBAM(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.8)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        x1 = self.relu(bn1)
        cbam = self.cbam(x1)
        conv2 = self.conv2(cbam)
        bn2 = self.bn1(conv2)
        out = bn2 + x
        return out

class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, activation='lrelu', stride=1):
        super().__init__()
        self.ActivationLayer = nn.LeakyReLU(inplace=True)
        if activation == 'relu':
            self.ActivationLayer = nn.ReLU(inplace=True)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            self.ActivationLayer,
        )

    def forward(self, x):
        return self.conv_relu(x)


class ConvTranspose2D(nn.Module):
    def __init__(self, in_channels, out_channels, activation='lrelu'):
        super().__init__()
        self.ActivationLayer = nn.LeakyReLU(inplace=True)
        if activation == 'relu':
            self.ActivationLayer = nn.ReLU(inplace=True)
        self.deconv_relu = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
            self.ActivationLayer,
        )

    def forward(self, x):
        return self.deconv_relu(x)


class MaxPooling2D(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        return self.maxpool(x)


class AvgPooling2D(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.avgpool(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(2,1,kernel_size, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, planes):
        super().__init__()
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()
    def forward(self, x):
        x = self.ca(x) * x
        out = self.sa(x) * x
        return x


class Concat(nn.Module):
    def forward(self, x, y):
        _, _, xh, xw = x.size()
        _, _, yh, yw = y.size()
        diffY = xh - yh
        diffX = xw - yw
        y = F.pad(y, (diffX // 2, diffX - diffX//2,
                      diffY // 2, diffY - diffY//2))
        return torch.cat((x, y), dim=1)

class DecomNet(nn.Module):
    def __init__(self, filters=32, activation='lrelu'):
        super().__init__()
        self.conv_input = Conv2D(3, filters)
        # top path build Reflectance map
        self.maxpool_r1 = MaxPooling2D()
        self.conv_r1 = Conv2D(filters, filters * 2)
        self.maxpool_r2 = MaxPooling2D()
        self.conv_r2 = Conv2D(filters * 2, filters * 4)
        self.deconv_r1 = ConvTranspose2D(filters * 4, filters * 2)
        self.concat_r1 = Concat()
        self.conv_r3 = Conv2D(filters * 4, filters * 2)
        self.deconv_r2 = ConvTranspose2D(filters * 2, filters)
        self.concat_r2 = Concat()
        self.conv_r4 = Conv2D(filters * 2, filters)
        self.conv_r5 = nn.Conv2d(filters, 3, kernel_size=3, padding=1)
        self.R_out = nn.Sigmoid()
        # bottom path build Illumination map
        self.conv_i1 = Conv2D(filters, filters)
        self.concat_i1 = Concat()
        self.conv_i2 = nn.Conv2d(filters * 2, 1, kernel_size=3, padding=1)
        self.I_out = nn.Sigmoid()

    def forward(self, x):
        conv_input = self.conv_input(x)
        # build Reflectance map
        maxpool_r1 = self.maxpool_r1(conv_input)
        conv_r1 = self.conv_r1(maxpool_r1)
        maxpool_r2 = self.maxpool_r2(conv_r1)
        conv_r2 = self.conv_r2(maxpool_r2)
        deconv_r1 = self.deconv_r1(conv_r2)
        concat_r1 = self.concat_r1(conv_r1, deconv_r1)
        conv_r3 = self.conv_r3(concat_r1)
        deconv_r2 = self.deconv_r2(conv_r3)
        concat_r2 = self.concat_r2(conv_input, deconv_r2)
        conv_r4 = self.conv_r4(concat_r2)
        conv_r5 = self.conv_r5(conv_r4)
        R_out = self.R_out(conv_r5)

        # build Illumination map
        conv_i1 = self.conv_i1(conv_input)
        concat_i1 = self.concat_i1(conv_r4, conv_i1)
        conv_i2 = self.conv_i2(concat_i1)
        I_out = self.I_out(conv_i2)

        return R_out, I_out


class IllumNet(nn.Module):
    def __init__(self, filters=32, activation='lrelu'):
        super().__init__()
        self.concat_input = Concat()
        # bottom path build Illumination map
        self.conv_i1 = Conv2D(2, filters)
        self.conv_i2 = Conv2D(filters, filters)
        self.conv_i3 = Conv2D(filters, filters)
        self.conv_i4 = nn.Conv2d(filters, 1, kernel_size=3, padding=1)
        self.I_out = nn.Sigmoid()

    def forward(self, I, ratio):
        with torch.no_grad():
            ratio_map = torch.ones_like(I) * ratio
        concat_input = self.concat_input(I, ratio_map)
        # build Illumination map
        conv_i1 = self.conv_i1(concat_input)
        conv_i2 = self.conv_i2(conv_i1)
        conv_i3 = self.conv_i3(conv_i2)
        conv_i4 = self.conv_i4(conv_i3)
        I_out = self.I_out(conv_i4)

        return I_out


class IllumNet_Custom(nn.Module):
    def __init__(self, filters=16, activation='lrelu', device='cuda'):
        super().__init__()
        self.concat_input = Concat()
        # Parameter
        self.Gauss = torch.as_tensor(
            np.array([[0.0947416, 0.118318, 0.0947416],
                      [0.118318, 0.147761, 0.118318],
                      [0.0947416, 0.118318, 0.0947416]]).astype(np.float32)
        )
        self.Gauss_kernel = self.Gauss.expand(1, 1, 3, 3).to(device)
        self.w = nn.Parameter(torch.FloatTensor(1), requires_grad=True).to(device).data.fill_(0.72)
        self.sigma = nn.Parameter(torch.FloatTensor(1), requires_grad=True).to(device).data.fill_(2.0)

        # bottom path build Illumination map
        self.conv_input = Conv2D(2, filters)
        self.res_block = nn.Sequential(
            ResConv(filters, filters),
            ResConv(filters, filters),
            ResConv(filters, filters)
        )
        # self.down1 = MaxPooling2D()
        # self.conv_2 = Conv2D(filters, filters*2)
        # self.down2 = MaxPooling2D()
        # self.conv_3 = Conv2D(filters*2, filters*4)
        # self.down3 = MaxPooling2D()
        # self.conv_4 = Conv2D(filters*4, filters*8)

        # self.d = nn.Dropout2d(0.5)

        # self.deconv_3 = ConvTranspose2D(filters*8, filters*4)
        # self.concat3 = Concat()
        # self.cbam3 = CBAM(filters*8)
        # self.deconv_2 = ConvTranspose2D(filters*8, filters*2)
        # self.concat2 = Concat()
        # self.cbam2 = CBAM(filters*4)
        # self.deconv_1 = ConvTranspose2D(filters*4, filters*1)
        # self.concat1 = Concat()
        # self.cbam1 = CBAM(filters*2)
        self.conv_out = nn.Conv2d(filters, 1, kernel_size=3, padding=1)

        self.I_out = nn.Sigmoid()

    def standard_illum_map(self, I, ratio=1, blur=False):
        self.w.clamp_(0.01, 0.99)
        self.sigma.clamp_(0.1, 10)
        # if blur: # low light image have much noisy
        #     I = torch.nn.functional.conv2d(I, weight=self.Gauss_kernel, padding=1)
        I = torch.log(I + 1.)
        I_mean = torch.mean(I, dim=[2, 3], keepdim=True)
        I_std = torch.std(I, dim=[2, 3], keepdim=True)
        I_min = I_mean - self.sigma * I_std
        I_max = I_mean + self.sigma * I_std
        I_range = I_max - I_min
        I_out = torch.clamp((I - I_min) / I_range, min=0.0, max=1.0)
        # Transfer to gamma correction, center intensity is w
        I_out = I_out ** (-1.442695 * torch.log(self.w))
        return I_out

    def set_parameter(self, w=None):
        if w is None:
            self.w.requires_grad = True
        else:
            self.w.data.fill_(w)
            self.w.requires_grad = False

    def get_parameter(self):
        if self.w.device.type == 'cuda':
            w = self.w.detach().cpu().numpy()
            sigma = self.sigma.detach().cpu().numpy()
        else:
            w = self.w.numpy()
            sigma = self.sigma.numpy()
        return w, sigma

    def forward(self, I, ratio):
        I_standard = self.standard_illum_map(I, ratio)
        concat_input = torch.cat([I, I_standard], dim=1)
        # build Illumination map
        conv_input = self.conv_input(concat_input)
        res_block = self.res_block(conv_input)
        # down1 = self.down1(conv_1)
        # conv_2 = self.conv_2(down1)
        # down2 = self.down2(conv_2)
        # conv_3 = self.conv_3(down2)
        # down3 = self.down3(conv_3)
        # conv_4 = self.conv_4(down3)
        # d = self.d(conv_4)
        # deconv_3 = self.deconv_3(d)

        # concat3 = self.concat3(conv_3, deconv_3)
        # cbam3 = self.cbam3(concat3)
        # deconv_2 = self.deconv_2(cbam3)

        # concat2 = self.concat2(conv_2, deconv_2)
        # cbam2 = self.cbam2(concat2)
        # deconv_1 = self.deconv_1(cbam2)

        # concat1 = self.concat1(conv_1, deconv_1)
        # cbam1 = self.cbam1(concat1)
        res_out = res_block + conv_input
        conv_out = self.conv_out(res_out)
        I_out = self.I_out(conv_out)

        return I_out, I_standard


class RestoreNet_MSIA(nn.Module):
    def __init__(self, filters=16, activation='relu'):
        super().__init__()
        # Illumination Attention
        self.i_input = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.i_att = nn.Sigmoid()

        # Network
        self.conv1_1 = Conv2D(3, filters, activation)
        self.conv1_2 = Conv2D(filters, filters * 2, activation)
        self.msia1 = MSIA(filters * 2, activation)

        self.conv2_1 = Conv2D(filters * 2, filters * 4, activation)
        self.conv2_2 = Conv2D(filters * 4, filters * 4, activation)
        self.msia2 = MSIA(filters * 4, activation)

        self.conv3_1 = Conv2D(filters * 4, filters * 8, activation)
        self.dropout = nn.Dropout2d(0.5)
        self.conv3_2 = Conv2D(filters * 8, filters * 4, activation)
        self.msia3 = MSIA(filters * 4, activation)

        self.conv4_1 = Conv2D(filters * 4, filters * 2, activation)
        self.conv4_2 = Conv2D(filters * 2, filters * 2, activation)
        self.msia4 = MSIA(filters * 2, activation)

        self.conv5_1 = Conv2D(filters * 2, filters * 1, activation)
        self.conv5_2 = nn.Conv2d(filters, 3, kernel_size=1, padding=0)
        self.out = nn.Sigmoid()

    def forward(self, R, I):
        # Illumination Attention
        i_input = self.i_input(I)
        i_att = self.i_att(i_input)

        # Network
        conv1 = self.conv1_1(R)
        conv1 = self.conv1_2(conv1)
        msia1 = self.msia1(conv1, i_att)

        conv2 = self.conv2_1(msia1)
        conv2 = self.conv2_2(conv2)
        msia2 = self.msia2(conv2, i_att)

        conv3 = self.conv3_1(msia2)
        conv3 = self.conv3_2(conv3)
        msia3 = self.msia3(conv3, i_att)

        conv4 = self.conv4_1(msia3)
        conv4 = self.conv4_2(conv4)
        msia4 = self.msia4(conv4, i_att)

        conv5 = self.conv5_1(msia4)
        conv5 = self.conv5_2(conv5)

        # out = self.out(conv5)
        out = conv5.clamp(min=0.0, max=1.0)
        return out


class RestoreNet_Unet(nn.Module):
    def __init__(self, filters=32, activation='lrelu'):
        super().__init__()
        self.conv1_1 = Conv2D(4, filters)
        self.conv1_2 = Conv2D(filters, filters)
        self.pool1 = MaxPooling2D()

        self.conv2_1 = Conv2D(filters, filters * 2)
        self.conv2_2 = Conv2D(filters * 2, filters * 2)
        self.pool2 = MaxPooling2D()

        self.conv3_1 = Conv2D(filters * 2, filters * 4)
        self.conv3_2 = Conv2D(filters * 4, filters * 4)
        self.pool3 = MaxPooling2D()

        self.conv4_1 = Conv2D(filters * 4, filters * 8)
        self.conv4_2 = Conv2D(filters * 8, filters * 8)
        self.pool4 = MaxPooling2D()

        self.conv5_1 = Conv2D(filters * 8, filters * 16)
        self.conv5_2 = Conv2D(filters * 16, filters * 16)
        self.dropout = nn.Dropout2d(0.5)

        self.upv6 = ConvTranspose2D(filters * 16, filters * 8)
        self.concat6 = Concat()
        self.conv6_1 = Conv2D(filters * 16, filters * 8)
        self.conv6_2 = Conv2D(filters * 8, filters * 8)

        self.upv7 = ConvTranspose2D(filters * 8, filters * 4)
        self.concat7 = Concat()
        self.conv7_1 = Conv2D(filters * 8, filters * 4)
        self.conv7_2 = Conv2D(filters * 4, filters * 4)

        self.upv8 = ConvTranspose2D(filters * 4, filters * 2)
        self.concat8 = Concat()
        self.conv8_1 = Conv2D(filters * 4, filters * 2)
        self.conv8_2 = Conv2D(filters * 2, filters * 2)

        self.upv9 = ConvTranspose2D(filters * 2, filters)
        self.concat9 = Concat()
        self.conv9_1 = Conv2D(filters * 2, filters)
        self.conv9_2 = Conv2D(filters, filters)

        self.conv10_1 = nn.Conv2d(filters, 3, kernel_size=1, stride=1)
        self.out = nn.Sigmoid()

    def forward(self, R, I):
        x = torch.cat([R, I], dim=1)
        conv1 = self.conv1_1(x)
        conv1 = self.conv1_2(conv1)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2_1(pool1)
        conv2 = self.conv2_2(conv2)
        pool2 = self.pool1(conv2)

        conv3 = self.conv3_1(pool2)
        conv3 = self.conv3_2(conv3)
        pool3 = self.pool1(conv3)

        conv4 = self.conv4_1(pool3)
        conv4 = self.conv4_2(conv4)
        pool4 = self.pool1(conv4)

        conv5 = self.conv5_1(pool4)
        conv5 = self.conv5_2(conv5)

        # d = self.dropout(conv5)
        up6 = self.upv6(conv5)
        up6 = self.concat6(conv4, up6)
        conv6 = self.conv6_1(up6)
        conv6 = self.conv6_2(conv6)

        up7 = self.upv7(conv6)
        up7 = self.concat7(conv3, up7)
        conv7 = self.conv7_1(up7)
        conv7 = self.conv7_2(conv7)

        up8 = self.upv8(conv7)
        up8 = self.concat8(conv2, up8)
        conv8 = self.conv8_1(up8)
        conv8 = self.conv8_2(conv8)

        up9 = self.upv9(conv8)
        up9 = self.concat9(conv1, up9)
        conv9 = self.conv9_1(up9)
        conv9 = self.conv9_2(conv9)

        conv10 = self.conv10_1(conv9)
        out = self.out(conv10)
        return out


class KinD_noDecom(nn.Module):
    def __init__(self, filters=32, activation='lrelu'):
        super().__init__()
        # self.decom_net = DecomNet()
        self.restore_net = RestoreNet_Unet()
        self.illum_net = IllumNet()

    def forward(self, R, I, ratio):
        I_final = self.illum_net(I, ratio)
        R_final = self.restore_net(R, I)
        I_final_3 = torch.cat([I_final, I_final, I_final], dim=1)
        output = I_final_3 * R_final
        return R_final, I_final, output


@BACKBONES.register_module()
class KinD(nn.Module):
    def __init__(self, filters=32, activation='lrelu'):
        super().__init__()
        self.decom_net = DecomNet()
        self.restore_net = RestoreNet_Unet()
        self.illum_net = IllumNet()
        self.KinD_noDecom = KinD_noDecom()
        self.KinD_noDecom.restore_net = self.restore_net
        self.KinD_noDecom.illum_net = self.illum_net

    def forward(self, L, ratio=0.5):
        R, I = self.decom_net(L)
        R_final, I_final, output = self.KinD_noDecom(R, I, ratio)
        return output

    def init_weights(self, pretrained=None):
        pass


class KinD_plus(nn.Module):
    def __init__(self, filters=32, activation='lrelu'):
        super().__init__()
        self.decom_net = DecomNet()
        self.restore_net = RestoreNet_MSIA()
        self.illum_net = IllumNet_Custom()

    def forward(self, L, ratio=0.5):
        R, I = self.decom_net(L)
        # R_final, I_final, output = self.KinD_noDecom(R, I, ratio)
        I_final, I_standard = self.illum_net(I, ratio)
        R_final = self.restore_net(R, I)
        I_final_3 = torch.cat([I_final, I_final, I_final], dim=1)
        output = I_final_3 * R_final
        return R_final, I_final, output