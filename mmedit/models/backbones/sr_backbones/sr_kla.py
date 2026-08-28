import math
import torch
from torch import nn
from mmedit.models.registry import BACKBONES


# 生成器
@BACKBONES.register_module()
class LKA_Generator(nn.Module):
    def __init__(self, scale_factor):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(LKA_Generator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        # 残差块
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = ResidualBlock(64)
        self.block8 = ResidualBlock(64)
        self.block9 = ResidualBlock(64)
        self.block10 = ResidualBlock(64)
        self.block11 = ResidualBlock(64)
        self.block12 = ResidualBlock(64)
        self.block13 = ResidualBlock(64)
        self.block14 = ResidualBlock(64)
        self.block15 = ResidualBlock(64)
        self.block16 = ResidualBlock(64)
        self.block17 = ResidualBlock(64)



        # conv ,此处删掉了原来的BN
        self.block18 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1))

        # 上采样+conv
        self.block19 = nn.Sequential(*[UpsampleBLock(64, 2) for _ in range(upsample_block_num)],
                                    nn.Conv2d(64, 3, kernel_size=9, padding=4))

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block7)
        block9 = self.block9(block8)
        block10 = self.block10(block9)
        block11 = self.block11(block10)
        block12 = self.block12(block11)
        block13 = self.block13(block12)
        block14 = self.block14(block13)
        block15 = self.block15(block14)
        block16 = self.block16(block15)
        block17 = self.block17(block16)
        block18 = self.block18(block17)
        block19 = self.block19(block1 + block18)  # 加了skipp connect

        return (torch.tanh(block19) + 1) / 2

    def init_weights(self, pretrained=None):
        pass

class ResidualBlock(nn.Module):
    """
    残差块
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        # 3x3conv卷积
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        # prelu 激活函数
        self.prelu = nn.PReLU()
        # 3x3conv卷积
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        # large kernel attention模块
        self.lka = LKA(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = residual + self.lka(residual)  # skipping connect

        return x + residual


class UpsampleBLock(nn.Module):
    """
    上采样模块
    使用pixelshuffle()方法
    """

    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        # 3x3卷积
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        # PixelShuffle 通道数减半，特征图分辨率翻倍
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x

class LKA(nn.Module):
    """
    large kernel attention
    """

    def __init__(self, channels):
        super().__init__()
        # 5x5的深度分离卷积
        self.conv0 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        # 带有空洞的深度分离卷积
        self.conv_spatial = nn.Conv2d(channels, channels, 7, stride=1, padding=9, groups=channels,
                                      dilation=3)
        self.conv1 = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        u = x.clone()
        x = self.conv0(x)
        x = self.conv_spatial(x)
        x = self.conv1(x)

        return u * x