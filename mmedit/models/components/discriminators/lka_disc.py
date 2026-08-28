# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule, build_conv_layer
from mmcv.runner import load_checkpoint
import torch
from mmedit.models.common import generation_init_weights
from mmedit.models.registry import COMPONENTS
from mmedit.utils import get_root_logger


@COMPONENTS.register_module()
class LKA_Discriminator(nn.Module):
    def __init__(self, negative_slope=0.2):
        super(LKA_Discriminator, self).__init__()
        self.net = nn.Sequential(
            # 3x3卷积+LeakyRelu激活函数
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            # 3x3conv卷积+SaBN+LeakyRelu激活函数
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope),

            # 3x3conv卷积+SaBN+LeakyRelu激活函数
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope),

            # 3x3conv卷积+SaBN+LeakyRelu激活函数
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope),

            # 3x3conv卷积+SaBN+LeakyRelu激活函数
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope),

            # 1x1卷积
            nn.Conv2d(512, 1, kernel_size=1)
        )

    def forward(self, x):
        x = self.net(x)
        x = torch.sigmoid(x.reshape(-1))  # 输出经过sigmoid激活
        return x

    def init_weights(self, pretrained=None):
        pass

# class SandwichBatchNorm2d(nn.Module):
#     def __init__(self, num_features):
#         super().__init__()
#         self.num_features = num_features
#         self.bn = nn.BatchNorm2d(num_features, affine=True)
#         self.embed = nn.Embedding(num_features, num_features * 2)
#         self.embed.weight.data[:, :num_features].normal_(
#             1, 0.02
#         )  # Initialise scale at N(1, 0.02)
#         self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         out = self.bn(x)
#         gamma, beta = self.embed(x).chunk(2, 1)
#         out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(
#             -1, self.num_features, 1, 1
#         )
#         return out
