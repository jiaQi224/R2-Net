# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.runner import load_checkpoint
from torch.nn.utils import spectral_norm

from mmedit.models.registry import COMPONENTS
from mmedit.utils import get_root_logger


@COMPONENTS.register_module()
class UNetDiscriminatorSN(nn.Module):
    """A simplified U-Net discriminator with spectral normalization.

    This is a more compact version of UNetDiscriminatorWithSpectralNorm
    designed for efficient adversarial training.

    Args:
        num_in_ch (int): Channel number of the input. Default: 3.
        num_feat (int, optional): Channel number of the intermediate
            features. Default: 64.
        skip_connection (bool, optional): Whether to use skip connection.
            Default: True.
    """

    def __init__(self, num_in_ch=3, num_feat=64, skip_connection=True):

        super().__init__()

        self.skip_connection = skip_connection

        # Initial convolution
        self.conv_0 = nn.Conv2d(
            num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)

        # Encoder (downsample)
        self.conv_1 = spectral_norm(
            nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv_2 = spectral_norm(
            nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv_3 = spectral_norm(
            nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))

        # Decoder (upsample)
        self.conv_4 = spectral_norm(
            nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv_5 = spectral_norm(
            nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv_6 = spectral_norm(
            nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))

        # Output layers
        self.conv_7 = spectral_norm(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv_8 = spectral_norm(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv_9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

        # Utilities
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results with shape (n, 1, h, w).
        """

        # Initial feature extraction
        feat_0 = self.lrelu(self.conv_0(x))

        # Encoder path
        feat_1 = self.lrelu(self.conv_1(feat_0))  # /2
        feat_2 = self.lrelu(self.conv_2(feat_1))  # /4
        feat_3 = self.lrelu(self.conv_3(feat_2))  # /8

        # Decoder path with skip connections
        feat_3_up = self.upsample(feat_3)  # /4
        feat_4 = self.lrelu(self.conv_4(feat_3_up))
        if self.skip_connection:
            feat_4 = feat_4 + feat_2

        feat_4_up = self.upsample(feat_4)  # /2
        feat_5 = self.lrelu(self.conv_5(feat_4_up))
        if self.skip_connection:
            feat_5 = feat_5 + feat_1

        feat_5_up = self.upsample(feat_5)  # /1
        feat_6 = self.lrelu(self.conv_6(feat_5_up))
        if self.skip_connection:
            feat_6 = feat_6 + feat_0

        # Final output layers
        out = self.lrelu(self.conv_7(feat_6))
        out = self.lrelu(self.conv_8(out))
        out = self.conv_9(out)

        return out

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
            strict (bool, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """

        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')
