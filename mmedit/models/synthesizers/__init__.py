# Copyright (c) OpenMMLab. All rights reserved.
from .cycle_gan import CycleGAN
from .pix2pix import Pix2Pix
from .pix2pixhd import Pix2PixHD
from .pix2pix_nogan import Pix2PixNoGan

__all__ = ['Pix2Pix', 'CycleGAN', 'Pix2PixHD', 'Pix2PixNoGan']
