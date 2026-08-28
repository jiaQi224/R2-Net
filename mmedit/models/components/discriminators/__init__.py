# Copyright (c) OpenMMLab. All rights reserved.
from .deepfill_disc import DeepFillv1Discriminators
from .gl_disc import GLDiscs
from .light_cnn import LightCNN
from .modified_vgg import ModifiedVGG
from .multi_layer_disc import MultiLayerDiscriminator
from .patch_disc import PatchDiscriminator
from .smpatch_disc import SoftMaskPatchDiscriminator
from .ttsr_disc import TTSRDiscriminator
from .unet_disc import UNetDiscriminatorWithSpectralNorm
from .lka_disc import LKA_Discriminator
from .gl_dis import GlobalLocalDiscriminator
from .unet_disc_sn import UNetDiscriminatorSN


__all__ = [
    'GLDiscs', 'ModifiedVGG', 'MultiLayerDiscriminator', 'TTSRDiscriminator',
    'DeepFillv1Discriminators', 'PatchDiscriminator', 'LightCNN',
    'UNetDiscriminatorWithSpectralNorm', 'SoftMaskPatchDiscriminator',
    'LKA_Discriminator', 'GlobalLocalDiscriminator', 'UNetDiscriminatorSN'
]
