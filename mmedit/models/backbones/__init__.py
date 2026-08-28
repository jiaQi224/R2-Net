# Copyright (c) OpenMMLab. All rights reserved.
from .encoder_decoders import (VGG16, ContextualAttentionNeck, DeepFillDecoder,
                               DeepFillEncoder, DeepFillEncoderDecoder,
                               DepthwiseIndexBlock, FBADecoder,
                               FBAResnetDilated, GLDecoder, GLDilationNeck,
                               GLEncoder, GLEncoderDecoder, HolisticIndexBlock,
                               IndexedUpsample, IndexNetDecoder,
                               IndexNetEncoder, PConvDecoder, PConvEncoder,
                               PConvEncoderDecoder, PlainDecoder,
                               ResGCADecoder, ResGCAEncoder, ResNetDec,
                               ResNetEnc, ResShortcutDec, ResShortcutEnc,
                               SimpleEncoderDecoder)
from .generation_backbones import ResnetGenerator, DehazingUNet
from .sr_backbones import (EDSR, LIIFEDSR, LIIFRDN, RDN, SRCNN, BasicVSRNet,
                           BasicVSRPlusPlus, DICNet, EDVRNet, GLEANStyleGANv2,
                           IconVSR, RealBasicVSRNet, RRDBNet,
                           TDANNet, TOFlow, TTSRNet, RCAN, LKA_Generator, SRResNetStage1, SRResNetStage2)
from .vfi_backbones import CAINNet, FLAVRNet, TOFlowVFINet

__all__ = [
    'VGG16', 'PlainDecoder', 'SimpleEncoderDecoder',
    'GLEncoderDecoder', 'GLEncoder', 'GLDecoder', 'GLDilationNeck',
    'PConvEncoderDecoder', 'PConvEncoder', 'PConvDecoder', 'ResNetEnc',
    'ResNetDec', 'ResShortcutEnc', 'ResShortcutDec', 'RRDBNet',
    'DeepFillEncoder', 'HolisticIndexBlock', 'DepthwiseIndexBlock',
    'ContextualAttentionNeck', 'DeepFillDecoder', 'EDSR', 'RDN', 'DICNet',
    'DeepFillEncoderDecoder', 'EDVRNet', 'IndexedUpsample', 'IndexNetEncoder',
    'IndexNetDecoder', 'TOFlow', 'ResGCAEncoder', 'ResGCADecoder', 'SRCNN',
    'DehazingUNet', 'ResnetGenerator', 'FBAResnetDilated', 'FBADecoder',
    'BasicVSRNet', 'IconVSR', 'TTSRNet', 'GLEANStyleGANv2', 'TDANNet',
    'LIIFEDSR', 'LIIFRDN', 'BasicVSRPlusPlus', 'RealBasicVSRNet', 'CAINNet',
    'TOFlowVFINet', 'FLAVRNet', 'RCAN', 'LKA_Generator', 'SRResNetStage1', 'SRResNetStage2'
]
