# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import auto_fp16
import torch.nn.functional as F
from ..builder import build_backbone, build_component, build_loss
from ..common import set_requires_grad
from ..registry import MODELS
from .basic_restorer import BasicRestorer
import mmcv
import numbers
import os.path as osp
from mmedit.core import InceptionV3, psnr, ssim, tensor2img


@MODELS.register_module()
class SRGAN(BasicRestorer):
    def __init__(self,
                 generator,
                 discriminator=None,
                 gan_loss=None,
                 pixel_loss=None,
                 perceptual_loss=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(BasicRestorer, self).__init__()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # generator
        self.generator = build_backbone(generator)
        # discriminator
        self.discriminator = build_component(
            discriminator) if discriminator else None

        # support fp16
        self.fp16_enabled = False

        # loss
        self.gan_loss = build_loss(gan_loss) if gan_loss else None
        self.pixel_loss = build_loss(pixel_loss) if pixel_loss else None
        self.perceptual_loss = build_loss(
            perceptual_loss) if perceptual_loss else None

        self.disc_steps = 1 if self.train_cfg is None else self.train_cfg.get(
            'disc_steps', 1)
        self.disc_init_steps = (0 if self.train_cfg is None else
                                self.train_cfg.get('disc_init_steps', 0))
        self.step_counter = 0  # counting training steps

        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
        """
        self.generator.init_weights(pretrained=pretrained)
        if self.discriminator:
            self.discriminator.init_weights(pretrained=pretrained)

    @auto_fp16(apply_to=('lq', ))
    def forward(self, lq, gt=None, test_mode=False, **kwargs):
        if test_mode:
            return self.forward_test(lq, gt, **kwargs)

        raise ValueError(
            'SRGAN model does not support `forward_train` function.')

    def train_step(self, data_batch, optimizer):
        # data
        lq = data_batch['lq']
        gt = data_batch['gt']

        # generator
        fake_g_output = self.generator(lq)

        losses = dict()
        log_vars = dict()

        # no updates to discriminator parameters.
        set_requires_grad(self.discriminator, False)

        if (self.step_counter % self.disc_steps == 0
                and self.step_counter >= self.disc_init_steps):
            if self.pixel_loss:
                losses['loss_pix'] = self.pixel_loss(fake_g_output, gt)
            if self.perceptual_loss:
                loss_percep, loss_style = self.perceptual_loss(
                    fake_g_output, gt)
                if loss_percep is not None:
                    losses['loss_perceptual'] = loss_percep
                if loss_style is not None:
                    losses['loss_style'] = loss_style
            # gan loss for generator
            fake_g_pred = self.discriminator(fake_g_output)
            losses['loss_gan'] = self.gan_loss(
                fake_g_pred, target_is_real=True, is_disc=False)

            # parse loss
            loss_g, log_vars_g = self.parse_losses(losses)
            log_vars.update(log_vars_g)

            # optimize
            optimizer['generator'].zero_grad()
            loss_g.backward()
            optimizer['generator'].step()

        # discriminator
        set_requires_grad(self.discriminator, True)
        # real
        real_d_pred = self.discriminator(gt)
        loss_d_real = self.gan_loss(
            real_d_pred, target_is_real=True, is_disc=True)
        loss_d, log_vars_d = self.parse_losses(dict(loss_d_real=loss_d_real))
        optimizer['discriminator'].zero_grad()
        loss_d.backward()
        log_vars.update(log_vars_d)
        # fake
        fake_d_pred = self.discriminator(fake_g_output.detach())
        loss_d_fake = self.gan_loss(
            fake_d_pred, target_is_real=False, is_disc=True)
        loss_d, log_vars_d = self.parse_losses(dict(loss_d_fake=loss_d_fake))
        loss_d.backward()
        log_vars.update(log_vars_d)

        optimizer['discriminator'].step()

        self.step_counter += 1

        log_vars.pop('loss')  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars,
            num_samples=len(gt.data),
            results=dict(lq=lq.cpu(), gt=gt.cpu(), output=fake_g_output.cpu()))

        return outputs


@MODELS.register_module()
class TwoStageSRGAN(BasicRestorer):
    def __init__(self,
                 stage1_generator,
                 stage2_generator,
                 discriminator=None,
                 gan_loss=None,
                 mid_loss=None,
                 final_loss=None,
                 perceptual_loss=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(BasicRestorer, self).__init__()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # generator
        self.stage1_generator = build_backbone(stage1_generator)
        self.stage2_generator = build_backbone(stage2_generator)
        # discriminator
        self.discriminator = build_component(
            discriminator) if discriminator else None

        # support fp16
        self.fp16_enabled = False

        # loss
        self.gan_loss = build_loss(gan_loss) if gan_loss else None
        self.mid_loss = mid_loss
        if mid_loss:
            self.mid_loss = build_loss(mid_loss)
        self.final_loss = build_loss(final_loss)
        self.perceptual_loss = build_loss(
            perceptual_loss) if perceptual_loss else None

        self.disc_steps = 1 if self.train_cfg is None else self.train_cfg.get(
            'disc_steps', 1)
        self.disc_init_steps = (0 if self.train_cfg is None else
                                self.train_cfg.get('disc_init_steps', 0))
        self.step_counter = 0  # counting training steps

        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        if self.discriminator:
            self.discriminator.init_weights(pretrained=pretrained)

    @auto_fp16(apply_to=('lq', ))
    def forward(self, lq, gt=None, test_mode=False, **kwargs):
        if test_mode:
            return self.forward_test(lq, gt, **kwargs)

        raise ValueError(
            'SRGAN model does not support `forward_train` function.')

    def forward_test(self,
                     lq,
                     gt=None,
                     meta=None,
                     save_image=False,
                     save_path=None,
                     iteration=None):

        stage1_output = self.stage1_generator(lq)
        fake_g_output = self.stage2_generator(stage1_output)
        if self.test_cfg is not None and self.test_cfg.get('metrics', None):
            assert gt is not None, (
                'evaluation with metrics must have gt images.')
            results = dict(eval_result=self.evaluate(fake_g_output, gt))
        else:
            results = dict(lq=lq.cpu(), output=fake_g_output.cpu())
            if gt is not None:
                results['gt'] = gt.cpu()

        # save image
        if save_image:
            lq_path = meta[0]['lq_path']
            folder_name = osp.splitext(osp.basename(lq_path))[0]
            if isinstance(iteration, numbers.Number):
                save_path = osp.join(save_path, folder_name,
                                     f'{folder_name}-{iteration + 1:06d}.png')
            elif iteration is None:
                save_path = osp.join(save_path, f'{folder_name}.png')
            else:
                raise ValueError('iteration should be number or None, '
                                 f'but got {type(iteration)}')
            mmcv.imwrite(tensor2img(fake_g_output), save_path)

        return results

    def train_step(self, data_batch, optimizer):
        # data
        lq = data_batch['lq']
        gt = data_batch['gt']

        # generator
        stage1_output = self.stage1_generator(lq)
        fake_g_output = self.stage2_generator(stage1_output)
        downsample_gt = F.interpolate(gt, scale_factor=0.25, mode='bilinear', align_corners=True)

        losses = dict()
        log_vars = dict()

        # no updates to discriminator parameters.
        set_requires_grad(self.discriminator, False)

        if (self.step_counter % self.disc_steps == 0
                and self.step_counter >= self.disc_init_steps):
            loss_final_pix = self.final_loss(fake_g_output, gt)
            losses['loss_final_pix'] = loss_final_pix
            if self.mid_loss:
                loss_mid_pix = self.mid_loss(stage1_output, downsample_gt)
                losses['loss_mid_pix'] = loss_mid_pix
            if self.perceptual_loss:
                loss_percep, loss_style = self.perceptual_loss(
                    fake_g_output, gt)
                if loss_percep is not None:
                    losses['loss_perceptual'] = loss_percep
                if loss_style is not None:
                    losses['loss_style'] = loss_style
            # gan loss for generator
            fake_g_pred = self.discriminator(fake_g_output)
            losses['loss_gan'] = self.gan_loss(
                fake_g_pred, target_is_real=True, is_disc=False)

            # parse loss
            loss_g, log_vars_g = self.parse_losses(losses)
            log_vars.update(log_vars_g)

            # optimize
            optimizer['stage1_generator'].zero_grad()
            optimizer['stage2_generator'].zero_grad()
            loss_g.backward()
            optimizer['stage1_generator'].step()
            optimizer['stage2_generator'].step()

        # discriminator
        set_requires_grad(self.discriminator, True)
        # real
        real_d_pred = self.discriminator(gt)
        loss_d_real = self.gan_loss(
            real_d_pred, target_is_real=True, is_disc=True)
        loss_d, log_vars_d = self.parse_losses(dict(loss_d_real=loss_d_real))
        optimizer['discriminator'].zero_grad()
        loss_d.backward()
        log_vars.update(log_vars_d)
        # fake
        fake_d_pred = self.discriminator(fake_g_output.detach())
        loss_d_fake = self.gan_loss(
            fake_d_pred, target_is_real=False, is_disc=True)
        loss_d, log_vars_d = self.parse_losses(dict(loss_d_fake=loss_d_fake))
        loss_d.backward()
        log_vars.update(log_vars_d)

        optimizer['discriminator'].step()

        self.step_counter += 1

        log_vars.pop('loss')  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars,
            num_samples=len(gt.data),
            results=dict(lq=lq.cpu(), gt=gt.cpu(), output=fake_g_output.cpu()))

        return outputs