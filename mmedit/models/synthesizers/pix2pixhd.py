# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import torch.nn.functional as F
import mmcv
import numpy as np
import torch
from mmcv.runner import auto_fp16
from torch import nn
from mmedit.core import tensor2img
from ..base import BaseModel
from ..builder import build_backbone, build_component, build_loss
from ..common import set_requires_grad
from ..registry import MODELS


@MODELS.register_module()
class Pix2PixHD(BaseModel):
    def __init__(self,
                 generator,
                 pixel_loss=None,
                 ssim_loss=None,
                 perceptual_loss=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # generator
        self.generator = build_backbone(generator)
        self.refine = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )


        # losses
        # assert gan_loss is not None  # gan loss cannot be None
        # self.gan_loss = build_loss(gan_loss)
        self.pixel_loss = build_loss(pixel_loss) if pixel_loss else None
        self.ssim_loss = build_loss(ssim_loss) if ssim_loss else None
        self.perceptual_loss = build_loss(perceptual_loss) if perceptual_loss else None

        self.disc_steps = 1 if self.train_cfg is None else self.train_cfg.get(
            'disc_steps', 1)
        self.disc_init_steps = (0 if self.train_cfg is None else
                                self.train_cfg.get('disc_init_steps', 0))
        if self.train_cfg is None:
            self.direction = ('a2b' if self.test_cfg is None else
                              self.test_cfg.get('direction', 'a2b'))
        else:
            self.direction = self.train_cfg.get('direction', 'a2b')
        self.step_counter = 0  # counting training steps

        self.show_input = (False if self.test_cfg is None else
                           self.test_cfg.get('show_input', False))

        # support fp16
        self.fp16_enabled = False
        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        """Initialize weights for the model.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Default: None.
        """
        self.generator.init_weights(pretrained=pretrained)

    def setup(self, img_a, img_b, meta):
        """Perform necessary pre-processing steps.

        Args:
            img_a (Tensor): Input image from domain A.
            img_b (Tensor): Input image from domain B.
            meta (list[dict]): Input meta data.

        Returns:
            Tensor, Tensor, list[str]: The real images from domain A/B, and \
                the image path as the metadata.
        """
        a2b = self.direction == 'a2b'
        real_a = img_a if a2b else img_b
        real_b = img_b if a2b else img_a
        image_path = [v['img_a_path' if a2b else 'img_b_path'] for v in meta]

        return real_a, real_b, image_path

    @auto_fp16(apply_to=('img_a', 'img_b'))
    def forward_train(self, img_a, img_b, meta):
        """Forward function for training.

        Args:
            img_a (Tensor): Input image from domain A.
            img_b (Tensor): Input image from domain B.
            meta (list[dict]): Input meta data.

        Returns:
            dict: Dict of forward results for training.
        """
        # necessary setup
        real_a, real_b, _ = self.setup(img_a, img_b, meta)
        x_512 = F.interpolate(real_a, (512, 512), mode='bilinear')
        midgray = self.generator(x_512)
        midgray = (midgray + 1.0) * 0.5
        A = (x_512 + 1.0) * 0.5
        fake_b = torch.where(midgray <= 0.5, 2 * A * midgray + A ** 2 * (1 - 2 * midgray), A)
        fake_b = torch.where(midgray > 0.5, 2 * A * (1 - midgray) + (2 * midgray - 1) * torch.sqrt(A), fake_b)
        fake_b = fake_b * 2.0 - 1.0

        # refine
        fake_b_1840 = F.interpolate(fake_b, (1840, 1840), mode='bilinear')
        high_input = torch.concat((real_a, fake_b_1840), dim=1)
        refine_out = self.refine(high_input) + fake_b_1840
        fake_b = refine_out
        results = dict(real_a=real_a, fake_b=fake_b, real_b=real_b)
        return results

    def forward_test(self,
                     img_a,
                     img_b,
                     meta,
                     save_image=False,
                     save_path=None,
                     iteration=None):
        """Forward function for testing.

        Args:
            img_a (Tensor): Input image from domain A.
            img_b (Tensor): Input image from domain B.
            meta (list[dict]): Input meta data.
            save_image (bool, optional): If True, results will be saved as
                images. Default: False.
            save_path (str, optional): If given a valid str path, the results
                will be saved in this path. Default: None.
            iteration (int, optional): Iteration number. Default: None.

        Returns:
            dict: Dict of forward and evaluation results for testing.
        """
        # No need for metrics during training for pix2pixHD. And
        # this is a special trick in pix2pixHD original paper & implementation,
        # collecting the statistics of the test batch at test time.
        self.train()

        # necessary setup
        real_a, real_b, image_path = self.setup(img_a, img_b, meta)
        x_512 = F.interpolate(real_a, (512, 512), mode='bilinear')
        midgray = self.generator(x_512)
        midgray = (midgray + 1.0) * 0.5
        A = (x_512 + 1.0) * 0.5
        fake_b = torch.where(midgray <= 0.5, 2 * A * midgray + A ** 2 * (1 - 2 * midgray), A)
        fake_b = torch.where(midgray > 0.5, 2 * A * (1 - midgray) + (2 * midgray - 1) * torch.sqrt(A), fake_b)
        fake_b = fake_b * 2.0 - 1.0
        # refine
        fake_b_1840 = F.interpolate(fake_b, (1840, 1840), mode='bilinear')
        high_input = torch.concat((real_a, fake_b_1840), dim=1)
        refine_out = self.refine(high_input) + fake_b_1840
        fake_b = refine_out
        results = dict(
            real_a=real_a.cpu(), fake_b=fake_b.cpu(), real_b=real_b.cpu())

        # save image
        if save_image:
            assert save_path is not None
            folder_name = osp.splitext(osp.basename(image_path[0]))[0]
            if self.show_input:
                if iteration:
                    save_path = osp.join(
                        save_path, folder_name,
                        f'{folder_name}-{iteration + 1:06d}-ra-fb-rb.png')
                else:
                    save_path = osp.join(save_path,
                                         f'{folder_name}-ra-fb-rb.png')
                output = np.concatenate([
                    tensor2img(results['real_a'], min_max=(-1, 1)),
                    tensor2img(results['fake_b'], min_max=(-1, 1)),
                    tensor2img(results['real_b'], min_max=(-1, 1))
                ],
                                        axis=1)
            else:
                if iteration:
                    save_path = osp.join(
                        save_path, folder_name,
                        f'{folder_name}-{iteration + 1:06d}-fb.png')
                else:
                    save_path = osp.join(save_path, f'{folder_name}-fb.png')
                output = tensor2img(results['fake_b'], min_max=(-1, 1))
            flag = mmcv.imwrite(output, save_path)
            results['saved_flag'] = flag

        return results

    def forward_dummy(self, img):
        """Used for computing network FLOPs.

        Args:
            img (Tensor): Dummy input used to compute FLOPs.

        Returns:
            Tensor: Dummy output produced by forwarding the dummy input.
        """
        out = self.generator(img)
        return out

    def forward(self, img_a, img_b, meta, test_mode=False, **kwargs):
        """Forward function.

        Args:
            img_a (Tensor): Input image from domain A.
            img_b (Tensor): Input image from domain B.
            meta (list[dict]): Input meta data.
            test_mode (bool): Whether in test mode or not. Default: False.
            kwargs (dict): Other arguments.
        """
        if test_mode:
            return self.forward_test(img_a, img_b, meta, **kwargs)

        return self.forward_train(img_a, img_b, meta)

    def backward_generator(self, outputs):
        """Backward function for the generator.

        Args:
            outputs (dict): Dict of forward results.

        Returns:
            dict: Loss dict.
        """
        losses = dict()
        if self.pixel_loss:
            losses['loss_pixel'] = self.pixel_loss(outputs['fake_b'],
                                                   outputs['real_b'])
        if self.ssim_loss:
            losses['loss_ssim'] = self.ssim_loss(outputs['fake_b'],
                                                   outputs['real_b'])

        if self.perceptual_loss:
            losses['loss_perceptual'], _ = self.perceptual_loss(outputs['fake_b'],
                                                  outputs['real_b'])

        loss_g, log_vars_g = self.parse_losses(losses)
        loss_g.backward()
        return log_vars_g

    def train_step(self, data_batch, optimizer):
        # data
        img_a = data_batch['img_a']
        img_b = data_batch['img_b']
        meta = data_batch['meta']

        # forward generator
        outputs = self.forward(img_a, img_b, meta, test_mode=False)

        log_vars = dict()

        optimizer['generator'].zero_grad()
        optimizer['refine'].zero_grad()
        log_vars.update(self.backward_generator(outputs=outputs))
        optimizer['generator'].step()
        optimizer['refine'].step()

        self.step_counter += 1

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        results = dict(
            log_vars=log_vars,
            num_samples=len(outputs['real_a']),
            results=dict(
                real_a=outputs['real_a'].cpu(),
                fake_b=outputs['fake_b'].cpu(),
                real_b=outputs['real_b'].cpu()))

        return results

    def val_step(self, data_batch, **kwargs):
        """Validation step function.

        Args:
            data_batch (dict): Dict of the input data batch.
            kwargs (dict): Other arguments.

        Returns:
            dict: Dict of evaluation results for validation.
        """
        # data
        img_a = data_batch['img_a']
        img_b = data_batch['img_b']
        meta = data_batch['meta']

        # forward generator
        results = self.forward(img_a, img_b, meta, test_mode=True, **kwargs)
        return results
