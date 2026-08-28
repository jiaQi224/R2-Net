# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcv.parallel import collate, scatter
import torch.nn.functional as F
import cv2
from mmedit.core import tensor2img
from mmedit.datasets.pipelines import Compose


def generation_inference(model, img, img_unpaired=None, patch_inference=True, patch_size=256):
    """Inference image with the model.

    Args:
        model (nn.Module): The loaded model.
        img (str): File path of input image.
        img_unpaired (str, optional): File path of the unpaired image.
            If not None, perform unpaired image generation. Default: None.

    Returns:
        np.ndarray: The predicted generation result.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = Compose(cfg.test_pipeline[0:])
    # prepare data
    if img_unpaired is None:
        data = dict(img_a_path=img)
    else:
        data = dict(A_path=img_unpaired, B_path=img_unpaired)
    data = test_pipeline(data)
    _, orig_W, orig_H = data['img_a'].shape
    new_H = ((orig_H + 255) // 256) * 256
    new_W = ((orig_W + 255) // 256) * 256
    size_new = max(new_H, new_W)
    data['img_a'] = F.interpolate(data['img_a'].unsqueeze(0), size=(size_new, size_new), mode='bilinear', align_corners=False)
    data['img_a'] = data['img_a'][0]
    data = collate([data], samples_per_gpu=1)
    if 'cuda' in str(device):
        data = scatter(data, [device])[0]

    if patch_inference:
        channel, orig_h, orig_w = data['img_a'].shape[1], data['img_a'].shape[2], data['img_a'].shape[3]
        number_h = data['img_a'].shape[2] // patch_size  #
        number_w = data['img_a'].shape[3] // patch_size  #
        patch_sequence = []  # 定义一个集合存放patch
        for h in range(number_h):
            for w in range(number_w):
                image_patch = data['img_a'][:, :, (h * patch_size):(h * patch_size + patch_size), (w * patch_size):(w * patch_size + patch_size)]
                patch_sequence.append(image_patch)
        predict_sequence = []
        with torch.no_grad():
            for image in patch_sequence:
                data['img_a'] = image
                results = model(test_mode=True, **data)
                predict_sequence.append(results['fake_b'])
        count = 0
        initial_patch = torch.ones(1, channel, orig_h, orig_h)
        for h in range(number_h):
            for w in range(number_w):
                initial_patch[:, :, h * patch_size:(h + 1) * patch_size, w * patch_size:(w + 1) * patch_size] = predict_sequence[count]
                count += 1
        output =  tensor2img(initial_patch, min_max=(0, 1))
    else:
        # forward the model
        with torch.no_grad():
            results = model(test_mode=True, **data)
        # process generation shown mode
        if img_unpaired is None:
            output = tensor2img(results['output'], min_max=(0, 1))
        else:
            if model.show_input:
                output = np.concatenate([
                    tensor2img(results['real_a'], min_max=(0, 1)),
                    tensor2img(results['fake_b'], min_max=(0, 1)),
                    tensor2img(results['real_b'], min_max=(0, 1)),
                    tensor2img(results['fake_a'], min_max=(0, 1))
                ],
                                        axis=1)
            else:
                output = tensor2img(results['fake_b'], min_max=(-1, 1))
                # output = tensor2img(results['fake_a'], min_max=(-1, 1))
    output = cv2.resize(output, (orig_H, orig_W))
    return output
