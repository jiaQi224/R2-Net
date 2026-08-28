# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

import cv2
from tqdm import tqdm
import mmcv
import torch

from mmedit.apis import init_model, restoration_inference
from mmedit.core import tensor2img
from mmedit.utils import modify_args


def parse_args():
    modify_args()
    parser = argparse.ArgumentParser(description='Restoration demo')
    parser.add_argument('--config',
                        required=True,
                        help='test config file path')
    parser.add_argument('--checkpoint',
                        required=True,
                        help='checkpoint file')
    parser.add_argument('--img_path_dir', default="./LSRW_datasets/test/low",
                        help='path to input image file')
    parser.add_argument('--save_path_dir', default="./results/r2_net",
                        help='path to save generation result')
    parser.add_argument(
        '--imshow', action='store_true', help='whether show image with opencv')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    parser.add_argument(
        '--ref-path', default=None, help='path to reference image file')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.device < 0 or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda', args.device)

    model = init_model(args.config, args.checkpoint, device=device)
    mmcv.mkdir_or_exist(args.save_path_dir)

    file_list = os.listdir(args.img_path_dir)
    for file in tqdm(file_list):
        img_path = os.path.join(args.img_path_dir, file)
        if args.ref_path:  # Ref-SR
            output = restoration_inference(model, img_path, args.ref_path)
        else:  # SISR
            output = restoration_inference(model, img_path)
        output = tensor2img(output)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        mmcv.imwrite(output, os.path.join(args.save_path_dir, file))
        if args.imshow:
            mmcv.imshow(output, 'predicted restoration result')


if __name__ == '__main__':
    main()
