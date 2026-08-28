# R2-Net: Efficient Low-Light Image Enhancement via Dynamic Routing and High-Frequency Refinement

这是论文 **R2-Net: Efficient Low-Light Image Enhancement via Dynamic Routing and High-Frequency Refinement** 的官方 PyTorch 实现。

CIKM 2026

Jiaqi Ruan, Yamin Li, Sihan Wu, Yucheng Wan, and Kansong Chen

## 简介

R2-Net 是一个面向低照度图像增强任务的高效网络，用于恢复自然光照、抑制噪声残留并保留图像高频细节。

模型包含两个核心模块：

- **Dynamic Routing Global Block (DRGB)**：用于内容感知的稀疏全局上下文建模。
- **High-Frequency Detail Residual Refinement (HDRR)**：用于边缘结构和纹理细节恢复。

完整模型结合全局光照建模和局部高频细化，旨在提升真实低照度场景下的增强质量与恢复鲁棒性。

## 模型变体

本仓库包含完整模型和两个消融模型：

| 名称 | 说明 |
| --- | --- |
| `r2_net.py` | 完整 R2-Net，包含 DRGB 和 HDRR |
| `r2_net_wo_drgb.py` | 移除 Dynamic Routing Global Block 的消融变体，用于评估稀疏全局上下文建模对增强性能的贡献 |
| `r2_net_wo_hdrr.py` | 移除 High-Frequency Detail Residual Refinement 模块的消融变体，用于评估高频细节恢复对增强性能的贡献 |

主模型实现：

```text
mmedit/models/backbones/generation_backbones/r2_net.py
```

模型注册位置：

```text
mmedit/models/backbones/generation_backbones/__init__.py
```

## 配置文件

LOL-v2-Real:

```text
my_config_lol/r2_net.py
my_config_lol/r2_net_wo_drgb.py
my_config_lol/r2_net_wo_hdrr.py
```

LSRW:

```text
my_config_lsrw/r2_net.py
my_config_lsrw/r2_net_wo_drgb.py
my_config_lsrw/r2_net_wo_hdrr.py
```

SICE:

```text
my_config_sice/r2_net.py
my_config_sice/r2_net_wo_drgb.py
my_config_sice/r2_net_wo_hdrr.py
```

## 环境安装

创建 conda 环境：

```bash
conda create -n r2net python=3.8 -y
conda activate r2net
```

请根据本机 CUDA 版本安装匹配的 PyTorch 和 MMCV，然后安装依赖：

```bash
pip install -r requirements.txt
pip install -e .
```

本项目使用 MMEditing 0.x 训练框架。

## 数据集准备

本仓库不包含数据集。请从官方渠道下载数据集，并按以下结构放置。

LOL-v2-Real:

```text
LOL_datasets/
├── train/
│   ├── low/
│   └── high/
└── test/
    ├── low/
    └── high/
```

LSRW:

```text
LSRW_datasets/
├── train/
│   ├── low/
│   └── high/
└── test/
    ├── low/
    └── high/
```

SICE:

```text
SICE_datasets/
├── train/
│   ├── low/
│   └── high/
└── test/
    ├── low/
    └── high/
```

## 训练

在 LOL-v2-Real 上训练 R2-Net：

```bash
bash tools/dist_train.sh my_config_lol/r2_net.py 1
```

在 LSRW 上训练 R2-Net：

```bash
bash tools/dist_train.sh my_config_lsrw/r2_net.py 1
```

在 SICE 上训练 R2-Net：

```bash
bash tools/dist_train.sh my_config_sice/r2_net.py 1
```

进行消融实验时，将 `r2_net.py` 替换为 `r2_net_wo_drgb.py` 或 `r2_net_wo_hdrr.py`。

## 测试

测试训练好的模型：

```bash
python tools/test.py my_config_lol/r2_net.py path/to/checkpoint.pth --save-path results/lol
```

LSRW 或 SICE 请使用对应配置文件。

## Demo

使用 restoration demo 进行推理：

```bash
python demo/restoration_demo.py \
  --config my_config_lol/r2_net.py \
  --checkpoint path/to/checkpoint.pth \
  --img_path_dir path/to/input_images \
  --save_path_dir results/demo
```

## 仓库结构

```text
R2-Net-release/
├── demo/
├── mmedit/
├── my_config_lol/
├── my_config_lsrw/
├── my_config_sice/
├── tools/
├── README.md
├── requirements.txt
└── setup.py
```

## License

请查看 `LICENSE` 文件。

## 引用

如果本仓库对你的研究有帮助，请引用：

```bibtex
@inproceedings{r2net2026,
  title={R2-Net: Efficient Low-Light Image Enhancement via Dynamic Routing and High-Frequency Refinement},
  author={Ruan, Jiaqi and Li, Yamin and Wu, Sihan and Wan, Yucheng and Chen, Kansong},
  booktitle={Proceedings of the 35th ACM International Conference on Information and Knowledge Management},
  year={2026}
}
```

## TODO

- [ ] 论文发表后补充论文链接。
- [ ] 补充更详细的数据集下载说明。
- [ ] 补充更多推理示例。
