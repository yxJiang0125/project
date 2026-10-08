# HSTA: Hybrid CNN–Swin Temporal Attention Network for Nighttime Video Anomaly Detection

**Paper**: A Hybrid CNN–Swin Temporal Attention Network for Nighttime Video Anomaly Detection  
**Authors**: Yuxuan Jiang, Yunhui Zeng, Yanlei Cui, Xin Jin  
**Venue**: ICIP 2026

## Key Results

| Dataset | AUC | FPS |
|---|---|---|
| Nighttime Traffic | 76.4% | 11.05 |
| UCSD Ped2 | 98.5% | — |
| CUHK Avenue | 91.1% | — |
| ShanghaiTech | 74.3% | — |

## Architecture

HSTA is a future-frame prediction network with three components:

1. **Hybrid Encoder** — ResNet-50 (local features) + Swin Transformer (global context), fused via cross-level concatenation
2. **Random Temporal Shift (RTS)** — stochastic channel-wise temporal shifts to enhance motion sensitivity
3. **TACAB Decoder** — Temporal-Aware Channel Attention Blocks combining residual channel attention (RCAB) with 3D convolutions for motion-aware decoding

## Directory Structure

```
HSTA/
├── models/
│   ├── hsta.py              # Full HSTA model (HybridCNN_Swin, RTS, TACAB, Network_Myself)
│   ├── basic_modules.py     # ConvBnRelu, ConvTransposeBnRelu, initialize_weights
│   └── __init__.py
├── datasets/
│   ├── video_data.py        # Video/TestVideo dataset loaders
│   ├── label_data.py        # Ground-truth label loader (.mat / .npy)
│   └── __init__.py
├── utils/
│   ├── anomaly_util.py      # PSNR, anomaly score, AUC computation
│   ├── train_util.py        # decode_input helper
│   ├── loss_util.py         # Intensity, Gradient, MS-SSIM, L2 losses
│   ├── optimizer_util.py    # Optimizer and scheduler setup
│   ├── log_util.py          # Logger creation
│   └── __init__.py
├── config/
│   ├── defaults.py          # Default config (yacs)
│   ├── avenue.yaml          # CUHK Avenue config
│   ├── shanghaitech.yaml    # ShanghaiTech config
│   ├── ped2.yaml            # UCSD Ped2 config
│   ├── highway.yaml         # Nighttime traffic config
│   └── __init__.py
├── train.py                 # Training script
├── test.py                  # Inference + AUC evaluation script
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Dataset Preparation

Organise each dataset under `datasets/` as follows:

```
datasets/
└── <dataset_name>/
    ├── training/
    │   └── frames/
    │       ├── video_01/
    │       │   ├── 0001.jpg
    │       │   └── ...
    │       └── ...
    ├── testing/
    │   └── frames/
    │       ├── video_01/
    │       └── ...
    └── <dataset_name>.mat   # ground-truth labels (ped2 / avenue / highway)
```

For **ShanghaiTech**, place per-video `.npy` label files under `datasets/shanghaitech/test_frame_mask/`.

The nighttime traffic dataset is available at:  
https://github.com/yxJiang0125/Dataset-for-nighttime-traffic-videos

## Inference

```bash
python test.py \
    --cfg config/avenue.yaml \
    --model-file pretrained/avenue.pth
```

Override any config key on the command line:

```bash
python test.py \
    --cfg config/avenue.yaml \
    --model-file pretrained/avenue.pth \
    GPUS "(0,1)" TEST.BATCH_SIZE_PER_GPU 2
```

Results are written to `output/<dataset>/<cfg_name>/anomaly_results.csv`.

## Training

```bash
python train.py --cfg config/avenue.yaml
```

Checkpoints are saved every `SAVE_CHECKPOINT_FREQ` epochs to `output/<dataset>/<cfg_name>/`.

## Config Options

Key fields in each YAML (override defaults in `config/defaults.py`):

| Key | Description |
|---|---|
| `DATASET.ROOT` | Root directory containing all datasets |
| `DATASET.DATASET` | Dataset name (subfolder under ROOT) |
| `MODEL.IMAGE_SIZE` | Resize resolution `[H, W]` (must be multiples of 32) |
| `MODEL.ENCODED_FRAMES` | Number of input frames (default: 4) |
| `TRAIN.LR` | Learning rate (default: 0.0002) |
| `TEST.ANOMALY_THRESHOLD` | PSNR threshold for frame-level anomaly flag |
| `GPUS` | GPU IDs as a tuple, e.g. `(0,)` or `(0,1,2)` |
