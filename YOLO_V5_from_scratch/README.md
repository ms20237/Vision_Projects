# YOLO V5 from Scratch

<p align="center">
  <img src="image/overview.png" alt="YOLO V5 Overview" width="600"/>
</p>

<p align="center">
  <em>YOLO V5 Overview.</em>
</p>

A PyTorch implementation of **YOLOv5** (Ultralytics), built from the ground up, following the architecture of the public [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) implementation. Unlike YOLOv1-v4, YOLOv5 was never published as an academic paper — this repository reproduces the "v6.0+"-style config: a `Conv`-stem backbone, `C3` CSP blocks, and `SPPF`.

This repository provides a complete, modular implementation including data loading, model architecture (CSPDarknet with SPPF and a C3-based PANet head), a custom multi-scale loss function with CIoU box regression and per-scale objectness balancing, a training pipeline with multi-scale training, and evaluation metrics.

---

## 📋 Table of Contents

- [✨ Features](#-features)
- [📁 Project Structure](#-project-structure)
- [🚀 Installation](#-installation)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
- [📊 Dataset Preparation](#-dataset-preparation)
- [⚙️ Configuration](#️-configuration)
- [🏃 Usage](#-usage)
  - [Training](#training)
  - [Command Line Arguments](#command-line-arguments)
  - [Available Arguments](#available-arguments)
- [🧠 Model Architecture](#-model-architecture)
  - [CSPDarknetV5 Backbone](#cspdarknetv5-backbone)
  - [SPPF + PANet Neck](#sppf--panet-neck)
  - [Model Scaling (n/s/m/l/x)](#model-scaling-nsmlx)
  - [Anchor Boxes](#anchor-boxes)
  - [Output Shape](#output-shape)
- [📉 Loss Function](#-loss-function)
- [📊 Evaluation Metrics](#-evaluation-metrics)
  - [mAP Calculation](#map-calculation)
- [📈 Results](#-results)
- [📝 License](#-license)

---

## ✨ Features

- **Complete YOLOv5 implementation** from scratch using PyTorch
- **CSPDarknet backbone** with a single 6×6 stride-2 `Conv` stem and `C3` ("CSP Bottleneck with 3 convolutions") stages — simpler than YOLOv4's hand-rolled CSP blocks
- **SiLU activation throughout** — backbone, neck, *and* head (YOLOv4 only used Mish in the backbone and kept LeakyReLU elsewhere)
- **SPPF (Spatial Pyramid Pooling - Fast)**: sequential max-pools instead of YOLOv4's parallel SPP, same effective receptive field, fewer operations
- **PANet neck** built from `C3` blocks: top-down path plus a bottom-up path, so features flow both ways before detection
- **Multi-scale prediction**: detects at 3 grid resolutions simultaneously (13×13, 26×26, 52×52 for a 416px input)
- **9 anchor boxes** (3 per scale)
- **Width/depth multipliers** for the full n/s/m/l/x model family (defaults to the "s" configuration)
- **CIoU loss** for box regression, with **per-scale objectness balancing** (finer grids weighted up, coarser grids weighted down) and optional **label smoothing**
- **Independent per-class logistic classifiers** (binary cross-entropy) instead of softmax, so multi-label predictions are possible
- **Multi-Scale Training** (320–608px) for robust performance across resolutions
- **Custom dataset loader** for PASCAL VOC format with label parsing
- **Mean Average Precision (mAP)** evaluation metric
- **Non-Maximum Suppression (NMS)** for post-processing, applied across all 3 scales' detections together
- **Flexible configuration** via YAML files and command-line arguments
- **Checkpoint saving/loading** for resuming training
- **Visualization utilities** for bounding boxes

---

## 📁 Project Structure

```
YOLO_V5_from_sratch/
├── configs/
│   └── train_config.yaml          # Configuration file for training
├── loss_f.py                      # YOLOv5 multi-scale loss function (CIoU + balanced BCE)
├── main.py                        # Training script entry point
├── model.py                       # YOLO V5 CSPDarknetV5 + SPPF + C3-based PANet head
├── utils.py                       # Utilities (IoU, NMS, mAP, target building, multi-scale, checkpointing)
├── intersection_Over_Union.py     # IoU calculation
└── dataset.py                     # dataset class for VOCDataset
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- CUDA-capable GPU (recommended — though the default "s" configuration is the lightest of the v3/v4/v5 implementations in this repo, at roughly 6M parameters)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/ms20237/Vision_Projects.git
cd Vision_Projects/YOLO_V5_from_sratch
```

2. Install dependencies:
```bash
pip install torch torchvision pandas pillow matplotlib numpy pyyaml tqdm
```

---

## 📊 Dataset Preparation

This implementation expects the PASCAL VOC dataset format. Prepare your dataset as follows:

1. **Directory structure:**
```
data/
├── images/                        # Training images
│   └── 000001.jpg
├── labels/                        # Label files (.txt)
│   └── 000001.txt
└── test/                          # Test images (optional)
    └── 000001.jpg
```

2. **Label format** (per line): `class_id x_center y_center width height`
   - Normalized coordinates (0-1)
   - Example: `0 0.5 0.5 0.2 0.3`

3. **Create CSV files** for training and testing:
   - Use `generate_csv.py` to create annotation CSV files
   - The CSV should have columns: `image_path, label_path`

> **Note on targets:** Same convention as the YOLOv3/v4 versions of this repo. Unlike a single-scale YOLO, `VOCDataset.__getitem__` should return `(image_tensor, boxes)` where `boxes` is a plain list of `(class_idx, x, y, w, h)` tuples in whole-image-relative `[0, 1]` units — **not** a pre-gridded target tensor. Which of the 3 grids/9 anchors each box belongs to is decided at train/eval time by `utils.build_targets`, since that depends on which scale's anchor best matches the box's shape (and, under multi-scale training, on which input size was chosen for that batch). Use `utils.yolo_multiscale_collate_fn` as your `DataLoader`'s `collate_fn` so these variable-length per-image box lists batch correctly. This target format, decoding, NMS, and mAP logic are identical to the YOLOv3/v4 implementations — only `model.py` and `loss_f.py` differ.

---

## ⚙️ Configuration

Create a `configs/train_config.yaml` file with your settings:

```yaml
# train_config.yaml

train_config_path: "./YOLO_V5_from-sratch/configs/train_config.yaml"
img_dir: "./data/images"
label_dir: "./data/labels"
test_path: "./data/test.csv"
dataset_ex_dir: "./data/100examples.csv"

load_model_file: "./YOLO_V5_from-sratch/overfit.pth.tar"

lr: 0.00002
device: "cuda"
batch_size: 16
epochs: 10
n_works: 2
seed: 123
load_model: false
weight_decay: 0.0
pin_memory: true
multi_scale: true
```

---

## 🏃 Usage

### Training

Run the training script:

```bash
python main.py --config configs/train_config.yaml
```

### Command Line Arguments

Override configuration values via command line:

```bash
python main.py \
    --img_dir data/images \
    --label_dir data/labels \
    --lr 0.001 \
    --batch_size 16 \
    --epochs 100 \
    --device cuda \
    --multi_scale \
    --load_model \
    --load_model_file checkpoint.pth.tar
```

## **Available Arguments**

| Argument            | Description                                          | Default                     |
| ------------------- | ----------------------------------------------------- | ---------------------------- |
| `--config`          | Path to YAML config file                              | `configs/train_config.yaml` |
| `--img_dir`         | Image directory path                                  | From config                 |
| `--label_dir`       | Label directory path                                  | From config                 |
| `--lr`              | Learning rate                                         | From config                 |
| `--batch_size`      | Batch size                                            | From config                 |
| `--epochs`          | Number of training epochs                             | From config                 |
| `--device`          | Device (`cuda`/`cpu`)                                 | Auto-detect                 |
| `--n_works`         | Number of data loading workers                        | From config                 |
| `--seed`            | Random seed                                           | From config                 |
| `--load_model`      | Load checkpoint                                       | `False`                     |
| `--load_model_file` | Checkpoint file path                                  | `None`                      |
| `--multi_scale`     | Enable multi-scale training (random size every 10 batches) | `True`                  |

---

## 🧠 Model Architecture

The implementation follows the public Ultralytics YOLOv5 architecture with:

- **CSPDarknet backbone**: a `Conv` stem + 4 downsample-then-`C3` stages, ending in `SPPF`
- **No fully connected layers**: the entire network is convolutional
- **Batch normalization** after every convolutional layer
- **SiLU activation everywhere** — backbone, neck, and head all share the same `Conv` block (unlike YOLOv4, which mixed Mish and LeakyReLU)
- **PANet neck**: top-down path followed by a bottom-up path, built from `C3` blocks instead of YOLOv4's 5-conv sets
- **Anchor-based, multi-scale detection**: 3 anchors per cell, at each of 3 grid resolutions (9 anchors total) — unchanged from YOLOv3/v4

### CSPDarknetV5 Backbone

Defined in `model.py`. The core building block is `C3` — simpler than YOLOv4's `CSPBlock`: the input feeds two parallel 1×1 convs (one straight through as a shortcut, one into a stack of `n` residual `Bottleneck` units), concatenated and fused with a final 1×1 conv:

```python
class CSPDarknetV5(nn.Module):
    # stem: Conv(3->w(64), kernel=6, stride=2)          # P1/2
    # -> Conv(stride2) + C3(w(128),  n=d(3))              # P2/4
    # -> Conv(stride2) + C3(w(256),  n=d(6))  --> route1   # P3/8
    # -> Conv(stride2) + C3(w(512),  n=d(9))  --> route2   # P4/16
    # -> Conv(stride2) + C3(w(1024), n=d(3))                # P5/32
    # -> SPPF(w(1024))                          --> route3   # P5/32, post-SPPF
    ...
```

`w()` and `d()` scale channel counts and block-repeat counts by `width_mult` / `depth_mult` respectively — see [Model Scaling](#model-scaling-nsmlx) below.

### SPPF + PANet Neck

For a 416×416 input (using the default "s" channel widths):

1. **SPPF**: `route3` is reduced to half its channels, then max-pooled through 3 *sequential* 5×5 max-pools (each stride 1, so spatial size is unchanged), and all four feature maps (input + 3 pooled) are concatenated and reduced back down — mathematically close to YOLOv4's parallel 5/9/13 SPP, but cheaper.
2. **Top-down (FPN-style)**: the SPPF output is reduced, upsampled 2×, and concatenated with `route2` → a `C3` block produces the P4-fused features. Those are reduced, upsampled 2×, and concatenated with `route1` → a `C3` block produces the P3-fused features (finest resolution, "small"-object head).
3. **Bottom-up (PANet)**: the P3-fused features are downsampled (stride-2 conv) and concatenated with the *pre-upsample* P4-reduction → a `C3` block produces the "medium"-object head input. That's downsampled again and concatenated with the *pre-upsample* P5-reduction → a `C3` block produces the "large"-object head input.
4. **Prediction**: bare 1×1 convs (no BN/activation) predict from each of the three PANet outputs.

### Model Scaling (n/s/m/l/x)

`YOLO_V5.__init__` exposes `width_mult` and `depth_mult`, matching how Ultralytics derives the whole model family from one architecture description:

| Variant | `width_mult` | `depth_mult` |
| ------- | ------------- | -------------- |
| n       | 0.25          | 0.33           |
| **s (default)** | **0.50** | **0.33** |
| m       | 0.75          | 0.67           |
| l       | 1.00          | 1.00           |
| x       | 1.25          | 1.33           |

```python
model = YOLO_V5(num_classes=20, width_mult=1.0, depth_mult=1.0)  # YOLOv5l-sized
```

### Anchor Boxes

Default anchor priors, in **pixels of a 416×416 input** (the standard COCO anchors, reused unchanged from the YOLOv3/v4 versions of this repo — Ultralytics calls the k-means re-fitting step "AutoAnchor"; re-run it on your own dataset for best results):

```python
COCO_ANCHORS = [
    [(116, 90), (156, 198), (373, 326)],  # scale 1: 13x13 grid -- large objects
    [(30, 61), (62, 45), (59, 119)],       # scale 2: 26x26 grid -- medium objects
    [(10, 13), (16, 30), (33, 23)],        # scale 3: 52x52 grid -- small objects
]
```

Each ground-truth box is matched to whichever of the 9 total anchors (across all 3 scales) has the closest shape. See `utils.build_targets`.

### Output Shape

- Input: `(batch_size, 3, 416, 416)` (default)
- Output: a **list of 3 tensors**, one per scale:
  - `(batch_size, 13, 13, 3 × (5 + C))`
  - `(batch_size, 26, 26, 3 × (5 + C))`
  - `(batch_size, 52, 52, 3 × (5 + C))`
  - where `C = 20` (VOC classes), so each scale has `3 × 25 = 75` output channels

**Multi-Scale Training**: input size can vary from 320 to 608 pixels (multiples of 32); the three output grids scale proportionally (e.g. 320px → 10×10 / 20×20 / 40×40).

---

## 📉 Loss Function

The custom YOLOv5 loss function (`loss_f.py`) sums a per-scale loss across all 3 prediction scales:

1. **Bounding box loss — CIoU**, unchanged from YOLOv4: predicted and target boxes are decoded into `(x, y, w, h)` and compared with a single **Complete IoU** score combining overlap, center-distance, and aspect-ratio consistency. Loss term is `1 - CIoU`, weighted by **λ_coord = 5**.

2. **Objectness loss — binary cross-entropy, per-scale balanced**: the finest grid (52×52) has far more cells than the coarsest (13×13), so its objectness loss is weighted differently per scale via `obj_scale_balance` (defaults `[0.4, 1.0, 4.0]` for large/medium/small in this repo's scale ordering) before being combined with **λ_obj**/**λ_noobj = 0.5**.

3. **Class loss** — independent per-class binary cross-entropy (multi-label logistic classifiers), with optional **label smoothing** (`label_smoothing`, default `0.0`) softening the `{0, 1}` BCE targets to reduce over-confidence.

**Key differences from YOLOv4**: (a) per-scale objectness-loss balancing was added; (b) optional label smoothing was added. Box-loss formula (CIoU), class-loss formula (BCE), and ahead-of-time anchor+scale assignment via `utils.build_targets` are all unchanged from YOLOv4.

---

## 📊 Evaluation Metrics

The implementation includes:

- **Intersection over Union (IoU)** calculation
- **Non-Maximum Suppression (NMS)** for removing duplicate detections — run once over the pooled detections from all 3 scales
- **Mean Average Precision (mAP)** at IoU threshold 0.5

### mAP Calculation

During training, mAP is computed on the validation set to monitor performance:

```python
pred_boxes, target_boxes = get_bboxes(
    train_loader, model, iou_threshold=0.5, threshold=0.4,
    anchors=COCO_ANCHORS, base_size=416, device=device, C=20,
)

mean_avg_prec = mean_average_precision(
    pred_boxes, target_boxes,
    iou_threshold=0.5,
    box_format="midpoint",
    num_classes=20,
)
```

---

## 📈 Results

*Note: Add your training results here including:*
- Final mAP score
- Sample detection visualizations
- Training loss curves

Example visualization using `plot_image()` function:

```python
from utils import plot_image
plot_image(image, nms_boxes)
```

---

## 📝 License

This project is licensed under the [MIT License](https://choosealicense.com/licenses/mit/).

---

**Happy Detecting!** 🎯
