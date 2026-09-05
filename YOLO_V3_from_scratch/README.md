# YOLO V3 from Scratch

A PyTorch implementation of **YOLOv3**, built from the ground up, following the original paper ["YOLOv3: An Incremental Improvement"](https://arxiv.org/abs/1804.02767) by Joseph Redmon and Ali Farhadi.

This repository provides a complete, modular implementation including data loading, model architecture (Darknet-53 with an FPN-style multi-scale head), a custom multi-scale loss function with anchor boxes, a training pipeline with multi-scale training, and evaluation metrics.

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
  - [Darknet-53 Backbone](#darknet-53-backbone)
  - [Multi-Scale Detection Head](#multi-scale-detection-head)
  - [Anchor Boxes](#anchor-boxes)
  - [Output Shape](#output-shape)
- [📉 Loss Function](#-loss-function)
- [📊 Evaluation Metrics](#-evaluation-metrics)
  - [mAP Calculation](#map-calculation)
- [📈 Results](#-results)
- [📝 License](#-license)

---

## ✨ Features

- **Complete YOLOv3 implementation** from scratch using PyTorch
- **Darknet-53 backbone** with residual (skip) connections, 53 convolutional layers
- **Multi-scale prediction**: detects at 3 grid resolutions simultaneously (13×13, 26×26, 52×52 for a 416px input), FPN-style, via upsample-and-concatenate
- **9 anchor boxes** (3 per scale) instead of a single anchor set
- **Independent per-class logistic classifiers** (binary cross-entropy) instead of softmax, so multi-label predictions are possible
- **Multi-Scale Training** (320–608px) for robust performance across resolutions
- **Custom dataset loader** for PASCAL VOC format with label parsing
- **YOLO-specific multi-scale loss function** with coordinate, objectness, and class components
- **Mean Average Precision (mAP)** evaluation metric
- **Non-Maximum Suppression (NMS)** for post-processing, applied across all 3 scales' detections together
- **Flexible configuration** via YAML files and command-line arguments
- **Checkpoint saving/loading** for resuming training
- **Visualization utilities** for bounding boxes

---

## 📁 Project Structure

```
YOLO_V3_from_sratch/
├── configs/
│   └── train_config.yaml          # Configuration file for training
├── loss_f.py                      # YOLOv3 multi-scale loss function implementation
├── main.py                        # Training script entry point
├── model.py                       # YOLO V3 Darknet-53 + FPN-style detection head
├── utils.py                       # Utilities (IoU, NMS, mAP, target building, multi-scale, checkpointing)
├── intersection_Over_Union.py     # IoU calculation
└── README.md
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- CUDA-capable GPU (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/ms20237/Vision_Projects.git
cd Vision_Projects/YOLO_V3_from_sratch
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

> **Note on targets:** Unlike a single-scale YOLO, `VOCDataset.__getitem__` should return `(image_tensor, boxes)` where `boxes` is a plain list of `(class_idx, x, y, w, h)` tuples in whole-image-relative `[0, 1]` units — **not** a pre-gridded target tensor. Which of the 3 grids/9 anchors each box belongs to is decided at train/eval time by `utils.build_targets_v3`, since that depends on which scale's anchor best matches the box's shape (and, under multi-scale training, on which input size was chosen for that batch). Use `utils.yolov3_collate_fn` as your `DataLoader`'s `collate_fn` so these variable-length per-image box lists batch correctly.

---

## ⚙️ Configuration

Create a `configs/train_config.yaml` file with your settings:

```yaml
# train_config.yaml

train_config_path: "./YOLO_V3_from-sratch/configs/train_config.yaml"
img_dir: "./data/images"
label_dir: "./data/labels"
test_path: "./data/test.csv"
dataset_ex_dir: "./data/100examples.csv"

load_model_file: "./YOLO_V3_from-sratch/overfit.pth.tar"

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

## Available Arguments

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

The implementation follows the original YOLOv3 architecture with:

- **Darknet-53 backbone**: 53 convolutional layers with residual (skip) connections
- **No fully connected layers**: the entire network is convolutional
- **Batch normalization** after each convolutional layer
- **Leaky ReLU** activations (negative slope 0.1)
- **FPN-style top-down path**: the deepest feature map is progressively upsampled and concatenated with earlier, higher-resolution backbone features before each subsequent, finer-grained prediction
- **Anchor-based, multi-scale detection**: 3 anchors per cell, at each of 3 grid resolutions (9 anchors total)

### Darknet-53 Backbone

Defined in `model.py`, with residual blocks (`ResidualBlock`) built from a 1×1 conv that halves the channel count followed by a 3×3 conv that restores it, wrapped in a skip connection:

```python
class Darknet53(nn.Module):
    # conv 32 -> conv 64 (s2) -> 1x ResidualBlock(64)
    # -> conv 128 (s2) -> 2x ResidualBlock(128)
    # -> conv 256 (s2) -> 8x ResidualBlock(256)   --> route1 (256ch, 52x52)
    # -> conv 512 (s2) -> 8x ResidualBlock(512)   --> route2 (512ch, 26x26)
    # -> conv 1024 (s2) -> 4x ResidualBlock(1024) --> route3 (1024ch, 13x13)
    ...
```

The three routed feature maps (`route1`, `route2`, `route3`) feed the multi-scale detection head below.

### Multi-Scale Detection Head

For a 416×416 input:

1. **Scale 1 (13×13, large objects):** a 5-conv block reduces `route3` (1024ch) to 512ch, which both (a) predicts directly and (b) is reduced to 256ch, upsampled 2×, and concatenated with `route2` (512ch) → 768ch.
2. **Scale 2 (26×26, medium objects):** a 5-conv block reduces the 768ch input to 256ch, which both (a) predicts directly and (b) is reduced to 128ch, upsampled 2×, and concatenated with `route1` (256ch) → 384ch.
3. **Scale 3 (52×52, small objects):** a 5-conv block reduces the 384ch input to 128ch and predicts directly.

This is what gives YOLOv3 its characteristic strength over YOLOv2 on small objects: the finest-resolution prediction has access to features derived from *all* three backbone depths, not just the shallowest one.

### Anchor Boxes

Default anchor priors, in **pixels of a 416×416 input** (the standard COCO YOLOv3 anchors — re-run k-means on your own dataset's box dimensions for best results, same idea as "Dimension Clusters" in the YOLOv2 paper):

```python
COCO_ANCHORS = [
    [(116, 90), (156, 198), (373, 326)],  # scale 1: 13x13 grid -- large objects
    [(30, 61), (62, 45), (59, 119)],       # scale 2: 26x26 grid -- medium objects
    [(10, 13), (16, 30), (33, 23)],        # scale 3: 52x52 grid -- small objects
]
```

Each ground-truth box is matched to whichever of the 9 total anchors (across all 3 scales) has the closest shape, exactly like YOLOv2's clustering-based assignment — just extended across scales. See `utils.build_targets_v3`.

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

The custom YOLOv3 loss function (`loss_f.py`) sums a per-scale loss across all 3 prediction scales:

1. **Bounding box coordinate loss**, same direct-location parametrization as YOLOv2:
   - `b_x = sigmoid(t_x) + c_x`
   - `b_y = sigmoid(t_y) + c_y`
   - `b_w = p_w * exp(t_w)`
   - `b_h = p_h * exp(t_h)`

2. **Objectness loss** — binary cross-entropy on a logistic (sigmoid) output, rather than YOLOv2's MSE-toward-IoU regression.

3. **No-object loss** — binary cross-entropy for cells/anchors without an assigned object, weighted by **λ_noobj = 0.5**.

4. **Class loss** — **independent per-class binary cross-entropy** (multi-label logistic classifiers) instead of a single softmax, since classes are no longer assumed mutually exclusive.

The loss still prioritizes bounding box coordinates with **λ_coord = 5**.

**Key differences from YOLOv2**: (a) loss is computed and summed over 3 scales instead of 1; (b) anchor *and scale* assignment for each ground-truth box happens once, ahead of time, while building the target tensors (`utils.build_targets_v3`) rather than being re-derived inside the loss; (c) objectness and class predictions are trained with binary cross-entropy rather than softmax/MSE, enabling multi-label detection.

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