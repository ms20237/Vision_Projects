# YOLO V2 from Scratch

A PyTorch implementation of the **YOLO (You Only Look Once) V2** object detection model built from the ground up, following the original paper ["YOLO9000: Better, Faster, Stronger"](https://arxiv.org/abs/1612.08242) by Joseph Redmon et al.

This repository provides a complete, modular implementation including data loading, model architecture (Darknet-19), custom loss function with anchor boxes, training pipeline with multi-scale training, and evaluation metrics.

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
  - [Darknet-19 Backbone](#darknet-19-backbone)
  - [Anchor Boxes](#anchor-boxes)
  - [Output Shape](#output-shape)
- [📉 Loss Function](#-loss-function)
- [📊 Evaluation Metrics](#-evaluation-metrics)
  - [mAP Calculation](#map-calculation)
- [📈 Results](#-results)
- [📝 License](#-license)

---

## ✨ Features

- **Complete YOLO V2 implementation** from scratch using PyTorch
- **Darknet-19 backbone** with 19 convolutional layers (fully convolutional)
- **Anchor box mechanism** with 5 VOC priors from k-means clustering
- **Multi-Scale Training** (320-608px) for robust performance across resolutions
- **Passthrough / Reorg layer** for fine-grained feature detection
- **Custom dataset loader** for PASCAL VOC format with label parsing
- **YOLO-specific loss function** with coordinate, objectness, and class components
- **Mean Average Precision (mAP)** evaluation metric
- **Non-Maximum Suppression (NMS)** for post-processing
- **Flexible configuration** via YAML files and command-line arguments
- **Checkpoint saving/loading** for resuming training
- **Visualization utilities** for bounding boxes

---

## 📁 Project Structure

```
YOLO_V2_from_sratch/
├── configs/
│   └── train_config.yaml          # Configuration file for training
├── data/                          # Dataset directory (created by user)
│   ├── create_dataset.py          # Dataset creation script
│   └── generate_csv.py            # Generate CSV of dataset ratio
├── dataset.py                     # VOC dataset loader with YOLOv2 target format
├── loss_f.py                      # YOLOv2 loss function implementation
├── main.py                        # Training script entry point
├── model.py                       # YOLO V2 Darknet-19 architecture
├── utils.py                       # Utilities (IoU, NMS, mAP, multi-scale, checkpointing)
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
cd Vision_Projects/YOLO_V2_from_sratch
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

---

## ⚙️ Configuration

Create a `configs/train_config.yaml` file with your settings:

```yaml
# train_config.yaml

train_config_path: "./YOLO_V2_from-sratch/configs/train_config.yaml"
img_dir: "./YOLO_V2_from-sratch/data/images"
label_dir: "./YOLO_V2_from-sratch/data/labels"
test_path: "./YOLO_V2_from-sratch/split_ratio/test.csv"
dataset_ex_dir: "./YOLO_V2_from-sratch/data/100examples.csv"

load_model_file: "./YOLO_V2_from-sratch/overfit.pth.tar"

lr: 0.00002
device: "cuda"
batch_size: 16
epochs: 10
n_works: 2
seed: 123
load_model: false
weight_decay: 0.0
pin_memory: true
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
    --load_model \
    --load_model_file checkpoint.pth.tar
```

## Available Arguments

| Argument            | Description                    | Default                     |
| ------------------- | ------------------------------ | --------------------------- |
| `--config`          | Path to YAML config file       | `configs/train_config.yaml` |
| `--img_dir`         | Image directory path           | From config                 |
| `--label_dir`       | Label directory path           | From config                 |
| `--lr`              | Learning rate                  | From config                 |
| `--batch_size`      | Batch size                     | From config                 |
| `--epochs`          | Number of training epochs      | From config                 |
| `--device`          | Device (`cuda`/`cpu`)          | Auto-detect                 |
| `--n_works`         | Number of data loading workers | From config                 |
| `--seed`            | Random seed                    | From config                 |
| `--load_model`      | Load checkpoint                | `False`                     |
| `--load_model_file` | Checkpoint file path           | `None`                      |

---

## 🧠 Model Architecture

The implementation follows the original YOLO V2 architecture with:

- **Darknet-19 backbone**: 19 convolutional layers (fully convolutional)
- **No fully connected layers**: The entire network is convolutional
- **Batch normalization** after each convolutional layer
- **Leaky ReLU** activations (negative slope 0.1)
- **Passthrough / Reorg layer**: Concatenates 26x26x512 feature maps with 13x13x1024 for fine-grained features
- **Anchor-based detection**: 5 anchors per cell from k-means clustering

### Darknet-19 Backbone

The model architecture is defined in `model.py` using two stages:

**Stage 1 (26x26x512 feature map):**
```python
DARKNET19_STAGE1 = [
    (3, 32, 1, 1), "M",
    (3, 64, 1, 1), "M",
    (3, 128, 1, 1), (1, 64, 1, 0), (3, 128, 1, 1), "M",
    (3, 256, 1, 1), (1, 128, 1, 0), (3, 256, 1, 1), "M",
    (3, 512, 1, 1), (1, 256, 1, 0), (3, 512, 1, 1), (1, 256, 1, 0), (3, 512, 1, 1),
]
```

**Stage 2 (13x13x1024 feature map):**
```python
DARKNET19_STAGE2 = [
    "M",
    (3, 1024, 1, 1), (1, 512, 1, 0), (3, 1024, 1, 1), (1, 512, 1, 0), (3, 1024, 1, 1),
]
```

### Anchor Boxes

Default VOC anchor priors (in 13x13 grid units):
```python
VOC_ANCHORS = [
    (1.08, 1.19),
    (3.42, 4.41),
    (6.63, 11.38),
    (9.42, 5.11),
    (16.62, 10.52),
]
```

### Output Shape

- Input: `(batch_size, 3, 416, 416)` (default)
- Output: `(batch_size, 13, 13, num_anchors × (5 + C))` where:
  - `num_anchors = 5`
  - `C = 20` (number of classes for VOC)
  - Output shape: `(batch_size, 13, 13, 125)`

**Multi-Scale Training**: Input size can vary from 320 to 608 pixels (multiples of 32), resulting in output grids from 10×10 to 19×19.

---

## 📉 Loss Function

The custom YOLOv2 loss function (`loss_f.py`) implements the anchor-based loss with the following components:

1. **Bounding box coordinate loss** with direct location prediction:
   - `b_x = sigmoid(t_x) + c_x`
   - `b_y = sigmoid(t_y) + c_y`
   - `b_w = p_w * exp(t_w)`
   - `b_h = p_h * exp(t_h)`

2. **Objectness loss** (for anchors with objects)

3. **No-object loss** (for anchors without objects, with λ_noobj = 0.5)

4. **Class loss** (classification error)

The loss prioritizes bounding box coordinates with **λ_coord = 5** and penalizes false positives with **λ_noobj = 0.5**.

**Key difference from YOLOv1**: Instead of YOLOv1's "winner takes all" approach, YOLOv2 assigns each ground-truth box to the anchor with the highest IoU, following the "Dimension Clusters" and "Direct location prediction" approach from the paper.

---

## 📊 Evaluation Metrics

The implementation includes:

- **Intersection over Union (IoU)** calculation
- **Non-Maximum Suppression (NMS)** for removing duplicate detections
- **Mean Average Precision (mAP)** at IoU threshold 0.5

### mAP Calculation

During training, mAP is computed on the validation set to monitor performance:

```python
mean_avg_prec = mean_average_precision(
    pred_boxes, target_boxes, 
    iou_threshold=0.5, 
    box_format="midpoint"
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
