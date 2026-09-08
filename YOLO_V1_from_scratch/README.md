# YOLO V1 from Scratch

A PyTorch implementation of the **YOLO (You Only Look Once) V1** object detection model built from the ground up, following the original paper ["You Only Look Once: Unified, Real-Time Object Detection"](https://arxiv.org/abs/1506.02640) by Joseph Redmon et al.

This repository provides a complete, modular implementation including data loading, model architecture, custom loss function, training pipeline, and evaluation metrics.

---

## 📋 Table of Contents

* [✨ Features](#-features)
* [📁 Project Structure](#-project-structure)
* [🚀 Installation](#-installation)

  * [Prerequisites](#prerequisites)
  * [Setup](#setup)
* [📊 Dataset Preparation](#-dataset-preparation)
* [⚙️ Configuration](#️-configuration)
* [🏃 Usage](#-usage)

  * [Training](#training)
  * [Command Line Arguments](#command-line-arguments)
  * [Available Arguments](#available-arguments)
* [🧠 Model Architecture](#-model-architecture)

  * [Architecture Configuration](#architecture-configuration)
  * [Output Shape](#output-shape)
* [📉 Loss Function](#-loss-function)
* [📊 Evaluation Metrics](#-evaluation-metrics)

  * [mAP Calculation](#map-calculation)
* [📈 Results](#-results)
* [📝 License](#-license)

---

## ✨ Features

- **Complete YOLO V1 implementation** from scratch using PyTorch
- **Custom dataset loader** for PASCAL VOC format with label parsing
- **YOLO-specific loss function** with bounding box, objectness, and class components
- **Modular architecture** following the original Darknet backbone
- **Mean Average Precision (mAP)** evaluation metric
- **Non-Maximum Suppression (NMS)** for post-processing
- **Flexible configuration** via YAML files and command-line arguments
- **Checkpoint saving/loading** for resuming training
- **Visualization utilities** for bounding boxes

---

## 📁 Project Structure

```
YOLO_V1_from_sratch/
├── configs/
│   └── train_config.yaml          # Configuration file for training
├── loss_f.py                      # YOLO loss function implementation
├── main.py                        # Training script entry point
├── model.py                       # YOLO V1 model architecture
├── utils.py                       # Utilities (IoU, NMS, mAP, checkpointing)
└── daatset.py                    # dataset class for VOCDataset 
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
cd Vision_Projects/YOLO_V1_from_sratch
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

train_config_path: "./YOLO_V1_from-sratch/configs/train_config.yaml"
img_dir: "./data/images"
label_dir: "./data/labels"
test_path: "./data/test.csv"
dataset_ex_dir: "./data/100examples.csv"

load_model_file: "./YOLO_V1_from-sratch/models/overfit.pth.tar"

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
    --load_model True \
    --load_model_file checkpoint.pth.tar
```

## **Available Arguments**

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


## 🧠 Model Architecture

The implementation follows the original YOLO V1 architecture with:

- **24 convolutional layers** from the Darknet backbone
- **2 fully connected layers** for final predictions
- **Leaky ReLU** activations (negative slope 0.1)
- **Batch normalization** after each convolutional layer

### Architecture Configuration

The model is defined using a configuration list in `model.py`:

```python
architecture_config = [
    (7, 64, 2, 3),          # Conv: kernel_size, filters, stride, padding
    "M",                    # MaxPool
    (3, 192, 1, 1),
    "M",
    # ... and so on
]
```

### Output Shape

- Input: `(batch_size, 3, 448, 448)`
- Output: `(batch_size, S × S × (C + B × 5))` where:
  - `S = 7` (grid size)
  - `B = 2` (bounding boxes per cell)
  - `C = 20` (number of classes for VOC)

---

## 📉 Loss Function

The custom YOLO loss function (`loss_f.py`) implements the five components from the paper:

1. **Bounding box coordinate loss** (with sqrt of width/height)
2. **Objectness loss** (for boxes with objects)
3. **No-object loss** (for boxes without objects, with λ_noobj = 0.5)
4. **Class loss** (classification error)

The loss prioritizes bounding box coordinates with **λ_coord = 5** and penalizes false positives with **λ_noobj = 0.5**.

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


**Happy Detecting!** 🎯
