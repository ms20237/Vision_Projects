# Vision_Projects
## About
This repository contains various computer vision projects implemented in Python. The projects focus on deep learning, object detection, and image processing, including implementations of models from scratch.
## Projects
- **YOLO_V1_from_scratch**:  A complete implementation of YOLO v1 for object detection, built from scratch to demonstrate fundamental understanding of unified detection frameworks. Features include the original Darknet backbone architecture (24 conv layers + 2 FC layers), custom loss function with coordinate/objectness/class components, grid-based detection (7×7), and NMS post-processing. This foundational implementation establishes core concepts in real-time object detection.
- **YOLO_V2_from_scratch**: A complete implementation of YOLO v2 (YOLO9000) for object detection, built from scratch with Darknet-19 backbone, anchor boxes, multi-scale training, and passthrough layers. The implementation demonstrates advanced understanding of modern object detection architectures, including dimension clusters, direct location prediction, and fully convolutional design. Achieves state-of-the-art performance through techniques like batch normalization, multi-scale training (320-608px), and fine-grained feature detection.
- **YOLO_V3_from_scratch**: A complete implementation of YOLO v3 for object detection, built from scratch with a Darknet-53 residual backbone and an FPN-style, multi-scale detection head. The implementation predicts at three grid resolutions simultaneously (13×13, 26×26, 52×52) using nine anchor boxes (three per scale), giving substantially better small-object detection than earlier versions. Class and objectness predictions use independent binary cross-entropy (logistic) classifiers rather than softmax, enabling multi-label detection, alongside multi-scale training (320-608px) carried over from YOLO v2.
- **YOLO_V4_from_scratch**: A complete implementation of YOLO v4 for object detection, built from scratch with a CSPDarknet53 backbone (Cross-Stage-Partial residual stages, Mish activation), an SPP block for cheap receptive-field expansion, and a PANet neck that adds a bottom-up path on top of YOLO v3's top-down FPN so features flow both ways before detection. Box regression uses CIoU loss — a single similarity score combining overlap, center-distance, and aspect-ratio consistency — instead of separate coordinate MSE terms, while objectness and class predictions keep YOLO v3's binary cross-entropy formulation.
- **YOLO_V5_from_scratch**: A complete implementation of YOLO v5 (Ultralytics-style; unlike v1-v4, never formally published as a paper) for object detection, built from scratch with a `Conv`-stem CSPDarknet backbone, `C3` CSP blocks, and `SPPF` (a faster, sequential-pooling alternative to YOLO v4's SPP). SiLU activation is used throughout the backbone, neck, and head. The CIoU loss from YOLO v4 is retained, with per-scale objectness-loss balancing and optional label smoothing added on top. Also introduces configurable width/depth multipliers, reproducing the full n/s/m/l/x model family from one architecture description.
## Features
- Python-based implementations
- Focus on deep learning and computer vision
- Educational examples for understanding model architectures and pipelines
## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ms20237/Vision_Projects.git
   ```
2. Navigate into the project you want to run:
   ```bash
   cd Vision_Projects/YOLO_Vx_from_scratch
   ```
   (options: `YOLO_V1_from_scratch` / `YOLO_V2_from_scratch` / `YOLO_V3_from_scratch` / `YOLO_V4_from_scratch` / `YOLO_V5_from_scratch`)
3. Install the shared dependencies:
   ```bash
   pip install torch torchvision pandas pillow matplotlib numpy pyyaml tqdm
   ```
4. Refer to that project's own `README.md` for dataset preparation, configuration, and training instructions specific to that YOLO version.

## Repository Structure
```
Vision_Projects/
├── YOLO_V1_from_scratch/
│   ├── configs/
│   │   └── train_config.yaml
│   ├── loss_f.py
│   ├── main.py
│   ├── model.py
│   ├── utils.py
│   ├── dataset.py
│   └── README.md
├── YOLO_V2_from_scratch/
│   ├── configs/
│   │   └── train_config.yaml
│   ├── loss_f.py
│   ├── main.py
│   ├── model.py
│   ├── utils.py
│   ├── dataset.py
│   └── README.md
├── YOLO_V3_from_scratch/
│   ├── configs/
│   │   └── train_config.yaml
│   ├── loss_f.py
│   ├── main.py
│   ├── model.py
│   ├── utils.py
│   ├── dataset.py
│   └── README.md
├── YOLO_V4_from_scratch/
│   ├── configs/
│   │   └── train_config.yaml
│   ├── loss_f.py
│   ├── main.py
│   ├── model.py
│   ├── utils.py
│   ├── dataset.py
│   └── README.md
├── YOLO_V5_from_scratch/
│   ├── configs/
│   │   └── train_config.yaml
│   ├── loss_f.py
│   ├── main.py
│   ├── model.py
│   ├── utils.py
│   ├── dataset.py
│   └── README.md
└── README.md                      # this file
```

Each project folder is self-contained: its own `model.py` (architecture), `loss_f.py` (training loss), `utils.py` (IoU, NMS, mAP, checkpointing, and version-specific training helpers), `main.py` (training entry point), and `README.md` with full setup and usage details for that specific YOLO version. From `YOLO_V3_from_scratch` onward, the multi-scale target format, box decoding, NMS, and mAP logic in `utils.py` are shared unchanged across versions — only `model.py` and `loss_f.py` change from one version to the next.

## Roadmap
Planned/possible additions as this repository grows:
- Mosaic/MixUp data augmentation and AutoAnchor (k-means anchor fitting) for the YOLO v4/v5 implementations
- YOLO v6/v7/v8-style improvements (anchor-free heads, decoupled detection heads, task-aligned label assignment)
- Additional detection architectures beyond the YOLO family (e.g. Faster R-CNN, SSD)
- Shared dataset utilities across projects to reduce duplication

## Contributing
Issues and pull requests are welcome. If you spot a bug or want to propose an improvement to one of the from-scratch implementations, please open an issue describing the change before submitting a PR.

## License
This project is licensed under the [MIT License](https://choosealicense.com/licenses/mit/).

---

**Happy Detecting!** 🎯