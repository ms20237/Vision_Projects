# Vision_Projects
## About
This repository contains various computer vision projects implemented in Python. The projects focus on deep learning, object detection, and image processing, including implementations of models from scratch.
## Projects
- **YOLO_V1_from_scratch**:  A complete implementation of YOLO v1 for object detection, built from scratch to demonstrate fundamental understanding of unified detection frameworks. Features include the original Darknet backbone architecture (24 conv layers + 2 FC layers), custom loss function with coordinate/objectness/class components, grid-based detection (7×7), and NMS post-processing. This foundational implementation establishes core concepts in real-time object detection.
- **YOLO_V2_from_scratch**: A complete implementation of YOLO v2 (YOLO9000) for object detection, built from scratch with Darknet-19 backbone, anchor boxes, multi-scale training, and passthrough layers. The implementation demonstrates advanced understanding of modern object detection architectures, including dimension clusters, direct location prediction, and fully convolutional design. Achieves state-of-the-art performance through techniques like batch normalization, multi-scale training (320-608px), and fine-grained feature detection.
- **YOLO_V3_from_scratch**: A complete implementation of YOLO v3 for object detection, built from scratch with a Darknet-53 residual backbone and an FPN-style, multi-scale detection head. The implementation predicts at three grid resolutions simultaneously (13×13, 26×26, 52×52) using nine anchor boxes (three per scale), giving substantially better small-object detection than earlier versions. Class and objectness predictions use independent binary cross-entropy (logistic) classifiers rather than softmax, enabling multi-label detection, alongside multi-scale training (320-608px) carried over from YOLO v2.
## Features
- Python-based implementations
- Focus on deep learning and computer vision
- Educational examples for understanding model architectures and pipelines
## Installation
1. Clone the repository:
```bash
   git clone https://github.com/ms20237/Vision_Projects.git
```