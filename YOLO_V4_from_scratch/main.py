import argparse
import yaml

from loss_f import YOLO_V4_LOSS
from model import YOLO_V4, COCO_ANCHORS
from dataset import VOCDataset

import torch
import torchvision.transforms as transforms
import torch.optim as optim

from torch.utils.data import DataLoader

from utils import load_checkpoint, get_bboxes, mean_average_precision, train_fn, yolo_multiscale_collate_fn


# Default Hyperparameters
DEFAULT_CONFIG_PATH = "./YOLO_V4_from_scratch/configs/train_config.yaml"

# 416 is the standard YOLOv4 base resolution; the 3 output grids
# (13x13, 26x26, 52x52) and the pixel-unit anchors in COCO_ANCHORS are
# both defined relative to it. Multi-scale training (see utils.train_fn)
# resizes away from this at random every few batches, but this is the
# reference size used to build fixed-size evaluation targets/anchors.
BASE_INPUT_SIZE = 416


def load_yaml_config(config_path):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Warning: Config file {config_path} not found. Using defaults.")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return {}


def init():
    """
    Initialize arguments with priority:
    1. Command line arguments (highest priority)
    2. YAML config file
    3. Default values (lowest priority)
    """
    parser = argparse.ArgumentParser(description="YOLO V4 Training Script")

    parser.add_argument('--config',
                        type=str,
                        default=DEFAULT_CONFIG_PATH,
                        help="Path to YAML config file")

    parser.add_argument('--train_config_path',
                        type=str,
                        default=None,
                        help="config path of train")

    parser.add_argument('--img_dir',
                        type=str,
                        default=None,
                        help="image path of train dataset")

    parser.add_argument('--label_dir',
                        type=str,
                        default=None,
                        help="label path of train dataset")

    parser.add_argument('--lr',
                        type=float,
                        default=None,
                        help="learning rate of train")

    parser.add_argument('--device',
                        type=str,
                        default=None,
                        help="device of train")

    parser.add_argument('--batch_size',
                        type=int,
                        default=None,
                        help="batch size of train")

    parser.add_argument('--epochs',
                        type=int,
                        default=None,
                        help="epochs number of train")

    parser.add_argument('--n_works',
                        type=int,
                        default=None,
                        help="number of works for train")

    parser.add_argument('--seed',
                        type=int,
                        default=None,
                        help="random seed")

    parser.add_argument('--load_model',
                        action='store_true',
                        default=None,
                        help="load model from checkpoint")

    parser.add_argument('--load_model_file',
                        type=str,
                        default=None,
                        help="path to model checkpoint file")

    parser.add_argument('--multi_scale',
                        action='store_true',
                        default=None,
                        help="enable multi-scale training (random input size every 10 batches)")

    args = parser.parse_args()

    yaml_config = load_yaml_config(args.config) if args.config else {}

    config = {
        'train_config_path': args.train_config_path if args.train_config_path is not None
                             else yaml_config.get('train_config_path'),
        'img_dir': args.img_dir if args.img_dir is not None
                   else yaml_config.get('img_dir'),
        'label_dir': args.label_dir if args.label_dir is not None
                     else yaml_config.get('label_dir'),
        'lr': args.lr if args.lr is not None
              else yaml_config.get('lr'),
        'device': args.device if args.device is not None
                  else yaml_config.get('device'),
        'batch_size': args.batch_size if args.batch_size is not None
                      else yaml_config.get('batch_size'),
        'epochs': args.epochs if args.epochs is not None
                  else yaml_config.get('epochs'),
        'n_works': args.n_works if args.n_works is not None
                   else yaml_config.get('n_works'),
        'seed': args.seed if args.seed is not None
                else yaml_config.get('seed'),
        'load_model': args.load_model if args.load_model is not None
                      else yaml_config.get('load_model'),
        'load_model_file': args.load_model_file if args.load_model_file is not None
                           else yaml_config.get('load_model_file'),
        'multi_scale': args.multi_scale if args.multi_scale is not None
                       else yaml_config.get('multi_scale', True),
        'test_path': yaml_config.get('test_path'),
        'dataset_ex_dir': yaml_config.get('dataset_ex_dir'),
        'weight_decay': yaml_config.get('weight_decay'),
        'pin_memory': yaml_config.get('pin_memory'),
    }

    return argparse.Namespace(**config)


# Plain resize to the base input size + tensor conversion. Multi-scale
# training then resizes further, on the GPU/CPU tensor, inside train_fn --
# this transform just needs to produce a consistent starting size.
class compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        return img, bboxes


transform = compose([transforms.Resize((BASE_INPUT_SIZE, BASE_INPUT_SIZE)), transforms.ToTensor()])


def main(img_dir: str,
         label_dir: str,
         lr: float,
         device: str,
         batch_size: int,
         epochs: int,
         n_works: int,
         seed: int,
         load_model: bool,
         load_model_file: str,
         test_path: str,
         dataset_ex_dir: str,
         weight_decay: float,
         pin_memory: bool,
         multi_scale: bool):
    """
    Main training function with configurable parameters
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    torch.manual_seed(seed)

    anchors = COCO_ANCHORS
    C = 20

    model = YOLO_V4(num_classes=C).to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_fn = YOLO_V4_LOSS(C=C, anchors=anchors, base_size=BASE_INPUT_SIZE)

    if load_model:
        load_checkpoint(torch.load(load_model_file), model, optimizer)
        print(f"Loaded model from {load_model_file}")

    # NOTE: VOCDataset (not included with the uploaded files) needs to
    # return (image_tensor, boxes) per item, where `boxes` is a plain list
    # of (class_idx, x, y, w, h) tuples in whole-image-relative [0, 1]
    # units -- NOT a pre-gridded target tensor. Binning into the 3
    # scale-specific grids happens on the fly via utils.build_targets_v3
    # (called from train_fn/get_bboxes), since with multi-scale training
    # the right grid size isn't fixed ahead of time. See loss_f.py's
    # module docstring for the exact target format this feeds into.
    train_dataset = VOCDataset(
        dataset_ex_dir,
        transform=transform,
        img_dir=img_dir,
        label_dir=label_dir
    )
    test_dataset = VOCDataset(
        test_path,
        transform=transform,
        img_dir=img_dir,
        label_dir=label_dir
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=n_works,
        pin_memory=pin_memory,
        shuffle=True,
        drop_last=True,
        collate_fn=yolo_multiscale_collate_fn,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=n_works,
        pin_memory=pin_memory,
        shuffle=True,
        drop_last=True,
        collate_fn=yolo_multiscale_collate_fn,
    )

    print(f"Training Configuration:")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch Size: {batch_size}")
    print(f"  - Learning Rate: {lr}")
    print(f"  - Device: {device}")
    print(f"  - Seed: {seed}")
    print(f"  - Multi-scale training: {multi_scale}")
    print("-" * 50)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4,
            anchors=anchors, base_size=BASE_INPUT_SIZE, device=device, C=C,
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=C
        )
        print(f"Train mAP: {mean_avg_prec:.4f}")

        train_fn(
            train_loader=train_loader,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            anchors=anchors,
            base_size=BASE_INPUT_SIZE,
            multi_scale=multi_scale,
            C=C,
        )


if __name__ == "__main__":
    args = init()
    main(
        img_dir=args.img_dir,
        label_dir=args.label_dir,
        lr=args.lr,
        device=args.device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        n_works=args.n_works,
        seed=args.seed,
        load_model=args.load_model,
        load_model_file=args.load_model_file,
        test_path=args.test_path,
        dataset_ex_dir=args.dataset_ex_dir,
        weight_decay=args.weight_decay,
        pin_memory=args.pin_memory,
        multi_scale=args.multi_scale,
    )