import argparse
import yaml
from tqdm import tqdm

from loss_f import YOLO_LOSS
from model import YOLO_V1, CNN_Block
from dataset import VOCDataset

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT

from torch.utils.data import DataLoader

from utils import load_checkpoint, get_bboxes, mean_average_precision, train_fn


# Default Hyperparameters 
DEFAULT_CONFIG_PATH = "./YOLO_V1_from_sratch/configs/train_config.yaml"


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
    # Create parser
    parser = argparse.ArgumentParser(description="YOLO V1 Training Script")
    
    # Command line arguments
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

    # Parse command line arguments
    args = parser.parse_args()
    
    # Load YAML config if it exists
    yaml_config = load_yaml_config(args.config) if args.config else {}
    
    # Create config dictionary with priority: CLI > YAML > Defaults
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
        'test_path': yaml_config.get('test_path'),
        'dataset_ex_dir': yaml_config.get('dataset_ex_dir'),
        'weight_decay': yaml_config.get('weight_decay'),
        'pin_memory': yaml_config.get('pin_memory'),
    }
    
    # Convert to SimpleNamespace for attribute-style access
    return argparse.Namespace(**config)


class compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        return img, bboxes


transform = compose([transforms.Resize((448, 448)), transforms.ToTensor()])

    
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
         pin_memory: bool):
    """
    Main training function with configurable parameters
    """
    # device config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    torch.manual_seed(seed)

    model = YOLO_V1(Split_size=7, num_boxes=2, num_classes=20).to(device)
    
    optimizer = optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    
    loss_fn = YOLO_LOSS()
    
    if load_model:
        load_checkpoint(torch.load(load_model_file), model, optimizer)
        print(f"Loaded model from {load_model_file}")

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
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=n_works,
        pin_memory=pin_memory,
        shuffle=True,
        drop_last=True,
    )   
    
    print(f"Training Configuration:")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch Size: {batch_size}")
    print(f"  - Learning Rate: {lr}")
    print(f"  - Device: {device}")
    print(f"  - Seed: {seed}")
    print("-" * 50)
    
    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Get predictions and targets for mAP calculation
        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"Train mAP: {mean_avg_prec:.4f}")

        # Train for one epoch
        train_fn(
            train_loader=train_loader, 
            model=model, 
            optimizer=optimizer, 
            loss_fn=loss_fn, 
            device=device
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
    )