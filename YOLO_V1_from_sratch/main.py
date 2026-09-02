import argparse 
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


# Hyperparameters 
DATASET_EX_DIR = "./YOLO_V1_from_sratch/data/100examples.csv"
IMG_DIR = "./YOLO_V1_from_sratch/data/images" 
LABEL_DIR = "./YOLO_V1_from_sratch/data/labels"

TEST_PATH = "./YOLO_V1_from_sratch/split_ratio/test.csv"
LOAD_MODEL_FILE = "./YOLO_V1_from_sratch/overfit.pth.tar"

LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16                                             # 64 in original paper 
WEIGHT_DECAY = 0                                            # 0.0005 in original paper but I don't need that much(Weight decay adds a penalty term,
                                                            # improve the model's generalization performance by avoiding overfitting)
EPOCHS = 10
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
SEED = 123


def init():
    """
    """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--img_dir',
                        type=str,
                        default=IMG_DIR,
                        help="image path of train dataset")
    
    parser.add_argument('--label_dir',
                        type=str,
                        default=LABEL_DIR,
                        help="label path of train dataset")
    
    parser.add_argument('--lr',
                        type=float,
                        default=LEARNING_RATE,
                        help="learning rate of train")

    parser.add_argument('--device',
                        type=str,
                        default=DEVICE,
                        help="device of train")

    parser.add_argument('--batch_size',
                        type=int,
                        default=BATCH_SIZE,
                        help="batch size of train")

    parser.add_argument('--epochs',
                        type=int,
                        default=EPOCHS,
                        help="epochs number of train")
    
    parser.add_argument('--n_works',
                        type=int,
                        default=NUM_WORKERS,
                        help="number of works for train")

    args = parser.parse_args()
    return args


class compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        return img, bboxes
transform = compose([transforms.Resize((448, 448)), transforms.ToTensor()])

    
def main(img_dir: str = IMG_DIR,
         label_dir: str = LABEL_DIR,
         lr: float = LEARNING_RATE,
         device: str = DEVICE,
         batch_size: int = BATCH_SIZE,
         epochs: int = EPOCHS,
         n_works: int = NUM_WORKERS,
         seed: int = SEED,


         ):
    # device config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    torch.manual_seed(seed)

    model = YOLO_V1(Split_size=7, num_boxes=2, num_classes=20).to(device)
    
    optimizer = optim.Adam(
        model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    
    loss_fn = YOLO_LOSS()
    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    train_dataset = VOCDataset(
        DATASET_EX_DIR,
        transform=transform,
        img_dir=img_dir,
        label_dir=label_dir
    )
    test_dataset = VOCDataset(
        TEST_PATH,
        transform=transform,
        img_dir=img_dir,
        label_dir=label_dir
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=n_works,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=n_works,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )   
    
    # train loop
    # for epoch in range(EPOCHS):
    #     for x, y in train_loader:
    #         x = x.to(DEVICE)
    #         for idx in range(2):
    #             bboxes = cellboxes_to_boxes(model(x))
    #             bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
    #             plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes)

    #         import sys
    #         sys.exit()
            
    #         pred_boxes, target_boxes, = get_bboxes(
    #             train_loader, model, iou_threshold=0.5, threshold=0.4
    #         )
    #         mean_avg_prec = mean_average_precision(
    #             pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
    #         )
    #         print(f"Train mAP: {mean_avg_prec}")
    #         if mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"):
    #             checkpoint = {
    #                 "state_dict": model.state_dict(),
    #                 "optimizer": optimizer.state_dict(),
    #             }
    #             save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
    #             import time
    #             time.sleep(10)

    #         train_fn(train_loader, model, optimizer, loss_fn)

    for epoch in range(EPOCHS):
    #   for x, y in train_loader:
    #     x = x.to(DEVICE)
    #     for idx in range(8):
    #         bboxes = cellboxes_to_boxes(model(x))
    #         bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
    #         plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes)

    #     import sys
    #     sys.exit()

        pred_boxes, target_boxes = get_bboxes(train_loader, model, iou_threshold=0.5, threshold=0.4)

        mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint")
        print(f"Train mAP: {mean_avg_prec}")

        #if mean_avg_prec > 0.9:
        #    checkpoint = {
        #        "state_dict": model.state_dict(),
        #        "optimizer": optimizer.state_dict(),
        #    }
        #    save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
        #    import time
        #    time.sleep(10)

        train_fn(train_loader=train_loader, 
                 model=model, 
                 optimizer=optimizer, 
                 loss_fn=loss_fn, 
                 device=device)


if __name__ == "__main__":
    args = init()
    main(args.img_dir,
         args.label_dir,
         args.lr,
         args.device,
         args.batch_size,
         args.epochs,
         args.n_works)


