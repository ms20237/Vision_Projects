import torch
import torchvision

from Loss_F import *
from utils import *
from Model import *
from dataset import *

import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT

from tqdm import tqdm
from torch.utils.data import DataLoader


# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Model
model = YOLO_V1()


seed = 123
torch.manual_seed(seed)

# Hyperparameters etc.
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16 # 64 in original paper but I don't have that much v
WEIGHT_DECAY = 0 # 0.0005 in original paper but I don't need that much(Weight decay adds a penalty term,
                                # improve the model's generalization performance by avoiding overfitting)
EPOCHS = 10
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "overfit.pth.tar"
IMG_DIR = r"data/images" 
LABEL_DIR = r"data/labels"


class compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        return img, bboxes
transform = compose([transforms.Resize((448, 448)), transforms.ToTensor()])


def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # update progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")
    
def main():
    model = YOLO_V1(Split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YOLO_LOSS()
    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    train_dataset = VOCDataset(
        r"data/100examples.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR
    )
    test_dataset = VOCDataset(
        r"test.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
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

        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"Train mAP: {mean_avg_prec}")

        #if mean_avg_prec > 0.9:
        #    checkpoint = {
        #        "state_dict": model.state_dict(),
        #        "optimizer": optimizer.state_dict(),
        #    }
        #    save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
        #    import time
        #    time.sleep(10)

        train_fn(train_loader, model, optimizer, loss_fn)
if __name__ == "__main__":
    main()


