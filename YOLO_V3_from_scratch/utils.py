import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from collections import Counter
from tqdm import tqdm


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union. Unchanged from YOLOv1/v2 -- IoU
    math doesn't depend on anchors, grid size, or number of scales.
    """
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Does Non Max Suppression given bboxes. Unchanged from YOLOv1/v2 -- it
    already operates on a flat [class, score, x, y, w, h] list, which is
    exactly what cellboxes_to_boxes_v3 produces once it's pooled boxes
    from all 3 scales together.
    """
    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20
):
    """
    Calculates mean average precision. Unchanged from YOLOv1/v2 -- operates
    on flat box lists, independent of grid size, anchor count, or scales.
    """
    average_precisions = []
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)


def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image. Unchanged from YOLOv1/v2."""
    im = np.array(image)
    height, width, _ = im.shape

    fig, ax = plt.subplots(1)
    ax.imshow(im)

    for box in boxes:
        box = box[2:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)

    plt.show()


# --------------------------------------------------------------------- #
#  YOLOv3-specific: multi-scale box decoding & target building          #
# --------------------------------------------------------------------- #
#
# YOLOv2's decoders (convert_cellboxes_v2 / cellboxes_to_boxes_v2) assumed
# a single S x S grid with a single anchor set. YOLOv3 predicts at 3 grids
# simultaneously with different anchors per grid, so decoding needs to
# happen per-scale and then be pooled together before NMS -- a detection
# from the coarse 13x13 grid competes on equal footing with one from the
# fine 52x52 grid once both are expressed in whole-image-relative units.

def decode_scale_predictions_v3(pred, anchors_px, base_size, C=20):
    """
    Decode one scale's raw network output into whole-image-relative boxes.

    pred: (B, S, S, A*(5+C)) raw logits for this scale
    anchors_px: list of (w, h) anchor priors for this scale, in PIXELS of
                a `base_size` x `base_size` network input (e.g. 416)
    base_size: the input resolution those anchor pixel values were defined
               at (anchors get rescaled to this scale's grid via its
               stride = base_size / S)
    returns: (B, S*S*A, 6) tensor of [class, score, x, y, w, h], all box
             coordinates in [0, 1] of the whole image
    """
    device = pred.device
    B_size, S, _, _ = pred.shape
    A = len(anchors_px)
    stride = base_size / S
    anchors_grid = torch.tensor(
        [(w / stride, h / stride) for (w, h) in anchors_px], dtype=torch.float32, device=device
    )  # (A, 2), in this scale's grid units

    pred = pred.reshape(B_size, S, S, A, 5 + C)
    tx, ty, tw, th, to = pred[..., 0], pred[..., 1], pred[..., 2], pred[..., 3], pred[..., 4]
    cls_logits = pred[..., 5:]

    cell_indices = torch.arange(S, device=device).repeat(B_size, S, 1).unsqueeze(-1)  # (B,S,S,1)
    cell_x = cell_indices.unsqueeze(3)                      # (B,S,S,1,1), varies along columns
    cell_y = cell_indices.permute(0, 2, 1, 3).unsqueeze(3)  # (B,S,S,1,1), varies along rows

    bx = (torch.sigmoid(tx).unsqueeze(-1) + cell_x) / S
    by = (torch.sigmoid(ty).unsqueeze(-1) + cell_y) / S
    bw = (torch.exp(tw).unsqueeze(-1) * anchors_grid[:, 0].view(1, 1, 1, A, 1)) / S
    bh = (torch.exp(th).unsqueeze(-1) * anchors_grid[:, 1].view(1, 1, 1, A, 1)) / S

    conf = torch.sigmoid(to).unsqueeze(-1)
    cls_prob = torch.sigmoid(cls_logits)  # independent logistic classifiers, not softmax
    best_cls_prob, best_cls = cls_prob.max(dim=-1, keepdim=True)
    score = conf * best_cls_prob

    boxes = torch.cat([best_cls.float(), score, bx, by, bw, bh], dim=-1)  # (B,S,S,A,6)
    return boxes.reshape(B_size, S * S * A, 6)


def cellboxes_to_boxes_v3(predictions, anchors, base_size=416, C=20):
    """
    List-of-lists version of decode_scale_predictions_v3, pooling boxes
    from all 3 scales together per image.

    predictions: list of 3 tensors, one per scale (model output)
    anchors: list of 3 lists of (w, h) anchor priors in pixels of base_size
    """
    all_scale_boxes = [
        decode_scale_predictions_v3(pred, anch, base_size, C=C)
        for pred, anch in zip(predictions, anchors)
    ]
    combined = torch.cat(all_scale_boxes, dim=1)  # (B, sum(S*S*A) across scales, 6)

    all_bboxes = []
    for ex_idx in range(combined.shape[0]):
        bboxes = [[x.item() for x in combined[ex_idx, i, :]] for i in range(combined.shape[1])]
        all_bboxes.append(bboxes)
    return all_bboxes


def build_targets_v3(boxes_per_image, anchors, grid_sizes, base_size=416, C=20, device="cpu"):
    """
    Build the 3 per-scale target tensors YOLO_V3_LOSS expects, from raw
    ground-truth boxes. This is the reference implementation dataset.py
    should mirror (or call directly from a custom collate_fn).

    boxes_per_image: list, len == batch size, of lists of
                      (class_idx, x, y, w, h) ground-truth boxes in
                      whole-image-relative [0, 1] units.
    anchors: list of 3 lists of 3 (w, h) anchor priors, in pixels of
             base_size (e.g. model.COCO_ANCHORS).
    grid_sizes: list of 3 ints, the S for each scale, e.g. [13, 26, 52]
                for base_size=416 (must satisfy S == base_size / stride).

    Each ground-truth box is assigned to whichever of the 9 total anchors
    (across all 3 scales) best matches its (w, h) shape -- comparing shapes
    in a common unit (pixels of base_size) exactly like YOLOv2's
    "Dimension Clusters" matching, just extended across scales.

    Returns: list of 3 tensors, each (B, S, S, 3, 5 + C).
    """
    B_size = len(boxes_per_image)
    A = len(anchors[0])
    targets = [torch.zeros(B_size, S, S, A, 5 + C, device=device) for S in grid_sizes]

    all_anchors_px = []
    anchor_owner = []  # (scale_idx, anchor_idx) for each of the 9 anchors
    for s_idx, scale_anchors in enumerate(anchors):
        for a_idx, (aw, ah) in enumerate(scale_anchors):
            all_anchors_px.append((aw, ah))
            anchor_owner.append((s_idx, a_idx))
    all_anchors_px_t = torch.tensor(all_anchors_px, dtype=torch.float32)  # (9, 2)

    for b, boxes in enumerate(boxes_per_image):
        for (cls_idx, x, y, w, h) in boxes:
            gt_w_px = w * base_size
            gt_h_px = h * base_size

            inter = torch.min(all_anchors_px_t[:, 0], torch.tensor(gt_w_px)) * \
                torch.min(all_anchors_px_t[:, 1], torch.tensor(gt_h_px))
            union = all_anchors_px_t[:, 0] * all_anchors_px_t[:, 1] + gt_w_px * gt_h_px - inter
            ious = inter / (union + 1e-6)
            best = ious.argmax().item()
            s_idx, a_idx = anchor_owner[best]

            S = grid_sizes[s_idx]
            stride = base_size / S
            anchor_w, anchor_h = anchors[s_idx][a_idx]
            anchor_w_grid, anchor_h_grid = anchor_w / stride, anchor_h / stride

            col = min(int(x * S), S - 1)
            row = min(int(y * S), S - 1)

            tx = x * S - col
            ty = y * S - row
            gt_w_grid = w * S
            gt_h_grid = h * S
            tw = float(torch.log(torch.clamp(torch.tensor(gt_w_grid / anchor_w_grid), min=1e-6)))
            th = float(torch.log(torch.clamp(torch.tensor(gt_h_grid / anchor_h_grid), min=1e-6)))

            targets[s_idx][b, row, col, a_idx, cls_idx] = 1.0
            targets[s_idx][b, row, col, a_idx, C] = 1.0
            targets[s_idx][b, row, col, a_idx, C + 1] = tx
            targets[s_idx][b, row, col, a_idx, C + 2] = ty
            targets[s_idx][b, row, col, a_idx, C + 3] = tw
            targets[s_idx][b, row, col, a_idx, C + 4] = th

    return targets


def yolov3_collate_fn(batch):
    """
    Custom collate_fn for a YOLOv3 dataset. Unlike YOLOv1/v2, targets are
    NOT pre-gridded into a fixed-size tensor by the Dataset -- they're kept
    as a plain per-image list of (class_idx, x, y, w, h) boxes, since the
    right grid to bin them into depends on which scale/anchor wins the
    match (and, under multi-scale training, on the input size chosen for
    that batch). Binning happens in build_targets_v3 at train/eval time.
    """
    images = torch.stack([item[0] for item in batch], dim=0)
    boxes = [item[1] for item in batch]  # list (len=batch) of lists of tuples
    return images, boxes


def get_bboxes(
    loader,
    model,
    iou_threshold,
    threshold,
    anchors,
    base_size=416,
    box_format="midpoint",
    device="cuda",
    C=20,
):
    """
    Same role as the YOLOv1/v2 version: run the model over a loader and
    collect NMS'd predicted boxes + ground-truth boxes for mAP. Updated for
    YOLOv3's list-of-3-scales predictions and raw-box-list targets (see
    yolov3_collate_fn) instead of a single pre-gridded target tensor.
    """
    all_pred_boxes = []
    all_true_boxes = []

    model.eval()
    train_idx = 0

    for batch_idx, (x, boxes) in enumerate(loader):
        x = x.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        pred_boxes = cellboxes_to_boxes_v3(predictions, anchors, base_size=base_size, C=C)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                pred_boxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for (cls_idx, bx, by, bw, bh) in boxes[idx]:
                all_true_boxes.append([train_idx, cls_idx, 1.0, bx, by, bw, bh])

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def train_fn(
    train_loader,
    model,
    optimizer,
    loss_fn,
    device,
    anchors,
    base_size=416,
    multi_scale=True,
    C=20,
):
    """
    One training epoch. Since targets are built on the fly from raw box
    lists (build_targets_v3) rather than pre-gridded by the Dataset,
    "Multi-Scale Training" here is simpler than in the YOLOv2 version: no
    re-gridding of an existing tensor is needed -- just resize the images
    and rebuild targets at the grid sizes implied by the new input size
    (S = size / stride for each of the 3 strides {32, 16, 8}).
    """
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    input_sizes = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
    current_size = base_size
    global_batch_idx = 0

    for batch_idx, (x, boxes) in enumerate(loop):
        if multi_scale and global_batch_idx % 10 == 0:
            current_size = int(np.random.choice(input_sizes))

        x = x.to(device)
        if x.shape[-1] != current_size:
            x = F.interpolate(x, size=(current_size, current_size), mode="bilinear", align_corners=False)

        grid_sizes = [current_size // 32, current_size // 16, current_size // 8]
        targets = build_targets_v3(boxes, anchors, grid_sizes, base_size=base_size, C=C, device=device)

        out = model(x)
        loss = loss_fn(out, targets)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        global_batch_idx += 1
        loop.set_postfix(loss=loss.item(), img_size=current_size)

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")