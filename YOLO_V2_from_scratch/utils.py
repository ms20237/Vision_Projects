import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from collections import Counter
from tqdm import tqdm


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union. Unchanged from YOLOv1 -- IoU math
    doesn't depend on anchors/grid size, only on (x, y, w, h) box values.

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
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
        box1_y2 = boxes_preds[..., 3:4]  # (N, 1)
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Does Non Max Suppression given bboxes. Unchanged from YOLOv1 -- this
    already operates on flat [class_pred, prob_score, x, y, w, h] lists
    regardless of how many anchors/cells produced them.

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU) 
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
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


def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20):
    """
    Calculates mean average precision. Unchanged from YOLOv1 -- this
    operates on flat box lists, independent of grid size or anchor count.

    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones 
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes

    Returns:
        float: mAP value across all classes given a specific IoU threshold 
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
    """Plots predicted bounding boxes on the image. Unchanged from YOLOv1."""
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
#  YOLOv2-specific box decoding                                         #
# --------------------------------------------------------------------- #
#
# YOLOv1's convert_cellboxes/cellboxes_to_boxes assumed a fixed 7x7 grid
# and exactly 2 raw-coordinate boxes per cell. YOLOv2 instead has:
#   - a grid size S that depends on the (possibly multi-scale) input size
#     (S = input_size / 32),
#   - `num_anchors` predictions per cell, each parameterized as
#     (t_x, t_y, t_w, t_h, t_o) offsets from an anchor prior rather than
#     raw coordinates -- see "Direct location prediction" in the paper.
#
# So decoding now needs S and the anchors, and there are two decoders:
# one for network *predictions* (which have an anchor dimension and are
# still in raw t_x/t_y/t_w/t_h/t_o form) and one for *targets* (single
# box per cell, already in offset/grid-unit form -- see loss_f.py's
# docstring for the exact target format).

def convert_cellboxes_v2(predictions, anchors, C=20):
    """
    Decode raw YOLOv2 network output into whole-image-relative box
    coordinates (x, y, w, h all in [0, 1] of the image), following:
        b_x = sigmoid(t_x) + c_x
        b_y = sigmoid(t_y) + c_y
        b_w = p_w * exp(t_w)
        b_h = p_h * exp(t_h)
    then dividing by S to go from grid units to image-relative units.

    predictions: (BATCH, S, S, num_anchors * (5 + C)) raw network output
    anchors: list of (w, h) tuples in grid-cell units
    returns: (BATCH, S*S*num_anchors, 6) tensor of
             [class_pred, confidence*class_prob, x, y, w, h]
    """
    device = predictions.device
    B_size, S, _, _ = predictions.shape
    A = len(anchors)
    anchors_t = torch.as_tensor(anchors, dtype=torch.float32, device=device)

    predictions = predictions.reshape(B_size, S, S, A, 5 + C)

    tx = predictions[..., 0]
    ty = predictions[..., 1]
    tw = predictions[..., 2]
    th = predictions[..., 3]
    to = predictions[..., 4]
    cls_logits = predictions[..., 5:]

    # Same row/col cell-index convention as YOLOv1's convert_cellboxes.
    cell_indices = torch.arange(S, device=device).repeat(B_size, S, 1).unsqueeze(-1)  # (B,S,S,1)
    cell_x = cell_indices.unsqueeze(3)                      # (B,S,S,1,1), varies along columns
    cell_y = cell_indices.permute(0, 2, 1, 3).unsqueeze(3)  # (B,S,S,1,1), varies along rows

    bx = (torch.sigmoid(tx).unsqueeze(-1) + cell_x) / S
    by = (torch.sigmoid(ty).unsqueeze(-1) + cell_y) / S
    bw = (torch.exp(tw).unsqueeze(-1) * anchors_t[:, 0].view(1, 1, 1, A, 1)) / S
    bh = (torch.exp(th).unsqueeze(-1) * anchors_t[:, 1].view(1, 1, 1, A, 1)) / S

    conf = torch.sigmoid(to).unsqueeze(-1)
    cls_prob = torch.softmax(cls_logits, dim=-1)
    best_cls_prob, best_cls = cls_prob.max(dim=-1, keepdim=True)
    score = conf * best_cls_prob

    boxes = torch.cat([best_cls.float(), score, bx, by, bw, bh], dim=-1)  # (B,S,S,A,6)
    return boxes.reshape(B_size, S * S * A, 6)


def cellboxes_to_boxes_v2(out, anchors, C=20):
    """List-of-lists version of convert_cellboxes_v2, for network predictions."""
    converted_pred = convert_cellboxes_v2(out, anchors, C=C)
    all_bboxes = []
    for ex_idx in range(out.shape[0]):
        bboxes = [
            [x.item() for x in converted_pred[ex_idx, box_idx, :]]
            for box_idx in range(converted_pred.shape[1])
        ]
        all_bboxes.append(bboxes)
    return all_bboxes


def convert_target_boxes_v2(target, C=20):
    """
    Decode a YOLOv2 *target* tensor (single box per cell, format documented
    in loss_f.py) into whole-image-relative box coordinates. Targets have
    no anchor dimension and are not passed through sigmoid/exp (they're
    already offsets/grid-units by construction), so this is intentionally
    separate from convert_cellboxes_v2.

    target: (BATCH, S, S, 5 + C)
    returns: (BATCH, S*S, 6) tensor of [class_pred, objectness, x, y, w, h]
    """
    device = target.device
    B_size, S, _, _ = target.shape

    obj = target[..., C]
    box_xy = target[..., C + 1:C + 3]
    box_wh = target[..., C + 3:C + 5]
    cls = target[..., 0:C]

    cell_indices = torch.arange(S, device=device).repeat(B_size, S, 1).unsqueeze(-1)  # (B,S,S,1)
    cell_x = cell_indices
    cell_y = cell_indices.permute(0, 2, 1, 3)

    bx = (box_xy[..., 0:1] + cell_x) / S
    by = (box_xy[..., 1:2] + cell_y) / S
    bw = box_wh[..., 0:1] / S
    bh = box_wh[..., 1:2] / S

    best_cls = cls.argmax(-1, keepdim=True).float()
    score = obj.unsqueeze(-1)

    boxes = torch.cat([best_cls, score, bx, by, bw, bh], dim=-1)  # (B,S,S,6)
    return boxes.reshape(B_size, S * S, 6)


def target_to_boxes_v2(target, C=20):
    """List-of-lists version of convert_target_boxes_v2, for ground-truth targets."""
    converted = convert_target_boxes_v2(target, C=C)
    all_bboxes = []
    for ex_idx in range(target.shape[0]):
        bboxes = [
            [x.item() for x in converted[ex_idx, box_idx, :]]
            for box_idx in range(converted.shape[1])
        ]
        all_bboxes.append(bboxes)
    return all_bboxes


def resize_targets_v2(target, old_S, new_S, C=20):
    """
    Re-grid a YOLOv2 target tensor built for an old_S x old_S grid onto a
    new_S x new_S grid.

    Needed for multi-scale training: the dataset builds targets once for a
    fixed grid (e.g. S=13 for a 416x416 image), but "Multi-Scale Training"
    re-feeds the same batch at a randomly chosen size (hence a different S)
    every 10 batches. This recovers each box's whole-image-relative
    coordinates from the old grid and re-bins them into the new grid,
    without needing access to the original raw annotations.

    Note: if two boxes from the old grid happen to land in the same new
    cell (more likely when shrinking resolution), the later one in
    iteration order overwrites the earlier one -- same "one box per cell"
    limitation the original YOLOv1/v2 target format already has.
    """
    if old_S == new_S:
        return target

    device = target.device
    B_size = target.shape[0]

    obj = target[..., C]
    box_xy = target[..., C + 1:C + 3]
    box_wh = target[..., C + 3:C + 5]
    cls = target[..., 0:C]

    old_cell_indices = torch.arange(old_S, device=device).repeat(B_size, old_S, 1).unsqueeze(-1)
    old_cell_x = old_cell_indices
    old_cell_y = old_cell_indices.permute(0, 2, 1, 3)

    norm_x = (box_xy[..., 0:1] + old_cell_x) / old_S
    norm_y = (box_xy[..., 1:2] + old_cell_y) / old_S
    norm_w = box_wh[..., 0:1] / old_S
    norm_h = box_wh[..., 1:2] / old_S

    new_target = torch.zeros(B_size, new_S, new_S, 5 + C, device=device, dtype=target.dtype)

    obj_idxs = (obj > 0).nonzero(as_tuple=False)  # (N, 3): batch, row, col
    for b, r, c in obj_idxs:
        b, r, c = b.item(), r.item(), c.item()

        nx = norm_x[b, r, c, 0].item()
        ny = norm_y[b, r, c, 0].item()
        nw = norm_w[b, r, c, 0].item()
        nh = norm_h[b, r, c, 0].item()

        new_col = min(int(nx * new_S), new_S - 1)
        new_row = min(int(ny * new_S), new_S - 1)

        new_target[b, new_row, new_col, 0:C] = cls[b, r, c]
        new_target[b, new_row, new_col, C] = 1.0
        new_target[b, new_row, new_col, C + 1] = nx * new_S - new_col
        new_target[b, new_row, new_col, C + 2] = ny * new_S - new_row
        new_target[b, new_row, new_col, C + 3] = nw * new_S
        new_target[b, new_row, new_col, C + 4] = nh * new_S

    return new_target


def get_bboxes(
    loader,
    model,
    iou_threshold,
    threshold,
    anchors=None,
    box_format="midpoint",
    device="cuda",
    C=20,
):
    """
    Same role as the YOLOv1 version: run the model over a loader and
    collect NMS'd predicted boxes + ground-truth boxes for mAP. Updated to
    decode anchor-based predictions (cellboxes_to_boxes_v2) and single-box
    targets (target_to_boxes_v2) separately, since they're no longer the
    same tensor shape/parametrization.
    """
    if anchors is None:
        from model import VOC_ANCHORS
        anchors = VOC_ANCHORS

    all_pred_boxes = []
    all_true_boxes = []

    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        true_bboxes = target_to_boxes_v2(labels, C=C)
        bboxes = cellboxes_to_boxes_v2(predictions, anchors, C=C)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

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


def train_fn(train_loader, model, optimizer, loss_fn, device, multi_scale=True, base_S=13, C=20):
    """
    One training epoch. Adds YOLOv2's "Multi-Scale Training": every 10
    batches, pick a new input size from {320, 352, ..., 608} (multiples of
    32), resize the batch to it, and re-grid the targets to match the
    resulting output grid (see resize_targets_v2). This is what lets a
    single trained YOLOv2 model run at multiple resolutions with an easy
    speed/accuracy tradeoff at inference time (Table 3 in the paper).
    """
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    input_sizes = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
    current_size = base_S * 32
    global_batch_idx = 0

    for batch_idx, (x, y) in enumerate(loop):
        if multi_scale and global_batch_idx % 10 == 0:
            current_size = int(np.random.choice(input_sizes))
            if hasattr(model, "set_input_size"):
                model.set_input_size(current_size)

        x, y = x.to(device), y.to(device)

        if multi_scale and x.shape[-1] != current_size:
            x = F.interpolate(x, size=(current_size, current_size), mode="bilinear", align_corners=False)

        new_S = current_size // 32
        if multi_scale and new_S != base_S:
            y = resize_targets_v2(y, old_S=base_S, new_S=new_S, C=C)

        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        global_batch_idx += 1
        loop.set_postfix(loss=loss.item(), img_size=current_size)

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")