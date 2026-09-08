"""
YOLOv4 loss function ("YOLOv4: Optimal Speed and Accuracy of Object
Detection", 2020).

The one loss-level change YOLOv4 makes over YOLOv3 in this codebase: box
regression switches from plain sum-squared-error on (t_x, t_y, t_w, t_h)
to **CIoU loss** (Complete IoU), one of the paper's "Bag of Freebies".
Instead of regressing 4 independent coordinates, CIoU directly optimizes a
single similarity score between the predicted and target box that jointly
accounts for:
    1. overlap area (plain IoU),
    2. normalized distance between box centers, and
    3. aspect ratio consistency,
which converges faster and more accurately than 4 separate MSE terms,
especially early in training when predicted boxes barely overlap the
target at all (plain IoU loss gives zero gradient in that regime; CIoU
does not, thanks to the center-distance term). Objectness and class loss
are unchanged from the YOLOv3 version (binary cross-entropy).

Expected target format: identical to the YOLOv3 loss -- a list of 3
tensors, one per scale, each (BATCH, S, S, num_anchors_per_scale, 5 + C):
    target[..., 0:C]      -> one-hot class label
    target[..., C]        -> objectness (1 if assigned, else 0)
    target[..., C+1]      -> t_x: x offset within the cell, in [0, 1]
    target[..., C+2]      -> t_y: y offset within the cell, in [0, 1]
    target[..., C+3]      -> t_w: log(gt_w / anchor_w), grid units
    target[..., C+4]      -> t_h: log(gt_h / anchor_h), grid units
See utils.build_targets_v3 (reused unchanged for YOLOv4 -- the anchor
assignment / target format didn't change) for a reference implementation
dataset.py can call.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_ciou(pred_boxes, target_boxes, eps=1e-7):
    """
    Complete IoU between two sets of (x, y, w, h) midpoint-format boxes,
    both in the same units (any consistent unit works -- grid cells here).

    pred_boxes, target_boxes: (..., 4)
    returns: (...,) CIoU values in [-1, 1] (1 = perfect match)
    """
    pred_x, pred_y, pred_w, pred_h = pred_boxes.unbind(-1)
    tgt_x, tgt_y, tgt_w, tgt_h = target_boxes.unbind(-1)

    pred_x1, pred_x2 = pred_x - pred_w / 2, pred_x + pred_w / 2
    pred_y1, pred_y2 = pred_y - pred_h / 2, pred_y + pred_h / 2
    tgt_x1, tgt_x2 = tgt_x - tgt_w / 2, tgt_x + tgt_w / 2
    tgt_y1, tgt_y2 = tgt_y - tgt_h / 2, tgt_y + tgt_h / 2

    # ---- IoU ----
    inter_x1 = torch.max(pred_x1, tgt_x1)
    inter_y1 = torch.max(pred_y1, tgt_y1)
    inter_x2 = torch.min(pred_x2, tgt_x2)
    inter_y2 = torch.min(pred_y2, tgt_y2)
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    pred_area = (pred_x2 - pred_x1).clamp(min=0) * (pred_y2 - pred_y1).clamp(min=0)
    tgt_area = (tgt_x2 - tgt_x1).clamp(min=0) * (tgt_y2 - tgt_y1).clamp(min=0)
    union = pred_area + tgt_area - inter_area + eps
    iou = inter_area / union

    # ---- center-distance penalty ----
    enc_x1 = torch.min(pred_x1, tgt_x1)
    enc_y1 = torch.min(pred_y1, tgt_y1)
    enc_x2 = torch.max(pred_x2, tgt_x2)
    enc_y2 = torch.max(pred_y2, tgt_y2)
    c2 = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + eps  # enclosing box diagonal^2

    rho2 = (pred_x - tgt_x) ** 2 + (pred_y - tgt_y) ** 2  # center distance^2

    # ---- aspect-ratio consistency penalty ----
    v = (4 / (torch.pi ** 2)) * torch.pow(
        torch.atan(tgt_w / (tgt_h + eps)) - torch.atan(pred_w / (pred_h + eps)), 2
    )
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = iou - (rho2 / c2 + alpha * v)
    return ciou


class YOLO_V4_LOSS(nn.Module):
    def __init__(self, C=20, anchors=None, base_size=416, num_scales=3):
        super(YOLO_V4_LOSS, self).__init__()
        if anchors is None:
            from model import COCO_ANCHORS
            anchors = COCO_ANCHORS

        self.C = C
        self.anchors = anchors      # list of `num_scales` lists of (w, h) in pixels of base_size
        self.base_size = base_size
        self.num_scales = num_scales

        # CIoU replaces the separate coordinate MSE terms, so lambda_coord
        # now weights a single (1 - CIoU) term per responsible box instead
        # of 4 squared-error terms; kept at the same relative scale as the
        # earlier versions' coordinate loss.
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
        self.lambda_obj = 1.0
        self.lambda_class = 1.0

    def _scale_loss(self, pred, target, scale_anchors_px):
        """Loss for a single scale. pred: (B,S,S,A*(5+C)) raw logits.
        target: (B,S,S,A,5+C), format documented above. scale_anchors_px:
        this scale's list of (w,h) anchors in pixels of self.base_size."""
        B_size, S, _, A, _ = target.shape
        C = self.C
        device = pred.device

        stride = self.base_size / S
        anchors_grid = torch.tensor(
            [(w / stride, h / stride) for (w, h) in scale_anchors_px], dtype=torch.float32, device=device
        )  # (A, 2), this scale's anchors in grid-cell units

        pred = pred.reshape(B_size, S, S, A, 5 + C)
        pred_tx = pred[..., 0]
        pred_ty = pred[..., 1]
        pred_tw = pred[..., 2]
        pred_th = pred[..., 3]
        pred_to = pred[..., 4]
        pred_cls_logits = pred[..., 5:]

        obj_mask = target[..., C]          # (B,S,S,A), 1 where a box is assigned
        noobj_mask = 1.0 - obj_mask

        target_xy = target[..., C + 1:C + 3]
        target_tw = target[..., C + 3]
        target_th = target[..., C + 4]
        target_cls = target[..., 0:C]

        # ============================= #
        #   BOX LOSS (CIoU, replaces    #
        #   YOLOv3's separate MSE       #
        #   coordinate terms)           #
        # ============================= #
        # Decode both predicted and target boxes into (x, y, w, h) in a
        # common unit -- grid cells. x/y are the within-cell offset in
        # [0, 1] (the cell index itself is deliberately omitted: pred and
        # target are compared within the same cell, so adding the same
        # constant offset to both would cancel out and doesn't affect
        # IoU/CIoU). w/h MUST be scaled by the actual anchor size (not
        # left as a raw exp(t) ratio) so they're in the same grid-cell
        # units as x/y -- otherwise position and size would use different
        # units and CIoU's distance/aspect-ratio terms would be wrong.
        an_w = anchors_grid[:, 0].view(1, 1, 1, A)
        an_h = anchors_grid[:, 1].view(1, 1, 1, A)

        pred_xy = torch.sigmoid(torch.stack([pred_tx, pred_ty], dim=-1))
        pred_w = torch.exp(pred_tw) * an_w
        pred_h = torch.exp(pred_th) * an_h
        pred_boxes = torch.cat([pred_xy, torch.stack([pred_w, pred_h], dim=-1)], dim=-1)  # (B,S,S,A,4)

        target_w = torch.exp(target_tw) * an_w
        target_h = torch.exp(target_th) * an_h
        target_boxes = torch.cat([target_xy, torch.stack([target_w, target_h], dim=-1)], dim=-1)  # (B,S,S,A,4)

        ciou = compute_ciou(pred_boxes, target_boxes)  # (B,S,S,A)
        box_loss = (obj_mask * (1.0 - ciou)).sum()

        # ==================================== #
        #   OBJECTNESS LOSS (binary, logistic) #
        # ==================================== #
        obj_loss_raw = F.binary_cross_entropy_with_logits(pred_to, obj_mask, reduction="none")
        object_loss = (obj_mask * obj_loss_raw).sum()
        no_object_loss = (noobj_mask * obj_loss_raw).sum()

        # ========================================= #
        #   CLASS LOSS (independent per-class BCE)  #
        # ========================================= #
        cls_loss_raw = F.binary_cross_entropy_with_logits(pred_cls_logits, target_cls, reduction="none")
        class_loss = (obj_mask.unsqueeze(-1) * cls_loss_raw).sum()

        return (
            self.lambda_coord * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )

    def forward(self, predictions, targets):
        """
        predictions: list of 3 tensors, one per scale, each
                     (B, S, S, A*(5+C)) raw network output.
        targets: list of 3 tensors, one per scale, each (B, S, S, A, 5+C).
        """
        assert len(predictions) == len(targets) == self.num_scales, \
            "YOLOv4 loss expects one prediction/target pair per scale"

        B_size = predictions[0].shape[0]
        total_loss = 0.0
        for scale_idx, (pred, target) in enumerate(zip(predictions, targets)):
            total_loss = total_loss + self._scale_loss(pred, target, self.anchors[scale_idx])

        return total_loss / B_size