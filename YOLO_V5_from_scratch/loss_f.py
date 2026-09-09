"""
YOLOv5 loss function (Ultralytics). Follows the public implementation's
loss design, which builds on YOLOv4's CIoU + BCE losses with one further
refinement:

- **Per-scale objectness balancing**: the finest grid (52x52, small
  objects) has far more cells than the coarsest (13x13, large objects), so
  its objectness loss would otherwise dominate/underweight differently
  than intended. Ultralytics counters this with fixed per-scale weights
  (`obj_scale_balance`, defaulting to Ultralytics' [4.0, 1.0, 0.4] for
  [small, medium, large] -- i.e. the finest grid's objectness loss is
  weighted up, the coarsest weighted down) before summing across scales.
- Box loss is still CIoU (unchanged from YOLOv4's version in this
  codebase); objectness and class loss are still binary cross-entropy
  (unchanged from YOLOv3/v4).
- Optional class-label smoothing, a training trick Ultralytics enables by
  default at a small value, which softens the {0, 1} BCE targets slightly
  to reduce over-confidence.

Expected target format: identical to the YOLOv3/v4 loss -- a list of 3
tensors, one per scale, each (BATCH, S, S, num_anchors_per_scale, 5 + C).
See utils.build_targets (shared, unchanged since YOLOv3) for a reference
implementation dataset.py can call.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_ciou(pred_boxes, target_boxes, eps=1e-7):
    """
    Complete IoU between two sets of (x, y, w, h) midpoint-format boxes,
    both in the same units (any consistent unit works -- grid cells here).
    Identical to the YOLOv4 version of this function.

    pred_boxes, target_boxes: (..., 4)
    returns: (...,) CIoU values in [-1, 1] (1 = perfect match)
    """
    pred_x, pred_y, pred_w, pred_h = pred_boxes.unbind(-1)
    tgt_x, tgt_y, tgt_w, tgt_h = target_boxes.unbind(-1)

    pred_x1, pred_x2 = pred_x - pred_w / 2, pred_x + pred_w / 2
    pred_y1, pred_y2 = pred_y - pred_h / 2, pred_y + pred_h / 2
    tgt_x1, tgt_x2 = tgt_x - tgt_w / 2, tgt_x + tgt_w / 2
    tgt_y1, tgt_y2 = tgt_y - tgt_h / 2, tgt_y + tgt_h / 2

    inter_x1 = torch.max(pred_x1, tgt_x1)
    inter_y1 = torch.max(pred_y1, tgt_y1)
    inter_x2 = torch.min(pred_x2, tgt_x2)
    inter_y2 = torch.min(pred_y2, tgt_y2)
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    pred_area = (pred_x2 - pred_x1).clamp(min=0) * (pred_y2 - pred_y1).clamp(min=0)
    tgt_area = (tgt_x2 - tgt_x1).clamp(min=0) * (tgt_y2 - tgt_y1).clamp(min=0)
    union = pred_area + tgt_area - inter_area + eps
    iou = inter_area / union

    enc_x1 = torch.min(pred_x1, tgt_x1)
    enc_y1 = torch.min(pred_y1, tgt_y1)
    enc_x2 = torch.max(pred_x2, tgt_x2)
    enc_y2 = torch.max(pred_y2, tgt_y2)
    c2 = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + eps

    rho2 = (pred_x - tgt_x) ** 2 + (pred_y - tgt_y) ** 2

    v = (4 / (torch.pi ** 2)) * torch.pow(
        torch.atan(tgt_w / (tgt_h + eps)) - torch.atan(pred_w / (pred_h + eps)), 2
    )
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = iou - (rho2 / c2 + alpha * v)
    return ciou


class YOLO_V5_LOSS(nn.Module):
    def __init__(self, C=20, anchors=None, base_size=416, num_scales=3,
                 obj_scale_balance=None, label_smoothing=0.0):
        super(YOLO_V5_LOSS, self).__init__()
        if anchors is None:
            from model import COCO_ANCHORS
            anchors = COCO_ANCHORS

        self.C = C
        self.anchors = anchors      # list of `num_scales` lists of (w, h) in pixels of base_size
        self.base_size = base_size
        self.num_scales = num_scales

        # Ultralytics' default per-scale objectness weights, ordered to
        # match this codebase's [large, medium, small] scale order (their
        # default is expressed as [small, medium, large] = [4.0, 1.0, 0.4],
        # i.e. finest-grid objectness loss weighted up, coarsest weighted
        # down -- reversed here to match our ordering).
        self.obj_scale_balance = obj_scale_balance if obj_scale_balance is not None else [0.4, 1.0, 4.0]

        self.label_smoothing = label_smoothing

        self.lambda_coord = 5
        self.lambda_noobj = 0.5
        self.lambda_obj = 1.0
        self.lambda_class = 1.0

    def _scale_loss(self, pred, target, scale_anchors_px, obj_balance):
        """Loss for a single scale. pred: (B,S,S,A*(5+C)) raw logits.
        target: (B,S,S,A,5+C), format documented above."""
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
        if self.label_smoothing > 0:
            # standard binary label smoothing: 1 -> 1-eps/2, 0 -> eps/2
            eps = self.label_smoothing
            target_cls = target_cls * (1 - eps) + 0.5 * eps

        # ============================= #
        #   BOX LOSS (CIoU)              #
        # ============================= #
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
        #   OBJECTNESS LOSS (binary, logistic, #
        #   weighted by this scale's balance)  #
        # ==================================== #
        obj_loss_raw = F.binary_cross_entropy_with_logits(pred_to, obj_mask, reduction="none")
        object_loss = obj_balance * (obj_mask * obj_loss_raw).sum()
        no_object_loss = obj_balance * (noobj_mask * obj_loss_raw).sum()

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
            "YOLOv5 loss expects one prediction/target pair per scale"

        B_size = predictions[0].shape[0]
        total_loss = 0.0
        for scale_idx, (pred, target) in enumerate(zip(predictions, targets)):
            total_loss = total_loss + self._scale_loss(
                pred, target, self.anchors[scale_idx], self.obj_scale_balance[scale_idx]
            )

        return total_loss / B_size