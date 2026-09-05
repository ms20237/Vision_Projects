"""
YOLOv3 loss function ("YOLOv3: An Incremental Improvement", 2018).

Differences from the YOLOv2 loss this file replaces:

- Multi-scale: YOLOv3 predicts at 3 grid resolutions at once, so this loss
  takes a *list* of 3 prediction tensors and a *list* of 3 target tensors
  (one pair per scale) and sums the loss across all three.
- Class loss: the paper switches from a single softmax per box (which
  assumes mutually-exclusive classes) to independent per-class logistic
  (sigmoid) classifiers trained with binary cross-entropy, so that
  multi-label cases (e.g. "Woman" and "Person" both being true) can be
  modeled. Objectness is likewise trained with binary cross-entropy on a
  sigmoid output rather than regressed toward MSE(IoU) as in this
  codebase's YOLOv2 version.
- Anchor assignment happens *before* the loss runs (baked into the target
  tensors -- see utils.build_targets_v3), not inside forward(), since with
  9 anchors across 3 scales, "which anchor is responsible" is also "which
  scale is responsible", and that's naturally decided while building
  per-image targets rather than re-derived every forward pass.

Expected target format: a list of 3 tensors, one per scale, each shaped
exactly like that scale's prediction tensor after reshaping --
(BATCH, S, S, num_anchors_per_scale, 5 + C) -- almost entirely zero, with
a filled entry at (b, row, col, anchor_idx) for every ground-truth box
assigned to that scale/anchor/cell:
    target[..., 0:C]      -> one-hot class label
    target[..., C]        -> objectness (1 if assigned, else 0)
    target[..., C+1]      -> t_x: x offset within the cell, in [0, 1]
    target[..., C+2]      -> t_y: y offset within the cell, in [0, 1]
    target[..., C+3]      -> t_w: log(gt_w / anchor_w), grid units
    target[..., C+4]      -> t_h: log(gt_h / anchor_h), grid units
(t_w/t_h are pre-computed in log-space -- unlike the YOLOv2 version, this
loss does NOT need to pick an anchor or convert units itself, since that
was already resolved when the target was built. See
utils.build_targets_v3 for a reference implementation dataset.py can call.)

NOTE: dataset.py was not included with the uploaded files, so it needs to
produce (or this loss needs to be fed via) targets in the above format --
see utils.build_targets_v3.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class YOLO_V3_LOSS(nn.Module):
    def __init__(self, C=20, num_scales=3):
        super(YOLO_V3_LOSS, self).__init__()
        self.C = C
        self.num_scales = num_scales

        self.mse = nn.MSELoss(reduction="sum")

        # Same relative weighting scheme as YOLOv1/v2 / the original
        # Darknet cfg files.
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
        self.lambda_obj = 1.0
        self.lambda_class = 1.0

    def _scale_loss(self, pred, target):
        """Loss for a single scale. pred: (B,S,S,A*(5+C)) raw logits.
        target: (B,S,S,A,5+C), format documented above."""
        B_size, S, _, A, _ = target.shape
        C = self.C

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

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #
        pred_xy = torch.sigmoid(torch.stack([pred_tx, pred_ty], dim=-1))  # (B,S,S,A,2)
        coord_xy_loss = self.mse(obj_mask.unsqueeze(-1) * pred_xy,
                                  obj_mask.unsqueeze(-1) * target_xy)

        # t_w / t_h targets are already in log-space (built alongside the
        # anchor assignment), so this is a direct regression -- no need to
        # re-derive anchors or take a log here, unlike the YOLOv2 loss.
        coord_wh_loss = self.mse(obj_mask * pred_tw, obj_mask * target_tw) + \
            self.mse(obj_mask * pred_th, obj_mask * target_th)

        box_loss = coord_xy_loss + coord_wh_loss

        # ==================================== #
        #   OBJECTNESS LOSS (binary, logistic)  #
        # ==================================== #
        obj_loss_raw = F.binary_cross_entropy_with_logits(pred_to, obj_mask, reduction="none")
        object_loss = (obj_mask * obj_loss_raw).sum()
        no_object_loss = (noobj_mask * obj_loss_raw).sum()

        # ========================================= #
        #   CLASS LOSS (independent per-class BCE)   #
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
            "YOLOv3 loss expects one prediction/target pair per scale"

        B_size = predictions[0].shape[0]
        total_loss = 0.0
        for pred, target in zip(predictions, targets):
            total_loss = total_loss + self._scale_loss(pred, target)

        return total_loss / B_size