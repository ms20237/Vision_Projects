"""
YOLOv2 loss function (arXiv:1612.08242).

The core difference from the YOLOv1 loss this file replaces: YOLOv1 has 2
raw predicted boxes per cell and picks whichever one currently has the
higher IoU with the ground truth to be "responsible" for the prediction.
YOLOv2 instead has `num_anchors` box *priors* per cell (from k-means
"Dimension Clusters"), and assigns each ground-truth box to whichever prior
shape is the closest match -- then only that anchor's prediction is trained
against the box, following the (t_x, t_y, t_w, t_h, t_o) parametrization
from "Direct location prediction":

    b_x = sigmoid(t_x) + c_x
    b_y = sigmoid(t_y) + c_y
    b_w = p_w * exp(t_w)
    b_h = p_h * exp(t_h)
    Pr(object) * IOU(b, object) = sigmoid(t_o)

Expected target tensor shape: (BATCH, S, S, 5 + C)
    target[..., 0:C]      -> one-hot class label
    target[..., C]        -> objectness (1 if a ground-truth box's center
                              falls in this cell, else 0)
    target[..., C+1:C+3]  -> (x, y) offset of the box center within the
                              cell, in [0, 1]
    target[..., C+3:C+5]  -> (w, h) of the box in S x S grid-cell units
                              (i.e. the same units the anchors are defined
                              in -- NOT normalized to [0, 1] of the whole
                              image, unlike YOLOv1's target format)

NOTE: dataset.py was not included with the uploaded files. The VOCDataset
class needs to build targets in the format above (S=13 for a 416x416
input) rather than YOLOv1's S=7 format for this loss to work correctly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from intersection_Over_Union import intersection_over_union


class YOLO_V2_LOSS(nn.Module):
    """
    Calculate the loss for a YOLOv2 model.
    """

    def __init__(self, S=13, C=20, anchors=None):
        super(YOLO_V2_LOSS, self).__init__()
        if anchors is None:
            from model import VOC_ANCHORS
            anchors = VOC_ANCHORS

        self.S = S
        self.C = C
        self.register_buffer_anchors(anchors)
        self.B = len(anchors)

        self.mse = nn.MSELoss(reduction="sum")

        # Same relative weighting scheme as YOLOv1 / the original Darknet
        # cfg files.
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
        self.lambda_obj = 1.0
        self.lambda_class = 1.0

    def register_buffer_anchors(self, anchors):
        # kept as a plain attribute (not nn.Module, so no need for a real
        # buffer) but moved to the right device lazily in forward()
        self.anchors = torch.tensor(anchors, dtype=torch.float32)

    def _anchor_ious(self, target_wh):
        """
        IoU between each ground-truth box's (w, h) -- both boxes centered
        at the origin -- and each anchor prior. Used only to decide which
        anchor is "responsible" for a given ground-truth box, following the
        distance metric d(box, centroid) = 1 - IOU(box, centroid) from
        "Dimension Clusters".

        target_wh: (N, 2)
        returns:   (N, A) IoU matrix
        """
        anchors = self.anchors.to(target_wh.device)  # (A, 2)

        gt_w = target_wh[:, 0:1]           # (N, 1)
        gt_h = target_wh[:, 1:2]
        an_w = anchors[:, 0].unsqueeze(0)   # (1, A)
        an_h = anchors[:, 1].unsqueeze(0)

        inter = torch.min(gt_w, an_w) * torch.min(gt_h, an_h)
        union = gt_w * gt_h + an_w * an_h - inter
        return inter / (union + 1e-6)

    def forward(self, predictions, target):
        B_size = predictions.shape[0]
        # Infer S from the network output rather than trusting self.S, so
        # this loss keeps working unmodified when the input resolution
        # changes under multi-scale training (predictions is fully
        # convolutional, so its spatial size tracks the input size).
        S = predictions.shape[1]
        A, C = self.B, self.C
        device = predictions.device
        anchors = self.anchors.to(device)  # (A, 2)

        predictions = predictions.reshape(B_size, S, S, A, 5 + C)

        pred_tx = predictions[..., 0]
        pred_ty = predictions[..., 1]
        pred_tw = predictions[..., 2]
        pred_th = predictions[..., 3]
        pred_to = predictions[..., 4]
        pred_cls_logits = predictions[..., 5:]

        exists_box = target[..., C].unsqueeze(-1)            # (B, S, S, 1), Iobj_i
        target_box_xy = target[..., C + 1:C + 3]              # (B, S, S, 2)
        target_box_wh = target[..., C + 3:C + 5]              # (B, S, S, 2), grid units
        target_cls = target[..., 0:C]                          # (B, S, S, C)

        # ------------------------------------------------------------- #
        #  Decide which anchor is responsible for each occupied cell    #
        # ------------------------------------------------------------- #
        flat_wh = target_box_wh.reshape(-1, 2)
        anchor_iou = self._anchor_ious(flat_wh)                        # (B*S*S, A)
        best_anchor = anchor_iou.argmax(dim=-1).reshape(B_size, S, S)   # (B, S, S)
        anchor_mask = F.one_hot(best_anchor, num_classes=A).float()     # (B, S, S, A)
        obj_ij = anchor_mask * exists_box                                # zero out empty cells

        # ------------------------------------------------------------- #
        #  Decode predicted boxes (needed for the objectness target,    #
        #  which regresses toward the actual predicted-vs-gt IoU, same  #
        #  as YOLOv1's Pr(object) * IOU target)                         #
        # ------------------------------------------------------------- #
        pred_xy = torch.sigmoid(torch.stack([pred_tx, pred_ty], dim=-1))       # (B,S,S,A,2)

        an_w = anchors[:, 0].view(1, 1, 1, A)
        an_h = anchors[:, 1].view(1, 1, 1, A)
        pred_w = torch.exp(pred_tw) * an_w
        pred_h = torch.exp(pred_th) * an_h
        pred_wh = torch.stack([pred_w, pred_h], dim=-1)                          # (B,S,S,A,2)

        target_xy_exp = target_box_xy.unsqueeze(3).expand(-1, -1, -1, A, -1)
        target_wh_exp = target_box_wh.unsqueeze(3).expand(-1, -1, -1, A, -1)

        pred_boxes_flat = torch.cat([pred_xy, pred_wh], dim=-1).reshape(-1, 4)
        target_boxes_flat = torch.cat([target_xy_exp, target_wh_exp], dim=-1).reshape(-1, 4)
        ious = intersection_over_union(pred_boxes_flat, target_boxes_flat, box_format="midpoint")
        ious = ious.reshape(B_size, S, S, A)

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #
        # x, y are regressed directly (already bounded to the cell via
        # sigmoid); w, h are regressed in log-space against the anchor,
        # i.e. the network target for t_w is log(gt_w / anchor_w).
        coord_xy_loss = self.mse(obj_ij.unsqueeze(-1) * pred_xy,
                                  obj_ij.unsqueeze(-1) * target_xy_exp)

        gt_w = target_box_wh[..., 0:1].expand(-1, -1, -1, A)
        gt_h = target_box_wh[..., 1:2].expand(-1, -1, -1, A)
        target_tw = torch.log(gt_w.clamp(min=1e-6) / an_w)
        target_th = torch.log(gt_h.clamp(min=1e-6) / an_h)

        coord_wh_loss = self.mse(obj_ij * pred_tw, obj_ij * target_tw) + \
            self.mse(obj_ij * pred_th, obj_ij * target_th)

        box_loss = coord_xy_loss + coord_wh_loss

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #
        pred_conf = torch.sigmoid(pred_to)                            # (B,S,S,A)
        object_loss = self.mse(obj_ij * pred_conf, obj_ij * ious.detach())

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #
        noobj_ij = 1.0 - obj_ij
        no_object_loss = self.mse(noobj_ij * pred_conf, torch.zeros_like(pred_conf))

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #
        target_cls_exp = target_cls.unsqueeze(3).expand(-1, -1, -1, A, -1)
        pred_cls_prob = torch.softmax(pred_cls_logits, dim=-1)
        class_loss = self.mse(obj_ij.unsqueeze(-1) * pred_cls_prob,
                               obj_ij.unsqueeze(-1) * target_cls_exp)

        loss = (
            self.lambda_coord * box_loss        # coordinate loss (x, y, w, h)
            + self.lambda_obj * object_loss      # objectness for responsible anchors
            + self.lambda_noobj * no_object_loss  # objectness for all other anchors
            + self.lambda_class * class_loss     # class loss
        )

        return loss / B_size