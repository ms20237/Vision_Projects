"""
YOLOv2 (from "YOLO9000: Better, Faster, Stronger", arXiv:1612.08242).

Key differences vs. the YOLOv1 model this file replaces:

- Backbone: Darknet-19 (19 conv layers, fully convolutional, Table 6 in the
  paper) instead of YOLOv1's 24-conv-layer backbone with 2 FC layers on top.
  There are no fully-connected layers anywhere in YOLOv2 -- the whole
  network is convolutional, which is also what lets it run at multiple
  input resolutions ("Multi-Scale Training").
- Anchor boxes: instead of directly regressing (x, y, w, h) per cell, the
  network predicts offsets relative to `num_anchors` box priors per cell
  ("Convolutional With Anchor Boxes" / "Dimension Clusters").
- Output grid is 13x13 for a 416x416 input (downsample factor 32) instead
  of YOLOv1's 7x7 for a 448x448 input.
- Passthrough / reorg layer: concatenates the earlier 26x26x512 feature map
  with the final 13x13x1024 map to give the detector access to finer
  grained features for small objects ("Fine-Grained Features").
"""

import torch
import torch.nn as nn


# (kernel_size, out_channels, stride, padding); "M" = 2x2 stride-2 maxpool.
# Stage 1 stops right before the 5th maxpool: its output is the 26x26x512
# feature map (for a 416x416 input) used by the passthrough layer.
DARKNET19_STAGE1 = [
    (3, 32, 1, 1), "M",
    (3, 64, 1, 1), "M",
    (3, 128, 1, 1), (1, 64, 1, 0), (3, 128, 1, 1), "M",
    (3, 256, 1, 1), (1, 128, 1, 0), (3, 256, 1, 1), "M",
    (3, 512, 1, 1), (1, 256, 1, 0), (3, 512, 1, 1), (1, 256, 1, 0), (3, 512, 1, 1),
]

# Stage 2: 5th maxpool through the last conv block of Darknet-19 (Table 6),
# ending at a 13x13x1024 feature map.
DARKNET19_STAGE2 = [
    "M",
    (3, 1024, 1, 1), (1, 512, 1, 0), (3, 1024, 1, 1), (1, 512, 1, 0), (3, 1024, 1, 1),
]

# Default anchor priors (width, height), in 13x13-grid units. These are the
# 5 VOC priors from the original Darknet yolov2-voc.cfg -- i.e. results of
# running k-means clustering on box dimensions as described in "Dimension
# Clusters". Swap these out if you re-run k-means on your own dataset.
VOC_ANCHORS = [
    (1.08, 1.19),
    (3.42, 4.41),
    (6.63, 11.38),
    (9.42, 5.11),
    (16.62, 10.52),
]


class CNN_Block(nn.Module):
    """Conv2d -> BatchNorm -> LeakyReLU(0.1). Every conv layer in YOLOv2 is
    batch-normalized (paper: "Batch Normalization" gives +2% mAP and lets
    us drop dropout), so this stays identical to the YOLOv1 block."""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNN_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batch_nom = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leaky_relu(self.batch_nom(self.conv1(x)))


class ReorgLayer(nn.Module):
    """
    Space-to-depth "passthrough" layer.

    (B, C, H, W) -> (B, C*stride^2, H/stride, W/stride), stacking spatially
    adjacent features into channels instead of discarding resolution, e.g.
    26x26x64 -> 13x13x256 for stride=2. This is exactly the mechanism
    described in "Fine-Grained Features": "concatenates the higher
    resolution features with the low resolution features by stacking
    adjacent features into different channels instead of spatial locations".
    """

    def __init__(self, stride=2):
        super(ReorgLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        B, C, H, W = x.shape
        s = self.stride
        assert H % s == 0 and W % s == 0, "feature map dims must be divisible by stride"

        x = x.view(B, C, H // s, s, W // s, s)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(B, C * s * s, H // s, W // s)
        return x


def _make_layers(cfg, in_channels):
    layers = []
    for item in cfg:
        if isinstance(item, tuple):
            k, out_c, stride, pad = item
            layers.append(CNN_Block(in_channels, out_c, kernel_size=k, stride=stride, padding=pad))
            in_channels = out_c
        elif item == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            raise ValueError(f"Unrecognized config entry: {item}")
    return nn.Sequential(*layers), in_channels


class YOLO_V2(nn.Module):
    """
    Darknet-19 backbone + passthrough layer + anchor-based detection head.

    Output shape: (BATCH, S, S, num_anchors * (5 + num_classes))
    The 5 values per anchor are (t_x, t_y, t_w, t_h, t_o) -- raw network
    outputs, *not* yet decoded into box coordinates. Decoding follows the
    paper's parametrization (see utils.convert_cellboxes_v2 / the loss
    function):

        b_x = sigmoid(t_x) + c_x
        b_y = sigmoid(t_y) + c_y
        b_w = p_w * exp(t_w)
        b_h = p_h * exp(t_h)
        Pr(object) * IOU(b, object) = sigmoid(t_o)
    """

    def __init__(self, in_channels=3, num_classes=20, anchors=None):
        super(YOLO_V2, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.anchors = anchors if anchors is not None else VOC_ANCHORS
        self.num_anchors = len(self.anchors)

        # Bookkeeping only -- the network itself is fully convolutional and
        # accepts any input size that's a multiple of 32 without changing
        # its weights. This just gives multi-scale training loops
        # (see "Multi-Scale Training") a place to record/query the current
        # size and the resulting output grid S = size / 32.
        self.current_input_size = 416
        self.S = self.current_input_size // 32

        self.stage1, stage1_out = _make_layers(DARKNET19_STAGE1, in_channels)
        self.stage2, stage2_out = _make_layers(DARKNET19_STAGE2, stage1_out)

        # 3 extra 3x3x1024 conv layers added on top of Darknet-19 for
        # detection ("Training for detection").
        self.conv_extra = nn.Sequential(
            CNN_Block(stage2_out, 1024, kernel_size=3, stride=1, padding=1),
            CNN_Block(1024, 1024, kernel_size=3, stride=1, padding=1),
            CNN_Block(1024, 1024, kernel_size=3, stride=1, padding=1),
        )

        # 1x1 conv compressing the passthrough features 512 -> 64 before the
        # reorg (matches the original Darknet yolov2 config), then reorg
        # 26x26x64 -> 13x13x256.
        self.passthrough_conv = CNN_Block(stage1_out, 64, kernel_size=1, stride=1, padding=0)
        self.reorg = ReorgLayer(stride=2)

        # Concat: 1024 (conv_extra) + 256 (reorg'd passthrough) = 1280 in.
        self.conv_final = CNN_Block(1024 + 256, 1024, kernel_size=3, stride=1, padding=1)

        # Final 1x1 conv: for every cell and every anchor, predict
        # (5 + num_classes) values. For VOC (5 anchors, 20 classes) this is
        # 5 * 25 = 125 filters, matching the paper.
        self.pred = nn.Conv2d(
            1024, self.num_anchors * (5 + num_classes), kernel_size=1, stride=1, padding=0
        )

    def set_input_size(self, size):
        """
        Record the input resolution the network is currently being fed at.
        Every 10 batches, "Multi-Scale Training" randomly picks a new size
        from {320, 352, ..., 608} (multiples of 32); call this so the
        resulting output grid size (self.S) stays accurate for anything
        that needs it (e.g. re-gridding targets, decoding boxes).
        """
        assert size % 32 == 0, "YOLOv2 input size must be a multiple of 32"
        self.current_input_size = size
        self.S = size // 32

    def forward(self, x):
        passthrough_feat = self.stage1(x)              # (B, 512, 26, 26) for 416 input
        x = self.stage2(passthrough_feat)               # (B, 1024, 13, 13)
        x = self.conv_extra(x)                           # (B, 1024, 13, 13)

        pt = self.passthrough_conv(passthrough_feat)     # (B, 64, 26, 26)
        pt = self.reorg(pt)                                # (B, 256, 13, 13)

        x = torch.cat([pt, x], dim=1)                    # (B, 1280, 13, 13)
        x = self.conv_final(x)                            # (B, 1024, 13, 13)
        out = self.pred(x)                                # (B, A*(5+C), 13, 13)

        # (B, A*(5+C), S, S) -> (B, S, S, A*(5+C)), same convention as YOLOv1
        out = out.permute(0, 2, 3, 1).contiguous()
        return out


def test(num_classes=20):
    model = YOLO_V2(num_classes=num_classes)
    x = torch.randn(2, 3, 416, 416)
    out = model(x)
    print(out.shape)  # expected: (2, 13, 13, 5 * (5 + 20)) = (2, 13, 13, 125)


if __name__ == "__main__":
    test()