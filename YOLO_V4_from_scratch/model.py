"""
YOLOv4 ("YOLOv4: Optimal Speed and Accuracy of Object Detection", Bochkovskiy,
Wang & Liao, 2020).

Structural differences vs. the YOLOv3 model this replaces:

- Backbone: CSPDarknet53 instead of plain Darknet53. Each residual stage is
  wrapped in a "Cross Stage Partial" (CSP) block: the stage input is split
  into two paths, one goes through the usual stack of residual units while
  the other skips straight to the end, and the two are concatenated. This
  cuts computation while preserving (or improving) accuracy by encouraging
  richer gradient flow. Backbone activations are Mish instead of LeakyReLU
  ("Bag of specials").
- SPP (Spatial Pyramid Pooling): right after the backbone, feature maps are
  max-pooled at 3 different kernel sizes (5, 9, 13, all stride 1 so spatial
  size is preserved) and concatenated with the un-pooled input. This
  cheaply enlarges the receptive field so the deepest feature map "sees"
  much more context before detection starts.
- Neck: PANet instead of a plain FPN. YOLOv3's neck only has a top-down
  path (upsample coarse features, fuse into finer ones). YOLOv4 adds a
  second, bottom-up path *after* that: the finest features are
  downsampled and fused back into the coarser maps, so information flows
  both ways before the 3 prediction heads.
- Still anchor-based, multi-scale (13x13 / 26x26 / 52x52 for 416 input),
  3 anchors per scale -- the box parametrization and overall detection
  head shape are unchanged from YOLOv3.
"""

import math
import torch
import torch.nn as nn


class Mish(nn.Module):
    """Mish(x) = x * tanh(softplus(x)). Used throughout the CSPDarknet53
    backbone ("Bag of specials" in the paper); smoother than LeakyReLU and
    empirically gives a small accuracy bump for the backbone specifically."""

    def forward(self, x):
        return x * torch.tanh(nn.functional.softplus(x))


class ConvBNMish(nn.Module):
    """Conv2d -> BatchNorm -> Mish. Backbone conv block."""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBNMish, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = Mish()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class CNN_Block(nn.Module):
    """Conv2d -> BatchNorm -> LeakyReLU(0.1). Used in the neck/head, same as
    YOLOv1-v3 (the original Darknet yolov4.cfg keeps LeakyReLU outside the
    backbone -- only the CSPDarknet53 backbone itself switches to Mish)."""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNN_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batch_nom = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leaky_relu(self.batch_nom(self.conv1(x)))


class ResUnit(nn.Module):
    """Same residual unit as Darknet53: 1x1 conv halves channels, 3x3 conv
    restores them, skip connection around the pair -- just with Mish."""

    def __init__(self, channels, hidden_channels=None):
        super(ResUnit, self).__init__()
        hidden_channels = hidden_channels or channels
        self.conv1 = ConvBNMish(channels, hidden_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvBNMish(hidden_channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class CSPBlock(nn.Module):
    """
    One CSPDarknet53 stage: a stride-2 downsampling conv, followed by a
    "Cross Stage Partial" split -- the downsampled feature map is split
    into two branches (via two separate 1x1 convs, NOT a literal tensor
    split, matching the original CSPDarknet implementation): one branch is
    a near-identity shortcut, the other runs through `num_blocks` residual
    units. The two branches are concatenated and fused with a final 1x1
    conv back to `out_channels`.

    `first=True` reproduces the first CSP stage's slightly different
    channel arrangement (no halving of the residual-branch channels),
    matching the original CSPDarknet53 architecture.
    """

    def __init__(self, in_channels, out_channels, num_blocks, first=False):
        super(CSPBlock, self).__init__()
        self.downsample = ConvBNMish(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

        hidden = out_channels if first else out_channels // 2

        self.shortcut_conv = ConvBNMish(out_channels, hidden, kernel_size=1, stride=1, padding=0)
        self.main_conv = ConvBNMish(out_channels, hidden, kernel_size=1, stride=1, padding=0)
        self.blocks = nn.Sequential(*[ResUnit(hidden, hidden) for _ in range(num_blocks)])
        self.main_transition = ConvBNMish(hidden, hidden, kernel_size=1, stride=1, padding=0)
        self.concat_conv = ConvBNMish(hidden * 2, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.downsample(x)
        shortcut = self.shortcut_conv(x)
        main = self.main_conv(x)
        main = self.blocks(main)
        main = self.main_transition(main)
        x = torch.cat([main, shortcut], dim=1)
        return self.concat_conv(x)


class CSPDarknet53(nn.Module):
    """
    CSPDarknet53 backbone. Same overall depth schedule as plain Darknet53
    (1, 2, 8, 8, 4 residual units per stage) but every stage is a CSPBlock.
    Returns the same three routed feature maps as the YOLOv3 backbone:
        route1: (B, 256,  52, 52)  -- for a 416x416 input
        route2: (B, 512,  26, 26)
        route3: (B, 1024, 13, 13)
    """

    def __init__(self, in_channels=3):
        super(CSPDarknet53, self).__init__()
        self.stem = ConvBNMish(in_channels, 32, kernel_size=3, stride=1, padding=1)

        self.stage1 = CSPBlock(32, 64, num_blocks=1, first=True)
        self.stage2 = CSPBlock(64, 128, num_blocks=2)
        self.stage3 = CSPBlock(128, 256, num_blocks=8)   # -> route1
        self.stage4 = CSPBlock(256, 512, num_blocks=8)   # -> route2
        self.stage5 = CSPBlock(512, 1024, num_blocks=4)  # -> route3

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        route1 = x                # 256 ch, 52x52

        x = self.stage4(x)
        route2 = x                # 512 ch, 26x26

        x = self.stage5(x)
        route3 = x                # 1024 ch, 13x13

        return route1, route2, route3


class SPPBlock(nn.Module):
    """
    Spatial Pyramid Pooling. Max-pools the input at several kernel sizes
    (all stride 1, padded to preserve spatial size) and concatenates the
    results with the original input, cheaply enlarging the receptive
    field of the deepest feature map. Channel count becomes
    in_channels * (len(kernel_sizes) + 1).
    """

    def __init__(self, kernel_sizes=(5, 9, 13)):
        super(SPPBlock, self).__init__()
        self.pools = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2) for k in kernel_sizes]
        )

    def forward(self, x):
        pooled = [x] + [pool(x) for pool in self.pools]
        return torch.cat(pooled, dim=1)


def _conv_set3(in_channels, out_channels):
    """1x1 -> 3x3 -> 1x1, halving to out_channels then doubling then back."""
    return nn.Sequential(
        CNN_Block(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
        CNN_Block(out_channels, out_channels * 2, kernel_size=3, stride=1, padding=1),
        CNN_Block(out_channels * 2, out_channels, kernel_size=1, stride=1, padding=0),
    )


def _conv_set5(in_channels, out_channels):
    """1x1 -> 3x3 -> 1x1 -> 3x3 -> 1x1, the standard 5-conv PANet fusion block."""
    return nn.Sequential(
        CNN_Block(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
        CNN_Block(out_channels, out_channels * 2, kernel_size=3, stride=1, padding=1),
        CNN_Block(out_channels * 2, out_channels, kernel_size=1, stride=1, padding=0),
        CNN_Block(out_channels, out_channels * 2, kernel_size=3, stride=1, padding=1),
        CNN_Block(out_channels * 2, out_channels, kernel_size=1, stride=1, padding=0),
    )


# Default anchor priors, same convention and default values as the YOLOv3
# version of this file: (width, height) in pixels of a 416x416 input,
# grouped by scale. Re-run k-means on your own dataset for best results.
COCO_ANCHORS = [
    [(116, 90), (156, 198), (373, 326)],  # scale 1: 13x13 grid -- large objects
    [(30, 61), (62, 45), (59, 119)],       # scale 2: 26x26 grid -- medium objects
    [(10, 13), (16, 30), (33, 23)],        # scale 3: 52x52 grid -- small objects
]


class YOLO_V4(nn.Module):
    """
    CSPDarknet53 backbone + SPP + PANet neck + 3 detection heads.

    forward() returns a list of 3 tensors, in the SAME order as
    COCO_ANCHORS: [13x13 (large objects), 26x26 (medium), 52x52 (small)],
    each shaped (BATCH, S, S, num_anchors_per_scale * (5 + num_classes)).
    """

    def __init__(self, in_channels=3, num_classes=20, anchors_per_scale=3):
        super(YOLO_V4, self).__init__()
        self.num_classes = num_classes
        self.A = anchors_per_scale
        pred_channels = self.A * (5 + num_classes)

        self.backbone = CSPDarknet53(in_channels)

        # ---- SPP stage on the deepest backbone feature (1024ch, 13x13) ----
        self.spp_pre = _conv_set3(1024, 512)          # -> 512ch
        self.spp = SPPBlock(kernel_sizes=(5, 9, 13))    # -> 512*4=2048ch
        self.spp_post = _conv_set3(2048, 512)            # -> 512ch, this is P5

        # ---- top-down path (like YOLOv3's FPN) ----
        self.p5_reduce = CNN_Block(512, 256, kernel_size=1, stride=1, padding=0)
        self.upsample1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.route2_reduce = CNN_Block(512, 256, kernel_size=1, stride=1, padding=0)
        self.p4_set = _conv_set5(512, 256)               # -> P4, 256ch, 26x26

        self.p4_reduce = CNN_Block(256, 128, kernel_size=1, stride=1, padding=0)
        self.upsample2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.route1_reduce = CNN_Block(256, 128, kernel_size=1, stride=1, padding=0)
        self.p3_set = _conv_set5(256, 128)               # -> P3, 128ch, 52x52 (finest)

        # ---- bottom-up path (PANet's addition over plain FPN) ----
        self.p3_downsample = CNN_Block(128, 256, kernel_size=3, stride=2, padding=1)
        self.n4_set = _conv_set5(512, 256)               # -> N4, 256ch, 26x26

        self.n4_downsample = CNN_Block(256, 512, kernel_size=3, stride=2, padding=1)
        self.n5_set = _conv_set5(1024, 512)              # -> N5, 512ch, 13x13

        # ---- prediction heads, one per final PANet output ----
        self.pred_large_conv = CNN_Block(512, 1024, kernel_size=3, stride=1, padding=1)
        self.pred_large = nn.Conv2d(1024, pred_channels, kernel_size=1, stride=1, padding=0)

        self.pred_medium_conv = CNN_Block(256, 512, kernel_size=3, stride=1, padding=1)
        self.pred_medium = nn.Conv2d(512, pred_channels, kernel_size=1, stride=1, padding=0)

        self.pred_small_conv = CNN_Block(128, 256, kernel_size=3, stride=1, padding=1)
        self.pred_small = nn.Conv2d(256, pred_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        route1, route2, route3 = self.backbone(x)   # 256@52, 512@26, 1024@13

        # SPP on the deepest feature map
        x = self.spp_pre(route3)
        x = self.spp(x)
        p5 = self.spp_post(x)                          # 512ch, 13x13

        # top-down (FPN)
        x = self.p5_reduce(p5)                           # 256, 13x13
        x = self.upsample1(x)                              # 256, 26x26
        r2 = self.route2_reduce(route2)                      # 256, 26x26
        x = torch.cat([x, r2], dim=1)                          # 512, 26x26
        p4 = self.p4_set(x)                                     # 256, 26x26

        x = self.p4_reduce(p4)                                   # 128, 26x26
        x = self.upsample2(x)                                      # 128, 52x52
        r1 = self.route1_reduce(route1)                              # 128, 52x52
        x = torch.cat([x, r1], dim=1)                                  # 256, 52x52
        p3 = self.p3_set(x)                                             # 128, 52x52 (finest)

        # bottom-up (PANet)
        x = self.p3_downsample(p3)                                      # 256, 26x26
        x = torch.cat([x, p4], dim=1)                                     # 512, 26x26
        n4 = self.n4_set(x)                                                # 256, 26x26

        x = self.n4_downsample(n4)                                          # 512, 13x13
        x = torch.cat([x, p5], dim=1)                                         # 1024, 13x13
        n5 = self.n5_set(x)                                                     # 512, 13x13

        out_large = self.pred_large(self.pred_large_conv(n5))     # A*(5+C), 13x13
        out_medium = self.pred_medium(self.pred_medium_conv(n4))  # A*(5+C), 26x26
        out_small = self.pred_small(self.pred_small_conv(p3))     # A*(5+C), 52x52

        outs = []
        for out in (out_large, out_medium, out_small):
            outs.append(out.permute(0, 2, 3, 1).contiguous())  # (B,S,S,A*(5+C))
        return outs


def test(num_classes=20):
    model = YOLO_V4(num_classes=num_classes)
    x = torch.randn(2, 3, 416, 416)
    outs = model(x)
    for i, o in enumerate(outs):
        print(f"scale {i}: {o.shape}")
    # expected: scale 0: (2,13,13,75), scale 1: (2,26,26,75), scale 2: (2,52,52,75)


if __name__ == "__main__":
    test()