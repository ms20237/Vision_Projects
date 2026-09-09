"""
YOLOv5 (Ultralytics, https://github.com/ultralytics/yolov5). Unlike v1-v4,
YOLOv5 was never published as an academic paper -- this follows the
architecture of the public Ultralytics implementation (specifically the
"v6.0+" style config: Conv-stem backbone, C3 blocks, SPPF).

Structural differences vs. the YOLOv4 model this replaces:

- Activation: SiLU (x * sigmoid(x), a.k.a. Swish) everywhere -- backbone
  AND neck/head. YOLOv4 only used Mish in the backbone and kept LeakyReLU
  in the neck/head; YOLOv5 is SiLU throughout.
- Backbone building block: `C3` ("CSP Bottleneck with 3 convolutions")
  replaces YOLOv4's hand-rolled `CSPBlock`. It's the same split/process/
  concat idea, but simpler: two parallel 1x1 convs (one straight through,
  one feeding a stack of residual `Bottleneck` units), concatenated and
  fused with a final 1x1 conv.
- Stem: a single 6x6 stride-2 conv (this replay of the "Focus" layer from
  early YOLOv5 versions is mathematically similar but simpler and faster;
  Ultralytics switched to it in v6.0) instead of Darknet-style stacked 3x3
  convs.
- SPPF replaces SPP: instead of 3 *parallel* max-pools at different kernel
  sizes (5, 9, 13) concatenated together, SPPF chains 3 *sequential* 5x5
  max-pools and concatenates all the intermediate outputs. Mathematically
  near-identical receptive field, meaningfully faster.
- Width/depth multipliers: channel counts and block-repeat counts are
  scaled by `width_mult` / `depth_mult`, matching how Ultralytics derives
  the n/s/m/l/x model family from one architecture description. This file
  defaults to the "s" (small) configuration.
- Neck is still PANet (top-down + bottom-up), same overall topology as
  YOLOv4's neck, just built from `C3` blocks instead of the 5-conv sets.
- Still anchor-based, multi-scale (13x13/26x26/52x52 for 416 input), 3
  anchors per scale -- box parametrization and detection head shape are
  unchanged from YOLOv3/v4.
"""

import math
import torch
import torch.nn as nn


def make_divisible(x, divisor=8):
    """Round a channel count up to the nearest multiple of `divisor`,
    matching Ultralytics' channel-scaling rule (keeps every layer's
    channel count hardware-friendly after applying width_mult)."""
    return int(math.ceil(x / divisor) * divisor)


class Conv(nn.Module):
    """Conv2d -> BatchNorm -> SiLU. The one conv block used everywhere in
    YOLOv5 -- backbone, neck, and head all share this (unlike YOLOv4,
    which used Mish in the backbone and LeakyReLU elsewhere)."""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, groups=1):
        super(Conv, self).__init__()
        if padding is None:
            padding = kernel_size // 2  # "same" padding for odd kernel sizes
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard residual bottleneck used inside C3: 1x1 conv reduces
    channels, 3x3 conv restores them, optional skip connection."""

    def __init__(self, channels, shortcut=True, expansion=0.5):
        super(Bottleneck, self).__init__()
        hidden = int(channels * expansion)
        self.cv1 = Conv(channels, hidden, kernel_size=1, stride=1)
        self.cv2 = Conv(hidden, channels, kernel_size=3, stride=1)
        self.add = shortcut

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C3(nn.Module):
    """
    "CSP Bottleneck with 3 convolutions" -- YOLOv5's version of a CSP
    stage, and simpler than YOLOv4's CSPBlock: the input feeds two
    parallel 1x1 convs (`cv1` into a stack of `n` Bottleneck units, `cv2`
    straight through as a shortcut), the two branches are concatenated,
    and a final 1x1 conv (`cv3`) fuses them back to `out_channels`.
    """

    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5):
        super(C3, self).__init__()
        hidden = int(out_channels * expansion)
        self.cv1 = Conv(in_channels, hidden, kernel_size=1, stride=1)
        self.cv2 = Conv(in_channels, hidden, kernel_size=1, stride=1)
        self.m = nn.Sequential(*[Bottleneck(hidden, shortcut=shortcut) for _ in range(n)])
        self.cv3 = Conv(hidden * 2, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        return self.cv3(torch.cat([self.m(self.cv1(x)), self.cv2(x)], dim=1))


class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling - Fast. Replaces YOLOv4's SPP (3 parallel
    max-pools at kernel sizes 5/9/13) with 3 *sequential* 5x5 max-pools,
    concatenating the input and every intermediate pooled result. This
    reaches the same effective receptive field as SPP's larger kernels
    (two 5x5 pools in a row see roughly as far as one 9x9 pool; three in a
    row roughly as far as one 13x13 pool) while being noticeably faster.
    """

    def __init__(self, in_channels, out_channels, kernel_size=5):
        super(SPPF, self).__init__()
        hidden = in_channels // 2
        self.cv1 = Conv(in_channels, hidden, kernel_size=1, stride=1)
        self.cv2 = Conv(hidden * 4, out_channels, kernel_size=1, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


class CSPDarknetV5(nn.Module):
    """
    YOLOv5's backbone: a Conv stem followed by 4 downsample-then-C3
    stages, then SPPF. Channel counts and C3 repeat counts are scaled by
    `width_mult` / `depth_mult` (defaults reproduce YOLOv5s).

    Returns the three routed feature maps used by the PANet neck:
        route1 (P3): stride 8,  (B, w(256),  52, 52) for a 416 input
        route2 (P4): stride 16, (B, w(512),  26, 26)
        route3 (P5): stride 32, (B, w(1024), 13, 13), after SPPF
    """

    def __init__(self, in_channels=3, width_mult=0.5, depth_mult=0.33):
        super(CSPDarknetV5, self).__init__()

        def w(c):
            return make_divisible(c * width_mult)

        def d(n):
            return max(round(n * depth_mult), 1)

        self.stem = Conv(in_channels, w(64), kernel_size=6, stride=2, padding=2)          # P1/2

        self.stage1 = nn.Sequential(
            Conv(w(64), w(128), kernel_size=3, stride=2),                                  # P2/4
            C3(w(128), w(128), n=d(3)),
        )
        self.stage2 = nn.Sequential(
            Conv(w(128), w(256), kernel_size=3, stride=2),                                 # P3/8
            C3(w(256), w(256), n=d(6)),
        )
        self.stage3 = nn.Sequential(
            Conv(w(256), w(512), kernel_size=3, stride=2),                                 # P4/16
            C3(w(512), w(512), n=d(9)),
        )
        self.stage4 = nn.Sequential(
            Conv(w(512), w(1024), kernel_size=3, stride=2),                                # P5/32
            C3(w(1024), w(1024), n=d(3)),
        )
        self.sppf = SPPF(w(1024), w(1024), kernel_size=5)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        route1 = x                       # P3, stride 8

        x = self.stage3(x)
        route2 = x                       # P4, stride 16

        x = self.stage4(x)
        x = self.sppf(x)
        route3 = x                       # P5, stride 32

        return route1, route2, route3


# Default anchor priors, same convention and values as the YOLOv3/v4
# versions of this file: (width, height) in pixels of a 416x416 input,
# grouped by scale. Re-run k-means (Ultralytics calls this "AutoAnchor")
# on your own dataset for best results.
COCO_ANCHORS = [
    [(116, 90), (156, 198), (373, 326)],  # scale 1: 13x13 grid -- large objects
    [(30, 61), (62, 45), (59, 119)],       # scale 2: 26x26 grid -- medium objects
    [(10, 13), (16, 30), (33, 23)],        # scale 3: 52x52 grid -- small objects
]


class YOLO_V5(nn.Module):
    """
    CSPDarknetV5 backbone + SPPF + PANet neck (built from C3 blocks) + 3
    detection heads.

    forward() returns a list of 3 tensors, in the SAME order as
    COCO_ANCHORS: [13x13 (large objects), 26x26 (medium), 52x52 (small)],
    each shaped (BATCH, S, S, num_anchors_per_scale * (5 + num_classes)).

    width_mult / depth_mult default to 0.5 / 0.33, reproducing YOLOv5s.
    Common presets (matching Ultralytics' model family):
        n: width=0.25, depth=0.33      s: width=0.50, depth=0.33 (default)
        m: width=0.75, depth=0.67      l: width=1.00, depth=1.00
        x: width=1.25, depth=1.33
    """

    def __init__(self, in_channels=3, num_classes=20, anchors_per_scale=3,
                 width_mult=0.5, depth_mult=0.33):
        super(YOLO_V5, self).__init__()
        self.num_classes = num_classes
        self.A = anchors_per_scale
        pred_channels = self.A * (5 + num_classes)

        def w(c):
            return make_divisible(c * width_mult)

        def d(n):
            return max(round(n * depth_mult), 1)

        self.backbone = CSPDarknetV5(in_channels, width_mult, depth_mult)

        c3, c4, c5 = w(256), w(512), w(1024)  # channel counts of route1/route2/route3

        # ---- top-down path (FPN-style, like YOLOv4's neck) ----
        self.reduce5 = Conv(c5, c4, kernel_size=1, stride=1)         # "layer10" in ultralytics yaml terms
        self.upsample1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.c3_p4 = C3(c4 * 2, c4, n=d(3), shortcut=False)            # -> P4-fused

        self.reduce4 = Conv(c4, c3, kernel_size=1, stride=1)         # "layer14"
        self.upsample2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.c3_p3 = C3(c3 * 2, c3, n=d(3), shortcut=False)            # -> P3-fused (finest, "small" head)

        # ---- bottom-up path (PANet) ----
        self.downsample1 = Conv(c3, c3, kernel_size=3, stride=2)
        self.c3_n4 = C3(c3 * 2, c4, n=d(3), shortcut=False)             # -> N4 ("medium" head)

        self.downsample2 = Conv(c4, c4, kernel_size=3, stride=2)
        self.c3_n5 = C3(c4 * 2, c5, n=d(3), shortcut=False)             # -> N5 ("large" head)

        # ---- prediction heads (bare 1x1 convs, no BN/activation -- same as v1-v4) ----
        self.pred_large = nn.Conv2d(c5, pred_channels, kernel_size=1, stride=1, padding=0)
        self.pred_medium = nn.Conv2d(c4, pred_channels, kernel_size=1, stride=1, padding=0)
        self.pred_small = nn.Conv2d(c3, pred_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        route1, route2, route3 = self.backbone(x)   # c3@52 (P3/8), c4@26 (P4/16), c5@13 (P5/32)

        p5r = self.reduce5(route3)                     # c4, 13x13
        x = self.upsample1(p5r)                          # c4, 26x26
        x = torch.cat([x, route2], dim=1)                  # c4*2, 26x26
        p4 = self.c3_p4(x)                                   # c4, 26x26

        p4r = self.reduce4(p4)                                # c3, 26x26
        x = self.upsample2(p4r)                                 # c3, 52x52
        x = torch.cat([x, route1], dim=1)                         # c3*2, 52x52
        p3 = self.c3_p3(x)                                          # c3, 52x52 (finest -- "small" head)

        x = self.downsample1(p3)                                     # c3, 26x26
        x = torch.cat([x, p4r], dim=1)                                 # c3*2, 26x26
        n4 = self.c3_n4(x)                                               # c4, 26x26 ("medium" head)

        x = self.downsample2(n4)                                          # c4, 13x13
        x = torch.cat([x, p5r], dim=1)                                      # c4*2, 13x13
        n5 = self.c3_n5(x)                                                    # c5, 13x13 ("large" head)

        out_large = self.pred_large(n5)      # A*(5+C), 13x13
        out_medium = self.pred_medium(n4)    # A*(5+C), 26x26
        out_small = self.pred_small(p3)      # A*(5+C), 52x52

        outs = []
        for out in (out_large, out_medium, out_small):
            outs.append(out.permute(0, 2, 3, 1).contiguous())  # (B,S,S,A*(5+C))
        return outs


def test(num_classes=20):
    model = YOLO_V5(num_classes=num_classes)  # defaults to YOLOv5s sizing
    x = torch.randn(2, 3, 416, 416)
    outs = model(x)
    for i, o in enumerate(outs):
        print(f"scale {i}: {o.shape}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.2f}M")
    # expected: scale 0: (2,13,13,75), scale 1: (2,26,26,75), scale 2: (2,52,52,75)


if __name__ == "__main__":
    test()