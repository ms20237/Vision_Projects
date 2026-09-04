"""
YOLOv3 ("YOLOv3: An Incremental Improvement", Redmon & Farhadi, 2018).

Biggest structural differences vs. the YOLOv2 model this replaces:

- Backbone: Darknet-53 -- 53 conv layers with residual ("skip") connections
  instead of YOLOv2's Darknet-19. Much deeper, comparable accuracy to
  ResNet-152 at roughly 2x the speed.
- Multi-scale prediction: YOLOv3 predicts boxes at THREE different grid
  resolutions (13x13, 26x26, 52x52 for a 416 input) instead of a single
  13x13 grid, using an FPN-like top-down pathway: the coarsest feature map
  is upsampled and concatenated with an earlier, higher-resolution
  backbone feature map before each subsequent prediction. This is what
  makes YOLOv3 noticeably better at detecting small objects than YOLOv2.
- 3 anchors per scale (9 total) instead of 5 anchors at one scale.
- Still fully convolutional / anchor-based like YOLOv2, so the box
  parametrization (b_x = sigmoid(t_x) + c_x, b_w = p_w * exp(t_w), ...) is
  unchanged from YOLOv2 -- only *how many* scales it's applied at changes.
"""

import torch
import torch.nn as nn


class CNN_Block(nn.Module):
    """Conv2d -> BatchNorm -> LeakyReLU(0.1), same block as YOLOv1/v2."""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNN_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batch_nom = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leaky_relu(self.batch_nom(self.conv1(x)))


class ResidualBlock(nn.Module):
    """
    Darknet-53's basic building block: a 1x1 conv that halves the channel
    count followed by a 3x3 conv that restores it, with a skip connection
    around the pair. This is what lets Darknet-53 go to 53 layers without
    the vanishing-gradient problems that plagued earlier, plain (no-skip)
    Darknet backbones.
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        reduced = channels // 2
        self.conv1 = CNN_Block(channels, reduced, kernel_size=1, stride=1, padding=0)
        self.conv2 = CNN_Block(reduced, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class Darknet53(nn.Module):
    """
    Darknet-53 backbone. Returns the three feature maps used for
    multi-scale detection:
        route1: (B, 256,  52, 52)  -- for a 416x416 input
        route2: (B, 512,  26, 26)
        route3: (B, 1024, 13, 13)
    """

    def __init__(self, in_channels=3):
        super(Darknet53, self).__init__()

        self.conv1 = CNN_Block(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = CNN_Block(32, 64, kernel_size=3, stride=2, padding=1)
        self.res1 = nn.Sequential(*[ResidualBlock(64) for _ in range(1)])

        self.conv3 = CNN_Block(64, 128, kernel_size=3, stride=2, padding=1)
        self.res2 = nn.Sequential(*[ResidualBlock(128) for _ in range(2)])

        self.conv4 = CNN_Block(128, 256, kernel_size=3, stride=2, padding=1)
        self.res3 = nn.Sequential(*[ResidualBlock(256) for _ in range(8)])  # -> route1

        self.conv5 = CNN_Block(256, 512, kernel_size=3, stride=2, padding=1)
        self.res4 = nn.Sequential(*[ResidualBlock(512) for _ in range(8)])  # -> route2

        self.conv6 = CNN_Block(512, 1024, kernel_size=3, stride=2, padding=1)
        self.res5 = nn.Sequential(*[ResidualBlock(1024) for _ in range(4)])  # -> route3

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x)
        x = self.conv3(x)
        x = self.res2(x)
        x = self.conv4(x)
        x = self.res3(x)
        route1 = x                      # 256 ch

        x = self.conv5(x)
        x = self.res4(x)
        route2 = x                      # 512 ch

        x = self.conv6(x)
        x = self.res5(x)
        route3 = x                      # 1024 ch

        return route1, route2, route3


def _conv_set(in_channels, out_channels):
    """
    The 5-conv "detection block" used before every scale's prediction head
    and before every upsample branch: 1x1 -> 3x3 -> 1x1 -> 3x3 -> 1x1,
    alternating between `out_channels` and `out_channels * 2`.
    """
    return nn.Sequential(
        CNN_Block(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
        CNN_Block(out_channels, out_channels * 2, kernel_size=3, stride=1, padding=1),
        CNN_Block(out_channels * 2, out_channels, kernel_size=1, stride=1, padding=0),
        CNN_Block(out_channels, out_channels * 2, kernel_size=3, stride=1, padding=1),
        CNN_Block(out_channels * 2, out_channels, kernel_size=1, stride=1, padding=0),
    )


# Default anchor priors (width, height) in PIXELS of a 416x416 network
# input, grouped by the scale they're used at. These are the standard
# COCO YOLOv3 anchors (from the original Darknet yolov3.cfg), ordered to
# match this model's forward() output order: [13x13, 26x26, 52x52].
# Re-run k-means on your own dataset's box dimensions for best results
# (same "Dimension Clusters" idea as YOLOv2).
COCO_ANCHORS = [
    [(116, 90), (156, 198), (373, 326)],  # scale 1: 13x13 grid -- large objects
    [(30, 61), (62, 45), (59, 119)],       # scale 2: 26x26 grid -- medium objects
    [(10, 13), (16, 30), (33, 23)],        # scale 3: 52x52 grid -- small objects
]


class YOLO_V3(nn.Module):
    """
    Darknet-53 backbone + FPN-style top-down path + 3 detection heads.

    forward() returns a list of 3 tensors (one per scale), each shaped
    (BATCH, S, S, num_anchors_per_scale * (5 + num_classes)), largest grid
    (finest resolution, smallest objects) LAST to match the anchor
    ordering convention above... actually returned in the order
    [13x13, 26x26, 52x52] to match COCO_ANCHORS' scale order.
    """

    def __init__(self, in_channels=3, num_classes=20, anchors_per_scale=3):
        super(YOLO_V3, self).__init__()
        self.num_classes = num_classes
        self.A = anchors_per_scale
        pred_channels = self.A * (5 + num_classes)

        self.backbone = Darknet53(in_channels)

        # ---- scale 1: 13x13, straight off the backbone's deepest feature ----
        self.set1 = _conv_set(1024, 512)
        self.pred1_conv = CNN_Block(512, 1024, kernel_size=3, stride=1, padding=1)
        self.pred1 = nn.Conv2d(1024, pred_channels, kernel_size=1, stride=1, padding=0)

        self.route1_conv = CNN_Block(512, 256, kernel_size=1, stride=1, padding=0)
        self.upsample1 = nn.Upsample(scale_factor=2, mode="nearest")

        # ---- scale 2: 26x26, upsample(scale1 features) + backbone route2 ----
        self.set2 = _conv_set(256 + 512, 256)
        self.pred2_conv = CNN_Block(256, 512, kernel_size=3, stride=1, padding=1)
        self.pred2 = nn.Conv2d(512, pred_channels, kernel_size=1, stride=1, padding=0)

        self.route2_conv = CNN_Block(256, 128, kernel_size=1, stride=1, padding=0)
        self.upsample2 = nn.Upsample(scale_factor=2, mode="nearest")

        # ---- scale 3: 52x52, upsample(scale2 features) + backbone route1 ----
        self.set3 = _conv_set(128 + 256, 128)
        self.pred3_conv = CNN_Block(128, 256, kernel_size=3, stride=1, padding=1)
        self.pred3 = nn.Conv2d(256, pred_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        route1, route2, route3 = self.backbone(x)   # 256@52, 512@26, 1024@13

        x = self.set1(route3)                          # 512, 13x13
        out1 = self.pred1(self.pred1_conv(x))           # A*(5+C), 13x13

        x = self.route1_conv(x)                          # 256, 13x13
        x = self.upsample1(x)                              # 256, 26x26
        x = torch.cat([x, route2], dim=1)                    # 768, 26x26

        x = self.set2(x)                                      # 256, 26x26
        out2 = self.pred2(self.pred2_conv(x))                  # A*(5+C), 26x26

        x = self.route2_conv(x)                                  # 128, 26x26
        x = self.upsample2(x)                                      # 128, 52x52
        x = torch.cat([x, route1], dim=1)                            # 384, 52x52

        x = self.set3(x)                                              # 128, 52x52
        out3 = self.pred3(self.pred3_conv(x))                          # A*(5+C), 52x52

        outs = []
        for out in (out1, out2, out3):
            # (B, A*(5+C), S, S) -> (B, S, S, A*(5+C))
            outs.append(out.permute(0, 2, 3, 1).contiguous())
        return outs


def test(num_classes=20):
    model = YOLO_V3(num_classes=num_classes)
    x = torch.randn(2, 3, 416, 416)
    outs = model(x)
    for i, o in enumerate(outs):
        print(f"scale {i}: {o.shape}")
    # expected: scale 0: (2,13,13,75), scale 1: (2,26,26,75), scale 2: (2,52,52,75)
    # (75 = 3 anchors * (5 + 20 classes))


if __name__ == "__main__":
    test()