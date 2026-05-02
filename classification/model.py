import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetJoint(nn.Module):
    """
    U-Net with:
      - Encoder (shared)
      - Bottleneck (shared)
      - Decoder → segmentation head (foreground/background)
      - Classifier head on bottleneck → breed (37 classes)
    """
    def __init__(self, in_channels=3, seg_classes=2,
                 num_breeds=37, features=[64, 128, 256, 512]):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc_blocks = nn.ModuleList()
        self.pools      = nn.ModuleList()

        ch = in_channels
        for f in features:
            self.enc_blocks.append(ConvBlock(ch, f))
            self.pools.append(nn.MaxPool2d(2))
            ch = f

        # ── Bottleneck ───────────────────────────────────────────────────────
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)
        # Output: (B, 1024, H/16, W/16)

        # ── Decoder (segmentation) ───────────────────────────────────────────
        self.ups        = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        ch = features[-1] * 2
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(ch, f, kernel_size=2, stride=2))
            self.dec_blocks.append(ConvBlock(f * 2, f))
            ch = f

        # Segmentation output head
        self.seg_head = nn.Conv2d(features[0], seg_classes, kernel_size=1)

        # ── Classifier head (on bottleneck) ──────────────────────────────────
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(features[-1] * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_breeds),
        )

    def forward(self, x, seg=True):
        # Encoder
        skips = []
        for block, pool in zip(self.enc_blocks, self.pools):
            x = block(x)
            skips.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Classification from bottleneck
        cls_out = self.cls_head(x)

        # Skip decoder if not needed
        if not seg:
            return None, cls_out

        # Decoder
        for up, block, skip in zip(self.ups, self.dec_blocks, reversed(skips)):
            x = up(x)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = block(x)

        seg_out = self.seg_head(x)
        return seg_out, cls_out