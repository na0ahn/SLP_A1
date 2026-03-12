"""
BC-ResNet (Broadcasted Residual Network) for Keyword Spotting.

Reference: "Broadcasted Residual Learning for Efficient Keyword Spotting"
           Kim & Kim, 2021

Key ideas:
  - Residual shortcuts using frequency-axis 1D conv (broadcast over time)
  - Main path: depthwise separable 2D conv
  - Very parameter-efficient

This implements BC-ResNet-Ext (extended channels) for better accuracy.

Input:  (B, 1, 80, 49)
Output: (B, 12)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SubSpectralNorm(nn.Module):
    """Sub-Spectral Normalization: normalize within sub-bands."""

    def __init__(self, n_groups: int = 8):
        super().__init__()
        self.n_groups = n_groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, F, T)
        B, C, F, T = x.shape
        g = self.n_groups
        # Normalize over (F/g, T) for each group
        x = x.view(B, C, g, F // g, T)
        x = F.layer_norm(x, [F // g, T])
        return x.view(B, C, F, T)


class BCResBlock(nn.Module):
    """
    BC-ResNet block.

    Two-path design:
      1. Frequency sub-sampling path (broadcasts over time) → shortcut
      2. Depthwise-separable 2D conv path → main

    Both paths added together.
    """

    def __init__(self, in_channels: int, out_channels: int, stride_f: int = 1):
        super().__init__()
        self.stride_f = stride_f

        # ── Frequency branch (shortcut with temporal averaging) ─────────────
        # Average over time → 1D freq conv → broadcast back
        self.freq_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 1),
                      stride=(stride_f, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # ── Main branch (2D depthwise-separable) ────────────────────────────
        self.dw_conv = nn.Conv2d(
            in_channels, in_channels, (3, 3),
            stride=(stride_f, 1), padding=(1, 1),
            groups=in_channels, bias=False,
        )
        self.dw_bn = nn.BatchNorm2d(in_channels)

        self.pw_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_channels)

        self.act = nn.SiLU(inplace=True)

        # Projection for shortcut if channels differ and no stride
        if in_channels != out_channels and stride_f == 1:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── Frequency branch ──────────────────────────────────────────────
        # Pool over time, apply freq conv, then broadcast
        x_avg = x.mean(dim=-1, keepdim=True)  # (B, C, F, 1)
        x_freq = self.freq_conv(x_avg)         # (B, out_C, F', 1)
        x_freq = x_freq.expand(-1, -1, -1, x.shape[-1])  # broadcast T

        # ── Main branch ───────────────────────────────────────────────────
        out = self.act(self.dw_bn(self.dw_conv(x)))
        out = self.pw_bn(self.pw_conv(out))

        # ── Combine ───────────────────────────────────────────────────────
        out = out + x_freq
        return self.act(out)


class BCResNet(nn.Module):
    """
    BC-ResNet for KWS.

    Architecture based on BC-ResNet-8 with extended channels.

    Input:  (B, 1, 80, 49)
    Output: (B, n_classes)
    """

    def __init__(self, n_classes: int = 12, dropout: float = 0.2):
        super().__init__()

        # Stem: (B, 1, 80, 49) → (B, 32, 40, 49)
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, (5, 5), stride=(2, 1), padding=(2, 2), bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )

        # Stage 1: 32 → 64 channels, freq stride=2
        # (B, 32, 40, 49) → (B, 64, 20, 49)
        self.stage1 = nn.Sequential(
            BCResBlock(32, 64, stride_f=2),
            BCResBlock(64, 64, stride_f=1),
        )

        # Stage 2: 64 → 128 channels, freq stride=2
        # (B, 64, 20, 49) → (B, 128, 10, 49)
        self.stage2 = nn.Sequential(
            BCResBlock(64, 128, stride_f=2),
            BCResBlock(128, 128, stride_f=1),
        )

        # Stage 3: 128 → 256 channels, freq stride=2
        # (B, 128, 10, 49) → (B, 256, 5, 49)
        self.stage3 = nn.Sequential(
            BCResBlock(128, 256, stride_f=2),
            BCResBlock(256, 256, stride_f=1),
        )

        # Stage 4: 256 → 256 channels, no stride
        # (B, 256, 5, 49) → (B, 256, 5, 49)
        self.stage4 = nn.Sequential(
            BCResBlock(256, 256, stride_f=1),
            BCResBlock(256, 256, stride_f=1),
        )

        # Head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(256, n_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.classifier(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_bcresnet(cfg: dict) -> BCResNet:
    """Build BC-ResNet from config."""
    model_cfg = cfg.get("model", {})
    return BCResNet(
        n_classes=model_cfg.get("n_classes", 12),
        dropout=model_cfg.get("dropout", 0.2),
    )


if __name__ == "__main__":
    model = BCResNet()
    x = torch.randn(2, 1, 80, 49)
    y = model(x)
    print(f"Input: {x.shape} → Output: {y.shape}")
    n_params = model.count_parameters()
    print(f"Trainable parameters: {n_params:,}")
    assert n_params <= 2_500_000, f"Parameter count {n_params} exceeds 2.5M limit!"
    print("Parameter count check PASSED.")
