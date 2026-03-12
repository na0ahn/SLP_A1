"""
Depthwise Separable CNN (DS-CNN) for Keyword Spotting.

Reference: "Hello Edge: Keyword Spotting on Microcontrollers" (Zhang et al. 2018)
           Extended with residual connections for better accuracy.

Input:  (B, 1, 80, 49) - batch of log-mel spectrograms
Output: (B, 12)        - class logits

Target: ~750K parameters, well below 2.5M limit.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DSBlock(nn.Module):
    """
    Depthwise Separable Conv block with optional residual connection.

    DW conv → BN → ReLU → PW conv → BN → ReLU
    Optional skip: 1×1 conv to match channels/stride
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_residual: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.use_residual = use_residual and (stride == 1)

        # Depthwise
        self.dw_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3,
            stride=stride, padding=1, groups=in_channels, bias=False
        )
        self.dw_bn = nn.BatchNorm2d(in_channels)

        # Pointwise
        self.pw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_channels)

        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # Residual shortcut (only when dimensions match or can be projected)
        if use_residual:
            if stride == 1 and in_channels == out_channels:
                self.shortcut = nn.Identity()
                self.use_residual = True
            elif stride == 1 and in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                )
                self.use_residual = True
            else:
                self.shortcut = None
                self.use_residual = False
        else:
            self.shortcut = None
            self.use_residual = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # Depthwise
        out = self.act(self.dw_bn(self.dw_conv(x)))
        # Pointwise
        out = self.pw_bn(self.pw_conv(out))
        out = self.dropout(out)

        # Residual
        if self.use_residual and self.shortcut is not None:
            out = out + self.shortcut(identity)
        elif self.use_residual:
            out = out + identity

        return self.act(out)


class DSCNN(nn.Module):
    """
    DS-CNN model with residual connections for KWS.

    Architecture:
      - Stem: standard Conv2d (1 → 64)
      - 6 DS blocks with progressive channel widening
      - Global Average Pooling
      - Dropout
      - Linear classifier

    Input:  (B, 1, 80, 49)
    Output: (B, n_classes)
    """

    def __init__(
        self,
        n_classes: int = 12,
        dropout: float = 0.2,
        channels: list = None,
        use_residual: bool = True,
    ):
        super().__init__()

        if channels is None:
            channels = [64, 128, 256, 256, 512]

        # Stem: (B, 1, 80, 49) → (B, channels[0], 40, 24)
        # Conv2d with stride 2 in both dims
        self.stem = nn.Sequential(
            nn.Conv2d(1, channels[0], kernel_size=(10, 4), stride=2, padding=(4, 1), bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )
        # After stem with input (1, 80, 49):
        # H_out = floor((80 + 2*4 - 10) / 2) + 1 = floor(78/2) + 1 = 40
        # W_out = floor((49 + 2*1 - 4) / 2) + 1 = floor(47/2) + 1 = 24
        # → (channels[0], 40, 24)

        # DS blocks: progressive downsampling and channel widening
        # 8 blocks: (in_ch, out_ch, stride)
        block_configs = [
            (channels[0], channels[0], 1),         # keep: (C0, 40, 24)
            (channels[0], channels[1], 2),          # down: (C1, 20, 12)
            (channels[1], channels[1], 1),          # keep: (C1, 20, 12)
            (channels[1], channels[2], 2),          # down: (C2, 10, 6)
            (channels[2], channels[2], 1),          # keep: (C2, 10, 6)
            (channels[2], channels[3], 1),          # keep: (C3, 10, 6)
            (channels[3], channels[4], 2),          # down: (C4, 5, 3)
            (channels[4], channels[4], 1),          # keep: (C4, 5, 3)
        ]

        self.blocks = nn.ModuleList()
        for in_c, out_c, stride in block_configs:
            self.blocks.append(
                DSBlock(
                    in_c, out_c, stride=stride,
                    use_residual=use_residual,
                    dropout=0.0,
                )
            )

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(channels[4], n_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, 80, 49) log-mel spectrogram

        Returns:
            logits: (B, n_classes)
        """
        x = self.stem(x)

        for block in self.blocks:
            x = block(x)

        x = self.gap(x)          # (B, 512, 1, 1)
        x = x.flatten(1)         # (B, 512)
        x = self.dropout(x)
        logits = self.classifier(x)  # (B, 12)

        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_dscnn(cfg: dict) -> DSCNN:
    """Build DS-CNN from config."""
    model_cfg = cfg.get("model", {})
    dscnn_cfg = model_cfg.get("dscnn", {})

    model = DSCNN(
        n_classes=model_cfg.get("n_classes", 12),
        dropout=model_cfg.get("dropout", 0.2),
        channels=dscnn_cfg.get("channels", [64, 128, 256, 256, 512]),
        use_residual=dscnn_cfg.get("use_residual", True),
    )

    return model


if __name__ == "__main__":
    model = DSCNN()
    x = torch.randn(2, 1, 80, 49)
    y = model(x)
    print(f"Input: {x.shape} → Output: {y.shape}")
    n_params = model.count_parameters()
    print(f"Trainable parameters: {n_params:,}")
    assert n_params <= 2_500_000, f"Parameter count {n_params} exceeds 2.5M limit!"
    print("Parameter count check PASSED.")
