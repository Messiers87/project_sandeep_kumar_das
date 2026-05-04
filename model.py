# =============================================================================
# Provides:
#   • GravitationalLensCNN    – custom baseline CNN  (primary model)
#   • GravitationalLensResNet – ResNet-18 transfer-learning variant (BONUS)
# =============================================================================

import torch
import torch.nn as nn
import torchvision.models as models

import config


# ---------------------------------------------------------------------------
# Helper: Convolutional Block
# ---------------------------------------------------------------------------

def _conv_block(in_ch: int, out_ch: int, kernel: int = 3,
                padding: int = 1) -> nn.Sequential:
    """Conv → BatchNorm → ReLU → MaxPool block."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),   # halves spatial dims
    )


# ---------------------------------------------------------------------------
# Primary Model: Custom CNN Baseline
# ---------------------------------------------------------------------------

class GravitationalLensCNN(nn.Module):
    """
    Baseline CNN for 3-class gravitational lens substructure classification.

    Architecture
    ────────────
    Input : (B, 1, 150, 150)

    Feature Extractor
      Block 1 : Conv(1→32)   → BN → ReLU → MaxPool2d  →  (B, 32,  75,  75)
      Block 2 : Conv(32→64)  → BN → ReLU → MaxPool2d  →  (B, 64,  37,  37)
      Block 3 : Conv(64→128) → BN → ReLU → MaxPool2d  →  (B, 128, 18,  18)
      Block 4 : Conv(128→256)→ BN → ReLU → MaxPool2d  →  (B, 256,  9,   9)
      Block 5 : Conv(256→256)→ BN → ReLU → MaxPool2d  →  (B, 256,  4,   4)

    Classifier
      AdaptiveAvgPool2d(4,4) → Flatten → FC(4096→512) → BN → ReLU → Dropout
      → FC(512→128) → ReLU → Dropout → FC(128→3)

    Output : (B, 3)  raw logits  (use CrossEntropyLoss during training)
    """

    def __init__(
        self,
        num_classes:    int = config.num_classes,
        input_channels: int = config.input_channels,
        dropout_rate:   float = 0.4,
    ):
        super().__init__()

        # ── Feature extraction ──────────────────────────────────────────
        self.features = nn.Sequential(
            _conv_block(input_channels, 32),   # 150 → 75
            _conv_block(32,  64),              #  75 → 37
            _conv_block(64,  128),             #  37 → 18
            _conv_block(128, 256),             #  18 →  9
            _conv_block(256, 256),             #   9 →  4
        )

        # ── Global spatial aggregation ───────────────────────────────────
        self.pool = nn.AdaptiveAvgPool2d((4, 4))   # ensures fixed (4,4) output

        # ── Classifier head ──────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),                          # 256 * 4 * 4 = 4096
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(128, num_classes),           # raw logits
        )

        # ── Weight initialisation ─────────────────────────────────────────
        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        """Kaiming He init for Conv/Linear; constant init for BN."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 1, H, W)  float32

        Returns
        -------
        logits : (B, num_classes)  float32
        """
        x = self.features(x)    # (B, 256, 4, 4)
        x = self.pool(x)        # (B, 256, 4, 4)  – no-op if already 4×4
        x = self.classifier(x)  # (B, 3)
        return x


# ---------------------------------------------------------------------------
# BONUS Model: ResNet-18 Transfer Learning
# ---------------------------------------------------------------------------

class GravitationalLensResNet(nn.Module):
    """
    Transfer-learning variant based on torchvision ResNet-18.

    Modifications
    ─────────────
    • First conv replaced: kernel 7→3, stride 2→1, padding 3→1
      so that 150-px grayscale images are not over-downsampled.
    • Input stem adapted for 1-channel (grayscale) input.
    • Final FC layer replaced: 512 → num_classes.
    • Optionally freeze backbone (feature_extract=True).

    Output : (B, num_classes)  raw logits
    """

    def __init__(
        self,
        num_classes:     int  = config.num_classes,
        input_channels:  int  = config.input_channels,
        pretrained:      bool = True,
        feature_extract: bool = False,   # True → freeze backbone weights
    ):
        super().__init__()

        # Load pretrained ResNet-18 backbone
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        # ── Adapt first conv for grayscale ───────────────────────────────
        orig_conv = backbone.conv1   # (64, 3, 7, 7)
        new_conv  = nn.Conv2d(
            in_channels  = input_channels,
            out_channels = orig_conv.out_channels,
            kernel_size  = 3,
            stride       = 1,
            padding      = 1,
            bias         = False,
        )
        # If pretrained, initialise new conv by averaging RGB weights
        if pretrained:
            with torch.no_grad():
                # Average R,G,B weight tensors along channel dim
                new_conv.weight.copy_(
                    orig_conv.weight.mean(dim=1, keepdim=True)
                )
        backbone.conv1 = new_conv

        # Remove original maxpool to preserve spatial resolution for small imgs
        backbone.maxpool = nn.Identity()

        # ── Optionally freeze backbone ────────────────────────────────────
        if feature_extract:
            for param in backbone.parameters():
                param.requires_grad = False

        # ── Replace classifier head ───────────────────────────────────────
        in_features = backbone.fc.in_features   # 512 for resnet18
        backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

        self.model = backbone

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dummy = torch.zeros(2, config.input_channels, config.resize_y, config.resize_x)

    cnn   = GravitationalLensCNN()
    resnet = GravitationalLensResNet(pretrained=False)

    out_cnn    = cnn(dummy)
    out_resnet = resnet(dummy)

    print(f"GravitationalLensCNN    output shape : {out_cnn.shape}")
    print(f"GravitationalLensResNet output shape : {out_resnet.shape}")

    total_params = sum(p.numel() for p in cnn.parameters() if p.requires_grad)
    print(f"Baseline CNN trainable parameters    : {total_params:,}")
