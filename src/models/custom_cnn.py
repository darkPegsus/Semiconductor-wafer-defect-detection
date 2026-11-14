"""Custom 4-Block Convolutional Neural Network baseline for wafer defect classification."""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Convolutional block: Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        pool_size: int = 2,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_size, stride=pool_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CustomWaferCNN(nn.Module):
    """Custom 4-Block CNN baseline architecture for 64x64 semiconductor wafer maps.

    Features:
    - 4 ConvBlocks expanding channels: in_channels -> 32 -> 64 -> 128 -> 256
    - Receptive field reduces spatial dimension: 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4
    - Adaptive average pooling to (2, 2)
    - Fully connected classification head with Dropout regularization
    - Returns unnormalized raw logits [B, num_classes]
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 9,
        dropout_rate: float = 0.30,
        fc_dim: int = 128,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Feature extractor blocks
        self.block1 = ConvBlock(in_channels, 32, kernel_size=3, padding=1)
        self.block2 = ConvBlock(32, 64, kernel_size=3, padding=1)
        self.block3 = ConvBlock(64, 128, kernel_size=3, padding=1)
        self.block4 = ConvBlock(128, 256, kernel_size=3, padding=1)

        # Adaptive pooling
        self.pool = nn.AdaptiveAvgPool2d((2, 2))

        # Classifier head
        in_features = 256 * 2 * 2  # 1024
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.67),
            nn.Linear(fc_dim, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Apply Kaiming normal initialization to conv layers and zero bias for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feature extractor blocks."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw unnormalized logits [B, num_classes]."""
        feat = self.extract_features(x)
        pooled = self.pool(feat)
        logits = self.classifier(pooled)
        return logits
