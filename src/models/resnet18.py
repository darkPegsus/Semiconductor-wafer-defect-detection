"""Adapted ResNet-18 architecture for semiconductor wafer defect classification."""

import urllib.error
from typing import Any

import torch
import torch.nn as nn
from loguru import logger
from torchvision.models import ResNet18_Weights, resnet18


class AdaptedResNet18(nn.Module):
    """ResNet-18 adapted for single-channel or 3-channel wafer maps and 9-class output.

    Supports:
    - Pretrained ImageNet weights or randomly initialized backbone.
    - Weight migration for single-channel input: averages RGB filters across channels
      to preserve edge and spatial hierarchy features.
    - Configurable dropout in classification head.
    - Backbone freezing/unfreezing helpers for two-stage transfer learning.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 9,
        pretrained: bool = True,
        dropout_rate: float = 0.40,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.pretrained = pretrained

        # Load torchvision ResNet-18
        backbone_loaded_pretrained = False
        if pretrained:
            try:
                weights = ResNet18_Weights.DEFAULT
                self.backbone = resnet18(weights=weights)
                backbone_loaded_pretrained = True
            except (urllib.error.URLError, ConnectionError, OSError) as e:
                logger.warning(
                    f"Pretrained weights unavailable in current offline/sandboxed environment: {e}. "
                    "Falling back to randomly initialized ResNet-18 backbone."
                )
                self.backbone = resnet18(weights=None)
        else:
            self.backbone = resnet18(weights=None)

        self.pretrained = backbone_loaded_pretrained

        # Adapt first convolutional layer if in_channels != 3
        old_conv1 = self.backbone.conv1
        if in_channels != 3:
            new_conv1 = nn.Conv2d(
                in_channels,
                out_channels=old_conv1.out_channels,
                kernel_size=old_conv1.kernel_size,
                stride=old_conv1.stride,
                padding=old_conv1.padding,
                bias=old_conv1.bias is not None,
            )
            if pretrained:
                # Average weights across RGB channels: [64, 3, 7, 7] -> [64, 1, 7, 7]
                with torch.no_grad():
                    if in_channels == 1:
                        new_conv1.weight.copy_(old_conv1.weight.mean(dim=1, keepdim=True))
                    else:
                        nn.init.kaiming_normal_(
                            new_conv1.weight,
                            mode="fan_out",
                            nonlinearity="relu",
                        )
            else:
                nn.init.kaiming_normal_(new_conv1.weight, mode="fan_out", nonlinearity="relu")

            self.backbone.conv1 = new_conv1

        # Replace 1000-class ImageNet classification head with 9-class head
        in_features = self.backbone.fc.in_features  # 512
        if dropout_rate > 0.0:
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes),
            )
        else:
            self.backbone.fc = nn.Linear(in_features, num_classes)

        # If not pretrained, initialize the new linear layer
        if not pretrained:
            self._initialize_head()

    def _initialize_head(self) -> None:
        """Initialize final linear classification layer."""
        if isinstance(self.backbone.fc, nn.Sequential):
            for m in self.backbone.fc.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        elif isinstance(self.backbone.fc, nn.Linear):
            nn.init.kaiming_normal_(self.backbone.fc.weight, mode="fan_out", nonlinearity="relu")
            if self.backbone.fc.bias is not None:
                nn.init.zeros_(self.backbone.fc.bias)

    def freeze_backbone(self) -> None:
        """Freeze all layers except the adapted input conv and classification head."""
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Keep conv1 and fc trainable
        for param in self.backbone.conv1.parameters():
            param.requires_grad = True
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self) -> None:
        """Unfreeze all parameters for end-to-end fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True

    def get_parameter_groups(
        self, lr_backbone: float = 1e-4, lr_head: float = 1e-3
    ) -> list[dict[str, Any]]:
        """Return parameter groups with differential learning rates for transfer learning."""
        backbone_params = []
        head_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "fc" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

        return [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr_head},
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw unnormalized logits [B, num_classes]."""
        return self.backbone(x)
