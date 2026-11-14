"""Unit tests for Custom CNN and Adapted ResNet-18 model architectures."""

import pytest
import torch

from src.models.custom_cnn import CustomWaferCNN
from src.models.resnet18 import AdaptedResNet18


@pytest.mark.parametrize("in_channels", [1, 3])
@pytest.mark.parametrize("batch_size", [2, 4])
def test_custom_cnn_forward_shapes(in_channels: int, batch_size: int):
    """Verify Custom CNN forward pass produces [B, 9] raw logits for single and 3-channel inputs."""
    model = CustomWaferCNN(in_channels=in_channels, num_classes=9)
    x = torch.randn(batch_size, in_channels, 64, 64)

    logits = model(x)
    assert logits.shape == (batch_size, 9)
    assert logits.dtype == torch.float32

    # Verify outputs are unnormalized raw logits (not softmax probabilities)
    assert not torch.allclose(logits.sum(dim=-1), torch.ones(batch_size), atol=1e-3)


def test_custom_cnn_backward_pass():
    """Verify loss calculation and gradient propagation through all trainable parameters."""
    model = CustomWaferCNN(in_channels=1, num_classes=9)
    x = torch.randn(4, 1, 64, 64)
    y = torch.tensor([0, 2, 4, 8], dtype=torch.long)

    criterion = torch.nn.CrossEntropyLoss()
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()

    # Check gradients exist and are non-zero across conv layers
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"
        assert not torch.all(param.grad == 0), f"Parameter {name} has all-zero gradients"


@pytest.mark.parametrize("in_channels", [1, 3])
@pytest.mark.parametrize("pretrained", [False, True])
def test_adapted_resnet18_forward_shapes(in_channels: int, pretrained: bool):
    """Verify Adapted ResNet-18 supports 1-channel, 3-channel, pretrained, and random init."""
    model = AdaptedResNet18(in_channels=in_channels, num_classes=9, pretrained=pretrained)
    x = torch.randn(2, in_channels, 64, 64)

    logits = model(x)
    assert logits.shape == (2, 9)


def test_adapted_resnet18_fine_tuning_freeze():
    """Verify backbone freezing and parameter groups for two-stage transfer learning."""
    model = AdaptedResNet18(in_channels=1, num_classes=9, pretrained=False)

    # Freeze backbone
    model.freeze_backbone()
    assert model.backbone.conv1.weight.requires_grad is True
    # layer1 should be frozen
    for param in model.backbone.layer1.parameters():
        assert param.requires_grad is False
    # fc should be trainable
    for param in model.backbone.fc.parameters():
        assert param.requires_grad is True

    # Unfreeze
    model.unfreeze_backbone()
    for param in model.parameters():
        assert param.requires_grad is True


def test_model_reproducibility():
    """Verify identical random seeds construct identical initial weights."""
    torch.manual_seed(42)
    model1 = CustomWaferCNN(in_channels=1, num_classes=9)

    torch.manual_seed(42)
    model2 = CustomWaferCNN(in_channels=1, num_classes=9)

    for p1, p2 in zip(model1.parameters(), model2.parameters(), strict=True):
        assert torch.all(p1 == p2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device not available")
def test_models_cuda_compatibility():
    """Verify forward and backward execution on NVIDIA GPU."""
    device = torch.device("cuda")

    # Custom CNN on CUDA
    cnn = CustomWaferCNN(in_channels=1, num_classes=9).to(device)
    x = torch.randn(4, 1, 64, 64, device=device)
    out = cnn(x)
    assert out.is_cuda
    loss = out.sum()
    loss.backward()

    # ResNet-18 on CUDA
    resnet = AdaptedResNet18(in_channels=1, num_classes=9, pretrained=False).to(device)
    out_res = resnet(x)
    assert out_res.is_cuda
    loss_res = out_res.sum()
    loss_res.backward()
