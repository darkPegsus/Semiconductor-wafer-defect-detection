"""Unit tests for training infrastructure: loss functions, samplers, early stopping, and metrics."""

import numpy as np
import pytest
import torch

from src.models.custom_cnn import CustomWaferCNN
from src.training.early_stopping import EarlyStopping
from src.training.losses import (
    compute_class_weights,
    get_loss_function,
)
from src.training.metrics import compute_metrics
from src.training.sampler import create_weighted_random_sampler
from src.training.scheduler import create_lr_scheduler


def test_compute_class_weights():
    """Verify class weights are inversely proportional to class frequencies."""
    # Class 0: 100 samples, Class 1: 10 samples
    labels = [0] * 100 + [1] * 10
    weights = compute_class_weights(labels, num_classes=2, mode="balanced")

    assert weights.shape == (2,)
    # Minority class (1) must receive higher weight than majority class (0)
    assert weights[1] > weights[0]
    assert weights[1].item() == pytest.approx(10.0 * weights[0].item(), rel=1e-2)


def test_loss_functions():
    """Verify loss calculations for standard CE, weighted CE, and MultiClassFocalLoss."""
    logits = torch.tensor([[2.0, -1.0, 0.0], [-1.0, 3.0, 0.0]], dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.long)

    # 1. Standard Cross-Entropy
    ce_fn = get_loss_function("cross_entropy")
    ce_loss = ce_fn(logits, targets)
    assert ce_loss.item() > 0.0

    # 2. Weighted Cross-Entropy
    weights = torch.tensor([1.0, 2.0, 1.0], dtype=torch.float32)
    w_ce_fn = get_loss_function("weighted_cross_entropy", class_weights=weights)
    w_ce_loss = w_ce_fn(logits, targets)
    assert w_ce_loss.item() > 0.0

    # 3. MultiClassFocalLoss
    focal_fn = get_loss_function("focal_loss", gamma=2.0)
    focal_loss = focal_fn(logits, targets)
    assert focal_loss.item() > 0.0

    # Focal loss should be lower than standard CE on well-classified samples due to modulating factor (1 - p_t)^gamma
    assert focal_loss.item() < ce_loss.item()


def test_weighted_random_sampler():
    """Verify weighted random sampler creates appropriate sample weights."""
    labels = np.array([0, 0, 0, 0, 1])  # Class 0 has 4, Class 1 has 1
    sampler = create_weighted_random_sampler(labels, num_classes=2)

    assert len(sampler) == 5
    weights = sampler.weights.numpy()
    # Minority sample (index 4) should have 4x the weight of majority samples
    assert weights[4] == pytest.approx(4.0 * weights[0], rel=1e-3)


def test_lr_scheduler_creation():
    """Verify learning rate scheduler factory."""
    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    scheduler = create_lr_scheduler(optimizer, name="CosineAnnealingLR", epochs=10, warmup_epochs=2)
    assert scheduler is not None


def test_early_stopping_and_checkpoint(tmp_path):
    """Verify checkpoint serialization, recovery, and early stopping trigger."""
    ckpt_path = tmp_path / "best_test_model.pt"
    early_stopping = EarlyStopping(patience=2, mode="max", checkpoint_path=ckpt_path)

    model = CustomWaferCNN(in_channels=1, num_classes=9)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Epoch 1: score 0.50 -> improvement, saves checkpoint
    saved = early_stopping.step(score=0.50, epoch=1, model=model, optimizer=optimizer)
    assert saved is True
    assert ckpt_path.exists()

    # Epoch 2: score 0.40 -> no improvement (counter = 1)
    saved = early_stopping.step(score=0.40, epoch=2, model=model, optimizer=optimizer)
    assert saved is False
    assert early_stopping.counter == 1
    assert early_stopping.should_stop is False

    # Epoch 3: score 0.40 -> no improvement (counter = 2) -> should stop!
    saved = early_stopping.step(score=0.40, epoch=3, model=model, optimizer=optimizer)
    assert saved is False
    assert early_stopping.counter == 2
    assert early_stopping.should_stop is True

    # Test checkpoint loading
    new_model = CustomWaferCNN(in_channels=1, num_classes=9)
    checkpoint = early_stopping.load_checkpoint(new_model)
    assert checkpoint["best_score"] == 0.50
    assert checkpoint["epoch"] == 1


def test_metrics_computation():
    """Verify multi-class evaluation metric calculations."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 0])  # One mistake: index 5 predicted as 0 instead of 2

    metrics = compute_metrics(y_true, y_pred, class_names={0: "none", 1: "Center", 2: "Donut"})

    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "per_class" in metrics
    assert "confusion_matrix" in metrics

    assert metrics["accuracy"] == pytest.approx(5 / 6, rel=1e-2)
    assert "none" in metrics["per_class"]
    assert "Center" in metrics["per_class"]
    assert "Donut" in metrics["per_class"]
