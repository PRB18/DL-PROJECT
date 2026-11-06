"""
tests/test_model.py
-------------------
Unit tests for model architectures, losses, and learning rate scheduling.
"""

import pytest
import torch
import torch.nn as nn

from src.models.baseline_cnn import BaselineCNN, build_baseline_cnn
from src.training.losses import LabelSmoothingCE, compute_cutmix_loss
from src.training.scheduler import build_scheduler, get_lr


class TestBaselineCNN:
    def test_output_shape(self):
        model = build_baseline_cnn(num_classes=10)
        x = torch.randn(4, 3, 64, 64)
        out = model(x)
        assert out.shape == (4, 10)

    def test_param_count(self):
        model = build_baseline_cnn(num_classes=38)
        total, trainable = model.count_parameters()
        assert total == trainable
        assert total < 2_000_000, f"BaselineCNN should be lightweight (<2M), got {total}"

    def test_forward_no_nan(self):
        model = build_baseline_cnn(num_classes=5)
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert not torch.isnan(out).any()


class TestTransferModel:
    def test_transfer_model_dummy(self):
        # Test custom dummy transfer head without downloading online weights
        class DummyBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3)
                self.pool = nn.AdaptiveAvgPool2d((1, 1))

            def forward(self, x):
                return torch.flatten(self.pool(self.conv(x)), 1)

        backbone = DummyBackbone()
        head = nn.Linear(16, 5)

        model = nn.Sequential(backbone, head)
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 5)

    def test_freeze_backbone_logic(self):
        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Linear(10, 10)
                self.head = nn.Linear(10, 2)

            def freeze_backbone(self):
                for p in self.backbone.parameters():
                    p.requires_grad = False

        net = SimpleNet()
        net.freeze_backbone()
        trainable_backbone = [p for p in net.backbone.parameters() if p.requires_grad]
        trainable_head = [p for p in net.head.parameters() if p.requires_grad]
        assert len(trainable_backbone) == 0
        assert len(trainable_head) > 0


class TestLosses:
    def test_label_smoothing_loss_positive(self):
        loss_fn = LabelSmoothingCE(num_classes=5, smoothing=0.1)
        logits = torch.randn(4, 5)
        targets = torch.tensor([0, 2, 4, 1])
        loss = loss_fn(logits, targets)
        assert loss.item() > 0.0

    def test_cutmix_loss(self):
        criterion = nn.CrossEntropyLoss()
        logits = torch.randn(4, 5)
        y_a = torch.tensor([0, 1, 2, 3])
        y_b = torch.tensor([4, 3, 2, 1])
        loss = compute_cutmix_loss(logits, y_a, y_b, lam=0.7, criterion=criterion)
        assert loss.item() > 0.0


class TestScheduler:
    def test_cosine_scheduler_lr_changes(self):
        param = [nn.Parameter(torch.randn(2, 2))]
        optimizer = torch.optim.SGD(param, lr=1.0)
        sched_dict = build_scheduler(
            optimizer,
            scheduler_type="cosine_warmup",
            warmup_epochs=2,
            total_epochs=10,
        )
        scheduler = sched_dict["scheduler"]

        lr_0 = get_lr(optimizer)
        scheduler.step()
        lr_1 = get_lr(optimizer)
        assert lr_1 > lr_0  # In warmup phase, LR increases
