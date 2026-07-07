"""Tests for optimizers and learning-rate schedulers."""

import numpy as np
import pytest

import babytorch
import babytorch.nn as nn
from babytorch.optim import SGD, Adam, AdamW, LambdaLR, StepLR, CosineWarmupLR


def _fit_line(optimizer_cls, **kwargs):
    """Fit y = 2x + 1 and return (first_loss, last_loss)."""
    babytorch.manual_seed(0)
    x = babytorch.randn(64, 1)
    y = x * 2.0 + 1.0
    model = nn.Linear(1, 1)
    opt = optimizer_cls(model.parameters(), **kwargs)
    crit = nn.MSELoss()
    first = last = None
    for i in range(200):
        loss = crit(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i == 0:
            first = loss.item()
        last = loss.item()
    return first, last


def test_sgd_reduces_loss():
    first, last = _fit_line(SGD, learning_rate=0.1)
    assert last < first * 0.1


def test_sgd_momentum_reduces_loss():
    first, last = _fit_line(SGD, learning_rate=0.1, momentum=0.9)
    assert last < 1e-3


def test_adam_reduces_loss():
    first, last = _fit_line(Adam, learning_rate=0.1)
    assert last < 1e-3


def test_adamw_reduces_loss():
    first, last = _fit_line(AdamW, learning_rate=0.1, weight_decay=0.01)
    assert last < first * 0.1


def test_zero_grad_clears_gradients():
    model = nn.Linear(2, 2)
    opt = SGD(model.parameters(), learning_rate=0.1)
    loss = model(babytorch.randn(3, 2)).sum()
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())
    opt.zero_grad()
    assert all(p.grad is None for p in model.parameters())


def test_empty_params_raises():
    with pytest.raises(ValueError):
        SGD([], learning_rate=0.1)


def test_lambda_lr_scales_base():
    opt = SGD(nn.Linear(1, 1).parameters(), learning_rate=0.1)
    sched = LambdaLR(opt, lr_lambda=lambda e: 0.5 ** e)
    sched.step(0)
    assert abs(opt.learning_rate - 0.1) < 1e-9
    sched.step(2)
    assert abs(opt.learning_rate - 0.1 * 0.25) < 1e-9


def test_step_lr():
    opt = SGD(nn.Linear(1, 1).parameters(), learning_rate=1.0)
    sched = StepLR(opt, step_size=10, gamma=0.1)
    sched.step(5)
    assert abs(opt.learning_rate - 1.0) < 1e-9
    sched.step(10)
    assert abs(opt.learning_rate - 0.1) < 1e-9
    sched.step(20)
    assert abs(opt.learning_rate - 0.01) < 1e-9


def test_cosine_warmup():
    opt = SGD(nn.Linear(1, 1).parameters(), learning_rate=1.0)
    sched = CosineWarmupLR(opt, warmup_steps=10, total_steps=100, min_lr=0.0)
    # during warmup: linearly increasing
    sched.step(0)
    lr0 = opt.learning_rate
    sched.step(5)
    lr5 = opt.learning_rate
    assert lr0 < lr5 <= 1.0
    # at end of warmup: peak
    sched.step(9)
    assert abs(opt.learning_rate - 1.0) < 1e-9
    # after total steps: min
    sched.step(100)
    assert abs(opt.learning_rate - 0.0) < 1e-9
