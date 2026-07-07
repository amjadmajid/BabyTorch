"""Gradient checks for every differentiable operation.

Each test builds a small scalar function of one tensor and asserts that
BabyTorch's analytic gradient matches a finite-difference estimate.  If a
backward pass has a bug, one of these will catch it.
"""

import numpy as np
import pytest

import babytorch
from babytorch import Tensor
from conftest import check_gradient


# ---------------------------------------------------------------------------
# Arithmetic (with broadcasting)
# ---------------------------------------------------------------------------

def test_add_broadcast():
    b = np.array([[1.0, 2.0, 3.0]])
    check_gradient(lambda x: (x + Tensor(b, dtype=babytorch.xp.float64)).sum(),
                   np.random.randn(4, 3))


def test_sub():
    b = np.random.randn(4, 3)
    check_gradient(lambda x: (x - Tensor(b, dtype=babytorch.xp.float64)).sum(),
                   np.random.randn(4, 3))


def test_mul_broadcast():
    b = np.array([[2.0, -1.0, 0.5]])
    check_gradient(lambda x: (x * Tensor(b, dtype=babytorch.xp.float64)).sum(),
                   np.random.randn(4, 3))


def test_div():
    b = np.random.randn(4, 3) + 3.0  # keep away from zero
    check_gradient(lambda x: (x / Tensor(b, dtype=babytorch.xp.float64)).sum(),
                   np.random.randn(4, 3))


def test_pow():
    check_gradient(lambda x: (x ** 3).sum(), np.random.randn(3, 3))


def test_sqrt():
    check_gradient(lambda x: x.sqrt().sum(), np.random.rand(3, 3) + 0.5)


def test_neg():
    check_gradient(lambda x: (-x).sum(), np.random.randn(3, 3))


# ---------------------------------------------------------------------------
# Matmul: vector, matrix, and batched (the transformer case)
# ---------------------------------------------------------------------------

def test_matmul_matrix():
    b = np.random.randn(3, 5)
    check_gradient(lambda x: (x @ Tensor(b, dtype=babytorch.xp.float64)).sum(),
                   np.random.randn(4, 3))


def test_matmul_vector():
    b = np.random.randn(3)
    check_gradient(lambda x: (x @ Tensor(b, dtype=babytorch.xp.float64)).sum(),
                   np.random.randn(4, 3))


def test_matmul_batched():
    # (2, 4, 3) @ (3, 5): weight broadcast across the batch -- the shape a
    # Transformer's linear layers actually see.
    b = np.random.randn(3, 5)
    check_gradient(lambda x: (x @ Tensor(b, dtype=babytorch.xp.float64)).sum(),
                   np.random.randn(2, 4, 3))


def test_matmul_batched_both():
    b = np.random.randn(2, 3, 5)
    check_gradient(lambda x: (x @ Tensor(b, dtype=babytorch.xp.float64)).sum(),
                   np.random.randn(2, 4, 3))


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------

def test_sum_all():
    check_gradient(lambda x: x.sum(), np.random.randn(3, 4))


def test_sum_axis():
    check_gradient(lambda x: (x.sum(axis=1) ** 2).sum(), np.random.randn(3, 4))


def test_sum_axis_keepdims():
    check_gradient(lambda x: (x.sum(axis=0, keepdims=True) ** 2).sum(),
                   np.random.randn(3, 4))


def test_mean():
    check_gradient(lambda x: x.mean(), np.random.randn(3, 4))


def test_mean_axis():
    check_gradient(lambda x: (x.mean(axis=1) ** 2).sum(), np.random.randn(3, 4))


def test_var():
    check_gradient(lambda x: x.var(axis=1).sum(), np.random.randn(3, 4))


def test_max_all():
    check_gradient(lambda x: x.max(), np.random.randn(3, 4))


def test_max_axis():
    check_gradient(lambda x: (x.max(axis=1) ** 2).sum(), np.random.randn(3, 4))


# ---------------------------------------------------------------------------
# Non-linearities
# ---------------------------------------------------------------------------

def test_relu():
    check_gradient(lambda x: x.relu().sum(), np.random.randn(3, 4))


def test_tanh():
    check_gradient(lambda x: x.tanh().sum(), np.random.randn(3, 4))


def test_sigmoid():
    check_gradient(lambda x: x.sigmoid().sum(), np.random.randn(3, 4))


def test_exp():
    check_gradient(lambda x: x.exp().sum(), np.random.randn(3, 4) * 0.5)


def test_log():
    check_gradient(lambda x: x.log().sum(), np.random.rand(3, 4) + 0.5)


def test_softmax():
    check_gradient(lambda x: (x.softmax(axis=-1) ** 2).sum(),
                   np.random.randn(3, 4))


def test_log_softmax():
    check_gradient(lambda x: x.log_softmax(axis=-1).sum(),
                   np.random.randn(3, 4))


# ---------------------------------------------------------------------------
# Shape manipulation
# ---------------------------------------------------------------------------

def test_reshape():
    check_gradient(lambda x: (x.reshape(2, 6) ** 2).sum(), np.random.randn(3, 4))


def test_transpose():
    check_gradient(lambda x: (x.T ** 2).sum(), np.random.randn(3, 4))


def test_transpose_axes():
    check_gradient(lambda x: (x.transpose((0, 2, 1)) ** 2).sum(),
                   np.random.randn(2, 3, 4))


def test_slice():
    check_gradient(lambda x: (x[1:3] ** 2).sum(), np.random.randn(5, 4))


def test_slice_repeated_indices():
    # The same rows read several times: gradients must accumulate.
    idx = np.array([0, 0, 1, 0])
    check_gradient(lambda x: (x[idx] ** 2).sum(), np.random.randn(3, 4))


def test_squeeze_unsqueeze():
    check_gradient(lambda x: (x.unsqueeze(1) ** 2).sum(), np.random.randn(3, 4))


# ---------------------------------------------------------------------------
# Composed expressions
# ---------------------------------------------------------------------------

def test_composed_expression():
    def f(x):
        y = (x @ Tensor(np.random.RandomState(1).randn(4, 4),
                        dtype=babytorch.xp.float64))
        return y.tanh().sum() + (x ** 2).mean()
    check_gradient(f, np.random.randn(3, 4))


def test_accumulation_reuse():
    # x used twice: gradients from both branches must add up.
    check_gradient(lambda x: (x * x + x).sum(), np.random.randn(3, 4))
