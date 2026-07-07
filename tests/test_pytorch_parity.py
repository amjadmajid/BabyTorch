"""Parity tests: BabyTorch should agree with PyTorch, number for number.

These tests recreate the original hand-written comparison scripts, but as
proper pytest cases that are **skipped automatically when PyTorch is not
installed** (BabyTorch itself never depends on torch).  They are the most
convincing demonstration of BabyTorch's promise: the same code, the same
results as the real thing.

Install torch to enable them::

    pip install torch
"""

import numpy as np
import pytest

import babytorch
from babytorch import Tensor

torch = pytest.importorskip("torch")   # skip the whole module without torch


def assert_close(bt_tensor, pt_tensor, tol=1e-5):
    a = babytorch.to_numpy(bt_tensor.data)
    b = pt_tensor.detach().numpy()
    assert np.allclose(a, b, atol=tol), f"values differ:\n{a}\n{b}"


def assert_grad_close(bt_tensor, pt_tensor, tol=1e-5):
    a = babytorch.to_numpy(bt_tensor.grad)
    b = pt_tensor.grad.detach().numpy()
    assert np.allclose(a, b, atol=tol), f"grads differ:\n{a}\n{b}"


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------

def test_add_parity():
    v1, v2 = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
    a_t, b_t = Tensor(v1, requires_grad=True), Tensor(v2, requires_grad=True)
    (a_t + b_t).sum().backward()
    a_p = torch.tensor(v1, requires_grad=True)
    b_p = torch.tensor(v2, requires_grad=True)
    (a_p + b_p).sum().backward()
    assert_grad_close(a_t, a_p)
    assert_grad_close(b_t, b_p)


def test_mul_parity():
    v1, v2 = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
    a_t, b_t = Tensor(v1, requires_grad=True), Tensor(v2, requires_grad=True)
    (a_t * b_t).sum().backward()
    a_p = torch.tensor(v1, requires_grad=True)
    b_p = torch.tensor(v2, requires_grad=True)
    (a_p * b_p).sum().backward()
    assert_grad_close(a_t, a_p)
    assert_grad_close(b_t, b_p)


def test_matmul_parity():
    m1 = np.random.randn(4, 3).astype(np.float32)
    m2 = np.random.randn(3, 5).astype(np.float32)
    a_t, b_t = Tensor(m1, requires_grad=True), Tensor(m2, requires_grad=True)
    (a_t @ b_t).sum().backward()
    a_p = torch.tensor(m1, requires_grad=True)
    b_p = torch.tensor(m2, requires_grad=True)
    (a_p @ b_p).sum().backward()
    assert_grad_close(a_t, a_p, tol=1e-4)
    assert_grad_close(b_t, b_p, tol=1e-4)


def test_batched_matmul_parity():
    m1 = np.random.randn(5, 3, 4).astype(np.float32)
    m2 = np.random.randn(5, 4, 2).astype(np.float32)
    a_t, b_t = Tensor(m1, requires_grad=True), Tensor(m2, requires_grad=True)
    (a_t @ b_t).sum().backward()
    a_p = torch.tensor(m1, requires_grad=True)
    b_p = torch.tensor(m2, requires_grad=True)
    (a_p @ b_p).sum().backward()
    assert_grad_close(a_t, a_p, tol=1e-4)
    assert_grad_close(b_t, b_p, tol=1e-4)


# ---------------------------------------------------------------------------
# Unary functions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fn_name", ["relu", "tanh", "sigmoid", "exp"])
def test_unary_parity(fn_name):
    x = np.random.randn(4, 3).astype(np.float32) * 0.5
    x_t = Tensor(x, requires_grad=True)
    getattr(x_t, fn_name)().sum().backward()
    x_p = torch.tensor(x, requires_grad=True)
    getattr(x_p, fn_name)().sum().backward()
    assert_grad_close(x_t, x_p, tol=1e-4)


def test_softmax_parity():
    x = np.random.randn(4, 5).astype(np.float32)
    x_t = Tensor(x, requires_grad=True)
    (x_t.softmax(axis=-1) ** 2).sum().backward()
    x_p = torch.tensor(x, requires_grad=True)
    (torch.softmax(x_p, dim=-1) ** 2).sum().backward()
    assert_grad_close(x_t, x_p, tol=1e-4)


# ---------------------------------------------------------------------------
# Shape ops
# ---------------------------------------------------------------------------

def test_squeeze_parity():
    v = np.array([[1.0], [-2.0], [3.0]], dtype=np.float32)
    a_t = Tensor(v, requires_grad=True)
    a_t.squeeze().sum().backward()
    a_p = torch.tensor(v, requires_grad=True)
    a_p.squeeze().sum().backward()
    assert_grad_close(a_t, a_p)


def test_unsqueeze_parity():
    v = [1.0, -2.0, 3.0]
    a_t = Tensor(v, requires_grad=True)
    a_t.unsqueeze(1).sum().backward()
    a_p = torch.tensor(v, requires_grad=True)
    a_p.unsqueeze(1).sum().backward()
    assert_grad_close(a_t, a_p)


# ---------------------------------------------------------------------------
# Iterability & subscription (behavioural, no torch needed for the asserts)
# ---------------------------------------------------------------------------

def test_iterable_and_subscriptable():
    t = Tensor([1.0, 2.0, 3.0, 4.0])
    assert [float(x) for x in t] == [1.0, 2.0, 3.0, 4.0]
    assert float(t[2].data) == 3.0
    t[2] = 5.0
    assert float(t[2].data) == 5.0


def test_cross_entropy_parity():
    logits = np.random.randn(6, 4).astype(np.float32)
    targets = np.array([0, 3, 1, 2, 0, 1])
    x_t = Tensor(logits, requires_grad=True)
    babytorch.nn.CrossEntropyLoss()(x_t, targets).backward()
    x_p = torch.tensor(logits, requires_grad=True)
    torch.nn.functional.cross_entropy(x_p, torch.tensor(targets)).backward()
    assert_grad_close(x_t, x_p, tol=1e-4)
