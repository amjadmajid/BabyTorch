"""Grader for chapter 6."""

import numpy as np

import babytorch
from babytorch import Tensor
from babytorch.backend import xp

from grading import load, exercise

impl = load("ch06_attention")


def reference_attention(q, k, v):
    """Plain NumPy reference, written independently of the exercise."""
    T, hs = q.shape
    scores = q @ k.T / np.sqrt(hs)
    scores = np.where(np.triu(np.ones((T, T)), k=1) == 1, -1e9, scores)
    scores = scores - scores.max(axis=-1, keepdims=True)
    w = np.exp(scores)
    w = w / w.sum(axis=-1, keepdims=True)
    return w @ v


@exercise
def test_attention_matches_reference():
    rng = np.random.default_rng(0)
    q, k, v = (rng.normal(size=(5, 8)).astype(np.float32) for _ in range(3))
    out = impl.causal_attention(Tensor(q), Tensor(k), Tensor(v))
    assert isinstance(out, Tensor)
    assert out.shape == (5, 8)
    assert np.allclose(out.numpy(), reference_attention(q, k, v), atol=1e-4)


@exercise
def test_first_position_can_only_see_itself():
    rng = np.random.default_rng(1)
    q, k, v = (rng.normal(size=(4, 6)).astype(np.float32) for _ in range(3))
    out = impl.causal_attention(Tensor(q), Tensor(k), Tensor(v)).numpy()
    assert np.allclose(out[0], v[0], atol=1e-4), \
        "position 0 has no past: its output must be exactly v[0]"


@exercise
def test_causality_the_future_cannot_leak():
    rng = np.random.default_rng(2)
    q, k, v = (rng.normal(size=(6, 8)).astype(np.float32) for _ in range(3))
    out1 = impl.causal_attention(Tensor(q), Tensor(k), Tensor(v)).numpy()

    # rewrite the FUTURE (positions 3..) with garbage
    k2, v2 = k.copy(), v.copy()
    k2[3:], v2[3:] = 100.0, -55.0
    out2 = impl.causal_attention(Tensor(q), Tensor(k2), Tensor(v2)).numpy()

    assert np.allclose(out1[:3], out2[:3], atol=1e-5), (
        "outputs at positions 0..2 changed when positions 3.. did -- "
        "the future is leaking through your mask")
    assert not np.allclose(out1[3:], out2[3:]), \
        "positions 3.. SHOULD change (they may attend to themselves)"


@exercise
def test_attention_is_differentiable():
    q = babytorch.randn(4, 6, requires_grad=True)
    k = babytorch.randn(4, 6, requires_grad=True)
    v = babytorch.randn(4, 6, requires_grad=True)
    impl.causal_attention(q, k, v).sum().backward()
    assert q.grad is not None and k.grad is not None and v.grad is not None


@exercise
def test_split_heads_layout():
    B, T, C, nh = 2, 3, 8, 2
    x = np.arange(B * T * C, dtype=np.float32).reshape(B, T, C)
    got = impl.split_heads(Tensor(x), nh)
    assert got.shape == (B, nh, T, C // nh)
    want = x.reshape(B, T, nh, C // nh).transpose(0, 2, 1, 3)
    assert np.allclose(got.numpy(), want), \
        "reshape (B,T,nh,hs) first, THEN swap the T and head axes"


@exercise
def test_merge_is_the_exact_inverse():
    x = babytorch.randn(2, 5, 12)
    back = impl.merge_heads(impl.split_heads(x, 3))
    assert back.shape == (2, 5, 12)
    assert np.allclose(back.numpy(), x.numpy(), atol=1e-6)
