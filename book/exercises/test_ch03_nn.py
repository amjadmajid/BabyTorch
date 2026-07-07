"""Grader for chapter 3."""

import numpy as np

import babytorch
from babytorch import Tensor
from babytorch.backend import xp

from grading import load, exercise, numeric_gradient

impl = load("ch03_nn")


@exercise
def test_rmsnorm_values():
    x = np.array([[1.0, 2.0, 3.0, 4.0], [-2.0, 0.5, 0.0, 1.0]],
                 dtype=np.float32)
    layer = impl.RMSNorm(4)
    out = layer(Tensor(x))
    rms = np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + 1e-5)
    assert out.shape == x.shape
    assert np.allclose(out.numpy(), x / rms, atol=1e-5), \
        "out = x / sqrt(mean(x^2) + eps) * gain, with gain starting at 1"


@exercise
def test_rmsnorm_gain_is_discovered_and_trains():
    layer = impl.RMSNorm(6)
    params = layer.parameters()
    assert len(params) == 1, \
        "RMSNorm has exactly one learnable tensor (the gain)"
    assert params[0].shape[-1] == 6
    assert np.allclose(babytorch.to_numpy(params[0].data), 1.0), \
        "the gain starts at ones (identity rescale)"

    x = babytorch.randn(3, 6)
    loss = (layer(x) ** 2).mean()
    loss.backward()
    assert params[0].grad is not None, \
        "gradients must reach the gain -- build forward from tensor ops only"


@exercise
def test_rmsnorm_works_on_3d_activations():
    x = babytorch.randn(2, 5, 8)          # (B, T, C), like a Transformer
    out = impl.RMSNorm(8)(x)
    assert out.shape == (2, 5, 8)


@exercise
def test_bce_values():
    p = Tensor([0.9, 0.2, 0.7], dtype=xp.float64)
    y = Tensor([1.0, 0.0, 1.0], dtype=xp.float64)
    want = -np.mean([np.log(0.9), np.log(1 - 0.2), np.log(0.7)])
    got = impl.bce_loss(p, y)
    assert isinstance(got, Tensor) and got.size == 1
    assert np.allclose(float(got.numpy()), want, atol=1e-5)


@exercise
def test_bce_gradient_matches_finite_differences():
    p0 = np.array([0.9, 0.2, 0.7, 0.4])
    y0 = np.array([1.0, 0.0, 1.0, 1.0])
    p = Tensor(p0, requires_grad=True, dtype=xp.float64)
    impl.bce_loss(p, Tensor(y0, dtype=xp.float64)).backward()

    def f(x):
        return float(-np.mean(y0 * np.log(x) + (1 - y0) * np.log(1 - x)))

    numeric = numeric_gradient(f, p0.copy())
    assert np.allclose(babytorch.to_numpy(p.grad), numeric, atol=1e-3)
