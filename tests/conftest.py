"""Shared test fixtures and the finite-difference gradient checker.

The tests run on the CPU (NumPy) so they work on any machine and give
byte-for-byte reproducible numbers.
"""

import os
os.environ.setdefault("BABYTORCH_DEVICE", "cpu")

import numpy as np
import pytest

import babytorch
from babytorch import Tensor


def numeric_gradient(f, x, eps=1e-5):
    """Estimate df/dx numerically with the central-difference formula.

        df/dx_i ~= ( f(x + eps e_i) - f(x - eps e_i) ) / (2 eps)

    where ``f`` maps an array to a single number.  We wiggle each element
    of ``x`` a tiny bit up and down and see how much ``f`` changes.  This
    is slow (two forward passes per element) but needs no calculus, so it
    is the perfect independent check on the analytic autograd gradients.
    """
    grad = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        original = x[idx]
        x[idx] = original + eps
        plus = f(x)
        x[idx] = original - eps
        minus = f(x)
        x[idx] = original
        grad[idx] = (plus - minus) / (2 * eps)
        it.iternext()
    return grad


def check_gradient(build_scalar, x0, eps=1e-5, tol=1e-4):
    """Assert autograd matches finite differences for ``build_scalar``.

    ``build_scalar`` takes a :class:`Tensor` and returns a scalar Tensor
    (the "loss").  We compare the gradient BabyTorch computes with
    ``backward()`` against the numerical estimate.
    """
    x0 = np.array(x0, dtype=np.float64)

    # Analytic gradient from BabyTorch's autograd.
    xt = Tensor(x0, requires_grad=True, dtype=babytorch.xp.float64)
    loss = build_scalar(xt)
    loss.backward()
    # to_numpy bridges the GPU->CPU gap so this harness also checks CuPy.
    analytic = babytorch.to_numpy(xt.grad).astype(np.float64)

    # Numerical gradient from finite differences.
    def f(x):
        scalar = build_scalar(Tensor(x, dtype=babytorch.xp.float64)).data
        return float(babytorch.to_numpy(scalar))
    numeric = numeric_gradient(f, x0.copy(), eps=eps)

    assert analytic.shape == numeric.shape, \
        f"grad shape {analytic.shape} != numeric shape {numeric.shape}"
    max_err = np.max(np.abs(analytic - numeric))
    assert max_err < tol, (
        f"gradient mismatch: max abs error {max_err:.2e} exceeds tol {tol:.1e}\n"
        f"analytic=\n{analytic}\nnumeric=\n{numeric}")


@pytest.fixture
def rng():
    return np.random.default_rng(0)
