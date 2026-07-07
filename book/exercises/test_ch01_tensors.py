"""Grader for chapter 1. Run:  pytest book/exercises/test_ch01_tensors.py -v"""

import numpy as np

import babytorch
from babytorch import Tensor

from grading import load, exercise

impl = load("ch01_tensors")


@exercise
def test_standardize_values():
    x = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [6.0, 60.0]],
                 dtype=np.float32)
    out = impl.standardize(Tensor(x))
    assert isinstance(out, Tensor), "return a Tensor, not a raw array"
    got = out.numpy()
    want = (x - x.mean(axis=0)) / x.std(axis=0)
    assert got.shape == x.shape
    assert np.allclose(got, want, atol=1e-5), \
        "columns should have mean 0 and std 1"
    assert np.allclose(got.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(got.std(axis=0), 1.0, atol=1e-4)


@exercise
def test_standardize_stays_in_the_graph():
    x = Tensor([[1.0, 4.0], [3.0, 8.0], [5.0, 6.0]], requires_grad=True)
    loss = (impl.standardize(x) ** 2).mean()
    loss.backward()
    assert x.grad is not None, \
        "use tensor operations only -- the computation graph must survive"


@exercise
def test_outer():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([10.0, 20.0], dtype=np.float32)
    out = impl.outer(Tensor(a), Tensor(b))
    assert isinstance(out, Tensor)
    assert out.shape == (3, 2), f"expected shape (3, 2), got {out.shape}"
    assert np.allclose(out.numpy(), np.outer(a, b))


@exercise
def test_outer_is_differentiable():
    a = babytorch.randn(4, requires_grad=True)
    b = babytorch.randn(5, requires_grad=True)
    impl.outer(a, b).sum().backward()
    assert a.grad is not None and b.grad is not None
