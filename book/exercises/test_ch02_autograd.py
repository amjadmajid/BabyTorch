"""Grader for chapter 2. Your backward passes face the same judge the
library's own operations face: finite differences (see tests/conftest.py)."""

import numpy as np

from babytorch import Tensor
from babytorch.backend import xp

from grading import load, exercise, numeric_gradient

impl = load("ch02_autograd")


def apply_op(op_result):
    """Wire an exercise op into the real graph, like Tensor methods do."""
    op, tensor, out_data = op_result
    return tensor._make_output(op, out_data, tensor.requires_grad, "ex")


def run_min(x, axis=None, keepdims=False):
    t = Tensor(x, requires_grad=True, dtype=xp.float64)
    op = impl.MinOperation()
    out = apply_op((op, t, op.forward(t, axis=axis, keepdims=keepdims)))
    return t, out


@exercise
def test_min_forward():
    x = np.array([[3.0, 1.0, 2.0], [0.5, 4.0, 6.0]])
    _, out = run_min(x)
    assert float(out.numpy()) == 0.5
    _, out = run_min(x, axis=1)
    assert np.allclose(out.numpy(), [1.0, 0.5])


@exercise
def test_min_backward_matches_finite_differences():
    x0 = np.array([[3.0, 1.0, 2.0], [0.5, 4.0, 6.0]])
    t, out = run_min(x0)
    out.backward()
    numeric = numeric_gradient(lambda x: float(np.min(x)), x0.copy())
    assert np.allclose(np.asarray(Tensor(t.grad).numpy()), numeric, atol=1e-4)


@exercise
def test_min_splits_gradient_between_ties():
    t, out = run_min(np.array([1.0, 1.0, 3.0]))
    out.backward()
    grad = np.asarray(Tensor(t.grad).numpy())
    # finite differences agree: nudging either tied element down moves the min
    numeric = numeric_gradient(lambda x: float(np.min(x)),
                               np.array([1.0, 1.0, 3.0]))
    assert np.allclose(grad, numeric, atol=1e-4), \
        "tied minima must share the gradient (0.5 each here)"
    assert np.isclose(grad.sum(), 1.0)


@exercise
def test_min_along_axis_backward():
    x0 = np.array([[3.0, 1.0, 2.0], [0.5, 4.0, 6.0]])
    t = Tensor(x0, requires_grad=True, dtype=xp.float64)
    op = impl.MinOperation()
    out = t._make_output(op, op.forward(t, axis=1), t.requires_grad, "ex")
    out.sum().backward()
    numeric = numeric_gradient(lambda x: float(np.min(x, axis=1).sum()),
                               x0.copy())
    assert np.allclose(np.asarray(Tensor(t.grad).numpy()), numeric, atol=1e-4)


@exercise
def test_abs_forward_and_backward():
    x0 = np.array([-2.0, -0.5, 1.5, 3.0])
    t = Tensor(x0, requires_grad=True, dtype=xp.float64)
    op = impl.AbsOperation()
    out = t._make_output(op, op.forward(t), t.requires_grad, "ex")
    assert np.allclose(out.numpy(), np.abs(x0))
    out.sum().backward()
    numeric = numeric_gradient(lambda x: float(np.abs(x).sum()), x0.copy())
    assert np.allclose(np.asarray(Tensor(t.grad).numpy()), numeric, atol=1e-4)


@exercise
def test_abs_at_zero_is_finite():
    t = Tensor(np.array([0.0, 2.0]), requires_grad=True, dtype=xp.float64)
    op = impl.AbsOperation()
    out = t._make_output(op, op.forward(t), t.requires_grad, "ex")
    out.sum().backward()
    assert np.all(np.isfinite(np.asarray(Tensor(t.grad).numpy()))), \
        "the subgradient at 0 must be a finite number (0 is the usual choice)"
