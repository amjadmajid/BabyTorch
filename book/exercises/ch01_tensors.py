"""Build-it exercises for Chapter 1 -- Tensors.

Implement the stubs, then grade yourself from the repo root:

    pytest book/exercises/test_ch01_tensors.py -v

Rules of the game: use whole-tensor operations only -- no Python loops
over elements. That constraint IS the lesson of chapter 1.
"""

import babytorch
from babytorch import Tensor


def standardize(t):
    """Return ``t`` standardized per column: zero mean, unit std.

    ``t`` is a Tensor of shape (rows, cols). For every column j,

        out[:, j] = (t[:, j] - mean_j) / std_j        std_j = sqrt(var_j)

    Return a **Tensor** of the same shape. Hints: ``t.mean(...)`` and
    ``t.var(...)`` take an ``axis`` and ``keepdims`` -- with
    ``keepdims=True`` the result broadcasts back over the rows by
    itself, exactly like the bias row in the chapter's figure.
    """
    raise NotImplementedError("your code here")


def outer(a, b):
    """CHALLENGE (*): the outer product, using broadcasting alone.

    ``a`` is a Tensor of shape (n,) and ``b`` of shape (m,). Return the
    Tensor of shape (n, m) with ``out[i, j] = a[i] * b[j]`` -- WITHOUT
    ``@``, without loops, without building the result element by
    element.

    Hint: what shapes would make ``a * b`` broadcast into (n, m)?
    ``reshape`` (or ``unsqueeze``) is all you need.
    """
    raise NotImplementedError("your code here")
