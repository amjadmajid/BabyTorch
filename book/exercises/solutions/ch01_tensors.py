"""Solutions for chapter 1. Struggle first -- that's where the learning is."""

import babytorch
from babytorch import Tensor


def standardize(t):
    # keepdims=True gives shapes (1, cols), which broadcast back over
    # the rows -- chapter 1's bias-row picture, exactly.
    mu = t.mean(axis=0, keepdims=True)
    std = t.var(axis=0, keepdims=True).sqrt()
    return (t - mu) / std


def outer(a, b):
    # (n, 1) * (m,) broadcasts to (n, m): a stretches across columns,
    # b stretches across rows. No matmul required.
    n = a.shape[0]
    return a.reshape(n, 1) * b
