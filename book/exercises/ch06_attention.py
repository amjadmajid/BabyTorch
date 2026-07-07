"""Build-it exercises for Chapter 6 -- Attention.

You have READ the attention code; now write it blind. The grader
includes the property that makes attention "causal": your output at
position t must not change when the future changes. Grade yourself:

    pytest book/exercises/test_ch06_attention.py -v
"""

import math

from babytorch import Tensor
from babytorch.backend import xp


def causal_attention(q, k, v):
    """One head of causal self-attention, from raw tensor operations.

    ``q``, ``k``, ``v`` are Tensors of shape (T, head_size) -- one
    sequence, one head, no batch dimension to worry about. Return the
    (T, head_size) Tensor:

        scores  = q @ k^T / sqrt(head_size)      (T, T)
        scores += mask        (0 on/below the diagonal, -1e9 above)
        weights = softmax(scores, axis=-1)
        out     = weights @ v

    Build the mask with ``xp.triu`` exactly like ``CausalSelfAttention``
    does (wrap it in a plain ``Tensor`` -- it is a rule, not a
    parameter). Useful pieces: ``q @ k.T``, ``t.softmax(axis=-1)``,
    ``math.sqrt``.
    """
    raise NotImplementedError("your code here")


def split_heads(x, n_head):
    """CHALLENGE (*), part 1: carve channels into heads.

    ``x`` is a Tensor of shape (B, T, C) with C divisible by ``n_head``.
    Return shape (B, n_head, T, C // n_head) -- the reshape/transpose
    two-step from the chapter (and from ``CausalSelfAttention.forward``).
    """
    raise NotImplementedError("your code here")


def merge_heads(x):
    """CHALLENGE (*), part 2: the exact inverse.

    ``x`` is (B, n_head, T, head_size); return (B, T, n_head * head_size)
    such that ``merge_heads(split_heads(y, n)) == y`` for any y.
    """
    raise NotImplementedError("your code here")
