"""Solutions for chapter 6. Struggle first -- that's where the learning is."""

import math

from babytorch import Tensor
from babytorch.backend import xp


def causal_attention(q, k, v):
    T, head_size = q.shape
    scores = (q @ k.T) * (1.0 / math.sqrt(head_size))
    mask = Tensor(xp.triu(xp.full((T, T), -1e9), k=1))   # a rule, not a parameter
    weights = (scores + mask).softmax(axis=-1)
    return weights @ v


def split_heads(x, n_head):
    B, T, C = x.shape
    # (B, T, C) -> (B, T, nh, hs) -> swap T and nh -> (B, nh, T, hs)
    return x.reshape(B, T, n_head, C // n_head).transpose((0, 2, 1, 3))


def merge_heads(x):
    B, n_head, T, head_size = x.shape
    # undo the swap first, then flatten the heads back into channels
    return x.transpose((0, 2, 1, 3)).reshape(B, T, n_head * head_size)
