"""Build-it exercises for Chapter 3 -- Neural networks.

Two pieces of real frameworks, built from BabyTorch tensor ops only
(write no backward code -- if you use tensor operations, chapter 2's
engine differentiates your layer for free). Grade yourself:

    pytest book/exercises/test_ch03_nn.py -v
"""

import babytorch
import babytorch.nn as nn
from babytorch import Tensor
from babytorch.backend import xp


class RMSNorm(nn.Module):
    """RMSNorm -- LayerNorm's leaner cousin, used by LLaMA and friends.

    Instead of subtracting the mean and dividing by the standard
    deviation, RMSNorm only rescales by the root-mean-square, then
    applies a learned gain:

        rms  = sqrt( mean(x**2, over the last axis) + eps )
        out  = x / rms * gain

    where ``gain`` is a learnable parameter vector of ``num_features``
    ones (compare ``LayerNorm.gamma`` in babytorch/nn/nn.py -- and
    remember: store it as an attribute with ``requires_grad=True`` and
    ``Module.parameters()`` will find it by itself).
    """

    def __init__(self, num_features, eps=1e-5):
        raise NotImplementedError("your code here")

    def forward(self, x):
        raise NotImplementedError("your code here")


def bce_loss(p, y):
    """CHALLENGE (*): binary cross-entropy.

    ``p`` holds predicted probabilities in (0, 1) -- e.g. a Sigmoid's
    output -- and ``y`` the true labels (0.0 or 1.0), both Tensors of
    the same shape. Return the scalar Tensor

        -mean( y * log(p)  +  (1 - y) * log(1 - p) )

    This is chapter 3's cross-entropy idea specialized to two classes:
    charge -log of the probability given to the correct label.
    Build it from tensor operations; no loops, no raw arrays.
    """
    raise NotImplementedError("your code here")
