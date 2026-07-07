"""Solutions for chapter 3. Struggle first -- that's where the learning is."""

import babytorch.nn as nn
from babytorch import Tensor
from babytorch.backend import xp


class RMSNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        # An attribute with requires_grad=True is all it takes for
        # Module.parameters() to find the gain.
        self.gain = Tensor(xp.ones(num_features), requires_grad=True)
        self.eps = eps

    def forward(self, x):
        rms = ((x ** 2).mean(axis=-1, keepdims=True) + self.eps).sqrt()
        return x / rms * self.gain


def bce_loss(p, y):
    # -mean( y*log(p) + (1-y)*log(1-p) ), straight from the definition.
    # Every piece is a tensor op, so backward() comes for free.
    return -(y * p.log() + (1.0 - y) * (1.0 - p).log()).mean()
