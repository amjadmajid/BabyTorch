"""Functional versions of common operations.

Same math as the layer classes in ``nn.py``, but as plain functions --
handy when you don't need a layer object, e.g. inside a custom
``forward``:

    import babytorch.nn.functional as F
    y = F.softmax(logits, axis=-1)
"""

import math

from ..backend import xp
from ..engine import Tensor
from .loss import CrossEntropyLoss, MSELoss


def relu(tensor, alpha=0.0):
    """max(0, x), or leaky ReLU when ``alpha > 0``."""
    return tensor.relu(alpha)


def sigmoid(tensor):
    """Squash values into (0, 1)."""
    return tensor.sigmoid()


def tanh(tensor):
    """Squash values into (-1, 1)."""
    return tensor.tanh()


def gelu(tensor):
    """Gaussian Error Linear Unit (tanh approximation, as in GPT-2)."""
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * tensor * (1.0 + ((tensor + 0.044715 * tensor ** 3) * c).tanh())


def softmax(tensor, axis=-1):
    """Turn raw scores into probabilities along ``axis``."""
    return tensor.softmax(axis=axis)


def log_softmax(tensor, axis=-1):
    """Numerically stable log(softmax(x))."""
    return tensor.log_softmax(axis=axis)


def cross_entropy(predictions, targets):
    """Cross-entropy between logits ``(n, classes)`` and integer targets ``(n,)``."""
    return CrossEntropyLoss()(predictions, targets)


def mse_loss(predictions, targets):
    """Mean squared error."""
    return MSELoss()(predictions, targets)


def one_hot(indices, num_classes):
    """Encode integer ids as one-hot vectors (a plain data helper,
    not differentiable): ``one_hot([1, 0], 3) -> [[0,1,0], [1,0,0]]``."""
    if isinstance(indices, Tensor):
        indices = indices.data
    indices = xp.asarray(indices).astype(xp.int64)
    out = xp.zeros(indices.shape + (num_classes,), dtype=xp.float32)
    out[..., :] = xp.eye(num_classes, dtype=xp.float32)[indices]
    return Tensor(out)
