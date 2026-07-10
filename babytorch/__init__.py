"""BabyTorch: a tiny deep learning framework you can read in an afternoon.

The small public API follows familiar PyTorch conventions so the core
concepts and training-loop structure transfer cleanly:

    import babytorch
    import babytorch.nn as nn
    from babytorch.optim import SGD

    x = babytorch.randn(32, 10, requires_grad=True)
    model = nn.Sequential(nn.Linear(10, 32, nn.ReLU()), nn.Linear(32, 1))
    loss = ((model(x) - 1) ** 2).mean()
    loss.backward()
"""

from .backend import xp, device, set_device, to_numpy
from .engine import Tensor, no_grad

__version__ = "0.3.0"


# ---------------------------------------------------------------------------
# Tensor factory functions (mirroring torch.zeros, torch.randn, ...)
# ---------------------------------------------------------------------------

def _normalize_shape(shape):
    """Allow both zeros(2, 3) and zeros((2, 3))."""
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        return tuple(shape[0])
    return shape


def tensor(data, requires_grad=False, dtype=xp.float32):
    """Create a tensor from data (a number, list, or array)."""
    return Tensor(data, requires_grad=requires_grad, dtype=dtype)


def zeros(*shape, requires_grad=False):
    """A tensor filled with zeros."""
    return Tensor(xp.zeros(_normalize_shape(shape)), requires_grad=requires_grad)


def ones(*shape, requires_grad=False):
    """A tensor filled with ones."""
    return Tensor(xp.ones(_normalize_shape(shape)), requires_grad=requires_grad)


def randn(*shape, requires_grad=False):
    """A tensor with standard-normal random values (mean 0, std 1)."""
    return Tensor(xp.random.standard_normal(_normalize_shape(shape)),
                  requires_grad=requires_grad)


def rand(*shape, requires_grad=False):
    """A tensor with uniform random values in [0, 1)."""
    return Tensor(xp.random.random(_normalize_shape(shape)),
                  requires_grad=requires_grad)


def arange(*args, requires_grad=False):
    """Evenly spaced values, like Python's range: arange(start, stop, step)."""
    return Tensor(xp.arange(*args), requires_grad=requires_grad)


def manual_seed(seed):
    """Seed the random number generator for reproducible experiments."""
    xp.random.seed(seed)


def __getattr__(name):
    # Import the visualizer lazily: it needs matplotlib/graphviz, which are
    # optional -- `import babytorch` should work without them.
    if name == "Grapher":
        from .visualization import Grapher
        return Grapher
    raise AttributeError(f"module 'babytorch' has no attribute '{name}'")
