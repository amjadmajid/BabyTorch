"""Build-it exercises for Chapter 2 -- Autograd.

You are about to do what the chapter says framework authors do: write an
operation's forward AND backward pass, then prove the calculus correct
against finite differences. Grade yourself from the repo root:

    pytest book/exercises/test_ch02_autograd.py -v

Model your code on ``MaxOperation`` and ``ReLUOperation`` in
``babytorch/engine/operations.py`` -- open them side by side.
"""

from babytorch.backend import xp
from babytorch.engine.operations import Operation


class MinOperation(Operation):
    """Minimum over all elements, or along an axis.

    The exact mirror of ``MaxOperation``: only the element(s) that
    *achieved* the minimum influenced the output, so only they receive
    gradient -- and if several elements tie, the gradient is split
    evenly between them.

    ``forward(a, axis=None, keepdims=False)`` receives a Tensor ``a``
    and must return a raw array (work on ``a.data``). Remember to save
    on ``self`` whatever ``backward`` will need.
    """

    def forward(self, a, axis=None, keepdims=False):
        raise NotImplementedError("your code here")

    def backward(self, grad):
        raise NotImplementedError("your code here")


class AbsOperation(Operation):
    """CHALLENGE (*): element-wise absolute value ``|a|``.

    Work out the local derivative yourself: what is d|a|/da when a is
    positive? Negative? (At exactly zero any value in [-1, 1] is a valid
    subgradient -- ``xp.sign`` gives 0 there, which is fine.)
    """

    def forward(self, a):
        raise NotImplementedError("your code here")

    def backward(self, grad):
        raise NotImplementedError("your code here")
