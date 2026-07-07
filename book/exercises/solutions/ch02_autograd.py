"""Solutions for chapter 2. Struggle first -- that's where the learning is."""

from babytorch.backend import xp
from babytorch.engine.operations import Operation


class MinOperation(Operation):
    # MaxOperation with the comparison flipped -- nothing more.

    def forward(self, a, axis=None, keepdims=False):
        self.a = a
        self.axis = axis
        self.keepdims = keepdims
        self.out = xp.min(a.data, axis=axis, keepdims=keepdims)
        return self.out

    def backward(self, grad):
        out = self.out
        if self.axis is not None and not self.keepdims:
            out = xp.expand_dims(out, self.axis)
            grad = xp.expand_dims(grad, self.axis)
        mask = (self.a.data == out)          # where the min was achieved
        count = mask.sum(axis=self.axis, keepdims=True)
        return mask * grad / count,          # ties share the gradient


class AbsOperation(Operation):
    def forward(self, a):
        self.a = a
        return xp.abs(a.data)

    def backward(self, grad):
        # d|a|/da = +1 for a > 0, -1 for a < 0; sign(0) = 0 is a valid
        # (and finite) subgradient at the kink.
        return grad * xp.sign(self.a.data),
