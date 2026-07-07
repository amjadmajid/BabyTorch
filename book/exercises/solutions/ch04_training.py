"""Solutions for chapter 4. Struggle first -- that's where the learning is."""

from babytorch.backend import xp
from babytorch.optim import Optimizer


def clip_grad_norm_(params, max_norm):
    total = float(xp.sqrt(sum(xp.sum(p.grad ** 2)
                              for p in params if p.grad is not None)))
    if total > max_norm:
        scale = max_norm / total
        for p in params:
            if p.grad is not None:
                p.grad *= scale
    return total


class RMSProp(Optimizer):
    def __init__(self, params, learning_rate=0.01, alpha=0.99, eps=1e-8):
        super().__init__(params, learning_rate)
        self.alpha = alpha
        self.eps = eps
        self.v = [xp.zeros_like(p.data) for p in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            self.v[i] = self.alpha * self.v[i] + (1 - self.alpha) * p.grad ** 2
            p.data -= self.learning_rate * p.grad / (xp.sqrt(self.v[i]) + self.eps)
