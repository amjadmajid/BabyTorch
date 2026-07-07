"""Build-it exercises for Chapter 4 -- Training.

Two tools every practitioner eventually reaches for: a guard against
exploding gradients, and an optimizer of your own. Grade yourself:

    pytest book/exercises/test_ch04_training.py -v

Open ``babytorch/optim/optim.py`` next to this file -- SGD and Adam are
your templates.
"""

import math

from babytorch.backend import xp
from babytorch.optim import Optimizer


def clip_grad_norm_(params, max_norm):
    """Rescale all gradients so their COMBINED norm is at most max_norm.

    Treat every parameter's ``.grad`` as one long vector; its total norm
    is ``sqrt( sum over all params of sum(grad**2) )``. If that exceeds
    ``max_norm``, multiply every gradient (in place, ``p.grad *= ...``)
    by ``max_norm / total_norm`` so the combined norm becomes exactly
    ``max_norm``. Gradients that are ``None`` are skipped.

    Return the total norm as a plain float, *measured before clipping*
    (training scripts print it to watch for instability).

    One unlucky batch can produce a huge gradient that catapults the
    weights; this cap is the standard seatbelt (the moons tutorial in
    tutorials/classification uses exactly this function).
    """
    raise NotImplementedError("your code here")


class RMSProp(Optimizer):
    """CHALLENGE (*): the missing link between SGD and Adam.

    RMSProp keeps ONE running average per parameter -- the mean squared
    gradient -- and divides the step by its square root:

        v  =  alpha * v  +  (1 - alpha) * grad**2
        p  =  p  -  learning_rate * grad / (sqrt(v) + eps)

    (Adam = this plus momentum on the gradient itself plus bias
    correction. Seeing RMSProp alone makes Adam obvious in hindsight.)

    Subclass ``Optimizer`` -- call ``super().__init__(params,
    learning_rate)`` and it gives you ``self.params`` (a list) and
    ``zero_grad()`` for free; you write ``__init__`` and ``step()``.
    Create one ``v`` buffer of zeros per parameter (see how Adam does
    it), and skip parameters whose ``.grad`` is None.
    """

    def __init__(self, params, learning_rate=0.01, alpha=0.99, eps=1e-8):
        raise NotImplementedError("your code here")

    def step(self):
        raise NotImplementedError("your code here")
