"""Optimizers: the rules for updating parameters from their gradients.

After ``loss.backward()`` fills every parameter's ``.grad``, an optimizer
decides how to nudge ``.data``.  All optimizers here implement the same
tiny interface:

* ``step()``      -- update all parameters using their current gradients;
* ``zero_grad()`` -- clear the gradients, ready for the next batch.

The classic training loop is therefore::

    prediction = model(x)
    loss = criterion(prediction, y)
    optimizer.zero_grad()   # 1. forget old gradients
    loss.backward()         # 2. compute fresh ones
    optimizer.step()        # 3. update the weights
"""

from ..backend import xp


class Optimizer:
    """Shared plumbing for all optimizers."""

    def __init__(self, params, learning_rate):
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive.")
        # `params` may be a generator; materialize it so we can iterate
        # over the parameters every step.
        self.params = list(params)
        if not self.params:
            raise ValueError("Optimizer got an empty parameter list.")
        self.learning_rate = learning_rate

    def zero_grad(self):
        """Reset all gradients (call once per batch, before backward)."""
        for p in self.params:
            p.grad = None

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent, optionally with momentum.

    Plain SGD takes a small step against the gradient::

        p = p - learning_rate * grad

    With ``momentum`` the update keeps a running *velocity*: a fraction of
    the previous step is added to the current one.  Like a heavy ball
    rolling downhill, it smooths out zig-zagging gradients and pushes
    through small bumps::

        v = momentum * v + grad
        p = p - learning_rate * v

    ``weight_decay`` adds L2 regularization: it constantly shrinks the
    weights a little, which discourages extreme values and overfitting.
    """

    def __init__(self, params, learning_rate=0.001, momentum=0.0, weight_decay=0.0):
        super().__init__(params, learning_rate)
        if not 0.0 <= momentum < 1.0:
            raise ValueError("Momentum must be in [0, 1).")
        if weight_decay < 0:
            raise ValueError("Weight decay cannot be negative.")
        self.momentum = momentum
        self.weight_decay = weight_decay
        # One velocity buffer per parameter (created lazily on first use).
        self.velocities = [None] * len(self.params)

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * p.data
            if self.momentum > 0:
                if self.velocities[i] is None:
                    self.velocities[i] = xp.zeros_like(p.data)
                self.velocities[i] = self.momentum * self.velocities[i] + grad
                grad = self.velocities[i]
            p.data -= self.learning_rate * grad


class Adam(Optimizer):
    """Adam: SGD with a per-parameter, self-tuning step size.

    Adam keeps two running averages for every parameter:

    * ``m`` -- the average gradient (like momentum: which way is downhill?)
    * ``v`` -- the average *squared* gradient (how noisy/steep is it?)

    The update divides one by the square root of the other::

        p = p - learning_rate * m / (sqrt(v) + eps)

    so parameters with consistently large gradients take relatively
    smaller steps, and rarely-updated parameters take relatively larger
    ones.  The ``1 - beta^t`` corrections remove the startup bias of the
    running averages (which begin at zero).
    """

    def __init__(self, params, learning_rate=0.001, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.0):
        super().__init__(params, learning_rate)
        self.beta1, self.beta2 = betas
        if not 0.0 <= self.beta1 < 1.0 or not 0.0 <= self.beta2 < 1.0:
            raise ValueError("Adam betas must each be in [0, 1).")
        if eps <= 0:
            raise ValueError("Adam epsilon must be positive.")
        if weight_decay < 0:
            raise ValueError("Weight decay cannot be negative.")
        self.eps = eps
        self.weight_decay = weight_decay
        # Bias correction is per parameter: a parameter may receive its first
        # gradient several optimizer steps after another parameter.
        self.steps = [0] * len(self.params)
        self.m = [xp.zeros_like(p.data) for p in self.params]
        self.v = [xp.zeros_like(p.data) for p in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            self.steps[i] += 1
            t = self.steps[i]
            grad = p.grad
            if self.weight_decay > 0:
                # Classic (L2) weight decay: folded into the gradient.
                grad = grad + self.weight_decay * p.data

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad * grad

            m_hat = self.m[i] / (1 - self.beta1 ** t)
            v_hat = self.v[i] / (1 - self.beta2 ** t)

            p.data -= self.learning_rate * m_hat / (xp.sqrt(v_hat) + self.eps)


class AdamW(Adam):
    """Adam with *decoupled* weight decay -- the optimizer used to train
    GPT-style models.

    In plain Adam, weight decay gets mixed into the gradient and then
    divided by ``sqrt(v)``, so the amount of decay varies per parameter.
    AdamW instead applies the decay directly to the weights, separately
    from the adaptive step -- simpler, and it regularizes better in
    practice (Loshchilov & Hutter, 2017).
    """

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            self.steps[i] += 1
            t = self.steps[i]
            grad = p.grad

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad * grad

            m_hat = self.m[i] / (1 - self.beta1 ** t)
            v_hat = self.v[i] / (1 - self.beta2 ** t)

            # Decoupled decay: shrink the weight directly...
            if self.weight_decay > 0:
                p.data -= self.learning_rate * self.weight_decay * p.data
            # ...then take the adaptive step.
            p.data -= self.learning_rate * m_hat / (xp.sqrt(v_hat) + self.eps)
