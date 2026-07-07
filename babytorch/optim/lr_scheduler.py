"""Learning-rate schedulers: change the step size as training progresses.

A good learning rate early in training is usually too large later on.
Schedulers adjust ``optimizer.learning_rate`` over time; call
``scheduler.step(t)`` once per epoch (or per iteration for LLM training).
"""

import math


class LRScheduler:
    """Base class: remembers the optimizer and its starting learning rate."""

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.base_lr = optimizer.learning_rate

    def step(self, t):
        raise NotImplementedError


class LambdaLR(LRScheduler):
    """Scale the base learning rate by a user-supplied function of time.

        scheduler = LambdaLR(optimizer, lambda epoch: 0.95 ** epoch)

    The lambda returns a *factor*; the actual learning rate becomes
    ``base_lr * factor`` (matching PyTorch's behaviour).
    """

    def __init__(self, optimizer, lr_lambda):
        super().__init__(optimizer)
        self.lr_lambda = lr_lambda

    def step(self, epoch):
        self.optimizer.learning_rate = self.base_lr * self.lr_lambda(epoch)


class StepLR(LRScheduler):
    """Multiply the learning rate by ``gamma`` every ``step_size`` epochs.

    E.g. ``StepLR(opt, step_size=10, gamma=0.1)`` divides the learning
    rate by 10 at epochs 10, 20, 30, ...
    """

    def __init__(self, optimizer, step_size, gamma=0.1):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def step(self, epoch):
        self.optimizer.learning_rate = \
            self.base_lr * self.gamma ** (epoch // self.step_size)


class CosineWarmupLR(LRScheduler):
    """Linear warmup followed by cosine decay -- the LLM training classic.

    * Steps ``0 .. warmup_steps``: learning rate ramps up linearly from 0.
      (Fresh models produce huge, noisy gradients; starting gently avoids
      an early explosion.)
    * Steps ``warmup_steps .. total_steps``: smooth cosine descent from
      ``base_lr`` down to ``min_lr``, easing the model into a good minimum.

    This is the schedule used by GPT-2/3, LLaMA, and most modern LLMs.
    """

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0.0):
        super().__init__(optimizer)
        assert 0 <= warmup_steps < total_steps
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr

    def step(self, t):
        if t < self.warmup_steps:
            lr = self.base_lr * (t + 1) / self.warmup_steps
        elif t >= self.total_steps:
            lr = self.min_lr
        else:
            progress = (t - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))   # 1 -> 0
            lr = self.min_lr + (self.base_lr - self.min_lr) * cosine
        self.optimizer.learning_rate = lr
