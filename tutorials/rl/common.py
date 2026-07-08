"""Shared machinery for the reinforcement-learning agents.

The four training scripts (``reinforce.py``, ``actor_critic.py``,
``dqn.py``, ``ppo.py`` and the Snake variants) stay short because the
pieces every agent needs live here:

* :class:`Categorical` -- sample an action and score it, the one thing
  BabyTorch has no built-in for (PyTorch calls it
  ``torch.distributions.Categorical``).  Everything else -- ``Linear``,
  ``Adam``, ``log_softmax`` -- BabyTorch already provides.
* small network classes (:class:`PolicyNet`, :class:`ValueNet`,
  :class:`QNet`, :class:`ConvNet`);
* the return / advantage maths (:func:`discounted_returns`,
  :func:`compute_gae`);
* a :class:`ReplayBuffer` and the target-network update for DQN;
* :func:`huber_loss` and :func:`clip_grad_value` for stable Q-learning;
* a headless learning-curve plot.

All of it is a few dozen lines: reinforcement learning is a *training
loop*, not new framework machinery.
"""

import random
from collections import deque

import numpy as np

import babytorch
import babytorch.nn as nn
from babytorch import Tensor
from babytorch.backend import xp, to_numpy


def set_seed(seed):
    """Seed every source of randomness we use, for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    babytorch.manual_seed(seed)


def to_tensor(x):
    """NumPy array (or list) -> a float32 BabyTorch tensor on the active device."""
    return Tensor(np.asarray(x, dtype=np.float32))


# ---------------------------------------------------------------------------
# The categorical distribution -- how a policy turns logits into an action
# ---------------------------------------------------------------------------

class Categorical:
    """A distribution over discrete actions, built from network *logits*.

    A policy network outputs one raw score (logit) per action.  This class
    turns those scores into a probability distribution and offers the three
    things every policy-gradient method asks for:

    * ``sample()``       -- draw an action (this is how the agent acts);
    * ``log_prob(a)``    -- log-probability of an action, the quantity whose
      gradient *is* the policy gradient;
    * ``entropy()``      -- how undecided the policy is, added to the loss to
      keep it exploring.

    ``log_prob`` and ``entropy`` return Tensors wired into the graph, so
    ``backward()`` flows through them; ``sample`` returns a plain integer.
    """

    def __init__(self, logits):
        self.logits = logits                     # (B, n_actions) Tensor
        self.log_probs = logits.log_softmax(axis=-1)   # (B, n_actions)

    def sample(self):
        """Return one sampled action id per row, as a NumPy int array."""
        probs = np.exp(to_numpy(self.log_probs.data))       # (B, n)
        # Inverse-CDF sampling, vectorised over the batch: draw u ~ U(0,1)
        # and take the first action whose cumulative probability exceeds it.
        u = np.random.random((probs.shape[0], 1))
        return (probs.cumsum(axis=1) > u).argmax(axis=1)

    def log_prob(self, actions):
        """Log-probabilities of ``actions`` (a 1-D int array) -> Tensor (B,)."""
        actions = xp.asarray(actions).astype(xp.int64)
        return self.log_probs[xp.arange(actions.shape[0]), actions]

    def entropy(self):
        """Entropy of each row, ``-sum(p * log p)`` -> Tensor (B,)."""
        return -(self.log_probs.exp() * self.log_probs).sum(axis=-1)


# ---------------------------------------------------------------------------
# Networks -- tiny MLPs, plus a ConvNet for the pixel version of Snake
# ---------------------------------------------------------------------------

def mlp(sizes, hidden_activation=nn.ReLU):
    """A plain multi-layer perceptron: ``Linear -> ReLU -> ... -> Linear``.

    The final layer has *no* activation, so it emits raw scores (logits or
    Q-values), which is what every head here wants.
    """
    layers = []
    for i in range(len(sizes) - 2):
        layers.append(nn.Linear(sizes[i], sizes[i + 1], hidden_activation()))
    layers.append(nn.Linear(sizes[-2], sizes[-1]))       # bare output layer
    return nn.Sequential(*layers)


class PolicyNet(nn.Module):
    """Maps an observation to one logit per action (the actor)."""

    def __init__(self, obs_dim, n_actions, hidden=128):
        self.net = mlp([obs_dim, hidden, n_actions])

    def forward(self, x):
        return self.net(x)                       # (B, n_actions) logits


class ValueNet(nn.Module):
    """Maps an observation to a single number V(s) (the critic)."""

    def __init__(self, obs_dim, hidden=128):
        self.net = mlp([obs_dim, hidden, 1])

    def forward(self, x):
        return self.net(x).reshape(-1)           # (B,)


class QNet(nn.Module):
    """Maps an observation to one Q-value per action."""

    def __init__(self, obs_dim, n_actions, hidden=128):
        self.net = mlp([obs_dim, hidden, hidden, n_actions])

    def forward(self, x):
        return self.net(x)                       # (B, n_actions)


class ConvNet(nn.Module):
    """A small convolutional Q-network for grid/image observations.

    Input shape ``(B, C, H, W)`` -> one Q-value per action.  Two conv
    layers read the board spatially, then two linear layers score the
    actions -- the same DQN, just with eyes instead of hand-built
    features.
    """

    def __init__(self, input_shape, n_actions, hidden=128):
        c, h, w = input_shape
        self.conv = nn.Sequential(
            nn.Conv2D(c, 16, kernel_size=3, stride=1), nn.ReLU(),
            nn.Conv2D(16, 32, kernel_size=3, stride=1), nn.ReLU())
        self.flatten = nn.Flatten()
        # Discover the flattened size by pushing one zero image through.
        with babytorch.no_grad():
            n_flat = self.flatten(self.conv(babytorch.zeros(1, c, h, w))).shape[1]
        self.head = mlp([n_flat, hidden, n_actions])

    def forward(self, x):
        return self.head(self.flatten(self.conv(x)))


# ---------------------------------------------------------------------------
# Returns and advantages -- turning a stream of rewards into learning signal
# ---------------------------------------------------------------------------

def discounted_returns(rewards, gamma):
    """Reward-to-go: ``G_t = r_t + gamma r_{t+1} + gamma^2 r_{t+2} + ...``.

    Walk the episode backwards so each return is built from the one after
    it in a single pass.  This is the target REINFORCE pushes the policy
    towards.
    """
    out = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        out[t] = running
    return out


def compute_gae(rewards, values, gamma, lam, last_value=0.0):
    """Generalised Advantage Estimation (Schulman et al., 2016).

    A knob-controlled trade-off between two ways to judge an action:

    * ``lam -> 0``: trust the critic -- advantage is the one-step TD error
      ``r + gamma V(s') - V(s)`` (low variance, biased by V's mistakes);
    * ``lam -> 1``: trust the real rewards -- advantage is the full
      Monte-Carlo return minus V (unbiased, high variance).

    ``lam`` around 0.95 blends them.  Returns ``(advantages, value_targets)``
    where the targets are what the critic should regress toward.
    """
    values = list(values) + [last_value]         # bootstrap the tail
    adv = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        running = delta + gamma * lam * running
        adv[t] = running
    returns = adv + np.array(values[:-1], dtype=np.float32)
    return adv, returns


def normalize(x, eps=1e-8):
    """Zero-mean, unit-variance -- steadies policy-gradient updates."""
    x = np.asarray(x, dtype=np.float32)
    return (x - x.mean()) / (x.std() + eps)


# ---------------------------------------------------------------------------
# Experience replay + target networks -- the two ideas that make DQN stable
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """A fixed-size ring of past transitions to sample mini-batches from.

    DQN cannot learn from consecutive steps directly -- they are far too
    correlated -- so it stores transitions and trains on random batches
    drawn from the whole recent history, which breaks the correlation.
    """

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Return a random batch as stacked arrays."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.int64),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


def hard_update(target, source):
    """Copy every parameter from ``source`` into ``target`` (a fresh clone)."""
    for pt, ps in zip(target.parameters(), source.parameters()):
        pt.data = xp.array(ps.data)


def soft_update(target, source, tau):
    """Nudge ``target`` a fraction ``tau`` of the way toward ``source``.

    ``target = tau * source + (1 - tau) * target``.  A slowly-trailing copy
    of the network gives the Q-learning target something stable to aim at,
    instead of chasing its own constantly-moving output.
    """
    for pt, ps in zip(target.parameters(), source.parameters()):
        pt.data = tau * ps.data + (1.0 - tau) * pt.data


# ---------------------------------------------------------------------------
# Losses / gradients
# ---------------------------------------------------------------------------

def huber_loss(pred, target, delta=1.0):
    """Huber (smooth-L1) loss: squared for small errors, linear for large.

    Like MSE near zero (smooth gradients) but linear in the tails, so a
    single wild Q-target can't produce a giant gradient and destabilise
    training.  Built from primitives -- ``relu`` gives us ``|x|`` -- so
    ``backward()`` differentiates it for free.
    """
    err = pred - target
    abs_err = err.relu() + (-err).relu()                 # |err|
    quadratic = 0.5 * err * err
    linear = delta * (abs_err - 0.5 * delta)
    # Pick the quadratic branch where |err| <= delta, the linear one beyond.
    small = Tensor((abs_err.data <= delta).astype(xp.float32))
    return (small * quadratic + (1.0 - small) * linear).mean()


def maximum(a, b):
    """Elementwise ``max(a, b)`` -- built from ``relu`` so it differentiates.

    ``max(a, b) = a + relu(b - a)``: the gradient flows to whichever side is
    larger, exactly as it should.  BabyTorch's ``Tensor.max`` reduces along
    an axis; PPO needs this *elementwise* version instead.
    """
    if not isinstance(b, Tensor):
        b = Tensor(np.full(a.shape, b, dtype=np.float32) if hasattr(a, "shape") else b)
    return a + (b - a).relu()


def minimum(a, b):
    """Elementwise ``min(a, b) = a - relu(a - b)`` (differentiable)."""
    if not isinstance(b, Tensor):
        b = Tensor(np.full(a.shape, b, dtype=np.float32) if hasattr(a, "shape") else b)
    return a - (a - b).relu()


def clip_tensor(x, low, high):
    """Clamp ``x`` into ``[low, high]``, elementwise and differentiably.

    PPO clips the probability ratio to keep each update small; the gradient
    is simply switched off wherever the ratio has been clipped.
    """
    return minimum(maximum(x, low), high)


def clip_grad_value(params, clip):
    """Clamp every gradient into ``[-clip, clip]`` before the optimizer step."""
    for p in params:
        if p.grad is not None:
            p.grad = xp.clip(p.grad, -clip, clip)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def evaluate(env, net, episodes=20):
    """Run the *greedy* policy and report ``(avg_return, solved_fraction)``.

    Works for any network whose largest output picks the action -- a policy
    net (largest logit = most likely action) or a Q-net (largest Q = best
    action).  Greedy means no exploration, so this measures what the agent
    has actually *learned*, apart from the randomness it explores with.
    """
    returns, solved = [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        done, total, info = False, 0.0, {}
        with babytorch.no_grad():
            while not done:
                scores = net(to_tensor(obs[None]))       # add a batch axis
                action = int(scores.argmax(axis=1)[0])
                obs, reward, done, info = env.step(action)
                total += reward
        returns.append(total)
        solved.append(bool(info.get("reached_goal", False)))
    return float(np.mean(returns)), float(np.mean(solved))


def moving_average(values, window):
    """Smooth a noisy learning curve with a trailing mean."""
    values = np.asarray(values, dtype=np.float32)
    if len(values) < window:
        return values
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(values, kernel, mode="valid")


def plot_returns(returns, path, title, window=50):
    """Save a learning curve (raw + smoothed) to ``path`` as a PNG."""
    import matplotlib
    matplotlib.use("Agg")                        # headless: no display needed
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(returns, color="#c9c7bf", linewidth=0.9, label="episode return")
    smooth = moving_average(returns, window)
    if len(smooth) > 1:
        ax.plot(range(window - 1, window - 1 + len(smooth)), smooth,
                color="#2a78d6", linewidth=2.0, label=f"{window}-episode average")
    ax.set_xlabel("episode")
    ax.set_ylabel("return")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Saved learning curve to {path}")
