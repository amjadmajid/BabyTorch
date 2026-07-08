"""Shared machinery for the diffusion models.

Both training scripts -- ``toy_diffusion.py`` (a 2-D point cloud) and
``mnist_diffusion.py`` (images) -- rest on the *same* four pieces, which is
the whole point: a diffusion model is a denoiser plus a fixed noise
schedule, and nothing about that machinery cares whether the data is a pair
of coordinates or a 28x28 image.

* :class:`NoiseSchedule` -- the fixed forward-process variances, precomputed
  once so training and sampling can just index them by timestep ``t``;
* :func:`timestep_embedding` -- turn the integer ``t`` into a vector the
  network can read (a Transformer positional encoding, reused);
* :func:`q_sample` -- the *forward* process: jump straight to a noised
  ``x_t`` in closed form, no step-by-step simulation needed;
* :func:`p_sample_loop` -- the *reverse* process: start from pure noise and
  walk back to a clean sample, one denoising step at a time.

Everything else -- ``Linear``, ``Conv2D``, ``Adam``, ``mean`` -- BabyTorch
already provides.  There is no new engine machinery here beyond the one op
the U-Net needs, ``nn.Upsample``.
"""

import random

import numpy as np

import babytorch
import babytorch.nn as nn
from babytorch import Tensor
from babytorch.backend import to_numpy


def set_seed(seed):
    """Seed every source of randomness we use, for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    babytorch.manual_seed(seed)


def to_tensor(x):
    """NumPy array (or list) -> a float32 BabyTorch tensor on the active device."""
    return Tensor(np.asarray(x, dtype=np.float32))


# ---------------------------------------------------------------------------
# The noise schedule -- how fast the data is destroyed
# ---------------------------------------------------------------------------

class NoiseSchedule:
    """The forward-process variances ``beta_t``, and everything derived from
    them, precomputed once.

    Diffusion destroys a data point over ``timesteps`` steps by repeatedly
    mixing in a little Gaussian noise.  ``beta_t`` sets how much noise step
    ``t`` adds; the schedule ramps it up linearly from ``beta_start`` to
    ``beta_end``.  From the betas everything else follows:

        alpha_t      = 1 - beta_t
        alpha_bar_t  = prod_{s <= t} alpha_s      (how much *signal* survives)

    ``alpha_bar_t`` runs from ~1 (barely touched) down to ~0 (pure noise),
    and it is the only quantity :func:`q_sample` needs.  We cache its square
    roots too, since the training and sampling code reach for them every step.
    """

    def __init__(self, timesteps=200, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        self.betas = np.linspace(beta_start, beta_end, timesteps, dtype=np.float32)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = np.cumprod(self.alphas).astype(np.float32)
        # Roots used all over training and sampling -- compute them once.
        self.sqrt_ab = np.sqrt(self.alpha_bars)
        self.sqrt_1m_ab = np.sqrt(1.0 - self.alpha_bars)


def timestep_embedding(t, dim):
    """Turn integer timesteps into ``(len(t), dim)`` sinusoidal vectors.

    This is exactly the Transformer's positional encoding (chapter 7), used
    here to tell the denoiser *how noisy* its input is.  It is a **fixed**
    function of ``t`` with no parameters and no gradient, so we build it in
    plain NumPy and hand it to the network as an extra input.
    """
    t = np.asarray(t, dtype=np.float32).reshape(-1, 1)          # (B, 1)
    half = dim // 2
    freqs = np.exp(-np.log(10000.0) * np.arange(half, dtype=np.float32) / half)
    args = t * freqs[None, :]                                   # (B, half)
    return np.concatenate([np.sin(args), np.cos(args)], axis=1)  # (B, dim)


# ---------------------------------------------------------------------------
# The forward process -- add noise (closed form, one jump)
# ---------------------------------------------------------------------------

def q_sample(schedule, x0, t, noise):
    """Sample the noised ``x_t`` directly from a clean ``x_0``:

        x_t = sqrt(alpha_bar_t) * x_0  +  sqrt(1 - alpha_bar_t) * noise

    No need to simulate all ``t`` little steps -- because a sum of Gaussians
    is again Gaussian, the whole forward chain collapses to this single
    blend of signal and noise.  ``x0`` and ``noise`` share a shape; ``t`` is
    a per-example integer array.  Returns ``x_t`` as a NumPy array.
    """
    x0 = np.asarray(x0, dtype=np.float32)
    noise = np.asarray(noise, dtype=np.float32)
    # Index the cached roots by each example's t, then reshape to
    # (B, 1, 1, ...) so they broadcast over whatever feature dims follow.
    bshape = [x0.shape[0]] + [1] * (x0.ndim - 1)
    signal = schedule.sqrt_ab[t].reshape(bshape)
    noise_scale = schedule.sqrt_1m_ab[t].reshape(bshape)
    return signal * x0 + noise_scale * noise


# ---------------------------------------------------------------------------
# The training objective -- predict the noise (Ho et al. 2020)
# ---------------------------------------------------------------------------

def train_step(schedule, model, optimizer, x0):
    """One optimization step of the *simple* DDPM loss.

    Corrupt a clean batch to a **random** timestep, ask the model to guess
    the noise that was added, and regress its guess toward the truth with
    plain MSE.  That is the entire objective -- a denoiser trained on every
    noise level at once.  Returns the scalar loss.
    """
    x0 = np.asarray(x0, dtype=np.float32)
    t = np.random.randint(0, schedule.timesteps, size=x0.shape[0])
    noise = np.random.standard_normal(x0.shape).astype(np.float32)
    x_t = q_sample(schedule, x0, t, noise)

    predicted_noise = model(to_tensor(x_t), t)
    loss = ((predicted_noise - to_tensor(noise)) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.data)


# ---------------------------------------------------------------------------
# The reverse process -- denoise from pure noise back to a sample
# ---------------------------------------------------------------------------

def p_sample_loop(schedule, model, shape, trajectory=False):
    """Generate samples by running the reverse process from pure noise.

    Start at ``x_T`` ~ N(0, I) and walk backward to ``x_0``.  At each step
    the model predicts the noise in ``x_t``; that prediction pins down the
    mean of ``x_{t-1}``.  We take that mean and add a little fresh noise
    (except on the very last step, which lands on the clean sample):

        x_{t-1} = (x_t - beta_t / sqrt(1 - alpha_bar_t) * eps) / sqrt(alpha_t)
                  + sqrt(beta_t) * z            (z ~ N(0, I), and 0 at t = 0)

    With ``trajectory=True`` it also returns a list of snapshots from noise
    to sample, handy for plotting the denoising in action.
    """
    x = np.random.standard_normal(shape).astype(np.float32)     # x_T: pure noise
    snapshots = [x.copy()]
    for t in reversed(range(schedule.timesteps)):
        with babytorch.no_grad():
            t_batch = np.full((shape[0],), t, dtype=np.int64)
            eps = to_numpy(model(to_tensor(x), t_batch).data)   # predicted noise

        beta_t = schedule.betas[t]
        alpha_t = schedule.alphas[t]
        mean = (x - beta_t / schedule.sqrt_1m_ab[t] * eps) / np.sqrt(alpha_t)
        if t > 0:
            z = np.random.standard_normal(shape).astype(np.float32)
            x = mean + np.sqrt(beta_t) * z
        else:
            x = mean                                            # no noise on the last step
        snapshots.append(x.copy())
    return (x, snapshots) if trajectory else x
