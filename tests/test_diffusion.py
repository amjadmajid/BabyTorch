"""Tests for the diffusion machinery in ``tutorials/diffusion/common.py``.

The four pieces a DDPM rests on -- the noise schedule, the closed-form
forward process ``q_sample``, the sinusoidal timestep embedding, and the
reverse sampling loop -- plus an end-to-end check that a denoiser trained
with ``train_step`` actually learns to remove noise.

``common.py`` is loaded by file path under a unique module name so it does
not collide with the identically named ``tutorials/rl/common.py`` that
``test_rl`` imports into the same pytest process.
"""

import importlib.util
import os

import numpy as np
import pytest

import babytorch
import babytorch.nn as nn
from babytorch import Tensor
from babytorch.optim import Adam


def _load_diffusion_common():
    path = os.path.join(os.path.dirname(__file__), "..",
                        "tutorials", "diffusion", "common.py")
    spec = importlib.util.spec_from_file_location("diffusion_common", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load_diffusion_common()


# ---------------------------------------------------------------------------
# The noise schedule
# ---------------------------------------------------------------------------

def test_schedule_monotone_and_bounded():
    s = C.NoiseSchedule(timesteps=200)
    assert s.betas.shape == (200,)
    # betas ramp up, and stay strictly inside (0, 1)
    assert s.betas[0] < s.betas[-1]
    assert np.all(s.betas > 0) and np.all(s.betas < 1)
    # alpha_bar (the surviving signal) falls monotonically from ~1 toward 0
    assert np.all(np.diff(s.alpha_bars) < 0)
    assert 0.98 < s.alpha_bars[0] < 1.0
    assert s.alpha_bars[-1] < 0.2
    # the cached roots really are the roots
    assert np.allclose(s.sqrt_ab, np.sqrt(s.alpha_bars))
    assert np.allclose(s.sqrt_1m_ab, np.sqrt(1.0 - s.alpha_bars))


# ---------------------------------------------------------------------------
# The forward process: q_sample
# ---------------------------------------------------------------------------

def test_q_sample_endpoints():
    s = C.NoiseSchedule(timesteps=200)
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal((512, 3)).astype(np.float32)
    noise = rng.standard_normal((512, 3)).astype(np.float32)

    # t = 0: almost all signal survives, so x_t ~= x0
    xt0 = C.q_sample(s, x0, np.zeros(512, dtype=np.int64), noise)
    assert np.abs(xt0 - x0).mean() < 0.05

    # t = T-1: almost all signal is gone, so x_t is dominated by the noise
    xtT = C.q_sample(s, x0, np.full(512, s.timesteps - 1, np.int64), noise)
    assert np.corrcoef(xtT.ravel(), noise.ravel())[0, 1] > 0.9


def test_q_sample_preserves_unit_variance():
    # With unit-variance data and noise, x_t stays ~unit variance at every t
    # because the coefficients satisfy ab + (1 - ab) = 1.  A direct check on
    # the sqrt coefficients being paired correctly.
    s = C.NoiseSchedule(timesteps=200)
    rng = np.random.default_rng(1)
    x0 = rng.standard_normal((4000, 2)).astype(np.float32)
    for t_val in (0, 50, 100, 199):
        t = np.full(4000, t_val, dtype=np.int64)
        noise = rng.standard_normal(x0.shape).astype(np.float32)
        xt = C.q_sample(s, x0, t, noise)
        assert abs(xt.std() - 1.0) < 0.1


# ---------------------------------------------------------------------------
# The timestep embedding
# ---------------------------------------------------------------------------

def test_timestep_embedding_shape_and_range():
    emb = C.timestep_embedding(np.arange(10), dim=16)
    assert emb.shape == (10, 16)
    assert emb.max() <= 1.0 and emb.min() >= -1.0
    # t = 0: sin(0) = 0 in the first half, cos(0) = 1 in the second half
    assert np.allclose(emb[0, :8], 0.0)
    assert np.allclose(emb[0, 8:], 1.0)


# ---------------------------------------------------------------------------
# A minimal denoiser honouring common's model(x, t) -> noise contract
# ---------------------------------------------------------------------------

class _TinyDenoiser(nn.Module):
    def __init__(self, dim, hidden=64, time_dim=32):
        self.time_dim = time_dim
        self.in_proj = nn.Linear(dim, hidden)
        self.t_proj = nn.Linear(time_dim, hidden)     # additive time conditioning
        self.out = nn.Sequential(
            nn.Linear(hidden, hidden, nn.GELU()),
            nn.Linear(hidden, dim),
        )

    def forward(self, x, t):
        temb = C.timestep_embedding(t, self.time_dim).astype(np.float32)
        h = self.in_proj(x) + self.t_proj(Tensor(temb))
        return self.out(h)


# ---------------------------------------------------------------------------
# The reverse process: shapes
# ---------------------------------------------------------------------------

def test_p_sample_loop_shape():
    s = C.NoiseSchedule(timesteps=20)
    model = _TinyDenoiser(dim=2)
    x = C.p_sample_loop(s, model, shape=(5, 2))
    assert x.shape == (5, 2)
    # with trajectory=True we get every intermediate state, x_T ... x_0
    x2, traj = C.p_sample_loop(s, model, shape=(5, 2), trajectory=True)
    assert x2.shape == (5, 2)
    assert len(traj) == s.timesteps + 1
    assert traj[0].shape == (5, 2)


# ---------------------------------------------------------------------------
# End to end: training with train_step teaches a model to denoise
# ---------------------------------------------------------------------------

def _four_cluster_data(n, rng):
    centers = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=np.float32)
    pick = centers[rng.integers(0, 4, size=n)]
    return (pick + 0.1 * rng.standard_normal((n, 2))).astype(np.float32)


def test_model_learns_to_denoise():
    """Train the DDPM loss on a simple 4-cluster distribution and check that
    (a) the loss trends down and (b) one-shot denoising beats the noisy input."""
    C.set_seed(0)
    rng = np.random.default_rng(0)
    s = C.NoiseSchedule(timesteps=50)
    model = _TinyDenoiser(dim=2)
    optimizer = Adam(model.parameters(), learning_rate=2e-3)

    data = _four_cluster_data(2000, rng)
    losses = []
    for _ in range(600):
        batch = data[rng.integers(0, len(data), size=128)]
        losses.append(C.train_step(s, model, optimizer, batch))

    # (a) the loss fell: late training is lower than early training
    assert np.mean(losses[-50:]) < np.mean(losses[:50])

    # (b) one-shot denoising at a moderate noise level recovers structure
    t_val = 20
    x0 = data[:512]
    t = np.full(512, t_val, dtype=np.int64)
    noise = rng.standard_normal(x0.shape).astype(np.float32)
    x_t = C.q_sample(s, x0, t, noise)
    with babytorch.no_grad():
        eps = babytorch.to_numpy(model(Tensor(x_t), t).data)
    x0_hat = (x_t - s.sqrt_1m_ab[t_val] * eps) / s.sqrt_ab[t_val]

    mse_noisy = float(((x_t - x0) ** 2).mean())
    mse_denoised = float(((x0_hat - x0) ** 2).mean())
    assert mse_denoised < 0.7 * mse_noisy
