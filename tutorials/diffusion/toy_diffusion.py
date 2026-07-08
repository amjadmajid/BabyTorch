"""Toy 2-D diffusion: watch a denoiser learn the *shape* of a distribution.

The smallest honest diffusion model there is.  The data is a cloud of 2-D
points shaped like two interleaving crescents (the classic "two moons").
There is no label to predict and no curve to fit -- the model's whole job is
to learn *where the points live*, so that we can then conjure brand-new
points out of pure noise and have them land on the same two crescents.

It rests entirely on the shared machinery in :mod:`common`:

    forward process   q_sample        -- noise a batch to a random timestep
    training loss      train_step      -- predict that noise with plain MSE
    reverse process    p_sample_loop   -- walk pure noise back to a sample

so the *only* thing this file adds is the denoiser itself -- a small MLP --
and the plumbing to make a picture out of it.  Because the data is 2-D and
the network is tiny, this is the **fast** tier: the defaults train to a
convincing result in about a minute on a CPU.  (The image tier,
``mnist_diffusion``, is the honest-but-slow one.)

Run it::

    python toy_diffusion.py                 # train, sample, show the plot
    python toy_diffusion.py --save out.png  # ... and write the figure to disk
    python toy_diffusion.py --no_plot       # just train + print the match check

The plots need the ``[viz]`` extra (``pip install -e ".[viz]"``); without it
the script still trains and prints its numeric verdict.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import babytorch.nn as nn
from babytorch.optim import Adam

import common as C


# ---------------------------------------------------------------------------
# The data -- two interleaving crescents
# ---------------------------------------------------------------------------

def make_two_moons(n, noise=0.08):
    """Draw ``n`` points shaped like two interleaving half-circles.

    Hand-rolled in NumPy so the tutorial depends on nothing beyond it; this
    is the same distribution scikit-learn's ``make_moons`` produces.  One
    crescent is the upper half of a circle, the other is a lower half shifted
    across and down so the two hook into each other.
    """
    n1 = n // 2
    n2 = n - n1
    t1 = np.pi * np.random.random(n1)          # angles along the top arc
    t2 = np.pi * np.random.random(n2)           # angles along the bottom arc
    upper = np.stack([np.cos(t1), np.sin(t1)], axis=1)
    lower = np.stack([1.0 - np.cos(t2), 0.5 - np.sin(t2)], axis=1)
    pts = np.concatenate([upper, lower], axis=0)
    pts += noise * np.random.standard_normal(pts.shape)
    return pts.astype(np.float32)


# ---------------------------------------------------------------------------
# The denoiser -- a plain MLP that guesses the noise in a point
# ---------------------------------------------------------------------------

class MLPDenoiser(nn.Module):
    """Predict the noise added to a 2-D point, given how noisy it is.

    The network reads two things: the (noised) point ``x_t``, and the
    timestep ``t`` that says how much noise is in it.  It returns its guess
    of the noise -- a vector the same shape as the point.

    Time is folded in **additively**: we project the point to a hidden vector,
    project the timestep's sinusoidal embedding to a hidden vector of the same
    width, and simply *add* them before the main stack of layers.  Additive
    conditioning is the cheap trick that lets the whole tutorial avoid a
    second new engine op -- there is no need to *concatenate* the time signal
    onto the input, so ``concat`` is never required (the U-Net later leans on
    the same idea for its skip connections).
    """

    def __init__(self, data_dim=2, hidden=128, time_dim=64):
        self.time_dim = time_dim
        # Two projections into a shared hidden space, then add them.
        self.in_proj = nn.Linear(data_dim, hidden)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden, nn.GELU()),
            nn.Linear(hidden, hidden),
        )
        # The body: a couple of nonlinear layers, then back down to 2-D.
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden, nn.GELU()),
            nn.Linear(hidden, hidden, nn.GELU()),
            nn.Linear(hidden, data_dim),
        )

    def forward(self, x, t):
        # timestep_embedding is a fixed, gradient-free function of t (plain
        # NumPy); the learnable time_mlp turns it into something the body can
        # use.  Adding the two projections is the "conditioning".
        temb = C.timestep_embedding(t, self.time_dim)          # (B, time_dim)
        h = self.in_proj(x) + self.time_mlp(C.to_tensor(temb))  # additive
        return self.net(h)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(schedule, model, data, steps, batch, lr, log_every=500):
    """Run the DDPM loss for ``steps`` minibatches and return the loss curve."""
    optimizer = Adam(model.parameters(), learning_rate=lr)
    losses = []
    for step in range(steps):
        idx = np.random.randint(0, data.shape[0], size=batch)
        loss = C.train_step(schedule, model, optimizer, data[idx])
        losses.append(loss)
        if step % log_every == 0 or step == steps - 1:
            print(f"step {step:5d} / {steps} | loss {loss:.4f}")
    return losses


# ---------------------------------------------------------------------------
# The check -- did the samples actually land on the data?
# ---------------------------------------------------------------------------

def _mean_nn_distance(a, b, exclude_self=False):
    """Mean over ``a`` of the distance from each point to its nearest in ``b``."""
    d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))  # (Na, Nb)
    if exclude_self:
        np.fill_diagonal(d, np.inf)          # a point is not its own neighbour
    return float(d.min(axis=1).mean())


def report_match(target, generated, cap=1000):
    """Print a quantitative verdict on how well ``generated`` matches ``target``.

    Two cheap, honest signals, both in the standardized space the model works
    in:

    * per-dimension mean and std -- a sample that matches should reproduce
      them (target is ~0 mean, ~1 std by construction here);
    * nearest-neighbour distance -- how far a generated point sits from the
      nearest real one, measured against the data's *own* point spacing.  If
      samples land on the crescents, the two distances are comparable; if the
      model collapsed to a blob, the sample->target distance blows up.
    """
    # Subsample so the O(N*M) distance matrix stays small.
    t = target[np.random.permutation(len(target))[:cap]]
    g = generated[np.random.permutation(len(generated))[:cap]]

    d_tt = _mean_nn_distance(t, t, exclude_self=True)   # the data's own spacing
    d_gt = _mean_nn_distance(g, t)                      # samples -> data
    ratio = d_gt / d_tt

    with np.printoptions(precision=3, suppress=True, sign=" "):
        print("\n--- does the sample match the target? ---")
        print(f"              target          generated")
        print(f"mean     {t.mean(0)}     {g.mean(0)}")
        print(f"std      {t.std(0)}     {g.std(0)}")
        print("nearest-neighbour distance (standardized units):")
        print(f"  target -> target : {d_tt:.3f}   (the data's own spacing)")
        print(f"  sample -> target : {d_gt:.3f}   (how close samples land)")

    std_ok = np.all(np.abs(g.std(0) / t.std(0) - 1.0) < 0.30)
    good = ratio < 2.5 and std_ok
    verdict = "PASS -- samples sit on the distribution" if good \
        else "OFF  -- samples stray from the data (train longer?)"
    print(f"=> distance ratio {ratio:.2f}, std {'ok' if std_ok else 'off'}"
          f"  ->  {verdict}\n")
    return good


# ---------------------------------------------------------------------------
# The picture
# ---------------------------------------------------------------------------

def plot(target, generated, losses, save=None):
    """Loss curve + target-vs-sample scatter.  No-op if matplotlib is absent."""
    try:
        import matplotlib
        if save:
            matplotlib.use("Agg")            # write a file without a display
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed -- skipping the plot; "
              "install the [viz] extra to see it)")
        return

    fig, (ax_loss, ax_pts) = plt.subplots(1, 2, figsize=(11, 5))

    ax_loss.plot(losses, linewidth=1)
    ax_loss.set_title("training loss (predict-the-noise MSE)")
    ax_loss.set_xlabel("step")
    ax_loss.set_ylabel("loss")

    ax_pts.scatter(target[:, 0], target[:, 1], s=6, alpha=0.4,
                   label="target data", color="tab:blue")
    ax_pts.scatter(generated[:, 0], generated[:, 1], s=6, alpha=0.5,
                   label="generated", color="tab:red")
    ax_pts.set_title("sampled from pure noise vs. the real data")
    ax_pts.set_aspect("equal")
    ax_pts.legend()

    fig.suptitle("BabyTorch toy diffusion (two moons)")
    fig.tight_layout()

    if save:
        os.makedirs(os.path.dirname(os.path.abspath(save)), exist_ok=True)
        fig.savefig(save, dpi=120)
        print(f"figure written to {save}")
    else:
        plt.show()


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Toy 2-D diffusion on two moons.")
    p.add_argument("--n_data", type=int, default=4000, help="dataset size")
    p.add_argument("--steps", type=int, default=3000, help="training steps")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--timesteps", type=int, default=200,
                   help="length of the noise schedule")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n_samples", type=int, default=2000,
                   help="points to generate for the plot and the check")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", type=str, default=None,
                   help="write the figure here instead of showing it")
    p.add_argument("--no_plot", action="store_true", help="skip plotting entirely")
    args = p.parse_args()

    C.set_seed(args.seed)

    # --- data: two moons, standardized to ~zero mean / unit std --------------
    # Diffusion measures everything against unit-variance Gaussian noise, so
    # the data has to live on that scale too.  We keep mean/std to undo the
    # standardization when we draw the final picture in the data's real units.
    raw = make_two_moons(args.n_data)
    mean, std = raw.mean(0), raw.std(0)
    data = (raw - mean) / std

    # --- the three fixed pieces + the denoiser -------------------------------
    schedule = C.NoiseSchedule(timesteps=args.timesteps)
    model = MLPDenoiser(data_dim=2, hidden=args.hidden)
    print(f"denoiser: {model.num_parameters()} parameters, "
          f"{args.timesteps}-step schedule\n")

    # --- train ---------------------------------------------------------------
    losses = train(schedule, model, data, args.steps, args.batch, args.lr)

    # --- sample: pure noise -> new points ------------------------------------
    print("\nsampling from pure noise ...")
    gen = C.p_sample_loop(schedule, model, shape=(args.n_samples, 2))

    # --- verify + picture (both in the data's real units) --------------------
    ok = report_match(data, gen)
    if not args.no_plot:
        plot(raw, gen * std + mean, losses, save=args.save)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
