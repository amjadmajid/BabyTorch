"""Image diffusion: a tiny U-Net learns to denoise -- and to dream -- digits.

Same diffusion recipe as ``toy_diffusion.py``, but the data is now MNIST, so
the denoiser has to be *convolutional*.  This is the honest, harder tier: a
small U-Net built from the three image ops BabyTorch provides -- ``Conv2D``,
``MaxPool2D`` and ``Upsample`` -- trained on real 28x28 digits.  In pure
NumPy/CuPy convolution is slow, so, exactly like the Snake ConvNet, keep the
images small and be patient (the default downsamples to 14x14).

It rests on the very same machinery in :mod:`common` as the toy did --
``NoiseSchedule``, ``q_sample``, ``train_step``, ``p_sample_loop`` -- which
is the whole point of Part IV: *nothing about diffusion changes when the data
becomes an image, only the denoiser's eyes do.*

The U-Net (encoder -> bottleneck -> decoder) folds two things in without any
new engine op:

* **additive skip connections** -- the decoder *adds* the matching encoder
  feature map back in (rather than concatenating it), so the fine detail lost
  to pooling is restored.  Addition needs no ``concat`` op;
* **additive time conditioning** -- the timestep's embedding is projected to
  a per-channel bias and *added* into every block, telling each layer how
  noisy its input is.

Run it::

    python mnist_diffusion.py                 # train + one-step denoise check
    python mnist_diffusion.py --size 28       # full resolution (slower)
    python mnist_diffusion.py --sample        # also dream digits from noise
    python mnist_diffusion.py --save out.png  # write the figure to disk

The plots need the ``[viz]`` extra; without it the script still trains and
prints its numeric checks (and an ASCII preview of a denoised digit).
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import babytorch
import babytorch.nn as nn
from babytorch.optim import Adam
from babytorch.backend import to_numpy
from babytorch.datasets import MNISTDataset

import common as C


# ---------------------------------------------------------------------------
# The data -- real MNIST digits, already scaled to [-1, 1]
# ---------------------------------------------------------------------------

def load_digits(root, size, limit=None):
    """Load MNIST as ``(N, 1, size, size)`` floats in [-1, 1].

    MNIST arrives at 28x28 already normalized to [-1, 1] -- the range
    diffusion wants, since it measures everything against unit-variance
    Gaussian noise.  If ``size == 14`` we shrink each image by averaging
    2x2 blocks, which quarters the number of pixels every convolution has
    to touch and turns a slow run into a merely patient one.
    """
    data = MNISTDataset(root=root, train=True, download=True).data  # (N,28,28)
    if limit is not None:
        data = data[:limit]
    data = data[:, None, :, :]                                       # add channel
    if size == 14:
        n = data.shape[0]
        data = data.reshape(n, 1, 14, 2, 14, 2).mean(axis=(3, 5))    # 2x2 average
    elif size != 28:
        raise ValueError("size must be 14 or 28")
    return data.astype(np.float32)


# ---------------------------------------------------------------------------
# The denoiser -- a tiny convolutional U-Net
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """One convolution that keeps the image size (padding=1), then GELU."""

    def __init__(self, in_ch, out_ch):
        self.conv = nn.Conv2D(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.conv(x))


class UNet(nn.Module):
    """Predict the noise in an image, conditioned on the timestep.

    Shape, for ``levels=2`` on a 28x28 input (``C`` = ``base``)::

        in  (B, 1, 28,28)
          enc0  ->  (B,  C, 28,28)  --- skip0 --------------------.
          pool  ->  (B,  C, 14,14)                                |
          enc1  ->  (B, 2C, 14,14)  --- skip1 ----------.         |
          pool  ->  (B, 2C,  7, 7)                       |        |
          mid   ->  (B, 2C,  7, 7)                       |        |
          up    ->  (B, 2C, 14,14)                       |        |
          dec1  ->  (B, 2C, 14,14)  + skip1  <-----------'        |
          up    ->  (B, 2C, 28,28)                                |
          dec0  ->  (B,  C, 28,28)  + skip0  <--------------------'
          head  ->  (B,  1, 28,28)   = predicted noise

    Every block's channels are chosen so the decoder feature and the encoder
    skip it adds *line up exactly* -- that matching is what lets the skip be
    an addition instead of a concatenation.
    """

    def __init__(self, in_ch=1, base=32, levels=2, time_dim=128):
        self.levels = levels
        self.time_dim = time_dim

        # A shared MLP turns the sinusoidal timestep embedding into a richer
        # vector; each block then projects *that* down to its own channel count.
        t_hidden = base * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, t_hidden, nn.GELU()),
            nn.Linear(t_hidden, t_hidden),
        )

        # channel widths: [in_ch, base, 2*base, ...], one per level boundary.
        chs = [in_ch] + [base * (2 ** i) for i in range(levels)]

        # --- encoder: a block then a 2x downsample, at each level -----------
        self.enc, self.enc_time = [], []
        prev = chs[0]
        for i in range(levels):
            self.enc.append(ConvBlock(prev, chs[i + 1]))
            self.enc_time.append(nn.Linear(t_hidden, chs[i + 1]))
            prev = chs[i + 1]
        self.pool = nn.MaxPool2D(2)

        # --- bottleneck: work at the smallest resolution --------------------
        self.mid = ConvBlock(prev, prev)
        self.mid_time = nn.Linear(t_hidden, prev)

        # --- decoder: a 2x upsample then a block, mirroring the encoder -----
        self.up = nn.Upsample(2)
        self.dec, self.dec_time = [], []
        for i in reversed(range(levels)):
            self.dec.append(ConvBlock(prev, chs[i + 1]))    # -> matches skip i
            self.dec_time.append(nn.Linear(t_hidden, chs[i + 1]))
            prev = chs[i + 1]

        self.head = nn.Conv2D(prev, in_ch, kernel_size=3, stride=1, padding=1)

    def _add_time(self, h, proj):
        """Add a per-channel time bias, broadcast over height and width."""
        b = h.shape[0]
        return h + proj.reshape(b, -1, 1, 1)

    def forward(self, x, t):
        temb = self.time_mlp(C.to_tensor(C.timestep_embedding(t, self.time_dim)))

        # encoder, stashing each level's feature map for the matching skip
        skips = []
        h = x
        for block, tproj in zip(self.enc, self.enc_time):
            h = self._add_time(block(h), tproj(temb))
            skips.append(h)
            h = self.pool(h)

        h = self._add_time(self.mid(h), self.mid_time(temb))

        # decoder, adding the time bias and then the mirror-image skip
        for block, tproj, skip in zip(self.dec, self.dec_time, reversed(skips)):
            h = self.up(h)
            h = self._add_time(block(h), tproj(temb))
            h = h + skip

        return self.head(h)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(schedule, model, data, steps, batch, lr, log_every=50):
    """Run the DDPM loss for ``steps`` minibatches; return the loss curve."""
    optimizer = Adam(model.parameters(), learning_rate=lr)
    losses = []
    t0 = time.time()
    for step in range(steps):
        idx = np.random.randint(0, data.shape[0], size=batch)
        loss = C.train_step(schedule, model, optimizer, data[idx])
        losses.append(loss)
        if step % log_every == 0 or step == steps - 1:
            rate = (step + 1) / (time.time() - t0)
            print(f"step {step:4d} / {steps} | loss {loss:.4f} "
                  f"| {rate:.1f} steps/s")
    return losses


# ---------------------------------------------------------------------------
# The check -- can the model denoise?
# ---------------------------------------------------------------------------

def denoise_once(schedule, model, x_t, t):
    """One-shot estimate of the clean image from a noised one.

    The model predicts the noise ``eps`` in ``x_t``; rearranging the forward
    process ``x_t = sqrt(ab) x0 + sqrt(1-ab) eps`` gives back an estimate of
    ``x0``.  A single network call -- no reverse loop -- so it is cheap enough
    to run as a sanity check.
    """
    with babytorch.no_grad():
        eps = to_numpy(model(C.to_tensor(x_t), t).data)
    ab = schedule.sqrt_ab[t].reshape(-1, 1, 1, 1)
    one_minus = schedule.sqrt_1m_ab[t].reshape(-1, 1, 1, 1)
    return (x_t - one_minus * eps) / ab


def denoise_report(schedule, model, images, frac=0.5):
    """Noise a batch to timestep ``frac*T``, denoise in one shot, and score it.

    If the denoiser learned anything, its one-shot reconstruction of the clean
    image must beat the noisy image itself -- i.e. recover more than we
    destroyed.  We report the per-pixel MSE before and after.
    """
    t_val = int(frac * (schedule.timesteps - 1))
    t = np.full((images.shape[0],), t_val, dtype=np.int64)
    noise = np.random.standard_normal(images.shape).astype(np.float32)
    x_t = C.q_sample(schedule, images, t, noise)
    x0_hat = np.clip(denoise_once(schedule, model, x_t, t), -1.0, 1.0)

    mse_noisy = float(((x_t - images) ** 2).mean())
    mse_denoised = float(((x0_hat - images) ** 2).mean())
    print(f"\n--- one-step denoising at t = {t_val}/{schedule.timesteps} ---")
    print(f"  MSE(noisy    , clean) : {mse_noisy:.4f}")
    print(f"  MSE(denoised , clean) : {mse_denoised:.4f}")
    good = mse_denoised < 0.6 * mse_noisy
    verdict = "PASS -- the denoiser recovers real structure" if good \
        else "OFF  -- denoising barely helps (train longer?)"
    print(f"=> denoised error is {mse_denoised / mse_noisy:.2f}x the noisy "
          f"error  ->  {verdict}\n")
    return good, (images, x_t, x0_hat)


# ---------------------------------------------------------------------------
# Pictures -- ASCII always, matplotlib if available
# ---------------------------------------------------------------------------

_RAMP = " .:-=+*#%@"


def ascii_image(img):
    """Render a single [-1, 1] image as ASCII shades (dark -> light)."""
    a = (np.clip(img.squeeze(), -1.0, 1.0) + 1.0) / 2.0     # -> [0, 1]
    rows = []
    for row in a:
        rows.append("".join(_RAMP[min(len(_RAMP) - 1, int(v * len(_RAMP)))]
                             for v in row))
    return "\n".join(rows)


def show_triplet_ascii(clean, noisy, denoised):
    """Print clean / noisy / denoised for the first image, side by side."""
    parts = [ascii_image(clean[0]), ascii_image(noisy[0]), ascii_image(denoised[0])]
    print("   clean" + " " * (clean.shape[-1] - 5)
          + "    noisy" + " " * (clean.shape[-1] - 5) + "    denoised")
    for lines in zip(*(p.split("\n") for p in parts)):
        print("  ".join(lines))


def plot(triplet, losses, generated=None, save=None):
    """Loss curve + a clean/noisy/denoised grid (+ samples).  Needs matplotlib."""
    try:
        import matplotlib
        if save:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed -- skipping the figure; "
              "install the [viz] extra to see it)")
        return

    clean, noisy, denoised = triplet
    has_samples = generated is not None
    n = min(6, clean.shape[0])
    rows = 4 if has_samples else 3
    fig, axes = plt.subplots(rows, n, figsize=(1.4 * n, 1.4 * rows))
    labels = ["clean", "noisy", "denoised"] + (["sampled"] if has_samples else [])
    banks = [clean, noisy, denoised] + ([generated] if has_samples else [])
    for r, (bank, label) in enumerate(zip(banks, labels)):
        for c in range(n):
            ax = axes[r, c]
            ax.imshow(np.clip(bank[c].squeeze(), -1, 1), cmap="gray",
                      vmin=-1, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(label, rotation=0, ha="right", va="center",
                              fontsize=10)
    fig.suptitle("BabyTorch MNIST diffusion")
    fig.tight_layout()

    if save:
        os.makedirs(os.path.dirname(os.path.abspath(save)), exist_ok=True)
        fig.savefig(save, dpi=120)
        print(f"figure written to {save}")
    else:
        plt.show()

    # The loss curve gets its own window/file.
    fig2 = plt.figure(figsize=(6, 4))
    plt.plot(losses, linewidth=1)
    plt.title("training loss (predict-the-noise MSE)")
    plt.xlabel("step")
    plt.ylabel("loss")
    if save:
        fig2.savefig(save.replace(".png", "_loss.png"), dpi=120)
    else:
        plt.show()


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="MNIST diffusion with a tiny U-Net.")
    p.add_argument("--size", type=int, default=14, choices=[14, 28],
                   help="image resolution; 14 is much faster")
    p.add_argument("--steps", type=int, default=600, help="training steps")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--base", type=int, default=16, help="U-Net base channels")
    p.add_argument("--timesteps", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--limit", type=int, default=8000,
                   help="use only this many training images (speed)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--root", type=str, default="./mnist_data")
    p.add_argument("--sample", action="store_true",
                   help="also run the full reverse process to dream new digits")
    p.add_argument("--save", type=str, default=None)
    p.add_argument("--no_plot", action="store_true")
    args = p.parse_args()

    C.set_seed(args.seed)
    print(f"device: {babytorch.device()}")

    # --- data ----------------------------------------------------------------
    data = load_digits(args.root, args.size, limit=args.limit)
    print(f"data: {data.shape} in [{data.min():.1f}, {data.max():.1f}]")

    # --- model: depth adapts to resolution (14 isn't divisible by 4) ---------
    levels = 2 if args.size % 4 == 0 else 1
    schedule = C.NoiseSchedule(timesteps=args.timesteps)
    model = UNet(in_ch=1, base=args.base, levels=levels)
    print(f"U-Net: {levels} level(s), {model.num_parameters()} parameters, "
          f"{args.timesteps}-step schedule\n")

    # --- train ---------------------------------------------------------------
    losses = train(schedule, model, data, args.steps, args.batch, args.lr)

    # --- verify: one-step denoising ------------------------------------------
    check = data[np.random.permutation(len(data))[:64]]
    ok, triplet = denoise_report(schedule, model, check)
    show_triplet_ascii(*(b[:1] for b in triplet))

    # --- optional: dream new digits from pure noise --------------------------
    generated = None
    if args.sample:
        print("\nsampling 6 digits from pure noise (slow -- full reverse loop) ...")
        generated = C.p_sample_loop(schedule, model, shape=(6, 1, args.size, args.size))

    if not args.no_plot:
        plot(triplet, losses, generated=generated, save=args.save)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
