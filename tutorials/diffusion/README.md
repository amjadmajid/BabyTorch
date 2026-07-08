# Diffusion with BabyTorch

Part II generated text one token at a time. **Diffusion** generates a
different way -- the way most images are made now, from Stable Diffusion to
the picture tools in chat assistants. It never builds the sample piece by
piece. It starts from pure noise and *cleans it up* into a sample, a little
at a time.

The recipe is small enough to read in one sitting. Add noise to real data
until it is gone -- that direction is fixed arithmetic, no learning. Then
train one network to undo a single step of it. Chain those learned steps from
noise back to data, and you have a generator. Both demos here rest on the
*same* four functions in [`common.py`](common.py); the only thing that
changes between them is the denoiser's eyes.

## The two models

| Script | Data | Denoiser | Tier |
|--------|------|----------|------|
| [`toy_diffusion.py`](toy_diffusion.py) | a 2-D "two moons" point cloud | a plain MLP | fast — ~1 min on a CPU |
| [`mnist_diffusion.py`](mnist_diffusion.py) | real MNIST digits | a tiny convolutional U-Net | honest-but-slow |

| File | What it holds |
|------|---------------|
| [`common.py`](common.py) | The shared machinery: `NoiseSchedule`, `timestep_embedding`, `q_sample` (forward process), `train_step` (predict-the-noise loss), `p_sample_loop` (reverse sampling). |

The whole of Part IV needed just **one** new engine op -- `nn.Upsample`, the
mirror of `MaxPool2D` that the U-Net's decoder climbs back up with. Skip
connections and timestep conditioning are folded in by *addition*, not
concatenation, so no `concat` op is ever required.

## Quickstart

```bash
pip install -e ".[viz]"      # from the repo root, once (viz = the plots)

cd tutorials/diffusion
python toy_diffusion.py      # learn a 2-D distribution, sample it back
python mnist_diffusion.py    # denoise MNIST with a U-Net (~3 min on a CPU)
```

Both scripts train, print an honest numeric check, and plot the result;
without the `[viz]` extra they still train and print (and show an ASCII
preview). Force the device with `BABYTORCH_DEVICE=cpu python toy_diffusion.py`.

### Toy: watch a distribution appear

```bash
python toy_diffusion.py                 # train, sample, show the scatter
python toy_diffusion.py --save out.png  # ... and write the figure to disk
```

The MLP learns *where the two crescents live*, then conjures 2000 new points
out of pure noise. The script scores the match with a nearest-neighbour
distance -- proof the samples sit **on** the data, not in a blob nearby --
and plots generated (red) over target (blue). It trains to a convincing
result in about a minute.

### MNIST: the honest, harder tier

```bash
python mnist_diffusion.py               # 14x14, one-step denoising check
python mnist_diffusion.py --sample      # also dream digits from pure noise
python mnist_diffusion.py --size 28     # full resolution (slower)
```

The U-Net trains, then denoises real digits in a **single** network call and
scores the reconstruction against the noisy input -- the recovered digits are
plainly legible, around a fifth of the noisy error. This is where "baby"
earns its name: convolution in pure NumPy/CuPy is slow, so the default
downsamples to 14×14 and keeps the model tiny. One-step denoising is crisp;
sampling from scratch with a model this small gives blurry-but-recognisable
digits -- enough to prove the loop closes, not a state-of-the-art generator.

## Where the ideas come from

The book's **Part IV** ([chapters 12–13](../../book/README.md)) explains all
of this from the ground up -- the forward and reverse processes and the
predict-the-noise loss (chapter 12), and the convolutional U-Net that denoises
images (chapter 13) -- using the exact code in these files. Everything is
checked in [`tests/test_diffusion.py`](../../tests/test_diffusion.py): the
schedule, the forward process, and a denoiser proven to learn.
