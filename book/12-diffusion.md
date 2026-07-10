# Chapter 12 — Diffusion

*Part IV, chapter 1 of 2. Generation becomes an iterative denoising process
rather than an autoregressive sequence.*

## Learning goals

By the end of this chapter, you will be able to:

- distinguish the fixed forward noising process from the learned reverse path;
- sample any noisy timestep directly using the cumulative noise schedule;
- derive the noise-prediction training objective; and
- trace iterative sampling from Gaussian noise to a data sample.

Part II built a generator that works one token at a time: predict the next
symbol, append it, repeat. That is how language models write. Images want a
different kind of generator -- and **diffusion** is the one that now draws
most of them, from Stable Diffusion to the image tools inside chat
assistants. It does not build a picture left to right. It starts from a
screenful of pure static and *cleans it up* into a sample, a little at a
time.

The trick behind it is almost too simple. Take a real data point and
gradually bury it in noise until nothing is left -- that direction is easy,
it is just adding random numbers. Now learn to run the tape **backward**:
given a noisy thing, guess what noise was added, and subtract a bit of it.
Chain enough of those backward steps together and you can start from noise
you made up yourself and arrive at something that looks like real data.

![The diffusion process as a chain of tiles from a clean sample on the left to pure noise on the right: a red forward arrow across the top adds a little Gaussian noise at each of T steps with no network, a blue reverse arrow along the bottom is the trained network predicting the noise and subtracting one step, and the closed form x_t equals sqrt(alpha-bar-t) times x_0 plus sqrt(1 minus alpha-bar-t) times epsilon lets you jump straight to any step](figures/fig-diffusion.svg)

Nothing here is specific to images. The same four pieces work on a cloud of
2-D points, which is exactly how this chapter will demonstrate them; chapter
13 swaps the denoiser's eyes for convolutions and points the identical
machinery at MNIST. Everything lives in one small shared file,
[`tutorials/diffusion/common.py`](../tutorials/diffusion/common.py) -- a
diffusion model is a denoiser plus a fixed noise schedule, and that file is
the schedule and the glue.

## The forward process: destroy the data in closed form

The forward process ramps a data point `x₀` up to pure noise over `T` steps.
Step `t` mixes in a little Gaussian noise, and how much is set by a fixed
**schedule** `βₜ` that grows from nearly nothing to a small ceiling. From the
betas, two derived quantities matter:

```
αₜ = 1 − βₜ                 (how much signal step t keeps)
ᾱₜ = α₁ · α₂ · … · αₜ        (how much signal survives all the way to t)
```

`ᾱₜ` slides from ~1 (barely touched) down toward ~0 (pure noise), and it is
the *only* number the forward process needs. Because a sum of Gaussians is
again Gaussian, the whole `t`-step chain collapses into a single blend --
there is no need to simulate the steps one by one:

```
xₜ  =  √ᾱₜ · x₀  +  √(1 − ᾱₜ) · ε ,      ε ~ N(0, I)
```

The schedule precomputes the betas, the running product, and the two square
roots once, so training and sampling can just index them by `t`:

<details>
<summary><b>How it's implemented</b> — <code>tutorials/diffusion/common.py</code> (the noise schedule)</summary>

```python
class NoiseSchedule:
    # ...
    def __init__(self, timesteps=200, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        self.betas = np.linspace(beta_start, beta_end, timesteps, dtype=np.float32)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = np.cumprod(self.alphas).astype(np.float32)
        # Roots used all over training and sampling -- compute them once.
        self.sqrt_ab = np.sqrt(self.alpha_bars)
        self.sqrt_1m_ab = np.sqrt(1.0 - self.alpha_bars)
```

</details>

The blend itself is one line of arithmetic. `q_sample` reads a batch of
clean points, a per-example timestep `t`, and a batch of noise, and returns
the noised `xₜ` -- reshaping the cached roots so they broadcast over whatever
feature dimensions the data happens to have (two coordinates here, or a whole
image in the next chapter):

<details>
<summary><b>How it's implemented</b> — <code>tutorials/diffusion/common.py</code> (the forward process, one jump)</summary>

```python
def q_sample(schedule, x0, t, noise):
    # ...
    x0 = np.asarray(x0, dtype=np.float32)
    noise = np.asarray(noise, dtype=np.float32)
    # Index the cached roots by each example's t, then reshape to
    # (B, 1, 1, ...) so they broadcast over whatever feature dims follow.
    bshape = [x0.shape[0]] + [1] * (x0.ndim - 1)
    signal = schedule.sqrt_ab[t].reshape(bshape)
    noise_scale = schedule.sqrt_1m_ab[t].reshape(bshape)
    return signal * x0 + noise_scale * noise
```

</details>

Notice there is no network in any of this. The forward process is fixed
arithmetic -- it never learns anything. All the learning is in undoing it.

## The training objective: predict the noise

Here is the whole idea of training, and it is startlingly plain. Take a clean
batch. Pick a **random** timestep `t` for each example. Noise them to that
level with `q_sample`. Then ask the network one question: *what noise did I
just add?* Score its guess against the truth with plain mean-squared error.

That is the entire loss (Ho et al., 2020) -- a denoiser trained on every
noise level at once, from "barely touched" to "almost pure static":

<details>
<summary><b>How it's implemented</b> — <code>tutorials/diffusion/common.py</code> (one DDPM training step)</summary>

```python
def train_step(schedule, model, optimizer, x0):
    # ...
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
```

</details>

It is the same four-step loop as every other model in this book -- forward,
loss, `backward()`, `step()` -- and the same `Adam` from chapter 4. Why
predict the *noise* rather than the clean point directly? The two are
algebraically equivalent (rearrange the blend and you can recover one from
the other), but regressing the noise keeps the target at a steady unit scale
for every `t`, which trains far more smoothly.

One subtlety: the loss is deliberately **jumpy**. Each step scores a random
mix of timesteps, and high-`t` examples (almost pure noise) are near
impossible to denoise, so their error stays high no matter how well the model
trains. The curve bounces around a downward drift rather than sliding to
zero -- healthy, not broken.

## Telling the network how noisy the input is

The network sees `xₜ` but not `t`, and it needs `t`: the right thing to do to
a barely-noised point is very different from the right thing to do to near-
static. So we hand it `t`, encoded the same way chapter 7 encoded token
positions -- a **sinusoidal embedding**, a fixed function of `t` with no
parameters and no gradient:

<details>
<summary><b>How it's implemented</b> — <code>tutorials/diffusion/common.py</code> (the timestep embedding)</summary>

```python
def timestep_embedding(t, dim):
    # ...
    t = np.asarray(t, dtype=np.float32).reshape(-1, 1)          # (B, 1)
    half = dim // 2
    freqs = np.exp(-np.log(10000.0) * np.arange(half, dtype=np.float32) / half)
    args = t * freqs[None, :]                                   # (B, half)
    return np.concatenate([np.sin(args), np.cos(args)], axis=1)  # (B, dim)
```

</details>

The denoiser projects this vector and **adds** it into its hidden
activations -- the same additive conditioning chapter 13 will lean on for its
skip connections. Addition, not concatenation, which quietly saves us from
needing any new engine op.

## The reverse process: sampling from noise

Training gives us a network `εθ(xₜ, t)` that guesses the noise. Generation
runs it in a loop. Start at `x_T`, pure `N(0, I)` noise you drew yourself. At
each step the model predicts the noise in `xₜ`; that prediction pins down
where `xₜ₋₁` should sit; take that mean and add back a little fresh noise
(except on the very last step, which lands on the clean sample):

<details>
<summary><b>How it's implemented</b> — <code>tutorials/diffusion/common.py</code> (the reverse process)</summary>

```python
def p_sample_loop(schedule, model, shape, trajectory=False):
    # ...
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
```

</details>

The whole loop runs under `no_grad` -- we are only *using* the network now,
not training it. `trajectory=True` keeps every intermediate `xₜ`, which is
what you would plot to watch a shapeless blob resolve into a sample.

## Try it: teaching a network the shape of two moons

The smallest honest diffusion model there is fits a cloud of 2-D points
shaped like two interleaving crescents. There is no label and no curve to
fit -- the network's whole job is to learn *where the points live*, so we can
then conjure new points out of noise and have them land on the same two
moons. The denoiser is a plain MLP; the machinery is the shared file above,
unchanged.

```bash
cd tutorials/diffusion
python toy_diffusion.py
```

It trains in about a minute on a CPU, samples 2000 points from pure noise,
and prints an honest verdict -- a nearest-neighbour distance that checks the
generated points really sit on the data, not in a blob nearby -- before
plotting the two side by side. The red cloud lands squarely on the blue: a
distribution learned, and sampled, with four functions and an MLP.

This is the fast tier. In chapter 13 the data becomes images, the denoiser
grows convolutions and a U-shape, and the exact same `q_sample`,
`train_step`, and `p_sample_loop` drive it -- because, as promised, none of
the diffusion machinery cares whether `x` is a pair of coordinates or a
28×28 picture.

## Key takeaways

- The forward process is fixed and admits direct sampling at any timestep.
- Training asks a timestep-conditioned network to predict the noise mixed into
  a clean sample, producing an ordinary mean-squared-error objective.
- Generation repeatedly applies a learned reverse update, trading many model
  evaluations for a flexible way to model complex distributions.

## Exercises

**Check yourself** (answers unfold):

**Q1.** The forward process has no parameters and is never trained. So where
does *all* the learning in a diffusion model happen?

<details><summary>Answer</summary>

Entirely in the denoiser `εθ(xₜ, t)` -- the network trained by `train_step`
to predict the noise. The forward process (`q_sample`, the schedule) is fixed
arithmetic that only ever *creates training data*: it turns a clean point
plus a random timestep into a noised input and the known noise to regress
against. Sampling then runs the learned denoiser backward.

</details>

**Q2.** `q_sample` jumps straight to `xₜ` in one line instead of looping `t`
times, adding a little noise each time. Why is that allowed?

<details><summary>Answer</summary>

Because each step adds independent Gaussian noise, and a sum of Gaussians is
itself Gaussian. Composing `t` small noising steps therefore has a
closed form: `xₜ = √ᾱₜ · x₀ + √(1 − ᾱₜ) · ε`, where `ᾱₜ` is the running
product of the `αₛ`. The single blend has exactly the distribution the
step-by-step chain would produce, so training can sample any `t` in O(1)
without simulating the walk.

</details>

**Q3.** During training the loss bounces around instead of sliding smoothly
to zero. Why is that expected here, and not a sign something is wrong?

<details><summary>Answer</summary>

Each step evaluates a *random* set of timesteps. For small `t` the input is
barely noised and the noise is easy to predict, so the error is low; for
large `t` the input is almost pure static and the added noise is nearly
unrecoverable, so the error stays high however well the model has trained.
The per-step loss is an average over whichever `t` were drawn, so it jitters
around a slowly falling trend rather than converging to zero.

</details>

**Build it** — add a **cosine noise schedule**. The linear `βₜ` here destroys
small images a little abruptly; Nichol & Dhariwal's cosine schedule sets
`ᾱₜ` directly from a cosine curve and often samples better. Add it as an
option to `NoiseSchedule` in
[`tutorials/diffusion/common.py`](../tutorials/diffusion/common.py) --
everything downstream reads the cached `ᾱ` arrays, so nothing else has to
change.

---

**The code:**
[`tutorials/diffusion/common.py`](../tutorials/diffusion/common.py) ·
[`tutorials/diffusion/toy_diffusion.py`](../tutorials/diffusion/toy_diffusion.py) ·
[`tests/test_diffusion.py`](../tests/test_diffusion.py) (schedule, forward process, and a denoiser proven to learn)

[← Chapter 11: Deep Q-Learning](11-deep-q-learning.md) | [Contents](README.md) | [Chapter 13: Image diffusion →](13-image-diffusion.md)
