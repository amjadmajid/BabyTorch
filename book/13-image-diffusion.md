# Chapter 13 — Image diffusion

Chapter 12 taught a plain MLP the shape of a 2-D cloud. The recipe --
`q_sample`, `train_step`, `p_sample_loop` -- does not change at all when the
data becomes an image. What changes is the denoiser's *eyes*. A picture is
too big and too structured to hand an MLP as a flat vector of pixels: the
model has to look at it the way chapter 3's `Conv2D` does, locally and with
shared filters. And it has to output a full-resolution noise image, the same
size as its input.

That combination -- see the *whole* picture, yet produce a *full-detail*
output -- is exactly what a **U-Net** is built for, and it is the only new
idea in this chapter. Everything else is the diffusion machinery you already
have.

## The problem: context and detail at the same time

A single convolution sees only a small patch. To understand a whole digit --
"this stroke and that one belong to the same 8" -- the network needs a wide
view, and the way to widen it is to **pool**: shrink the image so each later
pixel summarises a larger region. But pooling throws detail away, and a
denoiser must hand back a crisp, full-size image. Shrinking loses exactly
what the output needs.

The U-Net resolves the tension with a shape like the letter it is named for.
An **encoder** pools the image down step by step into a small, context-rich
stack of feature maps. A **decoder** then *upsamples* back to full
resolution. And across the middle, **skip connections** hand each decoder
level the encoder's feature map from the same resolution -- restoring the
fine detail pooling discarded, right where it is needed.

![A U-Net: on the left an encoder descends from the noised image through ConvBlocks with MaxPool halving the resolution 28x28 to 14x14 to a 7x7 bottleneck; on the right a decoder ascends through ConvBlocks with Upsample doubling resolution back up to an output convolution producing the predicted noise; dashed horizontal add-skip connections join each encoder level to the matching decoder level through a plus sign; and a timestep embedding box at the bottom is added as a per-channel bias into every block](figures/fig-unet.svg)

## Three ops, one of them new

The encoder is built from two ops you met in chapter 3: `Conv2D` to see, and
`MaxPool2D` to shrink. The decoder needs the opposite of pooling -- a way to
*grow* an image back up -- and that is the single engine addition Part IV
required: **`Upsample`**. It is the mirror image of max-pooling. Pooling
keeps one value per window; upsampling copies one value *into* a window. And
because it is just a copy, its backward pass is the copy's adjoint -- sum the
gradients back onto the source pixel:

<details>
<summary><b>How it's implemented</b> — <code>babytorch/engine/operations.py</code> (nearest-neighbour upsampling)</summary>

```python
class UpsampleOperation(Operation):
    # ...
    def forward(self, a):
        self.a = a
        s = self.scale
        # Repeat along height, then width, so each pixel becomes an s x s
        # block: out[.., h*s + i, w*s + j] = in[.., h, w] for all i, j < s.
        return xp.repeat(xp.repeat(a.data, s, axis=2), s, axis=3)

    def backward(self, grad):
        s = self.scale
        N, C, H, W = self.a.shape
        # Regroup each s x s output block (H*s -> (H, s), W*s -> (W, s)) and
        # sum it back onto the single source pixel it was copied from.
        return grad.reshape(N, C, H, s, W, s).sum(axis=(3, 5)),
```

</details>

That is the whole new op -- forward and backward, checked against finite
differences in [`tests/test_nn.py`](../tests/test_nn.py) exactly the way
chapter 2 checked every other op. From those three primitives the tutorial
builds one tiny reusable block -- a convolution that keeps the image size,
followed by GELU:

<details>
<summary><b>How it's implemented</b> — <code>tutorials/diffusion/mnist_diffusion.py</code> (the U-Net's building block)</summary>

```python
class ConvBlock(nn.Module):
    """One convolution that keeps the image size (padding=1), then GELU."""

    def __init__(self, in_ch, out_ch):
        self.conv = nn.Conv2D(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.conv(x))
```

</details>

## Additive skips, additive time -- and why there is no `concat`

Here is the design decision that keeps the whole of Part IV down to a single
new op. A textbook U-Net *concatenates* the skip connection onto the decoder
feature map -- stack the two along the channel axis -- and often splices the
timestep in the same way. Concatenation would need a new engine op. So we
**add** instead.

For that to work the shapes have to line up: each decoder level is given
exactly the channel count of the encoder level it will be added to, so the
skip is a plain sum. The timestep is folded in the same additive spirit --
project its embedding to one bias per channel and add it, broadcast over
every pixel:

<details>
<summary><b>How it's implemented</b> — <code>tutorials/diffusion/mnist_diffusion.py</code> (additive time conditioning)</summary>

```python
    def _add_time(self, h, proj):
        """Add a per-channel time bias, broadcast over height and width."""
        b = h.shape[0]
        return h + proj.reshape(b, -1, 1, 1)
```

</details>

With those two additions in hand, the forward pass is a clean descent and
climb. Go down the encoder, stashing each block's output for its skip and
pooling to the next level; cross the bottleneck; come back up, upsampling,
folding the time bias into each block, and finally **adding** the matching
skip:

<details>
<summary><b>How it's implemented</b> — <code>tutorials/diffusion/mnist_diffusion.py</code> (the U-Net forward pass)</summary>

```python
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
```

</details>

`self.time_mlp`, `_add_time`, `q_sample`, `p_sample_loop` -- the timestep
embedding is the very same sinusoidal function from chapter 12, and the
training loop is byte-for-byte the shared `train_step`. Only the module in
the middle grew a U-shape.

## Try it: denoising MNIST

```bash
cd tutorials/diffusion
python mnist_diffusion.py           # 14x14, ~3 minutes on a CPU
python mnist_diffusion.py --sample  # also dream digits from pure noise
```

The script trains, then runs the honest check: take real digits, noise them
halfway, and denoise in a **single** network call -- rearranging the forward
process, `x̂₀ = (xₜ − √(1 − ᾱₜ)·ε) / √ᾱₜ`. If the model learned anything, that
one-shot guess beats the noisy input, and it does: the reconstruction lands
around a fifth of the noisy error, and the recovered digits are plainly
legible. Run with `--sample` and it also walks pure noise all the way back
through `p_sample_loop` into brand-new digits.

This is where "baby" earns its name, the same way it did for the Snake
ConvNet in chapter 11. Convolution in pure NumPy/CuPy is slow, so the default
downsamples MNIST to 14×14 and keeps the U-Net tiny; `--size 28` is the full-
resolution, slower run. One-step denoising is crisp, but a model this small,
trained this briefly, *samples* only blurry-yet-recognisable digits -- enough
to prove the loop closes, not a state-of-the-art generator. The ideas,
though, are exactly the full-scale ones.

## Where diffusion goes

Between this and the image models people actually use lie scale and two
refinements, and neither is a new idea so much as more of these. **Latent
diffusion** (the "Stable" in Stable Diffusion) runs this identical loop not
on raw pixels but in the compressed latent space of an autoencoder, so the
U-Net works on something small. And **conditioning** is what turns a random
dream into a *directed* one: feed the denoiser a text embedding -- attended
to with the cross-attention of chapter 6 -- and the reverse process is
steered, step by step, toward "an astronaut riding a horse" instead of just
"a plausible image".

Which quietly reunites the whole book. The generator that writes text one
token at a time and the generator that resolves an image out of noise are
built from the same parts you have now seen every line of -- tensors and
`backward()`, `Conv2D` and attention, `Adam` and a training loop. Different
games, one machine.

## Exercises

**Check yourself** (answers unfold):

**Q1.** Why does a U-Net add skip connections from the encoder to the
decoder at all? What goes wrong without them?

<details><summary>Answer</summary>

Pooling deliberately throws away spatial detail to gain a wide, context-rich
view -- but the denoiser has to output a full-resolution image, and the
detail it needs was destroyed on the way down. The skip connection hands the
decoder the encoder's feature map from *before* that resolution was pooled
away, so fine structure (exact edges, thin strokes) is restored at each
level. Without them the decoder would have to hallucinate all the high-
frequency detail back from the tiny bottleneck, and outputs come out blurry.

</details>

**Q2.** BabyTorch never gained a `concat` op, yet the U-Net has both skip
connections and timestep conditioning -- both usually done with
concatenation. How does it manage?

<details><summary>Answer</summary>

By **adding** instead of concatenating. The decoder blocks are sized so each
one's channel count matches the encoder skip it receives, so the skip is a
plain elementwise sum rather than a channel-axis stack. The timestep
embedding is likewise projected to one value per channel and added as a bias,
broadcast over the image. Addition needs nothing beyond the elementwise `+`
the engine already has, so the only op Part IV had to add was `Upsample`.

</details>

**Q3.** The one-step denoising check is crisp, but sampling new digits from
pure noise looks blurry. Both use the same trained network -- why the gap?

<details><summary>Answer</summary>

The one-step check starts from a *real* image that was noised, so most of the
true structure is still present and the network only has to sharpen it. Full
sampling starts from pure noise with no structure at all and must build a
digit over hundreds of reverse steps, every one of which relies on the
network's own imperfect prediction; small errors compound. A tiny U-Net
trained for a few minutes denoises well but has not learned the data
distribution sharply enough to generate from scratch -- that is what scale
and longer training buy.

</details>

**Build it** — make the digits *chosen*, not random: add **class
conditioning**. Embed the digit label `0–9` into a vector, and add it into
every block right beside the timestep bias in
[`tutorials/diffusion/mnist_diffusion.py`](../tutorials/diffusion/mnist_diffusion.py)
(the same additive trick). Train on labels, then sample a specific digit on
demand -- a miniature of how text conditions a real image model.

---

**The code:**
[`tutorials/diffusion/mnist_diffusion.py`](../tutorials/diffusion/mnist_diffusion.py) ·
[`tutorials/diffusion/common.py`](../tutorials/diffusion/common.py) ·
[`babytorch/engine/operations.py`](../babytorch/engine/operations.py) (the `Upsample` op) ·
[`tests/test_diffusion.py`](../tests/test_diffusion.py)

[← Chapter 12: Diffusion](12-diffusion.md) | [Contents](README.md) | *End of the book — [back to the repository](../README.md)*
