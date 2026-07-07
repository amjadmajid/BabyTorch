# BabyTorch Test Suite

The tests are the ground truth that BabyTorch actually works. They run on
plain **NumPy (CPU)** so they pass on any machine, and the same suite also
runs on **CuPy (GPU)** — set `BABYTORCH_DEVICE=cuda`.

## Running

```bash
pip install -e ".[dev]"     # pytest + optional viz deps
pytest                      # run everything (CPU by default)
BABYTORCH_DEVICE=cuda pytest  # run the same suite on the GPU
```

## What each file covers

| File | What it checks |
|------|----------------|
| `test_autograd.py` | Every differentiable op, compared against a **finite-difference** gradient (`conftest.check_gradient`). This is the core correctness proof of the engine. |
| `test_nn.py` | Layers (Linear, Embedding, LayerNorm, Dropout, Conv2D…), the loss functions, and model save/load. |
| `test_optim.py` | SGD/Adam/AdamW actually reduce a loss; the LR schedulers produce the right learning rates. |
| `test_tokenizer.py` | Character and BPE tokenizers round-trip text and save/load. |
| `test_training.py` | End-to-end: linear regression, an MLP solving XOR, and a **tiny GPT overfitting a sequence** (the strongest single signal that the whole Transformer path is correct). |
| `test_pytorch_parity.py` | *Optional.* If `torch` is installed, checks BabyTorch's numbers match PyTorch's. Skipped automatically otherwise. |

## How the gradient check works

For a function `f(x)` we compare two gradients:

- the **analytic** one BabyTorch computes with `backward()`, and
- a **numerical** estimate `(f(x+ε) − f(x−ε)) / 2ε`, which needs no calculus.

If they agree to a tight tolerance, the backward pass is correct. See
`conftest.py`.
