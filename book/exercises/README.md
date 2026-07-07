# Exercises

Two tracks, matching the two ways people read the book:

**Check yourself** — quick questions at the end of every chapter, with
answers that unfold in place. No setup, a minute or two each. If you
just want to *understand* deep learning, these plus the chapters are
enough — you never need this directory.

**Build it** — this directory: you implement real pieces of a deep
learning framework yourself, and the test suite grades you. This is the
deeper track, and it is where understanding becomes permanent.

## How the Build-it track works

Each chapter has a starter file with `NotImplementedError` stubs and a
matching grader:

```bash
# 1. open the starter and implement the stubs
$EDITOR book/exercises/ch02_autograd.py

# 2. let the tests grade you (run from the repo root)
pytest book/exercises/test_ch02_autograd.py -v
```

Ungraded stubs show as **skipped** ("not implemented yet — your turn"),
so you always see exactly where you are: implement one exercise, one
skip turns into a pass or a failure with a reason. Work in any order;
chapters are independent.

| Starter | You implement | Chapter |
|---------|---------------|---------|
| `ch01_tensors.py` | `standardize` · ★ `outer` (broadcasting only) | [1 — Tensors](../01-tensors.md) |
| `ch02_autograd.py` | `MinOperation` · ★ `AbsOperation` (forward *and* backward) | [2 — Autograd](../02-autograd.md) |
| `ch03_nn.py` | `RMSNorm` (the LLaMA norm) · ★ `bce_loss` | [3 — Neural networks](../03-neural-networks.md) |
| `ch04_training.py` | `clip_grad_norm_` · ★ `RMSProp` optimizer | [4 — Training](../04-training.md) |
| `ch05_tokenization.py` | `WordTokenizer` with `<unk>` · ★ `most_frequent_pair` (BPE's inner step) | [5 — Tokenization](../05-tokenization.md) |
| `ch06_attention.py` | `causal_attention` from raw ops · ★ `split_heads`/`merge_heads` | [6 — Attention](../06-attention.md) |
| `ch07_transformer.py` | `count_gpt_parameters` (closed form) · ★ `TiedGPT` (weight tying) | [7 — The Transformer](../07-transformer.md) |
| `ch08_generation.py` | `top_p_filter` (nucleus sampling) · ★ `generate_greedy` | [8 — Training a GPT](../08-training-a-gpt.md) |

★ = the challenge of the pair. Do the unstarred one first.

## Stuck?

Struggle a while first — that is where the learning is — then look at
[`solutions/`](solutions/). Every solution is verified by the main test
suite (`tests/test_exercises.py` runs all graders against the
solutions), so the exercises are guaranteed to stay solvable as the
library evolves.

The graders themselves are worth reading afterwards: several check your
code the same way the library checks itself — against finite-difference
gradients, or against properties ("the output at position t must not
change when the future changes") rather than memorized numbers.
