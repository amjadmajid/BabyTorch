# BabyGPT — a tiny language model, built from scratch

This tutorial builds a real **decoder-only Transformer** (the GPT
architecture) out of nothing but BabyTorch tensors, and takes it through
the full life cycle of a language model:

```
    pretrain  ──►  finetune  ──►  generate
   (Shakespeare)  (nursery rhymes)  (new text)
```

Everything here is plain BabyTorch. There is no special "transformer
gradient" code anywhere — the autograd engine differentiates the whole
model automatically, exactly as it does for a two-line linear regression.

## Files

| File | What it is |
|------|-----------|
| [`model.py`](model.py) | The model: `CausalSelfAttention`, `MLP`, `Block`, `GPT`. **Read this first** — it is the heart of the tutorial, heavily commented. |
| [`train.py`](train.py) | Pretrain BabyGPT on a text corpus (Tiny Shakespeare by default). |
| [`finetune.py`](finetune.py) | Continue training a pretrained model on a smaller, stylistically different corpus. |
| [`generate.py`](generate.py) | Load a checkpoint and generate text (inference). |
| [`common.py`](common.py) | Small shared helpers (batching, checkpointing, sampling). |

## Quickstart

```bash
# from the repo root
pip install -e ".[viz]"          # add ".[gpu]" as well if you have an NVIDIA GPU

cd tutorials/llm

# 1. Pretrain on Shakespeare (downloads ~1 MB the first time).
#    On a GPU this takes a few minutes; on CPU use a smaller model (see below).
python train.py --steps 3000

# 2. Generate some Shakespeare-flavoured text.
python generate.py --prompt "ROMEO:" --tokens 400 --temperature 0.8

# 3. Finetune the same model on nursery rhymes, then generate again.
python finetune.py
python generate.py --checkpoint checkpoints/babygpt_finetuned --prompt "Twinkle"
```

Force the CPU (or GPU) for any command with the `--device` flag (or the
`BABYTORCH_DEVICE` environment variable, which works for any BabyTorch
program):

```bash
python train.py --device cpu --steps 1000 --n_embd 96 --n_layer 4
python train.py --device cuda
BABYTORCH_DEVICE=cpu python generate.py --prompt "ROMEO:"
```

### Running comfortably on a CPU

The defaults target a GPU. On a laptop CPU, shrink the model and the run
so it finishes in a couple of minutes — it will still learn to spell and
produce word-shaped text:

```bash
python train.py --steps 1500 --block_size 64 --n_embd 96 --n_head 4 --n_layer 4
```

## The architecture

![The full BabyGPT architecture, read bottom-up: token ids enter token and position embeddings, six identical Transformer blocks each apply LayerNorm, causal self-attention, a residual add, LayerNorm, the MLP and another residual add, and a final LayerNorm plus the Linear output head produce next-token logits — with a model card of the train.py defaults on the right](../../book/figures/fig-babygpt.svg)

Each Transformer block does two things: **attention** lets every position
gather information from earlier positions, and the **MLP** then processes
that information at each position. Residual connections and LayerNorm keep
the deep stack trainable.

For the full, gentle explanation — with diagrams — see **Part II of the
book**: [`../../book/README.md`](../../book/README.md).
