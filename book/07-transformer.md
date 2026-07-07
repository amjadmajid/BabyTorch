# Chapter 7 — The Transformer

*Part II, chapter 3 of 4. Attention is built. Now we assemble it into
the architecture behind GPT-2, GPT-4 and LLaMA — and see why a
24-layer-deep stack of it can be trained at all.*

## The other half: the MLP

Attention moves information *between* positions. It is followed by a
small two-layer network applied *at each position independently* —
the same weights for every position, no cross-talk
([`tutorials/llm/model.py`](../tutorials/llm/model.py)):

```python
class MLP(nn.Module):
    def __init__(self, n_embd, dropout=0.0):
        self.fc = nn.Linear(n_embd, 4 * n_embd)    # widen 4x
        self.gelu = nn.GELU()                      # bend
        self.proj = nn.Linear(4 * n_embd, n_embd)  # narrow back
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.proj(self.gelu(self.fc(x))))
```

A useful (if loose) division of labour: **attention gathers, the MLP
digests.** After a position has pulled in "there was a cat, thirteen
tokens ago", the MLP is where that information gets processed into
something the next layer can use. The temporary widening to `4·C`
(the standard Transformer ratio) gives it room to compute; GELU is
chapter 3's smooth ReLU, the GPT-family choice.

## The Block, and the two tricks that make depth possible

One Transformer **block** is just the two sublayers wired together —
but the wiring is everything:

```python
class Block(nn.Module):
    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # communicate
        x = x + self.mlp(self.ln2(x))    # compute
        return x
```

```
   x ──┬────────────────────────►(+)──┬────────────────────►(+)──► out
       │                          ▲   │                      ▲
       └─► LayerNorm ─► attention ┘   └─► LayerNorm ─► MLP ──┘
```

**Trick 1 — residual connections.** The sublayer's output is *added
to* its input, not substituted for it. Two consequences. For learning:
the block only needs to produce a useful *adjustment* to `x`, not
rebuild the whole representation — and if it has nothing to add,
"do nothing" is trivially available. For gradients: chapter 2 says
addition passes gradients through unchanged, so the chain of `+`s forms
an uninterrupted highway from the loss straight back to layer 1. Stack
20 plain layers and the repeated multiplication of local derivatives
shrinks (or blows up) the signal exponentially; the residual highway is
why deep Transformers don't suffer that fate.

**Trick 2 — pre-LayerNorm.** Each sublayer sees a normalized copy of
`x` (chapter 3's LayerNorm: zero mean, unit variance per position, then
a learned rescale). Meanwhile the residual stream itself accumulates
sums of many sublayer outputs and would otherwise drift in scale with
depth; normalizing *at each sublayer entrance* keeps every layer
operating on inputs of the same magnitude, whether it is layer 1 or
layer 24.

Neither trick adds expressive power. Both exist for one reason:
*trainability*. The Transformer's real innovation is not just attention
— it is an architecture whose gradients survive depth.

## The whole model

`GPT` is embeddings, a stack of blocks, and a readout:

```python
class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd=64, n_head=4,
                 n_layer=4, dropout=0.0):
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = [Block(n_embd, n_head, block_size, dropout)
                       for _ in range(n_layer)]
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
```

```
 token ids (B, T)
     │
     ├─► token_embedding ────► what each token is   ─┐
     │                                                (+) ─► dropout ─► x (B,T,C)
     └─► position_embedding ─► where each token sits ─┘
     ┌───────────────────────────────────────────┐
     │            Block × n_layer                │
     │   x = x + attn(ln1(x))   ← communicate    │
     │   x = x + mlp(ln2(x))    ← compute        │
     └───────────────────────────────────────────┘
     │
     ▼
   final LayerNorm ──► head: Linear(C → vocab_size)
     │
     ▼
 logits (B, T, vocab_size):  at every position, a score for
                             every token that could come next
```

Note what the output is: not one prediction, but a prediction **at
every position** — position `t`'s logits are its guess for token
`t+1`, having attended to positions `0..t` only. All `T` guesses train
in parallel.

The plain Python list of blocks deserves a remark: chapter 3's
`Module.parameters()` walks attribute lists too, so the stack's
parameters are found automatically, and `loss.backward()` reaches all
of them. No `ModuleList` ceremony needed.

## The loss is chapter 3's, unchanged

```python
def loss(self, idx, targets):
    logits = self.forward(idx)              # (B, T, V)
    logits = logits.reshape(B * T, V)       # every position = one example
    targets = targets.reshape(B * T)
    return nn.CrossEntropyLoss()(logits, targets)
```

Flatten batch and time together and next-token prediction *is*
classification with `V = vocab_size` classes — the exact loss from
chapter 3, fed `B·T` examples per step. This equivalence is the
punchline of the whole book: **a GPT is a classifier in a loop.**

## Size: where the parameters live

The tutorial's default configuration (`train.py`: `n_embd=192`,
`n_head=6`, `n_layer=6`, `block_size=128`, char vocab ≈ 65) weighs in
around **2.7 million parameters**:

| piece | count | share |
|-------|-------|-------|
| 6 blocks (attention + MLP + norms) | ≈ 2,669,000 | ~98% |
| position embeddings (128 × 192) | 24,576 | |
| token embeddings (65 × 192) | 12,480 | |
| final norm + output head | ≈ 12,900 | |

Almost everything is in the blocks, split roughly 1:2 between attention
and MLP weights — and every one of those matrices exists to serve `@`.
GPT-2 has the same skeleton at `n_embd=768, n_layer=12` (124M);
GPT-3 at `n_embd=12288, n_layer=96` (175B). The knobs and what they
buy:

* `n_embd` — width of every token's vector; capacity of the stream.
* `n_layer` — depth; more rounds of communicate-then-compute.
* `n_head` — how many attention patterns per layer (must divide `n_embd`).
* `block_size` — maximum context length; attention cost grows as T².
* `dropout` — regularization, for when the model outgrows the data.

**Try it**

```python
>>> import sys; sys.path.insert(0, "tutorials/llm")
>>> from model import GPT
>>> import babytorch
>>> model = GPT(vocab_size=65, block_size=32, n_embd=64, n_head=4, n_layer=2)
>>> model.num_parameters()
110529
>>> ids = (babytorch.rand(2, 10).data * 65).astype(int)   # 2 sequences, 10 tokens
>>> model(ids).shape
(2, 10, 65)                       # per position, a score for every token
```

Random weights, so the scores are noise — the model babbles. Chapter 8
turns the babble into Shakespeare.

---

**Source files for this chapter:**
[`tutorials/llm/model.py`](../tutorials/llm/model.py) (`MLP`, `Block`, `GPT`) ·
[`babytorch/nn/nn.py`](../babytorch/nn/nn.py) (`LayerNorm`, `Embedding`, `GELU`) ·
[`tests/test_training.py`](../tests/test_training.py) (a tiny GPT proven to learn)

[← Chapter 6: Attention](06-attention.md) | [Contents](README.md) | [Chapter 8: Training a GPT →](08-training-a-gpt.md)
