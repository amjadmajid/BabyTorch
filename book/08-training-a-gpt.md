# Chapter 8 — Training a GPT

*Part II, chapter 4 of 4. The model is built. Now we put it through a
language model's full life cycle — pretrain on a corpus, generate text,
finetune on a new voice — using the runnable scripts in
[`tutorials/llm/`](../tutorials/llm/).*

## Learning goals

By the end of this chapter, you will be able to:

- connect pretraining code to every component developed in Chapters 1-7;
- interpret language-model loss and compare it with a uniform baseline;
- explain temperature, top-k, nucleus sampling, and KV caching; and
- distinguish pretraining, task adaptation, and post-training.

## Pretraining: the loop meets a mountain of text

[`train.py`](../tutorials/llm/train.py) reads top to bottom as a recap
of the whole book:

```python
text = load_tiny_shakespeare()            # ~1.1 MB of raw text   (ch. 5)
tokenizer = CharTokenizer(text)           # vocab ≈ 65 characters (ch. 5)
data = encode_corpus(text, tokenizer)     # one long id array     (ch. 5)
train_data, val_data = data[:n], data[n:] # 90/10 honesty split   (ch. 4)

model = GPT(vocab_size, block_size=128,   # the Transformer       (ch. 6-7)
            n_embd=192, n_head=6, n_layer=6, dropout=0.1)
optimizer = AdamW(model.parameters(), learning_rate=3e-3,
                  weight_decay=0.1)       # the GPT optimizer     (ch. 4)
scheduler = CosineWarmupLR(optimizer, warmup_steps, total_steps)

for step in range(steps):
    scheduler.step(step)
    x, y = get_batch(train_data, block_size, batch_size)  # (B,T) windows
    loss = model.loss(x, y)               # forward + cross-entropy
    optimizer.zero_grad()
    loss.backward()                       # ch. 2 does the rest
    optimizer.step()
```

Nothing here is new — that is the point. "Pretraining" is not a special
procedure; it is chapter 4's four-step loop, fed self-supervised
batches from chapter 5, updating chapter 7's model. What is new is only
the *scale of the ambition*: absorb the statistics of a language into
the weights, with no labels beyond the text itself.

## Reading the loss curve

Cross-entropy gives training a meaningful yardstick. With 65 equally
likely characters, a clueless model's loss is `−ln(1/65) ≈ 4.17` — and
that is exactly where the curve starts. As it falls, the samples climb
a ladder of competence:

```
loss ~4.2   random keysmash        "xQj;wRk?vB"
loss ~3.0   letter frequencies     "e soaet htn re"
loss ~2.5   word-shaped strings    "ther sonot hind"
loss ~2.0   words, some grammar    "the king hath sent"
loss ~1.5   dialogue that scans    "ROMEO: What say you?"
```

Print both train and val loss (the script does, every 100 steps):
while they fall together the model is learning Shakespeare; when train
keeps falling and val turns up, it has begun memorizing the corpus —
chapter 4's overfitting, live.

## Generation: the loop that writes

A trained model maps context to next-token probabilities. Writing is
just applying it repeatedly — `GPT.generate` in
[`model.py`](../tutorials/llm/model.py):

![Generation is a classifier in a loop: the context (cropped to block_size) is forwarded through the frozen GPT to logits, which are divided by the temperature, filtered to the top-k, turned into probabilities by softmax, and sampled; the new token is appended and the loop goes around again — one token per lap](figures/fig-generation.svg)

Three practical details, all visible in the code:

* **Context cropping.** The model can only attend to `block_size`
  tokens, so generation feeds `idx[:, -block_size:]` — a sliding
  window. (Position embeddings exist only up to `block_size`; beyond
  that the oldest text simply falls out of view.)
* **`no_grad` + `eval`.** No backward pass will follow, so recording
  the graph (ch. 2) would waste memory; dropout must be off.
* **Sampling, not argmax.** Always picking the single most likely
  character locks the model into repetitive loops. We *sample* from the
  distribution — and control the randomness.

**Temperature** rescales the logits before softmax
(`logits / temperature`):

| temperature | effect |
|-------------|--------|
| → 0 | argmax: safest token every time; soon a repeating rut |
| 0.7–0.9 | sharper than learned: coherent, a little conservative |
| 1.0 | the model's honest learned distribution |
| > 1.2 | flattened: creative sliding into keysmash |

**Top-k** is the complementary guard: keep only the `k` most likely
tokens, zero the rest, renormalize. It cuts off the long tail of
individually-unlikely junk tokens whose *combined* probability is large
enough to derail a sentence.

**The KV cache** removes generation's hidden waste. Each lap of the
loop needs one row of logits — the newest — but a naive forward
recomputes attention over the *whole* context to produce it: the keys
and values of every old position are rebuilt from scratch, lap after
lap, only to come out identical every time. So `generate` keeps them.
The prompt is forwarded once and every block stores its `k` and `v`
arrays (**prefill**); after that, each lap forwards **only the token
sampled on the previous lap**, computes its single new q, k and v, and
attends to the stored past. Same logits, a fraction of the work —
every production LLM serves you through a cache like this.

Two details of the cache are worth reading in the code. A lone new
query needs no masking: every cached position *is* its past (the mask
row it slices is the last one — all zeros). And when the context
outgrows `block_size`, the window slides; our position embeddings are
absolute, so every cached key suddenly belongs to a shifted position
id, and the stale cache is rebuilt. `generate.py` times itself — run
it with `--no_cache` to feel what the cache buys.

<details>
<summary><b>How it's implemented</b> — <code>tutorials/llm/model.py</code> (the writing loop, unabridged)</summary>

```python
        idx = xp.asarray(idx).astype(xp.int64)
        if idx.ndim == 1:
            idx = idx[None, :]

        self.eval()
        caches = self.empty_kv_caches() if use_cache else None
        with babytorch.no_grad():
            for _ in range(max_new_tokens):
                # Never use more than block_size tokens of context.
                idx_cond = idx[:, -self.block_size:]

                if caches is None:
                    # No cache: forward the whole context, every lap.
                    logits = self.forward(idx_cond).data
                elif caches[0]["k"] is None:
                    # First lap ("prefill"): run the whole prompt once and
                    # let every block store its keys and values.
                    logits = self.forward(idx_cond, caches).data
                elif caches[0]["k"].shape[2] < self.block_size:
                    # Steady state: the past is cached -- forward only the
                    # token sampled on the previous lap.
                    logits = self.forward(idx[:, -1:], caches).data
                else:
                    # The context window is full and now slides one step
                    # per token.  Our position embeddings are absolute, so
                    # every cached key belongs to a position id that just
                    # shifted -- the cache is stale; rebuild it.
                    caches = self.empty_kv_caches()
                    logits = self.forward(idx_cond, caches).data

                logits = logits[:, -1, :] / temperature   # last position: (B, vocab)

                if top_k is not None:
                    k = min(top_k, logits.shape[-1])
                    # Zero out everything below the k-th largest logit.
                    kth = xp.sort(logits, axis=-1)[:, -k][:, None]
                    logits = xp.where(logits < kth, -xp.inf, logits)

                # Softmax -> probabilities, then sample one token per row.
                probs = _softmax_np(logits)
                next_id = _sample(probs)                  # (B, 1)
                idx = xp.concatenate([idx, next_id], axis=1)
        return idx
```

</details>

## Finetuning: same loop, new voice

[`finetune.py`](../tutorials/llm/finetune.py) is `train.py` with three
changed lines: **load the pretrained checkpoint instead of random
weights, use a much smaller learning rate, feed a different corpus**
(nursery rhymes). A few hundred steps later the model writes rhymes —
using spelling, rhythm and dialogue structure it learned from
Shakespeare.

That cheap transfer is the deepest fact in this book's second half.
The pretrained weights are not a lookup table of Shakespeare; they are
reusable machinery for *English-shaped text*, and a small nudge
repurposes them. Scale the same recipe up and it is how every deployed
LLM is made: pretrain once on the internet, then cheaply adapt —
to chat, to code, to your documents. (Finetune too long on the tiny
corpus, though, and the old skills wash away — *catastrophic
forgetting*: watch it happen by generating with a Shakespeare prompt
after many finetune steps.)

Checkpoints make this workflow possible — `save_checkpoint`
([`common.py`](../tutorials/llm/common.py)) writes three files: the
weights (chapter 3's `Module.save`, GPU-safe), the tokenizer
vocabulary, and the model config, so `load_checkpoint` can rebuild an
identical model anywhere.

## Run it

```bash
cd tutorials/llm
python train.py --steps 3000                    # pretrain (GPU: minutes)
python generate.py --prompt "ROMEO:" --tokens 400 --temperature 0.8
python attention_viz.py                         # draw what the heads learned
python finetune.py                              # adapt to nursery rhymes
python generate.py --checkpoint checkpoints/babygpt_finetuned --prompt "Twinkle"
```

On a CPU, shrink the model — it still climbs the same ladder, just to a
lower rung:

```bash
python train.py --steps 1500 --block_size 64 --n_embd 96 --n_head 4 --n_layer 4
```

Then experiment. Good first ones: swap `CharTokenizer` for
[`BPETokenizer`](../babytorch/text/tokenizers.py) and compare loss *per
character* (fair units — BPE predicts more text per token); train on
your own `--corpus file.txt`; draw your trained heads' `(T, T)`
attention weights with
[`attention_viz.py`](../tutorials/llm/attention_viz.py) and look for
structure — the previous-token diagonal, a bright column at a newline
(then read the script: it is just chapter 6's `att.data`, kept by a
flag); push `temperature` to extremes and watch the ladder in reverse.

## Where BabyGPT ends

Between this 2.7M-parameter model and a frontier LLM lie, honestly:
about five orders of magnitude of scale (parameters, data, compute);
engineering for that scale (fused GPU kernels, mixed precision,
batched serving, training sharded across thousands of devices); and
post-training -- supervised instruction tuning, preference optimization,
reinforcement learning, evaluation, and safety work aimed at useful behavior
rather than only next-token imitation.

What does **not** change is everything this book covered: tensors,
autograd, cross-entropy, AdamW, attention, residual blocks,
next-token sampling. Open any serious implementation — say, PyTorch
code for a production Transformer — and you will recognize every part,
because you have now read a complete one.

Remove the word "baby" and keep going.

## Key takeaways

- Pretraining is the familiar mini-batch loop applied to shifted windows of a
  large token stream, with validation loss as the main development signal.
- Generation repeatedly samples a next-token distribution; decoding controls
  behavior but cannot add knowledge the model did not learn.
- A KV cache reuses past attention projections during inference, while
  finetuning continues optimization on a narrower distribution or objective.

## Exercises

**Check yourself** (answers unfold):

**Q1.** A character-level model with a 65-token vocabulary starts
training at loss ≈ 4.17. Where does that number come from?

<details><summary>Answer</summary>

`−ln(1/65)`. Untrained weights spread probability roughly uniformly, so
the true next character gets `p ≈ 1/65`, and cross-entropy charges its
negative log. Any starting loss far from `ln(vocab_size)` is a bug tell.

</details>

**Q2.** One sample reads `"the the the the the"`, another
`"xq;Rd wke,pf"`. Which came from temperature 0.1 and which from 1.8?

<details><summary>Answer</summary>

The repetition is T = 0.1: sharpening pushes sampling toward argmax,
and greedy-ish decoding falls into loops. The keysmash is T = 1.8:
flattening gives junk tokens real probability. Usable text lives in
between — that is why 0.8 is the default.

</details>

**Q3.** Why does finetuning use a learning rate ~10× smaller than
pretraining?

<details><summary>Answer</summary>

The weights already encode something valuable; finetuning should
*nudge* them toward the new corpus, not overwrite them. Too large a
rate erases the pretrained knowledge — catastrophic forgetting, fast.

</details>

**Q4.** The KV cache holds 10 positions and `generate` forwards the
one token it just sampled. Inside each block, what are the shapes of
`q` and of the attention table `att` — and why does causality survive
without masking anything out?

<details><summary>Answer</summary>

`q` is `(B, n_head, 1, head_size)` — a single query — and `att` is
`(B, n_head, 1, 11)`: one row of weights over the 10 cached positions
plus the token itself. Nothing needs masking because everything in the
cache is this position's past; the row the code slices from the mask
is the last one, which is all zeros.

</details>

**Build it** — implement `top_p_filter` (nucleus sampling: keep a fixed
amount of *probability* instead of top-k's fixed *count*) and
★ `generate_greedy` (then watch greedy text loop — the reason we
sample), in
[`exercises/ch08_generation.py`](exercises/ch08_generation.py); run
`pytest book/exercises/test_ch08_generation.py -v`.
([How the exercises work](exercises/README.md).)

---

**Source files for this chapter:**
[`tutorials/llm/train.py`](../tutorials/llm/train.py) ·
[`tutorials/llm/generate.py`](../tutorials/llm/generate.py) ·
[`tutorials/llm/finetune.py`](../tutorials/llm/finetune.py) ·
[`tutorials/llm/attention_viz.py`](../tutorials/llm/attention_viz.py) ·
[`tutorials/llm/common.py`](../tutorials/llm/common.py)

[← Chapter 7: The Transformer](07-transformer.md) | [Contents](README.md) | [Chapter 9: Tabular Reinforcement Learning →](09-tabular-methods.md)
