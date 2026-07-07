# Chapter 8 — Training a GPT

*Part II, chapter 4 of 4. The model is built. Now we put it through a
language model's full life cycle — pretrain on a corpus, generate text,
finetune on a new voice — using the runnable scripts in
[`tutorials/llm/`](../tutorials/llm/).*

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
your own `--corpus file.txt`; print the `(T, T)` attention weights of a
trained head and look for structure (they are just `att.data` in
chapter 6's code); push `temperature` to extremes and watch the ladder
in reverse.

## Where BabyGPT ends

Between this 2.7M-parameter model and a frontier LLM lie, honestly:
about five orders of magnitude of scale (parameters, data, compute);
engineering for that scale (fused GPU kernels, mixed precision, KV
caches so generation doesn't recompute the past, training sharded
across thousands of devices); and post-training — instruction tuning
and reinforcement learning from human feedback, which is finetuning
(the kind you just did) aimed at "be helpful" rather than "rhyme".

What does **not** change is everything this book covered: tensors,
autograd, cross-entropy, AdamW, attention, residual blocks,
next-token sampling. Open any serious implementation — say, PyTorch
code for a production Transformer — and you will recognize every part,
because you have now read a complete one.

Remove the word "baby" and keep going.

---

**Source files for this chapter:**
[`tutorials/llm/train.py`](../tutorials/llm/train.py) ·
[`tutorials/llm/generate.py`](../tutorials/llm/generate.py) ·
[`tutorials/llm/finetune.py`](../tutorials/llm/finetune.py) ·
[`tutorials/llm/common.py`](../tutorials/llm/common.py)

[← Chapter 7: The Transformer](07-transformer.md) | [Contents](README.md) | *End of the book — [back to the repository](../README.md)*
