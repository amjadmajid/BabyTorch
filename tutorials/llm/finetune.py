"""Finetune a pretrained BabyGPT on a new, smaller corpus.

Run it::

    python train.py --steps 3000                 # 1. pretrain on Shakespeare
    python finetune.py --corpus data/rhymes.txt  # 2. specialize on rhymes

Finetuning = start from weights that already "know English", then keep
training on a narrower dataset so the model adopts its style.  It is far
cheaper than training from scratch, because the model only has to *adapt*
what it already knows -- the same reason real labs finetune a big base
model instead of pretraining a new one for every task.

Two practical touches versus pretraining:

* we **load** the pretrained weights and tokenizer instead of starting
  random;
* we use a **smaller learning rate**, so finetuning nudges the weights
  rather than overwriting everything the model learned.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import babytorch
from babytorch.optim import AdamW

from common import (load_checkpoint, get_batch, estimate_loss,
                    save_checkpoint, sample_text)

# A tiny built-in corpus so the demo runs with no external downloads.
DEFAULT_RHYMES = """
Twinkle, twinkle, little star,
How I wonder what you are!
Up above the world so high,
Like a diamond in the sky.

Humpty Dumpty sat on a wall,
Humpty Dumpty had a great fall.
All the king's horses and all the king's men
Couldn't put Humpty together again.

Jack and Jill went up the hill
To fetch a pail of water.
Jack fell down and broke his crown,
And Jill came tumbling after.
""" * 40


def main():
    p = argparse.ArgumentParser(description="Finetune BabyGPT.")
    p.add_argument("--checkpoint", default="checkpoints/babygpt",
                   help="pretrained checkpoint to start from")
    p.add_argument("--corpus", default=None,
                   help="finetuning .txt (default: built-in nursery rhymes)")
    p.add_argument("--out", default="checkpoints/babygpt_finetuned")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)  # smaller than pretraining
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    babytorch.manual_seed(args.seed)
    print(f"Device: {babytorch.device()}")

    # Load the pretrained model AND its tokenizer (we must reuse the exact
    # same vocabulary the model was trained with).
    model, tokenizer, config = load_checkpoint(args.checkpoint)
    block_size = config["block_size"]
    print(f"Loaded pretrained model: {model.num_parameters():,} parameters")

    # New corpus, filtered to characters the tokenizer already knows.
    if args.corpus:
        with open(args.corpus, encoding="utf-8") as f:
            text = f.read()
    else:
        text = DEFAULT_RHYMES
    known = set(tokenizer.chars)
    filtered = "".join(ch for ch in text if ch in known)
    dropped = len(text) - len(filtered)
    if dropped:
        print(f"Note: dropped {dropped} characters not in the pretrained vocab.")
    data = np.array(tokenizer.encode(filtered), dtype=np.int64)
    print(f"Finetuning corpus: {len(data):,} tokens")

    optimizer = AdamW(model.parameters(), learning_rate=args.lr, weight_decay=0.1)

    t0 = time.time()
    for step in range(args.steps):
        x, y = get_batch(data, block_size, args.batch_size)
        loss = model.loss(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0 or step == args.steps - 1:
            val = estimate_loss(model, data, block_size, args.batch_size, iters=10)
            print(f"step {step:4d} | train {loss.item():.3f} | val {val:.3f} "
                  f"| {time.time() - t0:.0f}s")

    save_checkpoint(args.out, model, tokenizer, config)
    print("\n----- sample after finetuning -----")
    print(sample_text(model, tokenizer, prompt="\n",
                      max_new_tokens=300, temperature=0.7))


if __name__ == "__main__":
    main()
