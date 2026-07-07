"""Pretrain BabyGPT on a text corpus (character-level).

Run it::

    python train.py                 # Tiny Shakespeare (downloads ~1 MB)
    python train.py --steps 500     # shorter run
    BABYTORCH_DEVICE=cpu python train.py   # force CPU

What "pretraining" means here: show the model a mountain of raw text and,
at every position, ask it to predict the next character.  There are no
human labels -- the text is its own supervision.  After enough steps the
model has absorbed the statistics of the language (spelling, common words,
a bit of grammar and style) into its weights.

The training loop is the same four steps as every other BabyTorch example:
forward -> loss -> backward -> optimizer step.  A Transformer is not
special to train; it is special in *what it can represent*.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import babytorch
from babytorch.optim import AdamW, CosineWarmupLR
from babytorch.text import CharTokenizer
from babytorch.datasets.text import load_tiny_shakespeare

from common import (encode_corpus, get_batch, estimate_loss,
                    save_checkpoint, sample_text)


def main():
    p = argparse.ArgumentParser(description="Pretrain BabyGPT.")
    p.add_argument("--corpus", default=None,
                   help="path to a .txt file (default: Tiny Shakespeare)")
    p.add_argument("--out", default="checkpoints/babygpt",
                   help="checkpoint path prefix")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--n_embd", type=int, default=192)
    p.add_argument("--n_head", type=int, default=6)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default=None, choices=["auto", "cpu", "cuda", "gpu"],
                   help="where to run (default: auto -- GPU if available)")
    args = p.parse_args()

    if args.device:
        babytorch.set_device(args.device)
    babytorch.manual_seed(args.seed)
    print(f"Device: {babytorch.device()}")

    # --- data ---------------------------------------------------------
    if args.corpus:
        with open(args.corpus, encoding="utf-8") as f:
            text = f.read()
    else:
        text = load_tiny_shakespeare(root="./data")
    print(f"Corpus: {len(text):,} characters")

    tokenizer = CharTokenizer(text)
    print(f"Vocabulary: {tokenizer.vocab_size} unique characters")
    data = encode_corpus(text, tokenizer)
    # 90/10 split so we can watch for overfitting.
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    # --- model --------------------------------------------------------
    config = dict(vocab_size=tokenizer.vocab_size, block_size=args.block_size,
                  n_embd=args.n_embd, n_head=args.n_head,
                  n_layer=args.n_layer, dropout=args.dropout)
    from model import GPT
    model = GPT(**config)
    print(f"Model: {model.num_parameters():,} parameters")

    optimizer = AdamW(model.parameters(), learning_rate=args.lr, weight_decay=0.1)
    scheduler = CosineWarmupLR(optimizer, warmup_steps=max(1, args.steps // 20),
                               total_steps=args.steps, min_lr=args.lr / 10)

    # --- training loop ------------------------------------------------
    t0 = time.time()
    for step in range(args.steps):
        scheduler.step(step)
        x, y = get_batch(train_data, args.block_size, args.batch_size)

        loss = model.loss(x, y)          # forward
        optimizer.zero_grad()
        loss.backward()                  # backward
        optimizer.step()                 # update

        if step % 100 == 0 or step == args.steps - 1:
            val = estimate_loss(model, val_data, args.block_size, args.batch_size)
            dt = time.time() - t0
            print(f"step {step:4d} | train {loss.item():.3f} | "
                  f"val {val:.3f} | lr {optimizer.learning_rate:.1e} | {dt:.0f}s")

    # --- save + sample ------------------------------------------------
    save_checkpoint(args.out, model, tokenizer, config)
    print("\n----- sample -----")
    print(sample_text(model, tokenizer, prompt="\n",
                      max_new_tokens=500, temperature=0.8))


if __name__ == "__main__":
    main()
