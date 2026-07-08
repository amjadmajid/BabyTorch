"""Look inside a trained BabyGPT: draw attention weights as heatmaps.

Run it (after training -- see train.py)::

    python attention_viz.py --prompt "First Citizen:"
    python attention_viz.py --layer 0 --head 2 --out head2.png

Chapter 6 of the book ends with a promise: the attention weights are not
an abstraction, they are a real ``(T, T)`` table you can *look at*.  This
script keeps that promise.  It runs one forward pass over your prompt,
asks a layer to keep its post-softmax weights (``store_attention`` in
``model.py``), and draws them: row i shows how much position i attended
to every position j <= i.  The upper triangle is empty -- that white
staircase *is* the causal mask.

Things to look for in a trained model:

* a bright diagonal or off-diagonal: heads that track the previous
  character(s) -- how the model learns spelling;
* bright columns: positions many rows attend to, e.g. the newline or
  space that starts a word or a speaker's name;
* heads that differ from each other: that is why multi-head exists.
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import babytorch
from common import load_checkpoint


def token_labels(tokenizer, ids):
    """One short printable label per token (chars: make whitespace visible)."""
    labels = []
    for t in ids:
        s = tokenizer.decode([int(t)])
        labels.append({"\n": "\\n", "\t": "\\t", " ": "·"}.get(s, s))
    return labels


def main():
    p = argparse.ArgumentParser(description="Plot BabyGPT attention heatmaps.")
    p.add_argument("--checkpoint", default="checkpoints/babygpt")
    p.add_argument("--prompt", default="First Citizen:\nWe are accounted poor")
    p.add_argument("--layer", type=int, default=-1,
                   help="which block to inspect (default: the last)")
    p.add_argument("--head", type=int, default=None,
                   help="one head only (default: every head, in a grid)")
    p.add_argument("--out", default="attention.png")
    p.add_argument("--device", default=None, choices=["auto", "cpu", "cuda", "gpu"],
                   help="where to run (default: auto -- GPU if available)")
    args = p.parse_args()

    if args.device:
        babytorch.set_device(args.device)
    model, tokenizer, config = load_checkpoint(args.checkpoint)

    # Encode the prompt (cropped to what the model can attend to) and run
    # ONE forward pass with the chosen layer recording its weights.
    ids = tokenizer.encode(args.prompt)[:config["block_size"]]
    layer = model.blocks[args.layer]
    layer.attn.store_attention = True
    model.eval()
    with babytorch.no_grad():
        model.forward([ids])
    att = babytorch.to_numpy(layer.attn.last_attention)[0]   # (n_head, T, T)

    n_head, T, _ = att.shape
    layer_no = args.layer if args.layer >= 0 else len(model.blocks) + args.layer
    heads = list(range(n_head)) if args.head is None else [args.head]
    labels = token_labels(tokenizer, ids)

    import matplotlib.pyplot as plt

    cols = min(len(heads), 3)
    rows = math.ceil(len(heads) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.0 * rows),
                             squeeze=False)

    # One shared scale so heads can be compared honestly; a single hue,
    # light -> dark, because the data is a magnitude (weight from 0 up).
    vmax = att[heads].max()
    for ax, h in zip(axes.flat, heads):
        im = ax.imshow(att[h], cmap="Oranges", vmin=0.0, vmax=vmax)
        ax.set_title(f"layer {layer_no}, head {h}", fontsize=10)
        ax.set_xticks(range(T), labels, fontsize=7, family="monospace")
        ax.set_yticks(range(T), labels, fontsize=7, family="monospace")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    for ax in axes.flat[len(heads):]:
        ax.axis("off")                                # unused grid cells

    fig.suptitle(f"Who attends to whom -- rows attend to columns "
                 f"({repr(args.prompt)[:48]})", fontsize=11)
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("attention weight", fontsize=9)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved {args.out}  (layer {layer_no}, "
          f"{len(heads)} head{'s' if len(heads) > 1 else ''}, T={T})")


if __name__ == "__main__":
    main()
