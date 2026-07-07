"""Generate text from a trained BabyGPT checkpoint (inference).

Run it::

    python generate.py --prompt "ROMEO:" --tokens 400
    python generate.py --prompt "Once upon a time" --temperature 1.0 --top_k 20

Inference is pure forward passes -- no gradients, no learning.  The model
reads your prompt, predicts a distribution over the next character, samples
one, appends it, and repeats.  The knobs shape *how* it samples:

* ``--temperature`` -- lower is safer/repetitive, higher is wilder.
* ``--top_k``       -- only ever sample from the k most likely characters.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import babytorch
from common import load_checkpoint, sample_text


def main():
    p = argparse.ArgumentParser(description="Generate text with BabyGPT.")
    p.add_argument("--checkpoint", default="checkpoints/babygpt")
    p.add_argument("--prompt", default="\n")
    p.add_argument("--tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=None)
    args = p.parse_args()

    print(f"Device: {babytorch.device()}")
    model, tokenizer, config = load_checkpoint(args.checkpoint)
    print(f"Loaded {model.num_parameters():,}-parameter model "
          f"(vocab {tokenizer.vocab_size}, block {config['block_size']})\n")

    text = sample_text(model, tokenizer, prompt=args.prompt,
                       max_new_tokens=args.tokens,
                       temperature=args.temperature, top_k=args.top_k)
    print(text)


if __name__ == "__main__":
    main()
