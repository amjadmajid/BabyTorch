"""Shared helpers for the BabyGPT training / finetuning / generation scripts.

Kept separate so the scripts themselves stay short and readable.
"""

import json
import os
import pickle

import numpy as np

import babytorch
from babytorch.backend import xp
from babytorch.text import CharTokenizer

from model import GPT


def encode_corpus(text, tokenizer):
    """Turn the whole training text into one long array of token ids."""
    return np.array(tokenizer.encode(text), dtype=np.int64)


def get_batch(data, block_size, batch_size):
    """Sample a random mini-batch of (context, next-token) pairs.

    We pick ``batch_size`` random start positions and cut ``block_size``
    tokens for the input ``x`` and the same window shifted one step to the
    right for the target ``y`` -- so ``y[t]`` is the token that really
    followed ``x[t]``.  Language modelling is next-token prediction, and
    this is where the "next token" labels come from, for free, from raw
    text.
    """
    ix = np.random.randint(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i:i + block_size] for i in ix])
    y = np.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x, y


def estimate_loss(model, data, block_size, batch_size, iters=20):
    """Average the loss over a few random batches (a cheap held-out check)."""
    model.eval()
    losses = []
    with babytorch.no_grad():
        for _ in range(iters):
            x, y = get_batch(data, block_size, batch_size)
            losses.append(model.loss(x, y).item())
    model.train()
    return float(np.mean(losses))


def save_checkpoint(path, model, tokenizer, config):
    """Save model weights, the tokenizer vocabulary, and the model config."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    model.save(path + ".model")
    tokenizer.save(path + ".tokenizer.json")
    with open(path + ".config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved checkpoint to {path}.*")


def load_checkpoint(path):
    """Rebuild a model + tokenizer from a checkpoint saved above."""
    with open(path + ".config.json") as f:
        config = json.load(f)
    tokenizer = CharTokenizer.load(path + ".tokenizer.json")
    model = GPT(**config)
    babytorch.nn.Module.load(path + ".model", model)
    return model, tokenizer, config


def sample_text(model, tokenizer, prompt="\n", max_new_tokens=300,
                temperature=0.8, top_k=None):
    """Generate a text continuation from ``prompt`` and return it as a string."""
    ids = tokenizer.encode(prompt) or [0]
    out = model.generate(np.array(ids), max_new_tokens=max_new_tokens,
                         temperature=temperature, top_k=top_k)
    return tokenizer.decode(babytorch.to_numpy(out)[0].tolist())
