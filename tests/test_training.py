"""End-to-end training tests: does loss actually go down?

These are the "does the whole machine turn" tests -- they wire together
tensors, layers, losses and optimizers and check that a real model learns.
"""

import os
import sys

import numpy as np
import pytest

import babytorch
import babytorch.nn as nn
from babytorch.optim import Adam, SGD

# Make the BabyGPT model importable from tutorials/llm.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tutorials", "llm"))
from model import GPT  # noqa: E402


def test_regression_learns_line():
    babytorch.manual_seed(0)
    x = babytorch.randn(128, 1)
    y = x * 3.0 - 2.0
    model = nn.Linear(1, 1)
    opt = SGD(model.parameters(), learning_rate=0.1)
    crit = nn.MSELoss()
    for _ in range(300):
        loss = crit(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    # recovered slope ~3 and intercept ~-2
    assert abs(model.w.item() - 3.0) < 0.1
    assert abs(model.b.item() + 2.0) < 0.1


def test_classification_learns_xor():
    """A 2-layer MLP should solve XOR -- the classic non-linear problem."""
    babytorch.manual_seed(0)
    X = babytorch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    model = nn.Sequential(
        nn.Linear(2, 8, nn.Tanh()),
        nn.Linear(8, 2),
    )
    opt = Adam(model.parameters(), learning_rate=0.1)
    crit = nn.CrossEntropyLoss()
    for _ in range(500):
        loss = crit(model(X), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    preds = model(X).argmax(axis=1)
    assert list(babytorch.to_numpy(preds)) == [0, 1, 1, 0]


def test_gpt_forward_shapes():
    model = GPT(vocab_size=17, block_size=8, n_embd=16, n_head=2, n_layer=2)
    idx = np.random.randint(0, 17, size=(3, 8))
    logits = model(idx)
    assert logits.shape == (3, 8, 17)


def test_gpt_overfits_tiny_sequence():
    """The strongest correctness signal for the whole framework.

    If every gradient in the Transformer is right, a tiny GPT can memorize
    a single repeated sequence and drive the loss toward zero.  If any
    backward pass is wrong, the loss plateaus.
    """
    babytorch.manual_seed(0)
    vocab_size, block_size = 12, 8
    model = GPT(vocab_size=vocab_size, block_size=block_size,
                n_embd=32, n_head=2, n_layer=2, dropout=0.0)
    opt = Adam(model.parameters(), learning_rate=0.01)

    # One fixed (input -> next token) pair to memorize.
    data = np.arange(block_size + 1) % vocab_size
    x = data[:-1][None, :]     # (1, block_size)
    y = data[1:][None, :]      # (1, block_size)

    first = last = None
    for i in range(120):
        loss = model.loss(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i == 0:
            first = loss.item()
        last = loss.item()

    assert last < first * 0.2, f"GPT did not learn: {first:.3f} -> {last:.3f}"


def test_gpt_generate_runs():
    model = GPT(vocab_size=12, block_size=8, n_embd=16, n_head=2, n_layer=1)
    out = model.generate(np.array([0, 1, 2]), max_new_tokens=5, top_k=3)
    assert out.shape[1] == 3 + 5      # context + generated
    assert babytorch.to_numpy(out).max() < 12  # valid token ids
