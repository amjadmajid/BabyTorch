"""Grader for chapter 7."""

import numpy as np
import pytest

import babytorch
from babytorch.optim import AdamW

from grading import load, exercise

impl = load("ch07_transformer")


CONFIGS = [
    dict(vocab_size=11, block_size=8, n_embd=8, n_head=2, n_layer=1),
    dict(vocab_size=20, block_size=16, n_embd=12, n_head=3, n_layer=2),
    dict(vocab_size=65, block_size=32, n_embd=48, n_head=4, n_layer=3),
]


@exercise
@pytest.mark.parametrize("config", CONFIGS,
                         ids=[f"L{c['n_layer']}-C{c['n_embd']}" for c in CONFIGS])
def test_parameter_count_matches_the_real_model(config):
    from model import GPT
    want = GPT(**config).num_parameters()
    got = impl.count_gpt_parameters(**config)
    assert got == want, (
        f"off by {got - want:+d} for {config} -- "
        "recheck biases and the two LayerNorm vectors per block")


@exercise
def test_tied_gpt_saves_exactly_the_head():
    from model import GPT
    config = dict(vocab_size=30, block_size=16, n_embd=24, n_head=4,
                  n_layer=2)
    plain = GPT(**config).num_parameters()
    tied = impl.TiedGPT(**config).num_parameters()
    saved = config["n_embd"] * config["vocab_size"] + config["vocab_size"]
    assert tied == plain - saved, \
        "tying should remove the head weight AND its bias, nothing else"


@exercise
def test_tied_gpt_forward_shape_and_gradient():
    babytorch.manual_seed(0)
    model = impl.TiedGPT(vocab_size=13, block_size=8, n_embd=16, n_head=2,
                         n_layer=1)
    ids = np.array([[1, 2, 3, 4, 5]])
    logits = model(ids)
    assert logits.shape == (1, 5, 13)

    loss = model.loss(ids[:, :-1], ids[:, 1:])
    loss.backward()
    assert model.token_embedding.weight.grad is not None, (
        "no gradient reached the embedding -- compute the logits with "
        "tensor ops (the transpose must stay in the graph)")


@exercise
def test_tied_gpt_learns():
    babytorch.manual_seed(1)
    model = impl.TiedGPT(vocab_size=10, block_size=8, n_embd=16, n_head=2,
                         n_layer=1)
    x = np.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    y = np.array([[2, 3, 4, 5, 6, 7, 8, 9]])
    optimizer = AdamW(model.parameters(), learning_rate=1e-2)
    first = None
    for _ in range(25):
        loss = model.loss(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        first = first if first is not None else loss.item()
    assert loss.item() < first * 0.7, \
        "25 steps on one tiny batch should clearly reduce the loss"
