"""Grader for chapter 8."""

import numpy as np

import babytorch
from babytorch import Tensor
from babytorch.backend import xp

from grading import load, exercise

impl = load("ch08_generation")


@exercise
def test_top_p_keeps_the_smallest_sufficient_set():
    probs = xp.asarray([0.5, 0.3, 0.15, 0.05])
    out = babytorch.to_numpy(impl.top_p_filter(probs, p=0.8))
    assert np.isclose(out.sum(), 1.0, atol=1e-6)
    assert np.count_nonzero(out) == 2, \
        "0.5 + 0.3 already reaches p=0.8 -- keep exactly those two"
    assert np.allclose(out, [0.625, 0.375, 0.0, 0.0], atol=1e-6)


@exercise
def test_top_p_keeps_positions_not_order():
    probs = xp.asarray([0.05, 0.3, 0.15, 0.5])      # unsorted on purpose
    out = babytorch.to_numpy(impl.top_p_filter(probs, p=0.8))
    assert out[3] > out[1] > 0 and out[0] == 0 and out[2] == 0, \
        "zero out the tail, but do NOT reorder the surviving tokens"


@exercise
def test_top_p_edges():
    probs = xp.asarray([0.5, 0.3, 0.15, 0.05])
    everything = babytorch.to_numpy(impl.top_p_filter(probs, p=1.0))
    assert np.allclose(everything, babytorch.to_numpy(probs), atol=1e-6), \
        "p=1.0 keeps the whole distribution"
    only_top = babytorch.to_numpy(impl.top_p_filter(probs, p=1e-9))
    assert np.allclose(only_top, [1.0, 0.0, 0.0, 0.0], atol=1e-6), \
        "always keep at least the single most likely token"


class _CycleModel:
    """Stub model: always scores (last_token + 1) mod vocab highest."""

    block_size = 4
    vocab = 7

    def eval(self):
        return self

    def forward(self, idx):
        idx = xp.asarray(idx)
        B, T = idx.shape
        logits = xp.zeros((B, T, self.vocab), dtype=xp.float32)
        for b in range(B):
            for t in range(T):
                logits[b, t, int(idx[b, t] + 1) % self.vocab] = 5.0
        return Tensor(logits)


@exercise
def test_greedy_follows_the_argmax():
    out = impl.generate_greedy(_CycleModel(), xp.asarray([0]), 6)
    got = list(babytorch.to_numpy(xp.asarray(out)).ravel().astype(int))
    assert got == [0, 1, 2, 3, 4, 5, 6], \
        "the stub model always votes for last+1: greedy must follow it"


@exercise
def test_greedy_survives_beyond_block_size():
    # 10 new tokens with block_size 4: cropping must kick in, not crash
    out = impl.generate_greedy(_CycleModel(), xp.asarray([3, 4]), 10)
    got = list(babytorch.to_numpy(xp.asarray(out)).ravel().astype(int))
    assert len(got) == 12
    assert got[:2] == [3, 4] and got[2:5] == [5, 6, 0], \
        "wrap-around continues: ... 5, 6, 0, 1, ..."
