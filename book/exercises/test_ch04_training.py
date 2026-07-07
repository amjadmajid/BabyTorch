"""Grader for chapter 4."""

import numpy as np

import babytorch
from babytorch import Tensor
from babytorch.backend import xp

from grading import load, exercise

impl = load("ch04_training")


def params_with_grads(grads):
    out = []
    for g in grads:
        p = Tensor(xp.zeros(2 if g is None else len(g)), requires_grad=True)
        p.grad = None if g is None else xp.asarray(g, dtype=xp.float32)
        out.append(p)
    return out


@exercise
def test_clip_leaves_small_gradients_alone():
    ps = params_with_grads([[3.0, 4.0]])          # norm 5
    total = impl.clip_grad_norm_(ps, max_norm=10.0)
    assert np.isclose(float(total), 5.0, atol=1e-5)
    assert np.allclose(babytorch.to_numpy(ps[0].grad), [3.0, 4.0]), \
        "below the cap, gradients must be untouched"


@exercise
def test_clip_rescales_to_the_cap():
    ps = params_with_grads([[6.0, 8.0], [0.0, 0.0]])   # combined norm 10
    total = impl.clip_grad_norm_(ps, max_norm=2.0)
    assert np.isclose(float(total), 10.0, atol=1e-4), \
        "return the norm measured BEFORE clipping"
    new_norm = np.sqrt(sum(float((xp.asarray(p.grad) ** 2).sum())
                           for p in ps))
    assert np.isclose(new_norm, 2.0, atol=1e-4), \
        "after clipping, the combined norm equals max_norm"
    assert np.allclose(babytorch.to_numpy(ps[0].grad), [1.2, 1.6], atol=1e-4)


@exercise
def test_clip_skips_none_gradients():
    ps = params_with_grads([[6.0, 8.0], None])
    impl.clip_grad_norm_(ps, max_norm=1.0)
    assert ps[1].grad is None


@exercise
def test_rmsprop_first_step_matches_the_math():
    p = Tensor(xp.zeros(2), requires_grad=True)
    p.grad = xp.asarray([1.0, -2.0], dtype=xp.float32)
    opt = impl.RMSProp([p], learning_rate=0.1, alpha=0.9, eps=1e-8)
    opt.step()
    # v = 0.9*0 + 0.1*g^2 ; p -= 0.1 * g / (sqrt(v) + eps)
    v = 0.1 * np.array([1.0, 4.0])
    want = -0.1 * np.array([1.0, -2.0]) / (np.sqrt(v) + 1e-8)
    assert np.allclose(babytorch.to_numpy(p.data), want, atol=1e-5), \
        "check the update rule: v = a*v + (1-a)*g**2, p -= lr*g/(sqrt(v)+eps)"


@exercise
def test_rmsprop_accumulates_v_across_steps():
    p = Tensor(xp.zeros(1), requires_grad=True)
    opt = impl.RMSProp([p], learning_rate=0.1, alpha=0.5, eps=0.0)
    p.grad = xp.asarray([2.0], dtype=xp.float32)
    opt.step()                                    # v = 2.0 -> step 0.1*2/sqrt(2)
    x1 = float(babytorch.to_numpy(p.data)[0])
    p.grad = xp.asarray([2.0], dtype=xp.float32)
    opt.step()                                    # v = 3.0 -> smaller step
    x2 = float(babytorch.to_numpy(p.data)[0])
    assert np.isclose(x1, -0.1 * 2 / np.sqrt(2.0), atol=1e-5)
    assert np.isclose(x2 - x1, -0.1 * 2 / np.sqrt(3.0), atol=1e-5), \
        "v must persist between steps (a running average, not recomputed)"


@exercise
def test_rmsprop_minimizes_a_bowl():
    babytorch.manual_seed(0)
    w = Tensor(xp.asarray([8.0, -6.0]), requires_grad=True)
    opt = impl.RMSProp([w], learning_rate=0.1)
    for _ in range(300):
        loss = ((w - 3.0) ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert np.allclose(babytorch.to_numpy(w.data), [3.0, 3.0], atol=0.2), \
        "300 steps on (w-3)^2 should land near w = 3"
