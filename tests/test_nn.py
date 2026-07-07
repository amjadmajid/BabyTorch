"""Tests for neural-network layers, losses, and save/load."""

import numpy as np
import pytest

import babytorch
import babytorch.nn as nn
from babytorch import Tensor
from conftest import check_gradient


def test_linear_shapes():
    layer = nn.Linear(4, 3)
    out = layer(babytorch.randn(8, 4))
    assert out.shape == (8, 3)
    # weight + bias
    assert len(layer.parameters()) == 2


def test_sequential_parameters_are_collected():
    model = nn.Sequential(
        nn.Linear(4, 8, nn.ReLU()),
        nn.Linear(8, 2),
    )
    # 2 layers x (weight + bias) = 4 parameter tensors
    assert len(model.parameters()) == 4
    assert model.num_parameters() == 4 * 8 + 8 + 8 * 2 + 2


def test_embedding_lookup_and_grad():
    emb = nn.Embedding(10, 4)
    idx = np.array([[1, 2, 3], [3, 3, 0]])
    out = emb(idx)
    assert out.shape == (2, 3, 4)
    out.sum().backward()
    # token 3 appears 3 times -> its row gets gradient 3
    assert np.allclose(babytorch.to_numpy(emb.weight.grad[3]), 3.0)
    assert np.allclose(babytorch.to_numpy(emb.weight.grad[9]), 0.0)


def test_layernorm_normalizes():
    ln = nn.LayerNorm(16)
    x = babytorch.randn(4, 16) * 5 + 3
    out = ln(x)
    mean = babytorch.to_numpy(out.data).mean(axis=-1)
    std = babytorch.to_numpy(out.data).std(axis=-1)
    assert np.allclose(mean, 0.0, atol=1e-4)
    assert np.allclose(std, 1.0, atol=1e-2)


def test_layernorm_gradient():
    ln = nn.LayerNorm(4)
    check_gradient(lambda x: ln(x).sum(), np.random.randn(3, 4))


def test_dropout_train_vs_eval():
    drop = nn.Dropout(0.5)
    x = babytorch.ones(1000, 1000)
    drop.train()
    out_train = babytorch.to_numpy(drop(x).data)
    # roughly half the values are zeroed in training
    frac_zero = (out_train == 0).mean()
    assert 0.45 < frac_zero < 0.55
    drop.eval()
    out_eval = babytorch.to_numpy(drop(x).data)
    assert np.allclose(out_eval, 1.0)  # identity at eval time


def test_gelu_gradient():
    check_gradient(lambda x: nn.GELU()(x).sum(), np.random.randn(3, 4))


def test_mse_loss_zero_when_perfect():
    pred = Tensor([[1.0, 2.0], [3.0, 4.0]])
    loss = nn.MSELoss()(pred, Tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert abs(loss.item()) < 1e-6


def test_cross_entropy_value():
    # Two examples, three classes; confident + correct -> low loss.
    logits = Tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    loss = nn.CrossEntropyLoss()(logits, [0, 1])
    assert loss.item() < 0.01
    # confident + wrong -> high loss
    loss_wrong = nn.CrossEntropyLoss()(logits, [1, 0])
    assert loss_wrong.item() > 5.0


def test_cross_entropy_gradient():
    labels = np.array([0, 2, 1])
    check_gradient(lambda x: nn.CrossEntropyLoss()(x, labels),
                   np.random.randn(3, 4))


def test_conv2d_shape_and_grad():
    conv = nn.Conv2D(3, 5, kernel_size=3, padding=1)
    x = babytorch.randn(2, 3, 8, 8)
    out = conv(x)
    assert out.shape == (2, 5, 8, 8)  # padding=1 keeps spatial size
    out.sum().backward()
    assert conv.w.grad is not None
    assert conv.b.grad is not None
    assert conv.b.grad.shape == conv.b.shape


def test_maxpool_halves():
    x = babytorch.randn(1, 1, 4, 4)
    out = x.maxpool2d(kernel_size=2, stride=2)
    assert out.shape == (1, 1, 2, 2)


def test_save_and_load_roundtrip(tmp_path):
    model = nn.Sequential(nn.Linear(4, 8, nn.ReLU()), nn.Linear(8, 2))
    x = babytorch.randn(3, 4)
    before = babytorch.to_numpy(model(x).data)

    path = str(tmp_path / "model.pkl")
    model.save(path)

    reloaded = nn.Sequential(nn.Linear(4, 8, nn.ReLU()), nn.Linear(8, 2))
    nn.Module.load(path, reloaded)
    after = babytorch.to_numpy(reloaded(x).data)

    assert np.allclose(before, after)


def test_train_eval_toggle_propagates():
    model = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.5))
    model.eval()
    assert all(m.training is False for m in model.modules())
    model.train()
    assert all(m.training is True for m in model.modules())
