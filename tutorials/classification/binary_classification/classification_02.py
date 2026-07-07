"""Binary classification of the two-moons dataset, with a decision boundary.

Two interleaved half-circles of points cannot be separated by a straight
line, so a linear model provably fails here -- this is the classic test
that a network's non-linearities are doing real work.

The dataset is generated with a few lines of NumPy (no sklearn needed).

Run it::

    python classification_02.py
"""

import numpy as np
import matplotlib.pyplot as plt

import babytorch
from babytorch import Tensor, Grapher
from babytorch.backend import xp
import babytorch.nn as nn
from babytorch.nn import Sequential, MSELoss
from babytorch.optim import SGD


def make_moons(n_samples=200, noise=0.1):
    """Two interleaved half-moons (same shape as sklearn's make_moons)."""
    n = n_samples // 2
    t = np.linspace(0, np.pi, n)
    outer = np.stack([np.cos(t), np.sin(t)], axis=1)
    inner = np.stack([1.0 - np.cos(t), 0.5 - np.sin(t)], axis=1)
    X = np.concatenate([outer, inner]).astype(np.float32)
    y = np.concatenate([np.zeros(n), np.ones(n)]).astype(np.float32)
    X += np.random.normal(scale=noise, size=X.shape).astype(np.float32)
    return X, y


def clip_gradients_norm(parameters, max_norm):
    """Rescale all gradients so their combined norm stays below max_norm.

    A safeguard against "exploding gradients": one unlucky batch with a
    huge gradient would otherwise catapult the weights far from anything
    useful.
    """
    total_norm = float(xp.sqrt(sum(xp.sum(p.grad ** 2)
                                   for p in parameters if p.grad is not None)))
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for p in parameters:
            if p.grad is not None:
                p.grad *= scale


# Hyperparameters
num_iterations = 8000
learning_rate = 0.01

# Generate the dataset and scale the labels to -1 / +1.
np.random.seed(0)
babytorch.manual_seed(0)
X_orig, y_orig = make_moons(n_samples=200, noise=0.1)
X = Tensor(X_orig)
y = Tensor(y_orig * 2 - 1).reshape(-1, 1)

# The model: two input features (the point's coordinates), one output score.
model = Sequential(
    nn.Linear(2, 64, nn.ReLU()),
    nn.Linear(64, 32, nn.ReLU()),
    nn.Linear(32, 16, nn.ReLU()),
    nn.Linear(16, 1),
)
optimizer = SGD(model.parameters(), learning_rate=learning_rate,
                weight_decay=0.0005)
criterion = MSELoss()

# Training loop
losses = []
for epoch in range(num_iterations):
    y_pred = model(X)
    loss = criterion(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    clip_gradients_norm(model.parameters(), max_norm=10.0)
    optimizer.step()
    losses.append(loss.item())
    if epoch % 1000 == 0:
        print(f"epoch {epoch:4d} | loss {loss.item():.5f}")

# Plot the loss
grapher = Grapher()
grapher.plot_loss(losses)
grapher.show()

# Visualize the decision boundary: classify every point of a grid and
# color the plane by the model's answer.
h = 0.25
x_min, x_max = X_orig[:, 0].min() - 1, X_orig[:, 0].max() + 1
y_min, y_max = X_orig[:, 1].min() - 1, X_orig[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Xmesh = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)

with babytorch.no_grad():
    scores = model(Tensor(Xmesh))
Z = (scores.numpy() > 0).reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X_orig[:, 0], X_orig[:, 1], c=y_orig, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title('BabyTorch decision boundary')
plt.show()
