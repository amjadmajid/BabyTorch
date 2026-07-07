"""Regression: fit a curve to noisy points -- the "hello world" of training.

A small MLP learns y = f(x) from 100 noisy examples.  The four-step loop
(forward -> loss -> backward -> step) is the same one that trains every
model in this repository, up to and including BabyGPT.

Run it::

    python regression_01.py

Requires the ``[viz]`` extra for the plots (``pip install -e ".[viz]"``).
"""

import numpy as np

import babytorch
import babytorch.nn as nn
from babytorch.nn import MSELoss
from babytorch.optim import SGD
from babytorch import Tensor, Grapher

babytorch.manual_seed(0)
np.random.seed(0)

num_iterations = 2000
learning_rate = 0.1

# --- data: noisy samples of a hidden curve -------------------------------
# The model never sees the formula, only the (x, y) points.
x_np = np.linspace(-2.0, 2.0, 100, dtype=np.float32).reshape(-1, 1)
y_np = 0.5 * np.sin(2.0 * x_np) + 0.5 \
    + np.random.normal(0.0, 0.05, x_np.shape).astype(np.float32)

x = Tensor(x_np)
y = Tensor(y_np)

# --- model: 1 input -> 1 output, with room to bend ------------------------
# Tanh suits smooth curves: it is itself smooth, while ReLU would
# approximate the sine with straight-line segments.
model = nn.Sequential(
    nn.Linear(1, 8, nn.Tanh()),
    nn.Linear(8, 16, nn.Tanh()),
    nn.Linear(16, 8, nn.Tanh()),
    nn.Linear(8, 1),
)

optimizer = SGD(model.parameters(), learning_rate=learning_rate)
criterion = MSELoss()

# --- training loop ---------------------------------------------------------
losses = []
for step in range(num_iterations):
    predictions = model(x)               # 1. forward
    loss = criterion(predictions, y)     # 2. how wrong?
    optimizer.zero_grad()                #    (forget old gradients)
    loss.backward()                      # 3. gradients for every parameter
    optimizer.step()                     # 4. small step downhill
    losses.append(loss.item())
    if step % 200 == 0:
        print(f"step {step:4d} | loss {loss.item():.5f}")

print(f"final loss: {losses[-1]:.5f}")

# --- evaluate and plot ------------------------------------------------------
with babytorch.no_grad():
    predictions = model(x)

grapher = Grapher()
grapher.plot_loss(losses)
grapher.show()

import matplotlib.pyplot as plt

plt.figure()
plt.scatter(x_np, y_np, color='red', s=12, label='noisy data')
plt.plot(x_np, predictions.numpy(), color='blue', linewidth=3, label='model')
plt.legend()
plt.title('BabyTorch regression')
plt.show()
