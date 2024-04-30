# Simple Regression Example with BabyTorch and PyTorch for Comparison

## Introduction
In this tutorial, we'll demonstrate how to implement a regression model using both BabyTorch and PyTorch. We'll predict values from a synthetic dataset created with `make_regression`.

## Setup and Data Preparation
```python
import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

# Create regression data
x, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=0)
y = (y.reshape(-1, 1) + 1) * .5
y /= np.max(y)
```

## BabyTorch Implementation

1. **Import Libraries**
   - Import necessary modules from BabyTorch.
   ```python
   import babytorch
   import babytorch.nn as nn
   from babytorch.optim import SGD
   ```

2. **Model Definition**
   - Define a neural network with multiple linear and ReLU layers.
   ```python
   model = nn.Sequential(
       nn.Linear(1, 8, nn.ReLU()),
       nn.Linear(8, 16, nn.ReLU()),
       nn.Linear(16, 8, nn.ReLU()),
       nn.Linear(8, 1)
   )
   ```

3. **Training**
   - Set up the optimizer and loss function, then train the model.
   ```python
   optimizer = SGD(model.parameters(), learning_rate=0.1)
   criterion = nn.MSELoss()

   for k in range(2000):
       y_pred = model(x)
       loss = criterion(y_pred, y)
       loss.backward()
       optimizer.step()
       model.zero_grad()
   ```

4. **Visualization**
   - Visualize the training losses and model predictions.
   ```python
   # Assuming Grapher() is set up for plotting
   Grapher().plot_loss(losses)
   ```

## PyTorch Implementation

1. **Import Libraries**
   - Use similar libraries from PyTorch.
   ```python
   import torch
   import torch.nn as nn
   import torch.optim as optim
   ```

2. **Model Definition**
   - Define an equivalent model in PyTorch.
   ```python
   model = nn.Sequential(
       nn.Linear(1, 8),
       nn.ReLU(),
       nn.Linear(8, 16),
       nn.ReLU(),
       nn.Linear(16, 8),
       nn.ReLU(),
       nn.Linear(8, 1)
   )
   ```

3. **Training**
   - Configure and execute the training loop in PyTorch.
   ```python
   optimizer = optim.SGD(model.parameters(), lr=0.1)
   criterion = nn.MSELoss()

   for k in range(2000):
       optimizer.zero_grad()
       y_pred = model(torch.tensor(x, dtype=torch.float32))
       loss = criterion(y_pred, torch.tensor(y, dtype=torch.float32))
       loss.backward()
       optimizer.step()
   ```

4. **Plot Results**
   - Plot the final predictions against the actual data.
   ```python
   plt.scatter(x, y, color='red')  # Actual data
   plt.plot(x_sorted, y_predictions_sorted, color='blue')  # Model predictions
   plt.show()
   ```

## Conclusion
This tutorial provides a side-by-side implementation of regression models in BabyTorch and PyTorch, highlighting their similarities and key syntax differences. Both frameworks are powerful tools for building neural networks, with BabyTorch serving as a simplified entry point for learning and transitioning to PyTorch.

## Full Code 
The full code is available [here](regression.py).
