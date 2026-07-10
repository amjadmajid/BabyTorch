# Regression with BabyTorch

## Introduction

Regression means predicting a *number* (rather than a class): here, a
small MLP learns a hidden curve `y = f(x)` from 100 noisy samples. This
is the smallest complete BabyTorch program — the same four-step training
loop later trains MNIST classifiers and BabyGPT.

For the concepts behind every line (autograd, MSE, SGD), see
[Part I of the book](../../book/README.md).

## Walkthrough

1. **Data** — noisy samples of a curve the model never sees in formula
   form:

   ```python
   x_np = np.linspace(-2.0, 2.0, 100, dtype=np.float32).reshape(-1, 1)
   y_np = 0.5 * np.sin(2.0 * x_np) + 0.5 + np.random.normal(0.0, 0.05, x_np.shape)

   x = Tensor(x_np)
   y = Tensor(y_np)
   ```

2. **Model** — one input, one output, three hidden layers so the
   network can bend. Tanh suits smooth targets: it is itself smooth,
   where ReLU would approximate the sine with straight-line segments.

   ```python
   model = nn.Sequential(
       nn.Linear(1, 8, nn.Tanh()),
       nn.Linear(8, 16, nn.Tanh()),
       nn.Linear(16, 8, nn.Tanh()),
       nn.Linear(8, 1),
   )
   ```

3. **Training** — the four-step loop:

   ```python
   optimizer = SGD(model.parameters(), learning_rate=0.1)
   criterion = MSELoss()

   for step in range(2000):
       predictions = model(x)               # 1. forward
       loss = criterion(predictions, y)     # 2. how wrong?
       optimizer.zero_grad()                #    forget old gradients
       loss.backward()                      # 3. gradients
       optimizer.step()                     # 4. small step downhill
   ```

4. **Visualization** — the loss curve, then the fitted curve on top of
   the data:

   ```python
   Grapher().plot_loss(losses)

   plt.scatter(x_np, y_np, color='red')                    # the data
   plt.plot(x_np, predictions.numpy(), color='blue')       # the model
   plt.show()
   ```

## The PyTorch equivalent

BabyTorch follows PyTorch's training-loop conventions, so this small example
ports with only a few API changes — activations become separate layers and
`learning_rate` becomes `lr`:

```python
import torch.nn as nn
import torch.optim as optim

model = nn.Sequential(
    nn.Linear(1, 8), nn.Tanh(),
    nn.Linear(8, 16), nn.Tanh(),
    nn.Linear(16, 8), nn.Tanh(),
    nn.Linear(8, 1),
)
optimizer = optim.SGD(model.parameters(), lr=0.1)
criterion = nn.MSELoss()
# ...the training loop is identical...
```

## Full code

The full code is available in
[`regression_01.py`](regression_01.py). Run it with:

```bash
python regression_01.py
```

It prints the falling loss and opens two plots (requires the `[viz]`
extra: `pip install -e ".[viz]"`).
