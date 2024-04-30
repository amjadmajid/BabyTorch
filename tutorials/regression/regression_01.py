
BABYTORCH = False

import random
import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

np.random.seed(0)
random.seed(0)

num_iterations = 2000
lr =0.1

# Create a regression dataset
x, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=0)

y = (y.reshape(-1, 1) + 1) * .5  # make y be between 0 and 1
y /= np.max(y)
print(f"{y[:5]=}")

if BABYTORCH: 

    import babytorch 
    import babytorch.nn as nn
    from babytorch.optim import SGD, LambdaLR
    from babytorch.visualization import Grapher

    x = babytorch.Tensor(x, require_grad=True)
    y = babytorch.Tensor(y, require_grad=True)

    model = nn.Sequential(
        nn.Linear(1, 8, nn.ReLU()),
        nn.Linear(8, 16, nn.ReLU()),
        nn.Linear(16, 8, nn.ReLU()),
        nn.Linear(8, 1)
    )

    optimizer = SGD(model.parameters(), learning_rate=lr)
    # optimizer = SGD(model.parameters(), weight_decay=0.001, learning_rate=lr)
    # scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0 - 0.9 * epoch / num_iterations)
    # scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: lr - lr * 0.9 * (epoch+1) / (num_iterations+1))
    criterion = nn.MSELoss() 

    # Training (model optimization)
    losses = []
    for k in range(num_iterations):
        y_predictions = model(x)
        loss = criterion(y_predictions, y)
        loss.backward()
        optimizer.step()
        # scheduler.step(k)
        model.zero_grad()
        losses.append(loss.data)
        # print(f"{k} - {loss.data}")

    # print(f"{losses=}")
    Grapher().plot_loss(losses)
    Grapher().show()

    # Testing
    with babytorch.no_grad():
        y_predictions = model(x)

    # Convert to numpy arrays for easier manipulation
    x = np.array(x).flatten()
    y_predictions = np.array(y_predictions).flatten()


else:

    # PyTorch version
    import random
    import torch
    import torch.nn as nn
    import torch.optim as optim

    torch.manual_seed(0)

    # Convert to PyTorch Tensors
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    model = nn.Sequential(
        nn.Linear(1, 8),
        nn.ReLU(),
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 8),
        nn.ReLU(),
        nn.Linear(8, 1)
    )

    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Training (model optimization)
    losses = []
    for k in range(num_iterations):
        y_predictions = model(x_tensor)
        loss = criterion(y_predictions, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        # print(f"{k} - {loss.item()}")

    print(losses[-1])
    plt.plot(losses)
    plt.title('Loss Plot')
    plt.show()

    # Testing
    with torch.no_grad():
        y_predictions = model(x_tensor)

    # Convert to numpy arrays for easier manipulation
    x = x.flatten()
    y_predictions = y_predictions.numpy().flatten()


# Get the sorted order of x and sort x and y_predictions accordingly
sorted_order = np.argsort(x)
x_sorted = x[sorted_order]
y_predictions_sorted = y_predictions[sorted_order]

# Plot the results
plt.scatter(x, y, color='red')
plt.scatter(x, y_predictions, color='green')
plt.plot(x_sorted, y_predictions_sorted, color='blue', linewidth=3)  # Plot sorted values
plt.show()
