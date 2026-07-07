"""Binary classification on a four-example toy dataset.

The BabyTorch model always runs.  If PyTorch happens to be installed, the
identical model is trained there too, so you can compare the two
frameworks' losses and predictions side by side -- and see that the API
really is the same with the word "baby" removed.

Run it::

    python classification_01.py
"""

import numpy as np

import babytorch.nn as b_nn
from babytorch.nn import MSELoss
from babytorch.optim import SGD
from babytorch import Tensor, Grapher


def babytorch_classification(x, target, num_iterations=1000, lr=0.001):
    print("babytorch_classification")
    model = b_nn.Sequential(
        b_nn.Linear(x.shape[1], 4, b_nn.Sigmoid()),
        b_nn.Linear(4, 1),
    )

    criterion = MSELoss()
    optimizer = SGD(model.parameters(), learning_rate=lr)

    losses = []
    for _ in range(num_iterations):
        predictions = model(x)
        loss = criterion(predictions, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return predictions, losses


def pytorch_classification(x, target, num_iterations=1000, lr=0.001):
    print("pytorch_classification")
    import torch
    import torch.nn as nn
    import torch.optim as optim

    model = nn.Sequential(
        nn.Linear(x.shape[1], 4),
        nn.Sigmoid(),
        nn.Linear(4, 1),
    )

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    losses = []
    for _ in range(num_iterations):
        predictions = model(x)
        loss = criterion(predictions, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return predictions.detach().numpy(), losses


if __name__ == '__main__':
    input_data = np.array([[1, 1, -1],
                           [1, 1, 1],
                           [1, -1, -1],
                           [1, -1, 1]], dtype=np.float32)

    # The target shape matches the model's output shape: (batch_size, 1).
    target = np.array([[1], [-1], [-1], [-1]], dtype=np.float32)

    babytorch_y_pred, babytorch_losses = babytorch_classification(
        Tensor(input_data), Tensor(target), num_iterations=10000, lr=0.01)

    try:
        import torch
        pytorch_y_pred, pytorch_losses = pytorch_classification(
            torch.tensor(input_data), torch.tensor(target),
            num_iterations=10000, lr=0.01)
    except ImportError:
        pytorch_y_pred, pytorch_losses = None, None
        print("PyTorch is not installed -- skipping the comparison run.")

    print("\nPredictions vs true labels:")
    predictions = babytorch_y_pred.numpy()
    if pytorch_y_pred is not None:
        print("BabyTorch | PyTorch | true")
        for b, p, t in zip(predictions, pytorch_y_pred, target):
            print(f"{b[0]:9.4f} | {p[0]:7.4f} | {t[0]:4.1f}")
    else:
        print("BabyTorch | true")
        for b, t in zip(predictions, target):
            print(f"{b[0]:9.4f} | {t[0]:4.1f}")

    grapher = Grapher()
    grapher.plot_loss(babytorch_losses, label='BabyTorch')
    if pytorch_losses is not None:
        grapher.plot_loss(pytorch_losses, label='PyTorch')
    grapher.show()
