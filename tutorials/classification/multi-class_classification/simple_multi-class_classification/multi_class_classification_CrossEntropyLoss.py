"""Multi-class classification with CrossEntropyLoss on a tiny dataset.

Five examples, four features each, three possible classes.  The model
outputs one raw score ("logit") per class and CrossEntropyLoss turns the
scores into probabilities and charges -log(p) of the correct class.

The BabyTorch model always runs; if PyTorch is installed, the identical
model is trained there too for comparison.

Run it::

    python multi_class_classification_CrossEntropyLoss.py
"""

import numpy as np

import babytorch
from babytorch import Tensor
import babytorch.nn as nn
from babytorch.optim import SGD

# Dummy data (5 samples, 4 features)
X_data = np.array([
    [0.1, 0.2, 0.3, 0.4],
    [0.5, 0.3, 0.2, 0.7],
    [0.6, 0.1, 0.1, 0.1],
    [0.9, 0.8, 0.2, 0.1],
    [0.5, 0.1, 0.1, 0.2],
], dtype=np.float32)

# Dummy targets: one class id (0, 1 or 2) per sample
y_data = np.array([0, 1, 2, 1, 0], dtype=np.int64)

test_data = np.array([[0.4, 0.2, 0.4, 0.5]], dtype=np.float32)

input_dim = 4    # number of input features
output_dim = 3   # number of classes


def babytorch_run(num_epochs=1000, lr=0.01):
    print("--- BabyTorch ---")
    X = Tensor(X_data)

    model = nn.Sequential(nn.Linear(input_dim, output_dim))
    criterion = nn.CrossEntropyLoss()   # targets are integer class ids
    optimizer = SGD(model.parameters(), learning_rate=lr)

    for epoch in range(num_epochs):
        outputs = model(X)
        loss = criterion(outputs, y_data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Test the model: the predicted class is the one with the largest score.
    with babytorch.no_grad():
        output = model(Tensor(test_data))
    print(f'Predicted label: {int(output.argmax())}')


def pytorch_run(num_epochs=1000, lr=0.01):
    print("--- PyTorch ---")
    import torch
    import torch.nn as t_nn
    import torch.optim as t_optim

    X = torch.tensor(X_data)
    y = torch.tensor(y_data)

    model = t_nn.Sequential(t_nn.Linear(input_dim, output_dim))
    criterion = t_nn.CrossEntropyLoss()
    optimizer = t_optim.SGD(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        outputs = model(X)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    with torch.no_grad():
        output = model(torch.tensor(test_data))
    print(f'Predicted label: {output.argmax().item()}')


if __name__ == '__main__':
    babytorch_run()
    try:
        import torch  # noqa: F401
    except ImportError:
        print("\nPyTorch is not installed -- skipping the comparison run.")
    else:
        print()
        pytorch_run()
