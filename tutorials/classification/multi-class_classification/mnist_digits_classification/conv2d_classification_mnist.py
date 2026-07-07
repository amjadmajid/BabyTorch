"""Classify MNIST digits with a convolutional layer.

Instead of flattening the image immediately, a Conv2D layer first slides
16 learned 3x3 filters over it, producing 16 feature maps that respond to
local patterns (strokes, curves, corners).  Only then does the network
flatten and classify.

Convolution is compute-heavy: this script is comfortable on a GPU
(``pip install -e ".[gpu]"``); on a CPU, reduce ``num_batches`` below or
expect a long first epoch.

Run it::

    python conv2d_classification_mnist.py
"""

import babytorch
from babytorch import no_grad, Tensor, Grapher
import babytorch.nn as nn
from babytorch.nn import Sequential, CrossEntropyLoss
from babytorch.optim import SGD
from babytorch.datasets import MNISTDataset, DataLoader

# Hyperparameters
hidden_size = 512
num_classes = 10
num_epochs = 1
batch_size = 16
learning_rate = 0.05
num_batches = None      # e.g. 300 for a quick CPU run; None = full epoch

print(f"Device: {babytorch.device()}")

# Initialize data loaders
train_loader = DataLoader(
    MNISTDataset(root='./mnist_data', train=True, download=True),
    batch_size=batch_size, shuffle=True)
test_loader = DataLoader(
    MNISTDataset(root='./mnist_data', train=False, download=True),
    batch_size=batch_size, shuffle=False)

# Initialize the model and optimizer.  Conv2D expects images with an
# explicit channel dimension: (batch, channels, height, width).
model = Sequential(
    nn.Conv2D(1, 16, kernel_size=3, stride=1, padding=1),  # (B,1,28,28) -> (B,16,28,28)
    nn.ReLU(),
    nn.Flatten(),                                          # -> (B, 16*28*28)
    nn.Linear(28 * 28 * 16, hidden_size, nn.ReLU()),
    nn.Linear(hidden_size, num_classes),
)
optimizer = SGD(model.parameters(), learning_rate=learning_rate,
                weight_decay=0.001)
criterion = CrossEntropyLoss()

# Training loop
losses = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        if num_batches is not None and i >= num_batches:
            break
        x = Tensor(images.reshape(len(images), 1, 28, 28))  # add channel dim

        # Forward and backward passes
        predictions = model(x)
        loss = criterion(predictions, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if i % 100 == 0:
            print(f"Epoch: {epoch + 1}, Batch: {i}, "
                  f"Loss: {loss.item():.6f}, "
                  f"Learning Rate: {optimizer.learning_rate:.4f}")

# Save the trained model
model.save("./mnist_conv.model")

# Plot the loss
grapher = Grapher()
grapher.plot_loss(losses)
grapher.show()

# Evaluate the model on the held-out test set
with no_grad():
    correct = 0
    total_samples = 0
    for images, labels in test_loader:
        outputs = model(Tensor(images.reshape(len(images), 1, 28, 28)))
        predicted = outputs.numpy().argmax(axis=1)
        correct += int((predicted == labels).sum())
        total_samples += len(labels)

accuracy = 100.0 * correct / total_samples
print(f"Accuracy of the model on {total_samples} test images: {accuracy:.2f}%")
