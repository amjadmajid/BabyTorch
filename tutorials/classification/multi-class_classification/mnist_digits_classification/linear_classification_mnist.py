"""Classify MNIST digits with a fully connected network.

Each 28x28 image is flattened to 784 numbers and pushed through three
Linear layers; CrossEntropyLoss trains the network to score the correct
digit highest.  Expect ~95% test accuracy after a single epoch.

The MNIST files (~11 MB) are downloaded automatically on the first run.

Run it::

    python linear_classification_mnist.py
"""

import babytorch
from babytorch import no_grad, Tensor, Grapher
import babytorch.nn as nn
from babytorch.nn import Sequential, CrossEntropyLoss
from babytorch.optim import SGD
from babytorch.datasets import MNISTDataset, DataLoader

# Hyperparameters
input_size = 28 * 28   # images are 28x28 pixels
hidden_size = 512
num_classes = 10
num_epochs = 1
batch_size = 16
learning_rate = 0.05

print(f"Device: {babytorch.device()}")

# Initialize data loaders.  Batches arrive as plain arrays of shape
# (batch, 28, 28); we flatten and wrap them in a Tensor per batch.
train_loader = DataLoader(
    MNISTDataset(root='./mnist_data', train=True, download=True),
    batch_size=batch_size, shuffle=True)
test_loader = DataLoader(
    MNISTDataset(root='./mnist_data', train=False, download=True),
    batch_size=batch_size, shuffle=False)

# Initialize the model and optimizer
model = Sequential(
    nn.Linear(input_size, hidden_size, nn.ReLU()),
    nn.Linear(hidden_size, hidden_size, nn.ReLU()),
    nn.Linear(hidden_size, num_classes),
)
optimizer = SGD(model.parameters(), learning_rate=learning_rate,
                weight_decay=0.001)
criterion = CrossEntropyLoss()

# Training loop
losses = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        x = Tensor(images.reshape(len(images), input_size))

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
model.save("./mnist.model")

# Plot the loss
grapher = Grapher()
grapher.plot_loss(losses)
grapher.show()

# Evaluate the model on the held-out test set
with no_grad():
    correct = 0
    total_samples = 0
    for images, labels in test_loader:
        outputs = model(Tensor(images.reshape(len(images), input_size)))
        predicted = outputs.numpy().argmax(axis=1)
        correct += int((predicted == labels).sum())
        total_samples += len(labels)

accuracy = 100.0 * correct / total_samples
print(f"Accuracy of the model on {total_samples} test images: {accuracy:.2f}%")
