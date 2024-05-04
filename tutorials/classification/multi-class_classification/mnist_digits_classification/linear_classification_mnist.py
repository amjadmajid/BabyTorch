import numpy as np
from babytorch import no_grad, Tensor
import babytorch.nn as nn
from babytorch.nn import Sequential, MSELoss, CrossEntropyLoss
from babytorch.optim import SGD
from babytorch.optim.lr_scheduler import LambdaLR
from babytorch.datasets import MNISTDataset, DataLoader
from babytorch import Grapher

# Hyperparameters
input_size = 28 * 28  # images are 28x28 pixels
hidden_size = 512
num_classes = 10
num_epochs = 1
batch_size = 16
learning_rate = 0.001


# Initialize data loaders
train_loader = DataLoader(MNISTDataset(root='./mnist_data', train=True, transform=Tensor.to_tensor, download=True), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(MNISTDataset(root='./mnist_data', train=False, transform=Tensor.to_tensor, download=True), batch_size=batch_size, shuffle=False)

# Initialize the model and optimizer
model = Sequential(
    nn.Linear(input_size, hidden_size, nn.ReLU()),
    nn.Linear(hidden_size, hidden_size, nn.ReLU()),
    nn.Linear(hidden_size, num_classes)
)
optimizer = SGD(model.parameters(), learning_rate=learning_rate, weight_decay=0.001)

criterion = CrossEntropyLoss()

# Training loop
losses = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Tensor(images.reshape(-1, input_size), requires_grad=True)

        # Forward and backward passes
        predictions = model(images)
        loss = criterion(predictions, labels)
        loss.backward()

        # Update weights
        optimizer.step()
        model.zero_grad()

        # Log progress
        if i % 100 == 0:
            print(f"Epoch: {epoch + 1}, Batch: {i}, Loss: {loss.data:.6f}, Learning Rate: {optimizer.learning_rate:.4f}")

        losses.append(loss.data)

# save the model
model.save("./mnist.model")

# Plot the loss
Grapher().plot_loss(losses)
Grapher().show()

# Evaluate the model
with no_grad():
    correct = 0
    total_samples = 0

    for images, labels in test_loader:
        images = Tensor(images.reshape(-1, input_size))
        outputs = model(images)
        correct += np.sum(np.argmax(outputs.data, axis=1) == labels)
        total_samples += batch_size

    accuracy = 100 * correct / total_samples
    print(f"Accuracy of the model on {total_samples} test images: {accuracy:.2f}%")
