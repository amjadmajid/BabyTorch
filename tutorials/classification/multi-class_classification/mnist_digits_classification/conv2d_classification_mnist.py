
babytorch = True  # Set to False to use PyTorch's implementation

if babytorch: 

    import cupy as cp
    from babytorch import no_grad, Tensor
    import babytorch.nn as nn
    from babytorch.nn import Sequential, MSELoss, CrossEntropyLoss
    from babytorch.optim import SGD
    from babytorch.optim.lr_scheduler import LambdaLR
    from babytorch.datasets import MNISTDataset, DataLoader
    from babytorch import Grapher

    # Hyperparameters
    input_size_after_conv = 28 * 28 * 16  # Adjusted for the output shape from Conv2D
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
        nn.Conv2D(1, 16, kernel_size=3, stride=1, padding=1),  # Assuming input is (Batch, 1, 28, 28)
        nn.ReLU(),
        nn.Flatten(),  # Flatten the tensor
        nn.Linear(28*28*16, hidden_size, nn.ReLU()),  # Update the input size to match the flattened output
        nn.Linear(hidden_size, num_classes)
    )
    optimizer = SGD(model.parameters(), learning_rate=learning_rate, weight_decay=0.001)

    criterion = CrossEntropyLoss()

    # Training loop
    losses = []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = Tensor(images.reshape(-1, 1, 28, 28), requires_grad=True)  # Add channel dimension

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
    model.save("./mnist_conv.model")

    # Plot the loss
    Grapher().plot_loss(losses)
    Grapher().show()

    # Evaluate the model
    with no_grad():
        correct = 0
        total_samples = 0

        for images, labels in test_loader:
            images = Tensor(images.reshape(-1, 1, 28, 28))  # Add channel dimension
            outputs = model(images)
            correct += cp.sum(cp.argmax(outputs.data, axis=1) == labels)
            total_samples += batch_size

        accuracy = 100 * correct / total_samples
        print(f"Accuracy of the model on {total_samples} test images: {accuracy:.2f}%")

# Use PyTorch's implementation
else: 
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    # Hyperparameters
    input_size_after_conv = 28 * 28 * 16  # Adjusted for the output shape from Conv2D
    hidden_size = 512
    num_classes = 10
    num_epochs = 1
    batch_size = 16
    learning_rate = 0.001

    # Initialize data loaders
    train_dataset = datasets.MNIST(root='./mnist_data', train=True, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets.MNIST(root='./mnist_data', train=False, transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model and optimizer
    model = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Flatten(),  # Flatten the tensor
        nn.Linear(input_size_after_conv, hidden_size),
        nn.Linear(hidden_size, num_classes)
    )

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    losses = []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Forward and backward passes
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Update weights
            optimizer.step()
            optimizer.zero_grad()

            # Log progress
            if i % 100 == 0:
                print(f"Epoch: {epoch + 1}, Batch: {i}, Loss: {loss.item():.6f}")

            losses.append(loss.item())

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total_samples = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total_samples
        print(f'Accuracy of the model on {total_samples} test images: {accuracy:.2f}%')

    # Optional: Save the model
    torch.save(model.state_dict(), './mnist_conv.pth')

