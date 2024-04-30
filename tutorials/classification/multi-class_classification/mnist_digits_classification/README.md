# MNIST Classification with BabyTorch: Linear and Convolutional Models

## Introduction
In this tutorial, we dive into classifying the MNIST dataset using BabyTorch. We'll build two types of models: a linear classifier and a convolutional neural network (CNN).

## Environment Setup and Data Loading

- Import necessary libraries and load and preprocess the MNIST dataset.
```python
    import numpy as np
    from babytorch import no_grad, Tensor
    import babytorch.nn as nn
    from babytorch.nn import Sequential, CrossEntropyLoss
    from babytorch.optim import SGD
    from babytorch.datasets import MNISTDataset, DataLoader
    from babytorch import Grapher

    train_loader = DataLoader(MNISTDataset(root='./mnist_data', train=True, transform=Tensor.to_tensor, download=True), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(MNISTDataset(root='./mnist_data', train=False, transform=Tensor.to_tensor, download=True), batch_size=batch_size, shuffle=False)
```

## Implementing a Linear Layer Classifier

1. **Set the hyperparameters for the Training**

   ```python
    input_size = 28 * 28  # images are 28x28 pixels
    hidden_size = 512
    num_classes = 10
    num_epochs = 1
    batch_size = 16
    learning_rate = 0.001
   ```

3. **Model Architecture**
   - Construct a sequential model consisting of fully connected layers.
   ```python
   model = Sequential(
       nn.Linear(784, 512, nn.ReLU()),  # Input layer to hidden layer with 512 neurons
       nn.Linear(512, 512, nn.ReLU()),  # Second hidden layer
       nn.Linear(512, 10)               # Output layer for 10 classes
   )
   ```

4. **Training the Model**
   - Set up the CrossEntropy loss function and the SGD optimizer.
   - Train the model, handling the reshaping of input data and tracking the loss.
   ```python
   criterion = CrossEntropyLoss()
   optimizer = SGD(model.parameters(), learning_rate=0.001, weight_decay=0.001)
   for epoch in range(num_epochs):
       for i, (images, labels) in enumerate(train_loader):
           images = Tensor(images.reshape(-1, 28*28), require_grad=True)
           predictions = model(images)
           loss = criterion(predictions, labels)
           loss.backward()
           optimizer.step()
           model.zero_grad()
   ```

5. **Model Evaluation**
   - Assess the model's performance on the test set without gradient updates.
   ```python
   with no_grad():
       correct = 0
       for images, labels in test_loader:
           images = Tensor(images.reshape(-1, 28*28))
           outputs = model(images)
           correct += np.sum(np.argmax(outputs.data, axis=1) == labels)
       accuracy = 100 * correct / len(test_loader.dataset)
       print(f"Test Accuracy: {accuracy:.2f}%")
   ```

## Implementing a Convolutional Layer Classifier

1. **Model Architecture**
   - Integrate convolutional layers for feature extraction.
   ```python
   model = Sequential(
       nn.Conv2D(1, 16, kernel_size=3, stride=1, padding=1),  # Convolution layer
       nn.ReLU(),
       nn.Flatten(),
       nn.Linear(28*28*16, 512, nn.ReLU()),
       nn.Linear(512, 10)
   )
   ```

2. **Training Adjustments**
   - Adapt the training loop to handle the 4D input required by the convolutional layers.
   ```python
   for epoch in range(num_epochs):
       for i, (images, labels) in enumerate(train_loader):
           images = Tensor(images.reshape(-1, 1, 28, 28), require_grad=True)
           predictions = model(images)
           loss = criterion(predictions, labels)
           loss.backward()
           optimizer.step()
           model.zero_grad()
   ```

3. **Evaluation and Testing**
   - Evaluate using the test dataset, ensuring input reshaping aligns with the convolutional layers.
   ```python
   with no_grad():
       correct = 0
       for images, labels in test_loader:
           images = Tensor(images.reshape(-1, 1, 28, 28))
           outputs = model(images)
           correct += np.sum(np.argmax(outputs.data, axis=1) == labels)
       accuracy = 100 * correct / len(test_loader.dataset)
       print(f"Test Accuracy: {accuracy:.2f}%")
   ```

## Conclusion
This detailed guide should equip you with the necessary skills to implement and understand basic image classification using both linear and convolutional models in BabyTorch, providing a strong foundation for further exploration into more complex neural network architectures.

## Full Code 

The full code for the linear classifier can be accessed [Here](linear_classification_mnist.py) and for convolutional one [here](conv2d_classification_mnist.py)