# MNIST Classification with BabyTorch: Linear and Convolutional Models

## Introduction

In this tutorial we classify the MNIST handwritten digits with two
models: a fully connected (linear) network, and a convolutional network
that first extracts local visual features. The dataset (~11 MB) is
downloaded automatically on first run.

## Environment setup and data loading

`MNISTDataset` parses the raw files; `DataLoader` shuffles and batches
them. Batches arrive as plain arrays — `images` has shape
`(batch, 28, 28)`, `labels` holds integer digits:

```python
import babytorch
from babytorch import no_grad, Tensor, Grapher
import babytorch.nn as nn
from babytorch.nn import Sequential, CrossEntropyLoss
from babytorch.optim import SGD
from babytorch.datasets import MNISTDataset, DataLoader

train_loader = DataLoader(MNISTDataset(root='./mnist_data', train=True,  download=True),
                          batch_size=16, shuffle=True)
test_loader  = DataLoader(MNISTDataset(root='./mnist_data', train=False, download=True),
                          batch_size=16, shuffle=False)
```

## Implementing a linear classifier

1. **Hyperparameters**

   ```python
   input_size = 28 * 28   # images are 28x28 pixels
   hidden_size = 512
   num_classes = 10
   num_epochs = 1
   batch_size = 16
   learning_rate = 0.05
   ```

2. **Model architecture** — flatten each image to 784 numbers, then
   stack fully connected layers:

   ```python
   model = Sequential(
       nn.Linear(784, 512, nn.ReLU()),
       nn.Linear(512, 512, nn.ReLU()),
       nn.Linear(512, 10),              # one score per digit
   )
   ```

3. **Training the model** — cross-entropy loss, SGD, and the standard
   loop; each batch is flattened and wrapped in a Tensor:

   ```python
   criterion = CrossEntropyLoss()
   optimizer = SGD(model.parameters(), learning_rate=0.05, weight_decay=0.001)

   for epoch in range(num_epochs):
       for images, labels in train_loader:
           x = Tensor(images.reshape(len(images), 784))
           predictions = model(x)
           loss = criterion(predictions, labels)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
   ```

4. **Model evaluation** — accuracy on the held-out test set, without
   gradient tracking:

   ```python
   with no_grad():
       correct = 0
       total = 0
       for images, labels in test_loader:
           outputs = model(Tensor(images.reshape(len(images), 784)))
           predicted = outputs.numpy().argmax(axis=1)
           correct += int((predicted == labels).sum())
           total += len(labels)
   print(f"Test accuracy: {100.0 * correct / total:.2f}%")
   ```

Expect roughly 95% after a single epoch.

## Implementing a convolutional classifier

1. **Model architecture** — a `Conv2D` layer slides 16 learned 3×3
   filters over the image before classification. Conv2D expects an
   explicit channel dimension, so images are reshaped to
   `(batch, 1, 28, 28)`:

   ```python
   model = Sequential(
       nn.Conv2D(1, 16, kernel_size=3, stride=1, padding=1),
       nn.ReLU(),
       nn.Flatten(),
       nn.Linear(28 * 28 * 16, 512, nn.ReLU()),
       nn.Linear(512, 10),
   )
   ```

2. **Training adjustments** — identical loop; only the reshape changes:

   ```python
   x = Tensor(images.reshape(len(images), 1, 28, 28))
   ```

   Expect ~97% test accuracy after one epoch — the convolutional
   features are worth a couple of points over the flat model.

3. **A note on speed** — convolution is compute-heavy. On a GPU
   (`pip install -e ".[gpu]"`) the epoch takes minutes; on a CPU, set
   `num_batches` in the script to something small (e.g. 300) for a quick
   run.

## Conclusion

Both models share the same loss, optimizer and loop — the only
difference is what the early layers do to the pixels. That is the
general pattern in deep learning: architectures change with the data;
training does not.

## Full code

The linear classifier: [`linear_classification_mnist.py`](linear_classification_mnist.py) ·
the convolutional one: [`conv2d_classification_mnist.py`](conv2d_classification_mnist.py)
