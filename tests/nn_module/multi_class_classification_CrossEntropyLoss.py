import torch
import torch.nn as nn
import torch.optim as optim

# Dummy data (5 samples, 4 features)
X = torch.FloatTensor([
    [0.1, 0.2, 0.3, 0.4],
    [0.5, 0.3, 0.2, 0.7],
    [0.6, 0.1, 0.1, 0.1],
    [0.9, 0.8, 0.2, 0.1],
    [0.5, 0.1, 0.1, 0.2]
])

# Dummy targets (5 samples, 1 label per sample)
y = torch.LongTensor([0, 1, 2, 1, 0])

# Initialize the model using nn.Sequential
input_dim = 4  # Number of input features
output_dim = 3  # Number of labels (classes)

model = nn.Sequential(
    nn.Linear(input_dim, output_dim)
)

criterion = nn.CrossEntropyLoss()  # Cross-Entropy Loss for multi-class problems
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training Loop
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X)
    # print("outputs: ", outputs)
    # exit()

    loss = criterion(outputs, y)

    # Zero gradients, backward pass, optimizer step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Test the model
with torch.no_grad():
    test_x = torch.FloatTensor([[0.4, 0.2, 0.4, 0.5]])
    output = model(test_x)
    print("output: ", output)
    _, predicted = torch.max(output.data, 1)
    print(f'Predicted label: {predicted.item()}')


print("#"*100)



import babytorch
from babytorch import Tensor
import babytorch.nn as nn
from babytorch.optim import SGD
import numpy as np

# Dummy data (5 samples, 4 features)
X = Tensor([
    [0.1, 0.2, 0.3, 0.4],
    [0.5, 0.3, 0.2, 0.7],
    [0.6, 0.1, 0.1, 0.1],
    [0.9, 0.8, 0.2, 0.1],
    [0.5, 0.1, 0.1, 0.2]
])

# Dummy targets (5 samples, 1 label per sample)
y = Tensor([0, 1, 2, 1, 0], dtype=np.int32) # int is necessary for the CrossEntropyLoss

# Initialize the model using nn.Sequential
input_dim = 4  # Number of input features
output_dim = 3  # Number of labels (classes)

model = nn.Sequential(
    nn.Linear(input_dim, output_dim)
)

criterion = nn.CrossEntropyLoss()  # Cross-Entropy Loss for multi-class problems
optimizer = SGD(model.parameters(), learning_rate=0.01)

# Training Loop
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)
    # print(f"{loss=}")

    # Zero gradients, backward pass, optimizer step
    model.zero_grad()  #TODO this has to be optimizer.zero_grad() to mach the pytorch implementation
    loss.backward()
    optimizer.step()

    # Print loss
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.data:.4f}')

# Test the model
with babytorch.no_grad():
    test_x = Tensor([[0.4, 0.2, 0.4, 0.5]])
    output = model(test_x)
    # print(f'{output=}')
    # predicted = output.max(axis=1)
    print(f'Predicted label: {output.data.argmax()}')
