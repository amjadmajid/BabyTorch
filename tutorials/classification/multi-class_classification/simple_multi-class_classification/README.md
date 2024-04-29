# Multi-Class Classification with BabyTorch: A Practical Guide

## Introduction
Multi-class classification involves predicting multiple categories. This tutorial will guide you through setting up a multi-class classifier using BabyTorch, illustrating its effectiveness for beginners and its seamless transition capabilities to PyTorch.

For comparison, the implementation is provided uisng PyTorch and BabyTorch.

## Step-by-Step Implementation

1. **Data Preparation**:
   - Start with a small dataset of five samples, each with four features, labeled into one of three categories.
   - ```python 
     X = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                       [2.0, 3.0, 4.0, 5.0],
                       [3.0, 4.0, 5.0, 6.0],
                       [4.0, 5.0, 6.0, 7.0],
                       [5.0, 6.0, 7.0, 8.0]])
     y = torch.tensor([0, 1, 2, 1, 0])
2. **Model Setup**:
   - Construct a linear classifier using BabyTorch's `Sequential` and `Linear` modules. The output layer’s size corresponds to the number of classes.
   - ```python
     model = Sequential(Linear(4, 3))
3. **Loss Function and Optimizer**:
   - Utilize CrossEntropyLoss for handling multiple classes. Optimize the model with SGD, setting an appropriate learning rate.
   - ```python
     criterion = CrossEntropyLoss()
     optimizer = optim.SGD(model.parameters(), lr=0.01)
4. **Training Loop**:
   - Execute a loop for training that includes forward and backward passes and updates the model parameters.
   - Periodically print the loss to monitor training progress.
   - ```python
        num_epochs = 1000
        for epoch in range(num_epochs):
            # Forward pass
            outputs = model(X)
            loss = criterion(outputs, y)
          
            # Zero gradients, backward pass, optimizer step
            model.zero_grad()  
            loss.backward()
            optimizer.step()

            # Print loss
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.data:.4f}')

5. **Testing and Predictions**:
   - Test the trained model with new data and output the predicted labels.
   - ```python 
        with babytorch.no_grad():
            test_x = Tensor([[0.4, 0.2, 0.4, 0.5]])
            output = model(test_x)
            print(f'Predicted label: {output.data.argmax()}')
## Conclusion
This tutorial demonstrates the essentials of setting up a multi-class classifier using BabyTorch, from model configuration to prediction. BabyTorch provides a simple yet powerful way to delve into machine learning, making the transition to more complex frameworks like PyTorch seamless.

## Full code
The full code can be found [here](multi_class_classification.py).