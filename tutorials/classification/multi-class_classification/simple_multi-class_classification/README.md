# Multi-Class Classification with BabyTorch: A Practical Guide

## Introduction

Multi-class classification means choosing among several categories. The
model outputs one raw score ("logit") per class, and `CrossEntropyLoss`
turns those scores into probabilities and charges the model `-log(p)` of
the probability it gave the correct class. (Chapter 3 of
[the book](../../../../book/README.md) explains why — and a language
model predicting its next token is this exact setup with thousands of
classes.)

The script trains the BabyTorch model always; if PyTorch is installed it
also trains the identical PyTorch model, so you can compare the two.

## Step-by-step implementation

1. **Data preparation**:
   - Five samples, four features each, labeled with one of three classes.
   ```python
   X = Tensor([[0.1, 0.2, 0.3, 0.4],
               [0.5, 0.3, 0.2, 0.7],
               [0.6, 0.1, 0.1, 0.1],
               [0.9, 0.8, 0.2, 0.1],
               [0.5, 0.1, 0.1, 0.2]])
   y = np.array([0, 1, 2, 1, 0], dtype=np.int64)   # integer class ids
   ```

2. **Model setup**:
   - A single linear layer: 4 features in, 3 class scores out.
   ```python
   model = nn.Sequential(nn.Linear(4, 3))
   ```

3. **Loss function and optimizer**:
   ```python
   criterion = nn.CrossEntropyLoss()
   optimizer = SGD(model.parameters(), learning_rate=0.01)
   ```

4. **Training loop**:
   ```python
   for epoch in range(1000):
       outputs = model(X)
       loss = criterion(outputs, y)

       optimizer.zero_grad()
       loss.backward()
       optimizer.step()

       if (epoch + 1) % 100 == 0:
           print(f'Epoch [{epoch + 1}/1000], Loss: {loss.item():.4f}')
   ```

5. **Testing and predictions**:
   - The predicted class is the one with the highest score.
   ```python
   with babytorch.no_grad():
       output = model(Tensor([[0.4, 0.2, 0.4, 0.5]]))
   print(f'Predicted label: {int(output.argmax())}')
   ```

## Conclusion

This is the essential shape of every classifier: scores per class,
cross-entropy, the four-step loop. The MNIST tutorial next door applies
it to real images, and BabyGPT applies it to text.

## Full code

The full code is in
[`multi_class_classification_CrossEntropyLoss.py`](multi_class_classification_CrossEntropyLoss.py).
