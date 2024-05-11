import cupy as cp
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_moons
from babytorch import Tensor, Grapher
import babytorch.nn as nn
from babytorch.nn import Sequential, MSELoss
from babytorch.optim import SGD

def clip_gradients_norm(parameters, max_norm):
    """Clips the gradients of the model parameters."""
    total_norm = cp.sqrt(sum(cp.sum(cp.square(p.grad)) for p in parameters))
    scale = max_norm / (total_norm + 1e-6)
    if total_norm > max_norm:
        for p in parameters:
            p.grad *= scale

# Hyperparameters
num_iterations = 8000
learning_rate = 0.01

# Generate dataset
# X_orig, y_orig = make_circles(n_samples=200, noise=0.1)
X_orig, y_orig = make_moons(n_samples=200, noise=0.1)
y_orig = y_orig * 2 - 1  # Convert y to -1 or 1
X = Tensor(X_orig, requires_grad=True)
y = Tensor(y_orig, requires_grad=True)
y = y.reshape(-1, 1)

# Initialize the model
model = Sequential(
    nn.Linear(2, 64, nn.ReLU()),
    nn.Linear(64, 32, nn.ReLU()),
    nn.Linear(32, 16, nn.ReLU()),
    nn.Linear(16, 1)
)
optimizer = SGD(model.parameters(), learning_rate=learning_rate, weight_decay=0.0005)
# scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: lr - lr * 0.9 * (epoch+1) / (num_iterations+1))
criterion = MSELoss()

# Training loop
losses = []
for epoch in range(num_iterations):
    y_pred = model(X)
    loss = criterion(y_pred, y)
    loss.backward()
    clip_gradients_norm(model.parameters(), max_norm=10.0)
    optimizer.step()
    # scheduler.step(k)
    model.zero_grad()
    losses.append(loss.data.get())
    # print(f"Epoch{epoch+1}/{num_iterations}, lr={optimizer.learning_rate}, loss={loss.data}")

# Plot the loss
Grapher().plot_loss(losses)
Grapher().show()

# Visualize the decision boundary
h = 0.25
x_min, x_max = X_orig[:, 0].min() - 1, X_orig[:, 0].max() + 1
y_min, y_max = X_orig[:, 1].min() - 1, X_orig[:, 1].max() + 1
xx, yy = cp.meshgrid(cp.arange(x_min, x_max, h), cp.arange(y_min, y_max, h))
Xmesh = cp.c_[xx.ravel(), yy.ravel()]
scores = model(Tensor(Xmesh))
Z = cp.array([s > 0 for s in scores])
Z = Z.reshape(xx.shape)

# Convert cupy arrays to numpy arrays for matplotlib
xx_np = xx.get()
yy_np = yy.get()
Z_np = Z.get()

# Use numpy arrays for plotting
plt.contourf(xx_np, yy_np, Z_np, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X_orig[:, 0], X_orig[:, 1], c=y_orig, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx_np.min(), xx_np.max())
plt.ylim(yy_np.min(), yy_np.max())
plt.show()

