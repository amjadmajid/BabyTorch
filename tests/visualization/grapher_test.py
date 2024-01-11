from babytorch import Tensor
from babytorch import Grapher

# Create two tensors
a = Tensor([[2.0, 4.5, 10.1],[0, 5, 1]], require_grad=True, label="a")
b = Tensor(3.0, require_grad=True, label="b")

# Perform some operations
c = a + b  # Suppose the Tensor class can track this operation
c.label = "c"
d = a * c  # Another operation
d.label = "d"
d.backward()

# Let's visualize the computation graph up to 'd'
grapher = Grapher()
grapher.show_graph(d)


