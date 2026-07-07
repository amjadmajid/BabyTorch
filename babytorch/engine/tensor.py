"""The Tensor: an n-dimensional array that remembers how it was made.

A :class:`Tensor` wraps an array (``.data``) plus three extra fields that
make automatic differentiation possible:

* ``requires_grad`` -- should gradients flow into this tensor?
* ``grad``          -- the accumulated gradient (filled in by ``backward``);
* ``operation``     -- the :class:`Operation` that produced this tensor,
  which in turn remembers *its* input tensors.

Chained together, ``operation`` links form the **computation graph**: a
recording of every step from the inputs to the final loss.  Calling
``loss.backward()`` replays that recording in reverse, applying the chain
rule one operation at a time.

The actual math of each operation lives in ``operations.py`` -- this file
is only the bookkeeping around it.
"""

from ..backend import xp, to_numpy
from .operations import *

# A single global flag, toggled by the `no_grad` context manager below.
# While it is True, new tensors are created without graph links, so no
# memory is spent remembering operations we will never differentiate.
NO_GRAD_CONTEXT = False


class no_grad:
    """Context manager that disables gradient tracking.

    Use it for evaluation and inference, where you only need the forward
    pass::

        with babytorch.no_grad():
            predictions = model(x)
    """

    def __enter__(self):
        global NO_GRAD_CONTEXT
        self.previous_value = NO_GRAD_CONTEXT
        NO_GRAD_CONTEXT = True
        return self

    def __exit__(self, type, value, traceback):
        global NO_GRAD_CONTEXT
        NO_GRAD_CONTEXT = self.previous_value


class Tensor:
    """A minimal tensor object that records operations for autodiff."""

    def __init__(self, data, requires_grad=False, dtype=None,
                 label="", _op_label=""):
        """Create a new tensor wrapping ``data``.

        Parameters
        ----------
        data : array-like
            Numeric data: a Python number, (nested) list, NumPy/CuPy array.
        requires_grad : bool, optional
            If ``True``, gradients will be accumulated in ``.grad``
            during backprop.  Model parameters set this; inputs usually
            don't need to.
        dtype : data type, optional
            Element type, ``float32`` by default (like PyTorch).
        label : str, optional
            Human-readable name, shown by the graph visualizer.
        _op_label : str, optional
            Name of the operation that produced this tensor (set
            internally; also used by the visualizer).
        """
        # dtype=None means "keep the data's own dtype if it already has one
        # (so operations preserve float64/float32 through the graph), and
        # fall back to float32 for raw Python numbers/lists" -- matching
        # PyTorch, whose default tensor type is float32.
        if dtype is None:
            dtype = getattr(data, "dtype", None)
            if dtype is None or not xp.issubdtype(dtype, xp.floating):
                dtype = xp.float32
        # asarray: reuse the buffer when possible instead of always copying
        self.data = xp.asarray(data, dtype=dtype)
        self.dtype = self.data.dtype
        self.requires_grad = requires_grad
        self.grad = None
        self.operation = None
        self.label = label
        self._op_label = _op_label

    # ------------------------------------------------------------------
    # Small conveniences
    # ------------------------------------------------------------------

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def T(self):
        return self.transpose()

    def item(self):
        """Return the value of a single-element tensor as a Python number."""
        return self.data.item()

    def numpy(self):
        """Return the data as a NumPy array on the CPU (copies from GPU)."""
        return to_numpy(self.data)

    def detach(self):
        """Return a new tensor with the same data but cut off from the
        computation graph (no gradient will flow through it)."""
        return Tensor(self.data, requires_grad=False, dtype=self.dtype)

    def argmax(self, axis=None):
        """Index of the maximum value (plain array, not differentiable)."""
        return self.data.argmax(axis=axis)

    def get_state(self):
        return {'data': self.data, 'grad': self.grad}

    def set_state(self, state):
        self.data = xp.asarray(state['data'])
        self.grad = None if state['grad'] is None else xp.asarray(state['grad'])

    @staticmethod
    def to_tensor(array, requires_grad=False):
        return Tensor(array, requires_grad=requires_grad)

    def _make_output(self, op, result, requires_grad, op_label):
        """Wrap an operation's raw result in a new Tensor and, when
        gradients are being tracked, link it into the computation graph."""
        requires_grad = requires_grad and not NO_GRAD_CONTEXT
        output = Tensor(result, requires_grad=requires_grad, _op_label=op_label)
        if requires_grad:
            output.operation = op   # <- this link *is* the graph edge
        return output

    # ------------------------------------------------------------------
    # Shape manipulation
    # ------------------------------------------------------------------

    def reshape(self, *new_shape):
        # accept both t.reshape(2, 3) and t.reshape((2, 3))
        if len(new_shape) == 1 and isinstance(new_shape[0], (tuple, list)):
            new_shape = tuple(new_shape[0])
        op = ReshapeOperation(self, *new_shape)
        return self._make_output(op, op.forward(), self.requires_grad, "reshape")

    def transpose(self, axes=None):
        op = TransposeOperation(axes)
        return self._make_output(op, op.forward(self), self.requires_grad, "transpose")

    def squeeze(self, axis=None):
        op = SqueezeOperation()
        return self._make_output(op, op.forward(self, axis), self.requires_grad, "squeeze")

    def unsqueeze(self, axis):
        op = UnsqueezeOperation()
        return self._make_output(op, op.forward(self, axis), self.requires_grad, "unsqueeze")

    def flatten(self):
        op = FlattenOperation()
        return self._make_output(op, op.forward(self), self.requires_grad, "flatten")

    def __getitem__(self, indices):
        # Allow tensors to be used as indices (e.g. labels for a lookup):
        # index arrays must be integers, so cast the float data.
        if isinstance(indices, Tensor):
            indices = indices.data.astype(xp.int64)
        elif isinstance(indices, tuple):
            indices = tuple(i.data.astype(xp.int64) if isinstance(i, Tensor) else i
                            for i in indices)
        op = SliceOperation(indices)
        return self._make_output(op, op.forward(self), self.requires_grad, "slice")

    def __setitem__(self, index, value):
        if isinstance(value, Tensor):
            value = value.data
        self.data[index] = value

    # ------------------------------------------------------------------
    # Arithmetic operators
    # ------------------------------------------------------------------

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        op = AddOperation()
        result = op.forward(self, other)
        return self._make_output(op, result,
                                 self.requires_grad or other.requires_grad, "+")

    def __radd__(self, other):
        return Tensor(other) + self

    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        op = SubOperation()
        result = op.forward(self, other)
        return self._make_output(op, result,
                                 self.requires_grad or other.requires_grad, "-")

    def __rsub__(self, other):
        return Tensor(other) - self

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        op = MulOperation()
        result = op.forward(self, other)
        return self._make_output(op, result,
                                 self.requires_grad or other.requires_grad, "*")

    def __rmul__(self, other):
        # 2 * tensor  ->  tensor * 2  (multiplication commutes)
        return self * other

    def __neg__(self):
        return self * -1.0

    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        op = DivOperation()
        result = op.forward(self, other)
        return self._make_output(op, result,
                                 self.requires_grad or other.requires_grad, "/")

    def __rtruediv__(self, other):
        return Tensor(other) / self

    def __pow__(self, exponent):
        if isinstance(exponent, Tensor):
            raise TypeError("Tensor exponents are not supported; "
                            "use a plain number, e.g. t ** 2 or t ** 0.5")
        op = PowOperation(exponent)
        return self._make_output(op, op.forward(self), self.requires_grad, f"**{exponent}")

    def sqrt(self):
        return self ** 0.5

    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        op = MatMulOperation()
        result = op.forward(self, other)
        return self._make_output(op, result,
                                 self.requires_grad or other.requires_grad, "@")

    # ------------------------------------------------------------------
    # Reductions
    # ------------------------------------------------------------------

    def sum(self, axis=None, keepdims=False):
        op = SumOperation()
        result = op.forward(self, axis=axis, keepdims=keepdims)
        return self._make_output(op, result, self.requires_grad, "sum")

    def max(self, axis=None, keepdims=False):
        op = MaxOperation()
        result = op.forward(self, axis, keepdims)
        return self._make_output(op, result, self.requires_grad, "max")

    def mean(self, axis=None, keepdims=False):
        """Average of all elements (or along ``axis``).

        Built by composition: a mean is just ``sum / count``, and both of
        those already know their own gradients -- so we get the backward
        pass for free.
        """
        if axis is None:
            count = self.data.size
        else:
            axes = axis if isinstance(axis, tuple) else (axis,)
            count = 1
            for ax in axes:
                count *= self.shape[ax]
        return self.sum(axis=axis, keepdims=keepdims) / float(count)

    def var(self, axis=None, keepdims=False):
        """Variance: the mean squared distance from the mean."""
        mu = self.mean(axis=axis, keepdims=True)
        squared_dev = (self - mu) ** 2
        return squared_dev.mean(axis=axis, keepdims=keepdims)

    # ------------------------------------------------------------------
    # Non-linearities
    # ------------------------------------------------------------------

    def relu(self, alpha=0.00):
        op = ReLUOperation(alpha)
        return self._make_output(op, op.forward(self), self.requires_grad, "relu")

    def tanh(self):
        op = TanhOperation()
        return self._make_output(op, op.forward(self), self.requires_grad, "tanh")

    def sigmoid(self):
        op = SigmoidOperation()
        return self._make_output(op, op.forward(self), self.requires_grad, "sigmoid")

    def exp(self):
        op = ExpOperation()
        return self._make_output(op, op.forward(self), self.requires_grad, "exp")

    def log(self):
        op = LogOperation()
        return self._make_output(op, op.forward(self), self.requires_grad, "log")

    def softmax(self, axis=-1):
        op = SoftmaxOperation(axis)
        return self._make_output(op, op.forward(self), self.requires_grad, "softmax")

    def log_softmax(self, axis=-1):
        """Numerically stable ``log(softmax(x))``.

        Composed from existing operations using the *log-sum-exp trick*:

            log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))

        The subtracted maximum is treated as a constant (detached).  That
        is safe because shifting every score by the same amount changes
        neither the softmax value nor its gradient -- but it does keep
        ``exp`` from overflowing.
        """
        shift = Tensor(self.data.max(axis=axis, keepdims=True))  # constant
        shifted = self - shift
        return shifted - shifted.exp().sum(axis=axis, keepdims=True).log()

    # ------------------------------------------------------------------
    # Convolution & pooling
    # ------------------------------------------------------------------

    def conv2d(self, other, stride=1, padding=0):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        op = Conv2DOperationOptim(self, other, stride, padding)
        return self._make_output(op, op.forward(),
                                 self.requires_grad or other.requires_grad, "conv2d")

    def maxpool2d(self, kernel_size, stride, padding=0):
        op = MaxPool2DOperation(kernel_size, stride, padding)
        return self._make_output(op, op.forward(self), self.requires_grad, "maxpool2d")

    # ------------------------------------------------------------------
    # Backpropagation
    # ------------------------------------------------------------------

    def backward(self, grad=None):
        """Run backpropagation from this tensor through the whole graph.

        Typically called on a scalar loss::

            loss.backward()

        After it returns, every tensor with ``requires_grad=True`` that
        contributed to ``loss`` holds ``dloss/dtensor`` in its ``.grad``.

        How it works, in three steps:

        1. Seed the output gradient: ``dloss/dloss = 1``.
        2. Sort the graph so every tensor comes *after* everything it
           depends on (a *topological sort*).
        3. Walk that order in reverse -- from the loss back to the
           inputs -- asking each operation to convert its output gradient
           into input gradients (chain rule), and *accumulating* them
           (a tensor used in several places sums the gradients from all
           of its uses).
        """
        if self.grad is None:
            if grad is not None:
                grad = xp.array(grad, dtype=self.data.dtype)
                assert grad.shape == self.data.shape, (
                    f"backward() gradient shape {grad.shape} must match "
                    f"tensor shape {self.data.shape}")
                self.grad = grad
            else:
                self.grad = xp.ones_like(self.data)

        if not self.requires_grad:
            return

        # -- step 2: topological sort ----------------------------------
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                if v.operation:
                    for tensor in v.operation.inputs():
                        build_topo(tensor)
                topo.append(v)

        build_topo(self)

        # -- step 3: chain rule in reverse order ------------------------
        for v in reversed(topo):
            if v.operation:
                grads = v.operation.backward(v.grad)
                if not isinstance(grads, tuple):
                    grads = (grads,)

                for tensor, tensor_grad in zip(v.operation.inputs(), grads):
                    if tensor.requires_grad:
                        if tensor.grad is None:
                            tensor.grad = tensor_grad
                        else:
                            tensor.grad = tensor.grad + tensor_grad

    # ------------------------------------------------------------------
    # Python protocol methods
    # ------------------------------------------------------------------

    def __repr__(self):
        return f"Tensor({str(self.data)}, requires_grad={self.requires_grad})"

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)
