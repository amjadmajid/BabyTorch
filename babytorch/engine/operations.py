"""Computation primitives used by :class:`~babytorch.engine.tensor.Tensor`.

Every operation in this file is one *node type* of the computation graph.
An operation knows two things:

* ``forward``  -- how to compute its output from its input arrays;
* ``backward`` -- given ``dLoss/dOutput`` (called ``grad``), how to compute
  ``dLoss/dInput`` for every input.  This is one application of the
  **chain rule**.

The operations work on raw ``xp`` arrays (NumPy on CPU, CuPy on GPU) and
know nothing about the :class:`Tensor` class beyond reading ``.data`` and
``.shape``.  Keeping the *math* here and the *bookkeeping* in ``tensor.py``
is the key design idea of the engine.

A note on broadcasting
----------------------
When shapes differ, the array library silently *broadcasts*: it stretches
size-1 dimensions (and prepends missing ones) so the shapes match, e.g.
``(32, 10) + (1, 10)`` repeats the second operand 32 times.  In the
backward pass this stretching must be undone: every copy that was created
in the forward pass funnels its gradient back into the *one* original
element.  Hence the golden rule implemented by :meth:`Operation.sum_to_shape`:

    **broadcast in the forward pass  <=>  sum in the backward pass**
"""

from ..backend import xp, scatter_add


class Operation:
    """Base class for all differentiable operations.

    Subclasses implement :meth:`forward` to perform the actual computation
    and :meth:`backward` to propagate gradients.  Inputs used during the
    forward pass are stored on the instance (``self.a``, ``self.b``) so
    that the backward pass can reuse them.
    """

    def forward(self):
        """Compute the output of the operation."""
        raise NotImplementedError

    def backward(self, grad):
        """Given the gradient of the loss w.r.t. our *output*, return the
        gradient(s) of the loss w.r.t. our *input(s)* as a tuple."""
        raise NotImplementedError

    def inputs(self):
        """Return the input tensors that took part in the forward pass."""
        attributes = []
        if hasattr(self, 'a'):
            attributes.append(self.a)
        if hasattr(self, 'b'):
            attributes.append(self.b)
        return tuple(attributes)

    @staticmethod
    def sum_to_shape(grad, shape):
        """Undo broadcasting: reduce ``grad`` back to ``shape`` by summing.

        If the forward pass broadcast a tensor of ``shape`` up to
        ``grad.shape``, each original element was copied into several
        output positions.  By the chain rule its gradient is the *sum* of
        the gradients at all those positions.

        Two things may have happened during broadcasting, undone in order:

        1. extra dimensions were prepended  -> sum them away entirely;
        2. size-1 dimensions were stretched -> sum them back to size 1.

        Example: ``bias`` of shape ``(1, 10)`` added to a batch of shape
        ``(32, 10)`` receives a ``(32, 10)`` gradient, which is summed
        over axis 0 back to ``(1, 10)``.
        """
        # 1) sum away prepended dimensions
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)
        # 2) sum stretched dimensions back to size 1
        axes = tuple(i for i, dim in enumerate(shape)
                     if dim == 1 and grad.shape[i] != 1)
        if axes:
            grad = grad.sum(axis=axes, keepdims=True)
        return grad


# ---------------------------------------------------------------------------
# Shape manipulation
# ---------------------------------------------------------------------------

class TransposeOperation(Operation):
    """Permute the axes of a tensor (default: reverse them all)."""

    def __init__(self, axes=None):
        # ``axes`` may arrive as any iterable (range, reversed, ...).  We
        # normalise it to a tuple immediately: an iterator could only be
        # walked once, but we need the axes again in the backward pass.
        self.axes = tuple(axes) if axes is not None else None

    def forward(self, a):
        self.a = a
        return xp.transpose(a.data, self.axes)  # axes=None reverses all axes

    def backward(self, grad):
        """Transposing back: apply the *inverse* permutation to the gradient.

        If axis ``i`` was moved to position ``j`` in the forward pass, the
        gradient's axis ``j`` must move back to position ``i``.  ``argsort``
        of the permutation gives exactly that inverse.
        """
        if self.axes is None:
            return xp.transpose(grad),
        inverse_axes = tuple(int(i) for i in xp.argsort(xp.array(self.axes)))
        return xp.transpose(grad, inverse_axes),


class ReshapeOperation(Operation):
    """Give the same data a new shape (total number of elements unchanged)."""

    def __init__(self, tensor, *new_shape):
        self.a = tensor
        self.new_shape = new_shape

    def forward(self):
        return self.a.data.reshape(*self.new_shape)

    def backward(self, grad):
        # Reshaping moves no data around, so the gradient simply flows
        # through -- reshaped back to the original shape.
        return grad.reshape(self.a.shape),


class SqueezeOperation(Operation):
    """Remove dimensions of size one, e.g. ``(3, 1, 2) -> (3, 2)``."""

    def forward(self, a, axis=None):
        self.a = a
        self.original_shape = a.shape
        return xp.squeeze(a.data, axis=axis)

    def backward(self, grad):
        return xp.reshape(grad, self.original_shape),


class UnsqueezeOperation(Operation):
    """Insert a dimension of size one, e.g. ``(3, 2) -> (3, 1, 2)``."""

    def forward(self, a, axis):
        self.a = a
        self.axis = axis
        return xp.expand_dims(a.data, axis=axis)

    def backward(self, grad):
        return xp.squeeze(grad, axis=self.axis),


class FlattenOperation(Operation):
    """Reshape to ``(batch, everything_else)``, keeping the batch dimension."""

    def forward(self, a):
        self.a = a
        self.original_shape = a.shape
        return a.data.reshape((a.shape[0], -1))

    def backward(self, grad):
        return grad.reshape(self.original_shape),


class SliceOperation(Operation):
    """Take a part of a tensor: ``a[indices]`` (slicing or fancy indexing).

    This one operation powers both ordinary slicing (``x[2:5]``) and
    integer-array lookups such as the Embedding layer (``table[token_ids]``)
    or picking out class probabilities (``probs[range(n), labels]``).
    """

    def __init__(self, indices):
        self.indices = indices

    def forward(self, a):
        self.a = a
        return a.data[self.indices]

    def backward(self, grad):
        """Scatter the gradient back to the positions we read from.

        Positions that were *not* selected get gradient 0.  If the same
        position was selected several times (repeated indices), its
        gradient contributions are *summed* -- that is why we use
        ``scatter_add`` and not plain assignment.
        """
        result = xp.zeros_like(self.a.data)
        # scatter_add returns the accumulated array: in place for NumPy/CuPy,
        # a fresh array for the functional MLX backend.
        result = scatter_add(result, self.indices, grad)
        return result,


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------

class AddOperation(Operation):
    """Element-wise addition ``a + b`` (with broadcasting)."""

    def forward(self, a, b):
        self.a = a
        self.b = b
        return a.data + b.data

    def backward(self, grad):
        # d(a+b)/da = 1 and d(a+b)/db = 1, so the gradient passes through
        # unchanged -- except that broadcasting must be undone.
        a_grad = Operation.sum_to_shape(grad, self.a.shape)
        b_grad = Operation.sum_to_shape(grad, self.b.shape)
        return a_grad, b_grad


class SubOperation(Operation):
    """Element-wise subtraction ``a - b`` (with broadcasting)."""

    def forward(self, a, b):
        self.a = a
        self.b = b
        return a.data - b.data

    def backward(self, grad):
        # d(a-b)/da = 1,  d(a-b)/db = -1
        a_grad = Operation.sum_to_shape(grad, self.a.shape)
        b_grad = Operation.sum_to_shape(-grad, self.b.shape)
        return a_grad, b_grad


class MulOperation(Operation):
    """Element-wise multiplication ``a * b`` (with broadcasting)."""

    def forward(self, a, b):
        self.a = a
        self.b = b
        return a.data * b.data

    def backward(self, grad):
        # Product rule: d(a*b)/da = b  and  d(a*b)/db = a.
        a_grad = Operation.sum_to_shape(grad * self.b.data, self.a.shape)
        b_grad = Operation.sum_to_shape(grad * self.a.data, self.b.shape)
        return a_grad, b_grad


class DivOperation(Operation):
    """Element-wise division ``a / b`` (with broadcasting)."""

    def forward(self, a, b):
        self.a = a
        self.b = b
        return a.data / b.data

    def backward(self, grad):
        # Quotient rule: d(a/b)/da = 1/b,   d(a/b)/db = -a / b^2
        a_grad = Operation.sum_to_shape(grad / self.b.data, self.a.shape)
        b_grad = Operation.sum_to_shape(
            grad * (-self.a.data / (self.b.data ** 2)), self.b.shape)
        return a_grad, b_grad


class PowOperation(Operation):
    """Raise a tensor to a constant power: ``a ** exponent``.

    The exponent is a plain Python number (not a tensor), which covers the
    common cases: squaring (``x**2``), square roots (``x**0.5``), inverses
    (``x**-1``)...
    """

    def __init__(self, exponent):
        self.exponent = exponent

    def forward(self, a):
        self.a = a
        return a.data ** self.exponent

    def backward(self, grad):
        # Power rule: d(a^n)/da = n * a^(n-1)
        n = self.exponent
        return grad * n * self.a.data ** (n - 1),


class MatMulOperation(Operation):
    """Matrix multiplication ``a @ b``.

    Supports vectors, matrices and *batched* matrices (3-D and higher),
    which is what a Transformer uses: an activation of shape
    ``(batch, time, features)`` multiplied by a weight of shape
    ``(features, out_features)``.

    The backward rule for matrices is worth memorising::

        C = A @ B
        dL/dA = dL/dC @ B^T
        dL/dB = A^T @ dL/dC

    (Check the shapes: they only fit together one way!)
    """

    def forward(self, a, b):
        self.a = a
        self.b = b
        return xp.matmul(a.data, b.data)

    def backward(self, grad):
        a, b = self.a.data, self.b.data

        # --- special cases involving 1-D vectors -------------------------
        if a.ndim == 1 and b.ndim == 1:
            # dot product -> grad is a scalar
            return grad * b, grad * a
        if a.ndim == 1 and b.ndim == 2:
            # (k,) @ (k, n) -> (n,)
            return xp.matmul(grad, b.T), xp.outer(a, grad)
        if a.ndim == 2 and b.ndim == 1:
            # (m, k) @ (k,) -> (m,)
            return xp.outer(grad, b), xp.matmul(a.T, grad)
        if a.ndim > 2 and b.ndim == 1:
            # (..., m, k) @ (k,) -> (..., m)
            a_grad = grad[..., :, None] * b            # outer product per row
            b_grad = (a * grad[..., :, None]).sum(axis=tuple(range(a.ndim - 1)))
            return a_grad, b_grad

        # --- general (possibly batched) matrix case ----------------------
        a_grad = xp.matmul(grad, xp.swapaxes(b, -1, -2))
        b_grad = xp.matmul(xp.swapaxes(a, -1, -2), grad)
        # If an operand was broadcast over batch dimensions (e.g. one
        # weight matrix shared by the whole batch), sum those dims away.
        a_grad = Operation.sum_to_shape(a_grad, a.shape)
        b_grad = Operation.sum_to_shape(b_grad, b.shape)
        return a_grad, b_grad


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------

class SumOperation(Operation):
    """Sum all elements, or the elements along the given axis/axes."""

    def forward(self, a, axis=None, keepdims=False):
        self.a = a
        self.axis = axis
        self.keepdims = keepdims
        return xp.sum(a.data, axis=axis, keepdims=keepdims)

    def backward(self, grad):
        """Every element contributed 1-to-1 to the sum, so each receives a
        copy of the gradient: the backward of *sum* is *broadcast*.
        (Compare with :meth:`Operation.sum_to_shape` -- the exact mirror.)
        """
        if self.axis is not None and not self.keepdims:
            # Re-insert the reduced axes as size-1 so broadcasting lines up:
            # e.g. (2,3).sum(axis=1) gave (2,), expand to (2,1) -> (2,3).
            grad = xp.expand_dims(grad, self.axis)
        return xp.broadcast_to(grad, self.a.shape),


class MaxOperation(Operation):
    """Maximum over all elements, or along an axis.

    Only the element(s) that *achieved* the maximum influenced the output,
    so only they receive gradient.  If several elements tie, the gradient
    is split evenly between them (a valid subgradient).
    """

    def forward(self, a, axis=None, keepdims=False):
        self.a = a
        self.axis = axis
        self.keepdims = keepdims
        self.out = xp.max(a.data, axis=axis, keepdims=keepdims)
        return self.out

    def backward(self, grad):
        out = self.out
        if self.axis is not None and not self.keepdims:
            out = xp.expand_dims(out, self.axis)
            grad = xp.expand_dims(grad, self.axis)
        # mask marks where the max was found (possibly several places)
        mask = (self.a.data == out)
        count = mask.sum(axis=self.axis, keepdims=True)
        return mask * grad / count,


# ---------------------------------------------------------------------------
# Element-wise non-linearities
# ---------------------------------------------------------------------------

class ExpOperation(Operation):
    """Element-wise exponential ``e**a``."""

    def forward(self, a):
        self.a = a
        self.out = xp.exp(a.data)
        return self.out

    def backward(self, grad):
        # The derivative of e^x is e^x itself -- reuse the saved output.
        return grad * self.out,


class LogOperation(Operation):
    """Element-wise natural logarithm.

    A tiny ``epsilon`` keeps ``log(0)`` (= -infinity) and ``1/0`` out of
    the computation when probabilities hit exactly zero.
    """

    EPSILON = 1e-8

    def forward(self, a):
        self.a = a
        return xp.log(a.data + self.EPSILON)

    def backward(self, grad):
        # d(log a)/da = 1/a
        return grad / (self.a.data + self.EPSILON),


class ReLUOperation(Operation):
    """Rectified Linear Unit: ``max(0, a)``.

    With ``alpha > 0`` it becomes *leaky* ReLU: negative inputs are scaled
    by ``alpha`` instead of being zeroed, so some gradient always flows.
    """

    def __init__(self, alpha=0.0):
        self.alpha = alpha

    def forward(self, a):
        self.a = a
        return xp.where(a.data > 0, a.data, self.alpha * a.data)

    def backward(self, grad):
        # Slope is 1 where the input was positive, alpha elsewhere.
        return grad * xp.where(self.a.data > 0, 1.0, self.alpha),


class TanhOperation(Operation):
    """Hyperbolic tangent, squashes to (-1, 1)."""

    def forward(self, a):
        self.a = a
        self.out = xp.tanh(a.data)
        return self.out

    def backward(self, grad):
        # d(tanh a)/da = 1 - tanh(a)^2
        return grad * (1.0 - self.out ** 2),


class SigmoidOperation(Operation):
    """Logistic sigmoid ``1 / (1 + e^-a)``, squashes to (0, 1)."""

    def forward(self, a):
        self.a = a
        self.out = 1 / (1 + xp.exp(-a.data))
        return self.out

    def backward(self, grad):
        # d(sigma)/da = sigma * (1 - sigma)
        return grad * (self.out * (1 - self.out)),


class SoftmaxOperation(Operation):
    """Softmax along an axis: turns raw scores into probabilities.

    ``softmax(x)_i = exp(x_i) / sum_j exp(x_j)``

    Forward: we first subtract the row maximum.  This changes nothing
    mathematically (it cancels in the fraction) but keeps ``exp`` from
    overflowing -- the classic *max trick*.

    Backward: with ``s = softmax(x)`` the Jacobian is
    ``ds_i/dx_j = s_i * (delta_ij - s_j)``, which contracted against the
    incoming gradient ``g`` collapses to the tidy expression::

        dx = s * (g - sum(g * s))        (sums along the softmax axis)

    Intuition: probabilities must keep summing to 1, so pushing one up
    pushes all the others down -- the second term carries that coupling.
    """

    def __init__(self, axis=-1):
        self.axis = axis

    def forward(self, a):
        self.a = a
        shifted = a.data - xp.max(a.data, axis=self.axis, keepdims=True)
        exps = xp.exp(shifted)
        self.out = exps / xp.sum(exps, axis=self.axis, keepdims=True)
        return self.out

    def backward(self, grad):
        s = self.out
        inner = xp.sum(grad * s, axis=self.axis, keepdims=True)
        return s * (grad - inner),


# ---------------------------------------------------------------------------
# Convolution & pooling (the vision detour)
# ---------------------------------------------------------------------------

class Conv2DOperation(Operation):
    """Naive 2D convolution, written with explicit loops.

    Kept for study purposes: it shows exactly what a convolution does,
    one output pixel at a time.  Use :class:`Conv2DOperationOptim` for
    real training -- it computes the same thing much faster.
    """

    def __init__(self, a, w, stride, padding):
        self.a = a  # input images   (N, C, H, W)
        self.w = w  # filter weights (F, C, K, K)
        self.stride = stride
        self.padding = padding

    def inputs(self):
        return (self.a, self.w)

    def forward(self):
        N, C, H, W = self.a.data.shape
        F, C, K, K = self.w.data.shape

        H_out = (H + 2 * self.padding - K) // self.stride + 1
        W_out = (W + 2 * self.padding - K) // self.stride + 1
        output = xp.zeros((N, F, H_out, W_out))

        if self.padding > 0:
            padded_data = xp.pad(self.a.data,
                                 ((0, 0), (0, 0),
                                  (self.padding, self.padding),
                                  (self.padding, self.padding)), 'constant')
        else:
            padded_data = self.a.data

        # Slide each filter over each image; every stop produces one number:
        # the dot product between the filter and the patch under it.
        for n in range(N):              # each image in the batch
            for f in range(F):          # each filter
                for i in range(H_out):  # each vertical stop
                    for j in range(W_out):  # each horizontal stop
                        i0, j0 = i * self.stride, j * self.stride
                        patch = padded_data[n, :, i0:i0 + K, j0:j0 + K]
                        output[n, f, i, j] = xp.sum(patch * self.w.data[f])
        return output

    def backward(self, grad_output):
        N, C, H, W = self.a.data.shape
        F, _, K, K = self.w.data.shape
        H_out, W_out = grad_output.shape[-2], grad_output.shape[-1]

        grad_input = xp.zeros((N, C, H + 2 * self.padding, W + 2 * self.padding))
        grad_kernel = xp.zeros_like(self.w.data)
        padded = xp.pad(self.a.data,
                        ((0, 0), (0, 0),
                         (self.padding, self.padding),
                         (self.padding, self.padding)), 'constant') \
            if self.padding > 0 else self.a.data

        # Each output pixel got its value from one K x K patch, so its
        # gradient flows back into that same patch (for the input) and
        # into the filter (scaled by the patch).
        for n in range(N):
            for f in range(F):
                for i in range(H_out):
                    for j in range(W_out):
                        i0, j0 = i * self.stride, j * self.stride
                        g = grad_output[n, f, i, j]
                        grad_input[n, :, i0:i0 + K, j0:j0 + K] += g * self.w.data[f]
                        grad_kernel[f] += g * padded[n, :, i0:i0 + K, j0:j0 + K]

        if self.padding > 0:
            grad_input = grad_input[:, :, self.padding:-self.padding,
                                    self.padding:-self.padding]
        return grad_input, grad_kernel


class Conv2DOperationOptim(Operation):
    """Fast 2D convolution using the *im2col* trick.

    Idea: copy every K x K patch of the input into one row of a big
    matrix (``im2col``).  Convolution then becomes a single matrix
    multiplication with the flattened filters -- and matmul is the one
    thing array libraries are extremely good at.  ``col2im`` scatters
    patch-gradients back to image positions (overlaps add up).
    """

    def __init__(self, a, w, stride, padding):
        self.a = a
        self.w = w
        self.stride = stride
        self.padding = padding

    def inputs(self):
        return (self.a, self.w)

    def _im2col(self, x, K, stride):
        N, C, H, W = x.shape
        H_out = (H - K) // stride + 1
        W_out = (W - K) // stride + 1

        x_col = xp.zeros((N, C, K, K, H_out, W_out), dtype=x.dtype)
        for i in range(K):
            for j in range(K):
                x_col[:, :, i, j, :, :] = \
                    x[:, :, i:i + H_out * stride:stride, j:j + W_out * stride:stride]

        return x_col.transpose(0, 4, 5, 1, 2, 3).reshape(N * H_out * W_out, -1)

    def _col2im(self, col, x_shape, K, stride):
        N, C, H, W = x_shape
        H_padded, W_padded = H + 2 * self.padding, W + 2 * self.padding
        H_out = (H_padded - K) // stride + 1
        W_out = (W_padded - K) // stride + 1
        col = col.reshape(N, H_out, W_out, C, K, K).transpose(0, 3, 4, 5, 1, 2)

        img = xp.zeros((N, C, H_padded, W_padded), dtype=col.dtype)
        for i in range(K):
            for j in range(K):
                img[:, :, i:i + H_out * stride:stride,
                    j:j + W_out * stride:stride] += col[:, :, i, j, :, :]

        if self.padding > 0:
            img = img[:, :, self.padding:-self.padding, self.padding:-self.padding]
        return img

    def forward(self):
        padded = xp.pad(self.a.data,
                        ((0, 0), (0, 0),
                         (self.padding, self.padding),
                         (self.padding, self.padding)), 'constant')
        K = self.w.data.shape[2]
        x_col = self._im2col(padded, K, self.stride)          # (N*H_out*W_out, C*K*K)
        w_col = self.w.data.reshape(self.w.data.shape[0], -1)  # (F, C*K*K)
        out = xp.dot(x_col, w_col.T)                           # one big matmul!

        N, C, H, W = self.a.data.shape
        F = self.w.data.shape[0]
        H_out = (H + 2 * self.padding - K) // self.stride + 1
        W_out = (W + 2 * self.padding - K) // self.stride + 1
        return out.reshape(N, H_out, W_out, F).transpose(0, 3, 1, 2)

    def backward(self, grad_output):
        F, _, K, _ = self.w.data.shape
        # (N, F, H_out, W_out) -> (N*H_out*W_out, F) to match im2col layout
        grad_output_col = grad_output.transpose(0, 2, 3, 1).reshape(-1, F)
        w_col = self.w.data.reshape(F, -1)

        grad_col = xp.dot(grad_output_col, w_col)
        grad_input = self._col2im(grad_col, self.a.data.shape, K, self.stride)

        padded = xp.pad(self.a.data,
                        ((0, 0), (0, 0),
                         (self.padding, self.padding),
                         (self.padding, self.padding)), 'constant')
        x_col = self._im2col(padded, K, self.stride)
        grad_kernel = xp.dot(grad_output_col.T, x_col) \
            .reshape(F, self.a.data.shape[1], K, K)

        return grad_input, grad_kernel


class MaxPool2DOperation(Operation):
    """2D max pooling: keep only the largest value in each window.

    In the backward pass the gradient flows *only* to the element that
    won the max in each window -- all others contributed nothing to the
    output, so their gradient is zero.
    """

    def __init__(self, kernel_size, stride, padding=0):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, a):
        self.a = a
        x = a.data
        if self.padding > 0:
            # Pad with -inf so padding can never win the max.
            x = xp.pad(x, ((0, 0), (0, 0),
                           (self.padding, self.padding),
                           (self.padding, self.padding)),
                       'constant', constant_values=-xp.inf)
        self.padded_shape = x.shape
        N, C, H, W = x.shape
        K = self.kernel_size

        H_out = (H - K) // self.stride + 1
        W_out = (W - K) // self.stride + 1

        output = xp.zeros((N, C, H_out, W_out), dtype=x.dtype)
        # For each window remember *where* the max was (flat index in the
        # window) so the backward pass can route the gradient there.
        self.indices = xp.zeros((N, C, H_out, W_out), dtype=xp.int64)

        for i in range(H_out):
            for j in range(W_out):
                i0, j0 = i * self.stride, j * self.stride
                patch = x[:, :, i0:i0 + K, j0:j0 + K]
                output[:, :, i, j] = xp.max(patch, axis=(2, 3))
                self.indices[:, :, i, j] = xp.argmax(patch.reshape(N, C, -1), axis=2)
        return output

    def backward(self, grad):
        N, C, H, W = self.padded_shape
        K = self.kernel_size
        H_out, W_out = grad.shape[-2], grad.shape[-1]

        grad_padded = xp.zeros(self.padded_shape, dtype=grad.dtype)

        for i in range(H_out):
            for j in range(W_out):
                i0, j0 = i * self.stride, j * self.stride
                # Build a (N, C, K, K) patch that is zero everywhere except
                # at the position that won the max.
                patch_grad = xp.zeros((N, C, K * K), dtype=grad.dtype)
                flat = xp.arange(N * C) * (K * K) + self.indices[:, :, i, j].ravel()
                patch_grad.ravel()[flat] = grad[:, :, i, j].ravel()
                grad_padded[:, :, i0:i0 + K, j0:j0 + K] += \
                    patch_grad.reshape(N, C, K, K)

        if self.padding > 0:
            return grad_padded[:, :, self.padding:-self.padding,
                               self.padding:-self.padding],
        return grad_padded,


class UpsampleOperation(Operation):
    """Nearest-neighbour 2D upsampling: copy each pixel into a ``scale x
    scale`` block, turning ``(N, C, H, W)`` into ``(N, C, H*scale, W*scale)``.

    It is the mirror image of pooling.  Max-pool *shrinks* an image by
    keeping one value per window; this *grows* it by copying one value into
    a whole window.  A U-Net's decoder uses it to climb back up to full
    resolution after the encoder has shrunk the picture down.

    Backward: each input pixel was copied into ``scale**2`` output
    positions, so by the chain rule its gradient is the *sum* of the
    gradients over that block -- a sum-pool, the exact adjoint of a copy.
    """

    def __init__(self, scale):
        self.scale = scale

    def forward(self, a):
        self.a = a
        s = self.scale
        # Repeat along height, then width, so each pixel becomes an s x s
        # block: out[.., h*s + i, w*s + j] = in[.., h, w] for all i, j < s.
        return xp.repeat(xp.repeat(a.data, s, axis=2), s, axis=3)

    def backward(self, grad):
        s = self.scale
        N, C, H, W = self.a.shape
        # Regroup each s x s output block (H*s -> (H, s), W*s -> (W, s)) and
        # sum it back onto the single source pixel it was copied from.
        return grad.reshape(N, C, H, s, W, s).sum(axis=(3, 5)),


# Backwards-compatible alias (the original class name had a typo).
RLeUOperation = ReLUOperation
