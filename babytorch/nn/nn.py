"""Neural network building blocks.

A network is assembled from :class:`Module` objects.  A module is anything
that (1) transforms tensors in its ``forward`` method and (2) may own
*parameters* -- tensors with ``requires_grad=True`` that the optimizer
will update.

The base :class:`Module` class walks an instance's attributes to find
parameters automatically, so a new layer only needs to store its weights
as attributes and implement ``forward`` -- exactly like PyTorch.
"""

import math
import pickle

from ..backend import xp, to_numpy
from ..engine import Tensor


class Module:
    """Base class for all neural network layers and models."""

    # Toggled by train()/eval().  Layers that behave differently during
    # training (e.g. Dropout) check this flag.
    training = True

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def parameters(self):
        """Collect every trainable tensor in this module, recursively.

        We look through the instance's attributes for:
        * tensors with ``requires_grad=True``  -> parameters of this module;
        * sub-modules                          -> ask them for theirs;
        * lists/tuples of sub-modules          -> same (e.g. Sequential,
          or the list of blocks in a Transformer).
        """
        params = []
        for value in vars(self).values():
            if isinstance(value, Tensor):
                if value.requires_grad:
                    params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
                    elif isinstance(item, Tensor) and item.requires_grad:
                        params.append(item)
        return params

    def named_parameters(self, prefix=""):
        """Yield ``(dotted_name, parameter)`` pairs, e.g. ``("layers.0.w", ...)``."""
        for name, value in vars(self).items():
            if isinstance(value, Tensor) and value.requires_grad:
                yield prefix + name, value
            elif isinstance(value, Module):
                yield from value.named_parameters(f"{prefix}{name}.")
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    if isinstance(item, Module):
                        yield from item.named_parameters(f"{prefix}{name}.{i}.")

    def modules(self):
        """Yield this module and every sub-module, recursively."""
        yield self
        for value in vars(self).values():
            if isinstance(value, Module):
                yield from value.modules()
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        yield from item.modules()

    def train(self, mode=True):
        """Put the model in training mode (Dropout active, etc.)."""
        for m in self.modules():
            m.training = mode
        return self

    def eval(self):
        """Put the model in evaluation mode (Dropout off, etc.)."""
        return self.train(False)

    def zero_grad(self):
        """Reset the gradients of all parameters.

        Call this after each optimizer step: ``backward()`` *accumulates*
        gradients, so leftovers from the previous batch would otherwise
        contaminate the next one.
        """
        for p in self.parameters():
            p.grad = None

    def num_parameters(self):
        """Total number of trainable scalars in the model."""
        return sum(p.data.size for p in self.parameters())

    def save(self, filename):
        """Save all parameters to ``filename``.

        Arrays are converted to NumPy first so a model trained on the GPU
        can still be loaded on a CPU-only machine.
        """
        with open(filename, 'wb') as f:
            states = [{'data': to_numpy(p.data), 'grad': None}
                      for p in self.parameters()]
            pickle.dump(states, f)

    @staticmethod
    def load(filename, model):
        """Load parameters saved by :meth:`save` into ``model``.

        The model must be constructed with the same architecture, since
        parameters are matched by position.
        """
        with open(filename, 'rb') as f:
            states = pickle.load(f)
        params = model.parameters()
        assert len(params) == len(states), (
            f"Checkpoint has {len(states)} parameter tensors but the model "
            f"has {len(params)} -- did the architecture change?")
        for p, state in zip(params, states):
            p.set_state(state)
        return model


# ---------------------------------------------------------------------------
# Activation layers
# ---------------------------------------------------------------------------

class ReLU(Module):
    """max(0, x): keep positives, zero out negatives."""

    def forward(self, x):
        return x.relu()

    def __repr__(self):
        return "ReLU"


class Tanh(Module):
    """Squash values into (-1, 1)."""

    def forward(self, x):
        return x.tanh()

    def __repr__(self):
        return "Tanh"


class Sigmoid(Module):
    """Squash values into (0, 1)."""

    def forward(self, x):
        return x.sigmoid()

    def __repr__(self):
        return "Sigmoid"


class GELU(Module):
    """Gaussian Error Linear Unit -- the activation used in Transformers.

    A smooth cousin of ReLU: instead of a hard cut at zero, inputs are
    scaled by (approximately) the probability that a standard normal
    variable is below them.  We use the common ``tanh`` approximation
    (the same one GPT-2 uses)::

        gelu(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 x^3) ))
    """

    def forward(self, x):
        c = math.sqrt(2.0 / math.pi)
        return 0.5 * x * (1.0 + ((x + 0.044715 * x ** 3) * c).tanh())

    def __repr__(self):
        return "GELU"


class Softmax(Module):
    """Turn raw scores into probabilities along ``axis``."""

    def __init__(self, axis=-1):
        self.axis = axis

    def forward(self, x):
        return x.softmax(axis=self.axis)

    def __repr__(self):
        return f"Softmax(axis={self.axis})"


# ---------------------------------------------------------------------------
# Layers with parameters
# ---------------------------------------------------------------------------

class Linear(Module):
    """Fully connected layer: ``y = x @ W + b``.

    Weights start as small random numbers drawn from
    ``U(-k, k) with k = 1/sqrt(in_features)`` (PyTorch's default).  The
    scaling keeps the output variance roughly equal to the input variance
    no matter how wide the layer is, which is what lets deep stacks of
    layers train at all.
    """

    def __init__(self, in_features, out_features, activation_function=None):
        k = 1.0 / math.sqrt(in_features)
        self.w = Tensor(xp.random.uniform(-k, k, (in_features, out_features)),
                        requires_grad=True)
        self.b = Tensor(xp.random.uniform(-k, k, (1, out_features)),
                        requires_grad=True)
        self.activation_function = activation_function

    def forward(self, x):
        out = x @ self.w + self.b
        if self.activation_function:
            out = self.activation_function(out)
        return out

    def __repr__(self):
        activation_str = f", activation={self.activation_function}" \
            if self.activation_function else ""
        return (f"Linear(in_features={self.w.shape[0]}, "
                f"out_features={self.w.shape[1]}{activation_str})")


class Embedding(Module):
    """A lookup table that maps integer ids to learned vectors.

    This is how token ids enter a language model: row ``i`` of ``weight``
    *is* the vector for token ``i``.  The forward pass is just indexing;
    the backward pass adds each output gradient back onto the row it was
    read from (rows used several times accumulate several gradients).
    """

    def __init__(self, num_embeddings, embedding_dim):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Small normal init, as used by GPT-2.
        self.weight = Tensor(
            xp.random.normal(0.0, 0.02, (num_embeddings, embedding_dim)),
            requires_grad=True)

    def forward(self, idx):
        """idx: integer array of any shape -> float tensor of shape ``idx.shape + (embedding_dim,)``."""
        if isinstance(idx, Tensor):
            idx = idx.data
        idx = xp.asarray(idx).astype(xp.int64)
        return self.weight[idx]

    def __repr__(self):
        return f"Embedding({self.num_embeddings}, {self.embedding_dim})"


class LayerNorm(Module):
    """Normalize each vector to zero mean and unit variance, then rescale.

    Applied over the *last* dimension (the feature dimension).  Two small
    learned parameters let the network undo the normalization where useful:
    ``gamma`` (scale, starts at 1) and ``beta`` (shift, starts at 0).

    Transformers place one of these before every attention and MLP block;
    without them, activations drift in scale as depth grows and training
    becomes unstable.
    """

    def __init__(self, num_features, eps=1e-5):
        self.gamma = Tensor(xp.ones(num_features), requires_grad=True)
        self.beta = Tensor(xp.zeros(num_features), requires_grad=True)
        self.eps = eps

    def forward(self, x):
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_hat = (x - mu) / (var + self.eps).sqrt()
        return x_hat * self.gamma + self.beta

    def __repr__(self):
        return f"LayerNorm({self.gamma.shape[0]})"


class Dropout(Module):
    """Randomly zero out a fraction ``p`` of the values during training.

    Dropout fights overfitting: because any value can vanish at any time,
    the network cannot rely on a single path and must spread knowledge out.

    We use *inverted* dropout: surviving values are scaled up by
    ``1/(1-p)`` during training, so at evaluation time the layer can
    simply pass values through unchanged.
    """

    def __init__(self, p=0.1):
        assert 0.0 <= p < 1.0, "dropout probability must be in [0, 1)"
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        mask = (xp.random.random(x.shape) < keep).astype(x.data.dtype) / keep
        return x * Tensor(mask)

    def __repr__(self):
        return f"Dropout(p={self.p})"


class Conv2D(Module):
    """2D convolution layer: slide learned filters over an image.

    Input  shape: ``(batch, in_channels, height, width)``
    Output shape: ``(batch, out_channels, out_height, out_width)``
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        k = 1.0 / math.sqrt(in_channels * kernel_size * kernel_size)
        self.w = Tensor(
            xp.random.uniform(-k, k, (out_channels, in_channels,
                                      kernel_size, kernel_size)),
            requires_grad=True)
        self.b = Tensor(xp.zeros(out_channels), requires_grad=True)

    def forward(self, x):
        out = x.conv2d(self.w, self.stride, self.padding)
        # Reshape the bias to (1, out_channels, 1, 1) so broadcasting adds
        # one bias value per output channel.  Doing this with tensor ops
        # (not raw arrays) keeps the bias inside the computation graph.
        return out + self.b.reshape(1, -1, 1, 1)

    def __repr__(self):
        return (f"Conv2D(in_channels={self.in_channels}, "
                f"out_channels={self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, "
                f"padding={self.padding})")


class MaxPool2D(Module):
    """Downsample an image by keeping the max of each window."""

    def __init__(self, kernel_size, stride=None, padding=0):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x):
        return x.maxpool2d(self.kernel_size, self.stride, self.padding)

    def __repr__(self):
        return (f"MaxPool2D(kernel_size={self.kernel_size}, "
                f"stride={self.stride}, padding={self.padding})")


class Flatten(Module):
    """Flatten everything except the batch dimension."""

    def forward(self, x):
        return x.flatten()

    def __repr__(self):
        return "Flatten"


class Sequential(Module):
    """Chain layers: the output of each is the input of the next."""

    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __repr__(self):
        return f"Sequential({', '.join(str(layer) for layer in self.layers)})"
