"""MLX adapter — the Apple-Silicon (Metal) backend, presented as ``xp``.

.. warning::

   **EXPERIMENTAL and, as shipped, UNTESTED on real hardware.**  MLX only
   installs and runs on Apple-Silicon Macs, so this module was written
   against MLX's documented API but could not be executed on the machine it
   was authored on (an Intel Mac).  Expect to fix things the first time you
   run it on an M-series Mac.  The CPU (NumPy) and CUDA (CuPy) backends are
   unaffected -- this module is only imported when you ``set_device("mps")``.

Why an adapter at all?  NumPy and CuPy share one API, so :mod:`backend`
points ``xp`` straight at the library.  MLX (``mlx.core``) is *close* to
NumPy but diverges in exactly the places BabyTorch leans on, so ``xp`` points
at this thin translation layer instead.  It covers only the ~35 ``xp.*``
calls the codebase actually makes (see ``grep -roE 'xp\\.[a-z_.]+' babytorch``).

Known divergences handled here
------------------------------
* **Randomness** — MLX is stateless/key-based (like JAX), not
  ``np.random``-stateful.  :class:`_Random` keeps a global key and splits it
  per draw so ``xp.random.*`` and ``manual_seed`` behave the familiar way.
* **Scatter-add** — MLX has no ``np.add.at``; :func:`scatter_add` uses the
  functional ``array.at[idx].add(...)`` and *returns* the result (the caller
  in :mod:`backend` reassigns it).
* **dtypes** — MLX exposes ``mx.float32`` etc. but not NumPy's dtype
  introspection, so :func:`issubdtype` / :data:`floating` are reimplemented.
* **``.get()``** — MLX arrays convert to host NumPy via ``np.array(arr)``;
  :func:`backend.to_numpy` already special-cases the ``mlx`` module.

Known risks NOT fully solved here (verify on device)
----------------------------------------------------
* **No float64.**  MLX/Metal is float32 (and below) only.  The finite-
  difference gradient checks in ``tests/conftest.py`` build float64 tensors,
  so those tests cannot run on this backend -- keep the numeric checks on the
  CPU.  ``float64`` below is aliased to ``float32`` so code that *names* it
  degrades instead of crashing, but precision differs.
* **Array *methods*.**  Ops call methods on the arrays directly
  (``a.reshape``, ``a.astype``, ``a.T``, ``a.sum(axis=)``, in-place
  ``grad[idx] += ...``).  MLX supports most of these, but any that differ
  must be fixed in ``engine/operations.py`` / ``engine/tensor.py`` -- this
  module cannot shim array methods, only the ``xp.*`` functions.
* **Lazy evaluation.**  MLX defers compute until an ``eval``/host copy.
  Correctness is fine (``.item()`` / ``to_numpy`` force it), but a long
  training loop may want periodic ``mx.eval(...)`` for memory.
"""

import mlx.core as mx


# ---------------------------------------------------------------------------
# dtypes  (MLX/Metal is float32-first: there is no float64 on the GPU)
# ---------------------------------------------------------------------------

float32 = mx.float32
float16 = mx.float16
# No float64 on Metal.  Alias it so code that *names* float64 keeps running
# (at float32 precision) rather than raising -- but see the module warning:
# the float64 gradient checks belong on the CPU backend.
float64 = mx.float32
int32 = mx.int32
int64 = getattr(mx, "int64", mx.int32)          # VERIFY: int64 support varies
bool_ = mx.bool_

inf = float("inf")

_FLOAT_DTYPES = (mx.float16, mx.float32, getattr(mx, "bfloat16", mx.float32))


class _FloatingKind:
    """Sentinel standing in for ``numpy.floating`` in :func:`issubdtype`."""


floating = _FloatingKind()


def issubdtype(dtype, kind):
    """Minimal ``np.issubdtype``: only the ``(dtype, floating)`` check is used."""
    if kind is floating or isinstance(kind, _FloatingKind):
        return dtype in _FLOAT_DTYPES
    return dtype == kind


# ---------------------------------------------------------------------------
# Array creation
# ---------------------------------------------------------------------------

def array(data, dtype=None):
    return mx.array(data, dtype=dtype) if dtype is not None else mx.array(data)


def asarray(data, dtype=None):
    return array(data, dtype=dtype)


def zeros(shape, dtype=float32):
    return mx.zeros(shape, dtype=dtype)


def ones(shape, dtype=float32):
    return mx.ones(shape, dtype=dtype)


def zeros_like(a):
    return mx.zeros_like(a)


def ones_like(a):
    return mx.ones_like(a)


def arange(*args, **kwargs):
    return mx.arange(*args, **kwargs)


def eye(n, *args, **kwargs):
    return mx.eye(n, *args, **kwargs)


def broadcast_to(a, shape):
    return mx.broadcast_to(a, shape)


# ---------------------------------------------------------------------------
# Elementwise / reductions / linear algebra
# ---------------------------------------------------------------------------

def exp(a):
    return mx.exp(a)


def log(a):
    return mx.log(a)


def sqrt(a):
    return mx.sqrt(a)


def tanh(a):
    return mx.tanh(a)


def where(condition, a, b):
    return mx.where(condition, a, b)


def sum(a, axis=None, keepdims=False):
    return mx.sum(a, axis=axis, keepdims=keepdims)


def max(a, axis=None, keepdims=False):
    return mx.max(a, axis=axis, keepdims=keepdims)


def matmul(a, b):
    return mx.matmul(a, b)


def dot(a, b):
    """``np.dot``: inner product for 1-D operands, matmul otherwise."""
    if a.ndim == 1 and b.ndim == 1:
        return mx.sum(a * b)
    return mx.matmul(a, b)


def outer(a, b):
    """``np.outer``: flatten both, then a column times a row."""
    a = mx.reshape(a, (-1, 1))
    b = mx.reshape(b, (1, -1))
    return a * b


# ---------------------------------------------------------------------------
# Shape manipulation
# ---------------------------------------------------------------------------

def reshape(a, shape):
    return mx.reshape(a, shape)


def transpose(a, axes=None):
    return mx.transpose(a) if axes is None else mx.transpose(a, axes)


def swapaxes(a, axis1, axis2):
    return mx.swapaxes(a, axis1, axis2)


def squeeze(a, axis=None):
    return mx.squeeze(a, axis=axis) if axis is not None else mx.squeeze(a)


def expand_dims(a, axis):
    return mx.expand_dims(a, axis)


def repeat(a, repeats, axis=None):
    return mx.repeat(a, repeats, axis=axis)


def concatenate(arrays, axis=0):
    return mx.concatenate(list(arrays), axis=axis)


def pad(a, pad_width, mode="constant", constant_values=0):
    # Only the constant mode is used in the codebase (conv/pool padding).
    # MLX's pad takes the same per-axis (low, high) pad_width.
    return mx.pad(a, pad_width, constant_values=constant_values)  # VERIFY signature


def argsort(a, axis=-1):
    return mx.argsort(a, axis=axis)


def argmax(a, axis=None):
    return mx.argmax(a, axis=axis)


# ---------------------------------------------------------------------------
# Scatter-add  (MLX is functional: this RETURNS the updated array)
# ---------------------------------------------------------------------------

def scatter_add(target, indices, values):
    """``target[indices] += values`` with repeated indices summed.

    MLX arrays are functional, so this returns a *new* array rather than
    mutating ``target`` in place; ``backend.scatter_add`` reassigns it.  The
    ``.at[idx].add`` form is MLX's accumulating scatter (repeated indices are
    summed, which the Embedding backward relies on).
    """
    idx = mx.array(indices)
    return target.at[idx].add(values)          # VERIFY: .at[].add API/semantics


# ---------------------------------------------------------------------------
# Randomness  (key-based; emulate a stateful np.random with a rolling key)
# ---------------------------------------------------------------------------

def _shape(size):
    if size is None:
        return ()
    if isinstance(size, int):
        return (size,)
    return tuple(size)


class _Random:
    """A stateful shim over MLX's key-based RNG, matching ``np.random``."""

    def __init__(self, seed=0):
        self._key = mx.random.key(seed)

    def _next(self):
        self._key, sub = mx.random.split(self._key)
        return sub

    def seed(self, s):
        self._key = mx.random.key(int(s))

    def uniform(self, low=0.0, high=1.0, size=None):
        return mx.random.uniform(low=low, high=high, shape=_shape(size),
                                 key=self._next())

    def random(self, size=None):
        return self.uniform(0.0, 1.0, size)

    def standard_normal(self, size=None):
        return mx.random.normal(shape=_shape(size), key=self._next())

    def normal(self, loc=0.0, scale=1.0, size=None):
        # Build from the standard normal so this works across MLX versions
        # (older ones lack loc/scale kwargs on random.normal).
        return loc + scale * mx.random.normal(shape=_shape(size), key=self._next())

    def randint(self, low, high=None, size=None):
        lo, hi = (0, low) if high is None else (low, high)
        return mx.random.randint(low=lo, high=hi, shape=_shape(size),
                                 key=self._next())


random = _Random()
