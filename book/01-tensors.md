# Chapter 1 — Tensors

*Part I, chapter 1 of 4. Everything in deep learning — images, sentences,
model weights, gradients — is stored in one data structure. This chapter
is about that structure.*

## One idea: a block of numbers with a shape

A **tensor** is an n-dimensional array: a block of numbers plus a
**shape** that says how the block is organized.

![Tensors are blocks of numbers with a shape — scalar, vector, matrix, 3-D — and broadcasting stretches a size-1 tensor to serve a whole batch](figures/fig-tensors.svg)

Why does deep learning insist on this structure instead of Python lists?
Two reasons:

1. **Meaning lives in the shape.** A batch of 32 sentences, each 128
   tokens long, each token a 192-number vector, is a tensor of shape
   `(32, 128, 192)`. The shape *is* the bookkeeping.
2. **Speed lives in the block.** Because the numbers sit in one
   contiguous block of memory, whole-tensor operations run as tight
   compiled loops (or GPU kernels) instead of interpreted Python. The
   golden rule of array programming: *never loop over elements in
   Python; say what you want done to the whole block.*

## Creating tensors

BabyTorch mirrors PyTorch's factory functions
([`babytorch/__init__.py`](../babytorch/__init__.py)):

```python
import babytorch

a = babytorch.tensor([[1., 2., 3.], [4., 5., 6.]])  # from data
z = babytorch.zeros(2, 3)         # all zeros
o = babytorch.ones(2, 3)          # all ones
r = babytorch.randn(2, 3)         # random, normal distribution (mean 0, std 1)
u = babytorch.rand(2, 3)          # random, uniform in [0, 1)
n = babytorch.arange(0, 10, 2)    # 0, 2, 4, 6, 8

babytorch.manual_seed(42)         # make the random ones reproducible
```

Elements are `float32` by default, exactly like PyTorch — deep learning
rarely needs more precision, and smaller numbers mean faster math.

**Try it**

```python
>>> import babytorch
>>> t = babytorch.tensor([[1., 2., 3.], [4., 5., 6.]])
>>> t.shape
(2, 3)
>>> t.ndim        # how many dimensions
2
>>> t.size        # how many numbers in total
6
>>> t.sum().item()   # .item() unwraps a single-element tensor to a Python number
21.0
```

## Reshaping: same numbers, different organization

The block of numbers can be reorganized without copying anything:

```python
x = babytorch.arange(6)      # shape (6,):    [0 1 2 3 4 5]
x.reshape(2, 3)              # shape (2, 3):  [[0 1 2] [3 4 5]]
x.reshape(3, 2)              # shape (3, 2):  [[0 1] [2 3] [4 5]]

m = babytorch.randn(2, 3)
m.T                          # transpose: shape (3, 2), rows <-> columns
m.unsqueeze(0)               # insert a size-1 axis:  (2, 3) -> (1, 2, 3)
m.unsqueeze(0).squeeze()     # remove size-1 axes:    (1, 2, 3) -> (2, 3)
```

These look cosmetic now, but Part II leans on them constantly: chapter 6
splits a `(B, T, C)` tensor into attention heads with nothing but
`reshape` and `transpose`.

## Math on whole tensors

Arithmetic applies **element-wise**, one aligned pair at a time:

```python
a = babytorch.tensor([1., 2., 3.])
b = babytorch.tensor([10., 20., 30.])
(a + b).data      # [11. 22. 33.]
(a * b).data      # [10. 40. 90.]  (element-wise, NOT matrix product)
(a ** 2).data     # [1. 4. 9.]
```

The `@` operator is real **matrix multiplication** — rows dotted with
columns. It is the single most important operation in deep learning; a
GPT spends nearly all of its time inside `@`:

```
        (2, 3)      @      (3, 4)      ->      (2, 4)
     [[. . .]           [[. . . .]           [[. . . .]
      [. . .]]           [. . . .]            [. . . .]]
                         [. . . .]]
          └── inner dimensions must match ──┘
              (3 columns) dot (3 rows)
```

```python
a = babytorch.randn(2, 3)
w = babytorch.randn(3, 4)
(a @ w).shape     # (2, 4)
```

Reductions collapse dimensions: `t.sum()`, `t.mean()`, `t.max()`,
`t.var()` — over everything, or along one axis
(`t.sum(axis=0)` sums the rows away).

## Broadcasting: small tensors stretch to fit

When shapes differ, the array library **broadcasts**: it (virtually)
copies size-1 dimensions until the shapes match.

```
      x: (32, 10)      +      b: (1, 10)

      [[..........]           [[----------]     <- the same row,
       [..........]    +       [----------]        virtually repeated
           ...                     ...             32 times
       [..........]]           [----------]]
```

```python
x = babytorch.randn(32, 10)   # a batch of 32 examples
b = babytorch.ones(1, 10)     # ONE bias row
y = x + b                     # b is stretched across all 32 rows
y.shape                       # (32, 10)
```

This is how one bias vector serves a whole batch, and how one weight
matrix multiplies every sentence in a batch at once. The rule, aligning
shapes from the right: two dimensions are compatible when they are equal,
or one of them is 1 (missing leading dimensions count as 1).

Remember broadcasting — it returns in chapter 2 with a twist: whatever
was *copied* on the way forward must be *summed* on the way back.

## The same code on CPU and GPU

BabyTorch never says `numpy` directly. Every module imports the array
library under the neutral name `xp` from
[`babytorch/backend.py`](../babytorch/backend.py):

```python
from babytorch.backend import xp    # numpy on CPU, cupy on GPU

xp.zeros((2, 3))                    # works identically on both
```

`backend.py` picks the library — **CuPy** if an NVIDIA GPU is
available, otherwise **NumPy** — and `xp` is a tiny proxy that forwards
every call to the current choice. CuPy is a deliberate drop-in clone of
NumPy's API, so this single choice is the entire GPU story — there is no
other GPU-specific code in the framework. Control it from code, or from
the environment:

```python
>>> import babytorch
>>> babytorch.set_device("cpu")    # "cpu", "cuda", or "auto"
'cpu'
>>> babytorch.device()
'cpu'
>>> t.numpy()    # copy back to a NumPy array on the CPU (for plotting, saving...)
```

```bash
BABYTORCH_DEVICE=cpu  python train.py    # initial device via the environment
BABYTORCH_DEVICE=cuda python train.py    # require the GPU
```

One rule: pick the device *before* building tensors or models — arrays
do not migrate to the new library after they are created. (On macOS
there is no CUDA, so everything runs on the CPU — which works out of
the box.)

<details>
<summary><b>How it's implemented</b> — <code>babytorch/backend.py</code></summary>

```python
class _XP:
    """Proxy that forwards every attribute to the active array library.

    Because modules bind ``xp`` once (``from .backend import xp``) but
    every *use* is an attribute access (``xp.zeros``), routing the
    lookup through ``__getattr__`` lets :func:`set_device` swap the
    library underneath all of them at once.  For MLX the "library" is the
    :mod:`babytorch.mlx_backend` adapter module rather than MLX itself.
    """

    _lib = None  # numpy, cupy, or the mlx_backend adapter; set by set_device()

    def __getattr__(self, name):
        return getattr(_XP._lib, name)
```

```python
def set_device(name):
    """Select the array library: ``"cpu"``, ``"cuda"`` (or ``"gpu"``),
    ``"mps"`` (or ``"mlx"``), or ``"auto"``.  Returns the name of the device
    actually selected.

    Call it *before* creating tensors or models (see the module docstring
    for why).  ``"cuda"`` and ``"mps"`` raise with an explanation if their
    GPU stack is not present; ``"auto"`` never raises.
    """
    global DEVICE
    name = name.lower()
    if name in ("cuda", "gpu"):
        _XP._lib = _cuda_library()
        DEVICE = "cuda"
    elif name in ("mps", "mlx"):
        _XP._lib = _mlx_library()
        DEVICE = "mps"
    elif name == "cpu":
        import numpy
        _XP._lib = numpy
        DEVICE = "cpu"
    elif name == "auto":
        try:
            return set_device("cuda")
        except Exception:
            return set_device("cpu")
    else:
        raise ValueError(
            f"unknown device {name!r}: expected 'cpu', 'cuda', 'mps' or 'auto'")
    return DEVICE
```

</details>

## A tensor that remembers where it came from

So far, nothing here needed a framework — plain NumPy does all of it.
Here is the actual beginning of BabyTorch's `Tensor`
([`babytorch/engine/tensor.py`](../babytorch/engine/tensor.py)):

```python
self.data = xp.asarray(data, dtype=dtype)   # the block of numbers
self.requires_grad = requires_grad          # should gradients flow here?
self.grad = None                            # filled in by backward()
self.operation = None                       # the Operation that produced this
```

<details>
<summary><b>How it's implemented</b> — <code>babytorch/engine/tensor.py</code></summary>

```python
class Tensor:
    """A minimal tensor object that records operations for autodiff."""

    def __init__(self, data, requires_grad=False, dtype=None,
                 label="", _op_label=""):
    # ...
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
    # ...
    @property
    def shape(self):
        return self.data.shape
    # ...
    def item(self):
        """Return the value of a single-element tensor as a Python number."""
        return self.data.item()

    def numpy(self):
        """Return the data as a NumPy array on the CPU (copies from GPU)."""
        return to_numpy(self.data)
```

</details>

Three extra fields. `data` we have covered; the other three exist for
one purpose: **every tensor remembers which operation created it, and
that operation remembers its input tensors.** Follow those links
backwards and you recover the entire history of a computation — every
`+`, `@` and `reshape` that led from the inputs to the result.

That recorded history is called the *computation graph*, and it is the
whole secret of deep learning frameworks: a network can only *learn* if
we can compute how the loss changes when each weight moves, and the
graph lets the framework work that out automatically — backwards, one
recorded step at a time.

That is chapter 2.

## Exercises

**Check yourself** (a minute each — answers unfold):

**Q1.** What is the shape of `babytorch.randn(3, 1) * babytorch.randn(4)`?

<details><summary>Answer</summary>

`(3, 4)`. Aligning shapes from the right, `(3, 1)` meets `(4,)` — the
missing dimension counts as 1, so both stretch: the column of 3 repeats
across 4 columns, the row of 4 repeats across 3 rows.

</details>

**Q2.** `(32, 10) + (10,)` broadcasts happily, but `(32, 10) + (32,)`
raises an error. Why?

<details><summary>Answer</summary>

Alignment is from the **right**: `(32,)` lines up against the size-10
last dimension, and 32 ≠ 10 with neither being 1. (To add one number
per *row* you'd reshape to `(32, 1)`.)

</details>

**Q3.** Why is `float32` the default element type instead of the more
precise `float64`?

<details><summary>Answer</summary>

Half the memory and twice the throughput, and gradient descent is
noisy anyway — batch noise dwarfs the extra rounding error, so the
precision would buy nothing. (Frontier models go *lower* still:
16-bit and below.)

</details>

**Build it** — the deeper track, graded by tests: implement
`standardize` and ★ `outer` (broadcasting only, no `@`) in
[`exercises/ch01_tensors.py`](exercises/ch01_tensors.py), then run
`pytest book/exercises/test_ch01_tensors.py -v`.
([How the exercises work](exercises/README.md).)

---

**Source files for this chapter:**
[`babytorch/__init__.py`](../babytorch/__init__.py) (factories) ·
[`babytorch/backend.py`](../babytorch/backend.py) (CPU/GPU selection) ·
[`babytorch/engine/tensor.py`](../babytorch/engine/tensor.py) (the Tensor class)

[Contents](README.md) | [Chapter 2: Autograd →](02-autograd.md)
