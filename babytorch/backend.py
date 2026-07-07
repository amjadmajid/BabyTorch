"""Array backend selection for BabyTorch.

BabyTorch runs on top of an *array library* that does the actual number
crunching.  Two libraries share (almost exactly) the same API:

* **NumPy** -- runs on the CPU, installed everywhere (Linux, macOS,
  Windows).
* **CuPy**  -- runs on an NVIDIA GPU, drop-in replacement for NumPy.

Every other BabyTorch module imports the active library from here under
the neutral name ``xp`` (a common convention meaning "numpy-or-cupy"):

    from babytorch.backend import xp

    xp.zeros((2, 3))        # works on CPU *and* GPU

``xp`` is a tiny *proxy* object rather than the library itself, so the
active library can be swapped at runtime -- that is the whole trick
behind :func:`set_device`.

Choosing a device
-----------------
Two ways, same effect:

1. **In code** (a plain function call, e.g. from a notebook)::

       import babytorch
       babytorch.set_device("cpu")     # always NumPy
       babytorch.set_device("cuda")    # require an NVIDIA GPU
       babytorch.set_device("auto")    # GPU if available, else CPU

2. **From the environment**, which sets the *initial* device before any
   code runs::

       BABYTORCH_DEVICE=cpu  python train.py
       BABYTORCH_DEVICE=cuda python train.py

The default is ``auto``: use the GPU if CuPy *and* a CUDA device are
available, otherwise fall back to the CPU.

One rule of thumb: **pick the device before building tensors or
models.**  Arrays do not migrate when the device changes -- a model
created on the GPU keeps its CuPy arrays even after
``set_device("cpu")``, and mixing the two libraries in one expression
fails.  (Real frameworks move data per-tensor with ``.to(device)``;
BabyTorch keeps one global choice so the entire GPU story fits in this
one file.)

A note on macOS
---------------
BabyTorch fully supports macOS on the CPU.  GPU acceleration currently
requires CUDA, i.e. an NVIDIA GPU, which Macs do not have -- so
``set_device("cuda")`` on a Mac raises an explanatory error instead of
suggesting an impossible install.  An Apple-Silicon backend (via MLX)
is on the roadmap; see ``TODO.md``.
"""

import os
import sys


class _XP:
    """Proxy that forwards every attribute to the active array library.

    Because modules bind ``xp`` once (``from .backend import xp``) but
    every *use* is an attribute access (``xp.zeros``), routing the
    lookup through ``__getattr__`` lets :func:`set_device` swap the
    library underneath all of them at once.
    """

    _lib = None  # numpy or cupy, set by set_device()

    def __getattr__(self, name):
        return getattr(_XP._lib, name)

    def __repr__(self):
        return f"<xp -> {_XP._lib.__name__}>"


xp = _XP()
DEVICE = None


def _cuda_library():
    """Import CuPy and check that a CUDA device actually exists."""
    if sys.platform == "darwin":
        raise RuntimeError(
            "CUDA GPUs are not available on macOS. BabyTorch runs on the "
            "CPU there (which works out of the box); an Apple-Silicon "
            "backend is on the roadmap, see TODO.md.")
    try:
        import cupy
    except ImportError as e:
        raise RuntimeError(
            "GPU support needs CuPy, which is not installed. "
            "Install it with:  pip install -e \".[gpu]\"") from e
    cupy.cuda.runtime.getDeviceCount()  # raises if no GPU / no driver
    return cupy


def set_device(name):
    """Select the array library: ``"cpu"``, ``"cuda"`` (or ``"gpu"``), or
    ``"auto"``.  Returns the name of the device actually selected.

    Call it *before* creating tensors or models (see the module
    docstring for why).  ``"cuda"`` raises with an explanation if no
    usable GPU stack is present; ``"auto"`` never raises.
    """
    global DEVICE
    name = name.lower()
    if name in ("cuda", "gpu"):
        _XP._lib = _cuda_library()
        DEVICE = "cuda"
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
            f"unknown device {name!r}: expected 'cpu', 'cuda' or 'auto'")
    return DEVICE


def device():
    """Return the active device name: ``"cpu"`` or ``"cuda"``."""
    return DEVICE


def to_numpy(array):
    """Return ``array`` as a NumPy array on the CPU.

    Plotting libraries (matplotlib) and file formats only understand
    NumPy arrays, so GPU arrays must be copied back to main memory
    first.  The check is on the *array itself* (CuPy arrays have a
    ``.get()`` method, NumPy arrays don't), so this works even for
    arrays created before a device switch.
    """
    if hasattr(array, "get"):
        return array.get()
    return array


def scatter_add(target, indices, values):
    """In-place ``target[indices] += values`` that handles *repeated* indices.

    Plain fancy-index assignment (``target[indices] += values``) silently
    drops all but one contribution when the same index appears twice.
    Gradients must *accumulate*, so we need the "unbuffered" version
    (``add.at``, which NumPy and CuPy both provide).

    This matters for e.g. the Embedding layer: if the word "the" occurs
    five times in a batch, its embedding row receives five gradient
    contributions that must all be summed.
    """
    xp.add.at(target, indices, values)


# Pick the initial device from the environment (default: auto).
_requested = os.environ.get("BABYTORCH_DEVICE", "auto").lower()
if _requested not in ("cpu", "cuda", "gpu", "auto"):
    _requested = "auto"
set_device(_requested)
