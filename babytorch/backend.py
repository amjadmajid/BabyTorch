"""Array backend selection for BabyTorch.

BabyTorch runs on top of an *array library* that does the actual number
crunching.  Three of them share (nearly) the same NumPy API:

* **NumPy** -- runs on the CPU, installed everywhere (Linux, macOS,
  Windows).
* **CuPy**  -- runs on an NVIDIA GPU (CUDA), a near drop-in for NumPy.
* **MLX**   -- runs on an Apple-Silicon GPU (Metal).  MLX's API is *close*
  to NumPy but not identical, so it is reached through a small adapter
  (:mod:`babytorch.mlx_backend`) that re-presents it under the same ``xp``
  surface.  This backend is **experimental** -- opt in with
  ``set_device("mps")``.

Every other BabyTorch module imports the active library from here under
the neutral name ``xp`` (a common convention meaning "numpy-or-cupy"):

    from babytorch.backend import xp

    xp.zeros((2, 3))        # works on CPU, CUDA *and* Metal

``xp`` is a tiny *proxy* object rather than the library itself, so the
active library can be swapped at runtime -- that is the whole trick
behind :func:`set_device`.

Choosing a device
-----------------
Two ways, same effect:

1. **In code** (a plain function call, e.g. from a notebook)::

       import babytorch
       babytorch.set_device("cpu")     # always NumPy
       babytorch.set_device("cuda")    # require an NVIDIA GPU (CuPy)
       babytorch.set_device("mps")     # require an Apple-Silicon GPU (MLX)
       babytorch.set_device("auto")    # CUDA if available, else CPU

2. **From the environment**, which sets the *initial* device before any
   code runs::

       BABYTORCH_DEVICE=cpu  python train.py
       BABYTORCH_DEVICE=cuda python train.py
       BABYTORCH_DEVICE=mps  python train.py

The default is ``auto``: use a CUDA GPU if CuPy and a device are
available, otherwise fall back to the CPU.  ``auto`` deliberately chooses
only between CUDA and the CPU -- the experimental Metal backend is never
selected implicitly; ask for ``"mps"`` by name.

One rule of thumb: **pick the device before building tensors or
models.**  Arrays do not migrate when the device changes -- a model
created on the GPU keeps its arrays even after ``set_device("cpu")``, and
mixing two libraries in one expression fails.  (Real frameworks move data
per-tensor with ``.to(device)``; BabyTorch keeps one global choice so the
entire GPU story fits in this one file.)

A note on macOS
---------------
BabyTorch fully supports macOS on the CPU.  On an **Apple-Silicon** Mac,
``set_device("mps")`` additionally runs on the GPU through MLX (install it
with ``pip install -e ".[mlx]"``); this path is newer than the CPU and
CUDA ones and still settling, which is why ``auto`` will not pick it for
you.  On an **Intel** Mac neither MLX nor CUDA exists, so both ``"mps"``
and ``"cuda"`` raise an explanatory error there -- the CPU is the way to
run.
"""

import os
import sys


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

    def __repr__(self):
        return f"<xp -> {_XP._lib.__name__}>"


xp = _XP()
DEVICE = None


def _cuda_library():
    """Import CuPy and check that a CUDA device actually exists."""
    if sys.platform == "darwin":
        raise RuntimeError(
            "CUDA GPUs are not available on macOS. BabyTorch runs on the "
            "CPU there (which works out of the box); on Apple Silicon try "
            "set_device('mps') for the experimental Metal backend.")
    try:
        import cupy
    except ImportError as e:
        raise RuntimeError(
            "GPU support needs CuPy, which is not installed. "
            "Install it with:  pip install -e \".[gpu]\"") from e
    cupy.cuda.runtime.getDeviceCount()  # raises if no GPU / no driver
    return cupy


def _mlx_library():
    """Import MLX (Apple Silicon only) and return the ``xp`` adapter module.

    MLX ships wheels only for Apple-Silicon Macs, so importing it fails on
    Intel Macs, Linux and Windows.  We translate that failure into the same
    kind of explanatory error :func:`_cuda_library` raises, rather than
    letting a bare ``ImportError`` surface.
    """
    if sys.platform != "darwin":
        raise RuntimeError(
            "The MLX (Metal) backend is Apple-Silicon only; this is not "
            "macOS. Use 'cpu', or 'cuda' on an NVIDIA GPU.")
    try:
        import mlx.core  # noqa: F401  (import is the availability check)
    except ImportError as e:
        raise RuntimeError(
            "The MLX backend needs the 'mlx' package, which installs only on "
            "Apple-Silicon Macs. Install it there with:  "
            "pip install -e \".[mlx]\"  -- on Intel Macs, Linux or Windows "
            "MLX is unavailable, so use the CPU (or a CUDA GPU) instead.") from e
    from . import mlx_backend
    return mlx_backend


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


def device():
    """Return the active device name: ``"cpu"``, ``"cuda"`` or ``"mps"``."""
    return DEVICE


def to_numpy(array):
    """Return ``array`` as a NumPy array on the CPU.

    Plotting libraries (matplotlib) and file formats only understand NumPy
    arrays, so GPU arrays must be copied back to main memory first.  The
    check is on the *array itself*, not the global device, so this works
    even for arrays created before a device switch: CuPy arrays expose
    ``.get()``, MLX arrays come from the ``mlx`` module (and convert via
    ``numpy.array``), and NumPy arrays are already there.
    """
    if hasattr(array, "get"):                        # CuPy -> host copy
        return array.get()
    if type(array).__module__.split(".")[0] == "mlx":  # MLX -> host copy
        import numpy
        return numpy.array(array)
    return array                                     # already NumPy


def scatter_add(target, indices, values):
    """``target[indices] += values`` that handles *repeated* indices, and
    returns the result.

    Plain fancy-index assignment (``target[indices] += values``) silently
    drops all but one contribution when the same index appears twice.
    Gradients must *accumulate*, so we need the "unbuffered" version.  NumPy
    and CuPy provide ``add.at`` (in place); MLX is functional and returns a
    new array, so callers use the return value either way.

    This matters for e.g. the Embedding layer: if the word "the" occurs five
    times in a batch, its embedding row receives five gradient contributions
    that must all be summed.
    """
    if DEVICE == "mps":
        return _XP._lib.scatter_add(target, indices, values)
    xp.add.at(target, indices, values)
    return target


# Pick the initial device from the environment (default: auto).
_requested = os.environ.get("BABYTORCH_DEVICE", "auto").lower()
if _requested not in ("cpu", "cuda", "gpu", "mps", "mlx", "auto"):
    _requested = "auto"
set_device(_requested)
