"""Array backend selection for BabyTorch.

BabyTorch runs on top of an *array library* that does the actual number
crunching.  Two libraries share (almost exactly) the same API:

* **NumPy** -- runs on the CPU, installed everywhere.
* **CuPy**  -- runs on an NVIDIA GPU, drop-in replacement for NumPy.

This module picks one of them **once**, at import time, and every other
BabyTorch module imports the chosen library from here under the neutral
name ``xp`` (a common convention meaning "numpy-or-cupy"):

    from babytorch.backend import xp

    xp.zeros((2, 3))        # works on CPU *and* GPU

Selection rules
---------------
The environment variable ``BABYTORCH_DEVICE`` controls the choice:

* ``auto`` (default) -- use the GPU if CuPy *and* a CUDA device are
  available, otherwise fall back to the CPU.
* ``cpu``            -- always use NumPy.
* ``cuda`` / ``gpu`` -- require CuPy; raise an error if unavailable.

Example::

    BABYTORCH_DEVICE=cpu python train.py     # force CPU
"""

import os

_requested = os.environ.get("BABYTORCH_DEVICE", "auto").lower()


def _try_gpu():
    """Import CuPy and check that a CUDA device actually exists."""
    import cupy
    cupy.cuda.runtime.getDeviceCount()  # raises if no GPU / no driver
    return cupy


if _requested in ("cuda", "gpu"):
    xp = _try_gpu()
    DEVICE = "cuda"
elif _requested == "cpu":
    import numpy as xp
    DEVICE = "cpu"
else:  # "auto"
    try:
        xp = _try_gpu()
        DEVICE = "cuda"
    except Exception:
        import numpy as xp
        DEVICE = "cpu"


def device():
    """Return the active device name: ``"cpu"`` or ``"cuda"``."""
    return DEVICE


def to_numpy(array):
    """Return ``array`` as a NumPy array on the CPU.

    Plotting libraries (matplotlib) and file formats only understand
    NumPy arrays, so GPU arrays must be copied back to main memory first.
    """
    if DEVICE == "cuda":
        return xp.asnumpy(array)
    return array


def scatter_add(target, indices, values):
    """In-place ``target[indices] += values`` that handles *repeated* indices.

    Plain fancy-index assignment (``target[indices] += values``) silently
    drops all but one contribution when the same index appears twice.
    Gradients must *accumulate*, so we need the "unbuffered" version:

    * NumPy calls it ``np.add.at``
    * CuPy  calls it ``cupyx.scatter_add``

    This matters for e.g. the Embedding layer: if the word "the" occurs
    five times in a batch, its embedding row receives five gradient
    contributions that must all be summed.
    """
    xp.add.at(target, indices, values)
