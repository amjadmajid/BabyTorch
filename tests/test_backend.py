"""Runtime device selection: babytorch.set_device.

The suite as a whole runs on whatever BABYTORCH_DEVICE selects (CPU by
default); these tests exercise *switching* devices from code.
"""

import numpy as np
import pytest

import babytorch


def _has_gpu():
    try:
        import cupy
        cupy.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False


def _has_mlx():
    try:
        import mlx.core  # noqa: F401  (present only on Apple Silicon)
        return True
    except Exception:
        return False


@pytest.fixture
def restore_device():
    """Put the device back the way the rest of the suite expects it."""
    original = babytorch.device()
    yield
    babytorch.set_device(original)


def test_set_device_cpu_roundtrip(restore_device):
    assert babytorch.set_device("cpu") == "cpu"
    assert babytorch.device() == "cpu"

    t = babytorch.randn(4, 3)
    assert isinstance(t.data, np.ndarray)

    # A tiny training step works end to end on the selected device.
    import babytorch.nn as nn
    from babytorch.optim import SGD
    model = nn.Linear(3, 1)
    optimizer = SGD(model.parameters(), learning_rate=0.1)
    loss = ((model(t) - 1.0) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert np.isfinite(loss.item())


def test_set_device_invalid_name(restore_device):
    with pytest.raises(ValueError):
        babytorch.set_device("tpu")


def test_auto_never_raises(restore_device):
    assert babytorch.set_device("auto") in ("cpu", "cuda")


@pytest.mark.skipif(not _has_gpu(), reason="no CUDA GPU available")
def test_set_device_cuda_roundtrip(restore_device):
    assert babytorch.set_device("cuda") == "cuda"

    t = babytorch.randn(4, 3)
    assert type(t.data).__module__.split(".")[0] == "cupy"
    # .numpy() copies back to the CPU regardless of the active device.
    assert isinstance(t.numpy(), np.ndarray)

    # Switching back: new tensors are NumPy again...
    babytorch.set_device("cpu")
    assert isinstance(babytorch.ones(2, 2).data, np.ndarray)
    # ...and arrays created before the switch still convert cleanly,
    # because to_numpy checks the array itself, not the global device.
    assert isinstance(babytorch.to_numpy(t.data), np.ndarray)


@pytest.mark.skipif(_has_gpu(), reason="only meaningful without a GPU")
def test_cuda_unavailable_raises_helpfully(restore_device):
    with pytest.raises(RuntimeError):
        babytorch.set_device("cuda")


@pytest.mark.skipif(_has_mlx(), reason="only meaningful without MLX installed")
def test_mps_unavailable_raises_helpfully(restore_device):
    # 'mps' and its alias 'mlx' are *recognised* device names: without an MLX
    # install they raise RuntimeError ("known but unavailable"), exactly like
    # 'cuda' without a GPU -- and a leftover failed switch must not disturb
    # the active (CPU) backend.
    for name in ("mps", "mlx"):
        with pytest.raises(RuntimeError):
            babytorch.set_device(name)
    assert isinstance(babytorch.ones(2, 2).data, np.ndarray)
    # A genuine typo is still a ValueError, not the unavailable-RuntimeError.
    with pytest.raises(ValueError):
        babytorch.set_device("metal")


@pytest.mark.skipif(not _has_mlx(), reason="no MLX / not Apple Silicon")
def test_set_device_mps_roundtrip(restore_device):
    # Runs only on an Apple-Silicon Mac with MLX installed -- this is the
    # real validation of babytorch/mlx_backend.py (authored without a device
    # to run it on).  If the adapter has bugs, they surface here.
    assert babytorch.set_device("mps") == "mps"

    t = babytorch.randn(4, 3)
    assert type(t.data).__module__.split(".")[0] == "mlx"
    assert isinstance(t.numpy(), np.ndarray)      # host copy works

    # A tiny training step, end to end on the Metal backend.
    import babytorch.nn as nn
    from babytorch.optim import SGD
    model = nn.Linear(3, 1)
    optimizer = SGD(model.parameters(), learning_rate=0.1)
    loss = ((model(t) - 1.0) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert np.isfinite(loss.item())

    # Switching back to the CPU: MLX arrays made earlier still convert cleanly.
    babytorch.set_device("cpu")
    assert isinstance(babytorch.ones(2, 2).data, np.ndarray)
    assert isinstance(babytorch.to_numpy(t.data), np.ndarray)
