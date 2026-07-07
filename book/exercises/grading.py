"""Shared plumbing for the exercise graders.

* :func:`load` imports a starter module -- or, when the environment
  variable ``EXERCISES_SOLUTIONS`` is set, the corresponding module from
  ``solutions/``. The main test suite uses that switch to prove every
  exercise stays solvable.
* :func:`exercise` turns ``NotImplementedError`` into a pytest *skip*,
  so untouched exercises read as "your turn", not as failures.
"""

import functools
import importlib.util
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    """Import ``book/exercises/<name>.py`` (or its solution)."""
    if os.environ.get("EXERCISES_SOLUTIONS"):
        path = os.path.join(HERE, "solutions", name + ".py")
        modname = "solutions_" + name
    else:
        path = os.path.join(HERE, name + ".py")
        modname = "exercise_" + name
    spec = importlib.util.spec_from_file_location(modname, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exercise(test_fn):
    """Skip (don't fail) while the exercise is still a stub."""
    @functools.wraps(test_fn)
    def wrapper(*args, **kwargs):
        try:
            return test_fn(*args, **kwargs)
        except NotImplementedError:
            pytest.skip("not implemented yet -- your turn!")
    return wrapper


def numeric_gradient(f, x, eps=1e-5):
    """Central-difference gradient of a scalar function of an array.

    The same independent check the library's own tests use (see
    ``tests/conftest.py`` and chapter 2): no calculus, just nudging.
    """
    import numpy as np
    x = np.array(x, dtype=np.float64)
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        original = x[idx]
        x[idx] = original + eps
        plus = f(x)
        x[idx] = original - eps
        minus = f(x)
        x[idx] = original
        grad[idx] = (plus - minus) / (2 * eps)
        it.iternext()
    return grad
