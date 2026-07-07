"""Tests for the Grapher (headless-safe: nothing pops open a window)."""

import pytest

import babytorch
from babytorch import Tensor
from babytorch.visualization import Grapher


def test_build_computation_graph():
    graphviz = pytest.importorskip("graphviz")
    a = Tensor([[2.0, 4.5], [0.0, 5.0]], requires_grad=True, label="a")
    b = Tensor(3.0, requires_grad=True, label="b")
    c = a + b
    d = (a * c).sum()
    d.backward()

    dot = Grapher().plot_graph(d)          # builds the graph, does not display
    source = dot.source
    assert "data" in source and "grad" in source
    # every operation on the path should appear as a node label
    for op in ("+", "*", "sum"):
        assert op in source


def test_array_to_string_handles_large_and_none():
    g = Grapher()
    big = babytorch.randn(10, 10)
    assert "shape(10, 10)" in g.array_to_string(big.data)
    assert g.array_to_string(None) == "None"


def test_plot_loss_headless():
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend, no display needed
    g = Grapher()
    g.plot_loss([3.0, 2.0, 1.0, 0.5], label="loss")   # should not raise
