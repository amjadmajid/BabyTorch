"""Drawing tools for understanding what BabyTorch is doing.

Two kinds of picture:

* :meth:`Grapher.show_graph` / :meth:`Grapher.plot_graph` draw the
  **computation graph** -- every tensor and operation between the inputs
  and a chosen output, in the style of Karpathy's micrograd.  Great for
  *seeing* backprop on a small example.
* :meth:`Grapher.plot_loss` draws the familiar **loss curve** over
  training iterations.

Both matplotlib and graphviz are optional dependencies, imported lazily
so that ``import babytorch`` works on a machine that has neither.

Adapted and extended from
https://github.com/karpathy/micrograd/blob/master/trace_graph.ipynb
"""

import datetime
import os

from ..backend import to_numpy


def _to_cpu(array):
    """Return a NumPy view of a possibly-GPU array (or None unchanged)."""
    if array is None:
        return None
    return to_numpy(array)


class Grapher:
    def __init__(self):
        self.caller_id = 1

    # ------------------------------------------------------------------
    # Formatting node labels
    # ------------------------------------------------------------------

    def array_to_string(self, arr):
        """Compact text for a node label.

        Small arrays are printed element by element; anything larger than a
        handful of numbers is summarized by its shape, so the graph stays
        readable for real tensors.
        """
        if arr is None:
            return "None"
        arr = _to_cpu(arr)
        if arr.size == 1:
            return f"{arr.item():.4f}"
        if arr.size <= 6 and arr.ndim == 1:
            return '[' + ', '.join(f"{v:.2f}" for v in arr) + ']'
        return f"shape{tuple(arr.shape)}"

    # ------------------------------------------------------------------
    # Computation graph
    # ------------------------------------------------------------------

    def trace(self, root):
        """Walk backwards from ``root`` collecting all nodes and edges."""
        nodes, edges = set(), set()

        def build(v):
            if v not in nodes:
                v.graph_idx = self.caller_id
                self.caller_id += 1
                nodes.add(v)
                if v.operation:
                    for child in v.operation.inputs():
                        edges.add((child, v))
                        build(child)

        build(root)
        return nodes, edges

    def plot_graph(self, root):
        """Return a graphviz ``Digraph`` of the computation behind ``root``."""
        try:
            from graphviz import Digraph
        except ImportError as e:
            raise ImportError(
                "Drawing computation graphs needs the 'graphviz' package "
                "(pip install graphviz) and the Graphviz system binaries "
                "(e.g. apt install graphviz).") from e

        dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})  # left to right
        nodes, edges = self.trace(root)

        for n in nodes:
            uid = str(id(n))
            label_data = self.array_to_string(n.data)
            label_grad = self.array_to_string(n.grad)
            dot.node(name=uid,
                     label=f"{{ {n.label} | data {label_data} | grad {label_grad} }}",
                     shape='record')
            if n._op_label:
                dot.node(name=uid + n._op_label, label=n._op_label)
                dot.edge(uid + n._op_label, uid)

        for n1, n2 in edges:
            dot.edge(str(id(n1)), str(id(n2)) + n2._op_label)

        return dot

    def ensure_graphs_folder(self):
        if not os.path.exists('graphs'):
            os.makedirs('graphs')

    def show_graph(self, output, view=True):
        """Render the computation graph behind ``output`` to graphs/*.svg."""
        self.ensure_graphs_folder()
        stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        file_name = f'graphs/output_graph_{stamp}'
        dot = self.plot_graph(output)
        dot.render(filename=file_name, format='svg', cleanup=True)
        if view:
            try:
                dot.view(filename=file_name, cleanup=True)
            except Exception:
                pass  # headless machine: the file is still written
        return file_name + '.svg'

    # ------------------------------------------------------------------
    # Loss curves
    # ------------------------------------------------------------------

    def _plt(self):
        try:
            import matplotlib.pyplot as plt
            return plt
        except ImportError as e:
            raise ImportError("Plotting needs matplotlib "
                              "(pip install matplotlib).") from e

    def plot_loss(self, losses, label=""):
        """Plot a list of loss values over training iterations."""
        plt = self._plt()
        losses = [float(_to_cpu(l)) if hasattr(l, 'shape') else float(l)
                  for l in losses]
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Loss over Iterations')
        plt.grid(True)
        plt.plot(losses, label=label)
        if label:
            plt.legend()

    def show(self):
        self._plt().show()

    def savefig(self, path):
        self._plt().savefig(path)
