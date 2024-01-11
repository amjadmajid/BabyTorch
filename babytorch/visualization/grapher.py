# This is an extended version of the visualization code in https://github.com/karpathy/micrograd/blob/master/trace_graph.ipynb 
import datetime
from graphviz import Digraph
import matplotlib.pyplot as plt
import numpy as np
import os

class Grapher:
    def __init__(self):
        self.caller_id = 1

    def ndarray_to_string(self, arr):
        if arr.size == 1:
            return f"{arr.item():.2f}"
        elif arr.ndim == 1:
            return '[' + ', '.join(map(lambda val: f"{val.item():.2f}", arr)) + ']'
        
        rows = []
        for row in arr:
            formatted_row = ', '.join(map(lambda val: f"{val:.2f}", row))
            rows.append(f"[{formatted_row}]")
        return ', '.join(rows) 

    def trace(self, root):
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
        dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right

        nodes, edges = self.trace(root)

        for n in nodes:
            uid = str(id(n))

            if isinstance(n.data, np.ndarray):
                label_data = self.ndarray_to_string(n.data)
                label_grad = self.ndarray_to_string(n.grad)
            else:
                label_data = "{:.4f}".format(n.data)
                label_grad = "{:.4f}".format(n.grad)

            dot.node(name = uid, label=f"{n.label} | data {label_data} | grad  {label_grad}", shape='record')

            if n._op_label:
                # if this value is a result of some operation, create an op node for it
                dot.node(name = uid + n._op_label, label=n._op_label)
                # and connect this node to it 
                dot.edge(uid + n._op_label, uid)
        for n1, n2 in edges:
            # connect n1 to the op node of n2
            dot.edge(str(id(n1)), str(id(n2)) + n2._op_label)

        return dot

    def ensure_graphs_folder(self):
        if not os.path.exists('graphs'):
            os.makedirs('graphs')

    def show_graph(self, o):
        self.ensure_graphs_folder()
        file_name = 'graphs/output_graph_' + str(datetime.datetime.now())
        dot = self.plot_graph(o)
        dot.render(filename=file_name, format='png', cleanup=True)
        dot.view(filename=file_name, cleanup=True)


    def plot_loss(self, losses, label=""):
        """Plots the loss over iterations."""
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Loss over Iterations')
        plt.grid(True)
        plt.plot(losses, label=label)
        if label:
            plt.legend()
    

    def show(self):
        plt.show()

