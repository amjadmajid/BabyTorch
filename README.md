<div align="center"> <img alt="BabyTorch Logo" src="/images/babyTorchLogo.jpg"></div>
<div style="text-align:center; font-size:2em; font-weight:bold;"> A Minimalist Educational Deep Learning Framework with Similar API to PyTorch </div>

## Introduction
BabyTorch is a lightweight, educational deep learning framework inspired by PyTorch. It's designed to offer a similar API to PyTorch, but with a minimal implementation. This simplicity makes it an excellent tool for learning and experimenting with deep learning concepts. The framework is structured to be easily understandable and modifiable, making it perfect for educational purposes.

## Installation
To install BabyTorch, follow these steps:

1. Clone the BabyTorch repository:
   ```bash
   git clone https://github.com/amjadmajid/BabyTorch.git
   ```
2. Navigate to the BabyTorch directory:
   ```bash
   cd BabyTorch
   ```
3. For a regular installation, run:
   ```bash
   python setup.py install
   ```
4. For developing BabyTorch itself, run:
   ```bash
   pip install -e . --user 
   ```

## Features
BabyTorch includes the following modules, mirroring the structure of PyTorch:

- `datasets`: Data loading utilities and predefined datasets like MNIST.
- `engine`: Core engine for operations and tensor manipulations.
- `nn`: Neural network layers and loss functions.
- `optim`: Optimization algorithms and learning rate schedulers.
- `visualization`: Tools for visualizing data and model graphs.

## Example Usage
Below are some examples of how to use BabyTorch, which also serve as basic tests:

1. **Neural Network Module Tests**:
   - Basic neural network operations: `tests/nn_module/nn_basic.py`
   - MNIST classification demo: `tests/nn_module/05_classification_mnist.py`

2. **Tensor Tests**:
   - Iterability and subscription tests: `tests/tensor_tests/iteration_subscribtion/iterablility_subscribtion_tests.py`
   - Mathematical operations: `tests/tensor_tests/math_operations/add_mul_tests.py`

3. **Visualization Tests**:
   - Graph visualization: `tests/visualization_tests/grapher_test.py`

These tests provide practical examples of implementing and using various components of BabyTorch.

## Contributing
We welcome contributions to __BabyTorch. It is in an early development stage__. If you're interested in improving BabyTorch or adding new features, please check `TODO.md` for upcoming tasks or propose your own ideas.


---

## BabyTorch Architecture Design

The BabyTorch framework is designed with simplicity and educational value in mind. Here’s an overview of its main components:

### Core Modules

1. **Datasets (`datasets`)**
   - Handles data loading and preprocessing.
   - Includes implementations for standard datasets like MNIST.
   - Provides a foundation for custom dataset integration.

2. **Engine (`engine`)**
   - The backbone of the framework, handling core operations.
   - It contains two files: `operations.py` and `tensor.py`.
   - `operations.py` implements the underlying computation engine. For each operation the forward and backward passes are implemented.
   - `tensor.py` implements the tensor data structure and its operations. For any operation, the corresponding operation in `operations.py` is called but the result is stored in a tensor. In this way we separate the computation and the data.

3. **Neural Networks (`nn`)**
   - Provides building blocks for neural networks.
   - Includes layers, activation functions, and loss functions.
   - Allows easy stacking and customization of layers to build various models.

4. **Optimizers (`optim`)**
   - Implements optimization algorithms.
   - Contains optimizers like SGD.
   - Features learning rate schedulers for controlling the learning rate during training.

5. **Visualization (`visualization`)**
   - Tools for visualizing data, model architectures, and training progress.
   - Helps in understanding model behavior and performance.

### Testing and Examples

- The `tests` directory contains numerous tests and examples demonstrating the use of BabyTorch's components. It serves as both a testing suite and a source of practical examples for users.

### Extensibility

- Designed to be easily extensible, allowing users to add new functionalities or modify existing ones.
- Encourages experimentation and exploration in deep learning.

### Directory Structure
```bash
.
├── README.md
├── TODO.md
├── babytorch
│   ├── __init__.py
│   ├── __pycache__
│   ├── datasets
│   ├── engine
│   ├── nn
│   ├── optim
│   └── visualization
├── setup.py
└── tests
    ├── nn_module
    ├── tensor_tests
    └── visualization_tests
```
---

This addition to the README provides a clear, high-level view of BabyTorch's architecture, making it easier for users to navigate the framework and understand its capabilities. You can tailor the descriptions to match the specific implementations and features of BabyTorch.

## License
This project is licensed under the [MIT License](LICENSE).

---

Happy Learning! 🚀
