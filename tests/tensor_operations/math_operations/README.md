# Test Suite for Tensor Operations

This test suite validates the tensor operations implemented in `babytorch` by comparing the results with PyTorch. The tests cover various operations, including addition, multiplication, broadcasting, squeezing and activation functions.

### Required Libraries:
  The following libraries must be installed in your environment `babytorch`, `torch`, `numpy`, and `unittest`.


### [Addition & Multiplication](add_mul_tests.py)
This file contains tests for scalar, vector, and matrix addition and multiplication operations. It ensures that the results and gradients obtained from `babytorch` match those from PyTorch.

To run the tests, you'll need:
- Python 3.x
- `babytorch`: A lightweight deep learning framework
- `PyTorch`: A popular deep learning framework`

**Running the Tests**

To run the tests, simply execute the following command in your terminal:

```bash
cd tests/tensor_operations/math_operations
python add_mul_tests.py
```
The following tests will be executed and the results will be displayed in the console:

1. **Addition Tests**:
   - Scalar Addition
   - Vector Addition
   - Matrix Addition
   - Scalar-Vector Addition
   - Scalar-Matrix Addition
   - Vector-Matrix Addition
   - Batched Matrix Addition
   - Batched Matrix Addition with Broadcasting

2. **Multiplication Tests**:
   - Scalar Multiplication
   - Vector Multiplication
   - Matrix Multiplication
   - Matrix-Matrix Multiplication with Different Shapes
   - Matrix-Vector Multiplication with Different Shapes
   - Batched Matrix Multiplication with Broadcasting

### [Tensor Squeeze and Unsqueeze](squeeze_unsqueeze_tests.py)

To run the tests, simply execute the following command in your terminal:

```bash
cd tests/tensor_operations/math_operations
python squeeze_unsqueeze_tests.py
```

The following tests will be executed and the results will be displayed in the console:


1. **Squeeze Tests**:
   - Vector Squeeze
   
2. **Unsqueeze Tests**:
   - Scalar Unsqueeze
   - Vector Unsqueeze


### [Unary Operations](unary_operations_tests.py)

This file contains tests for various activation functions such as ReLU, Tanh, and Exponential implemented in `babytorch`. The tests validate the correctness of the implementations by comparing the results and gradients obtained from `babytorch` with those from PyTorch.

**Running the Tests**

To run the tests for activation functions, execute the following command in your terminal:

```bash
cd tests/tensor_operations/activation_functions
python unary_operations_tests.py
```

The following tests will be executed and the results will be displayed in the console:

1. **ReLU Tests**:
   - Scalar ReLU
   - Vector ReLU
   - Matrix ReLU

2. **Tanh Tests**:
   - Scalar Tanh
   - Vector Tanh
   - Matrix Tanh

3. **Exponential Tests**:
   - Scalar Exp
   - Vector Exp
   - Matrix Exp


### Conclusion

This test suite serves as a comprehensive validation tool for the tensor operations and activation functions implemented in `babytorch`. By comparing the results with PyTorch, it ensures the correctness and consistency of the framework's functionalities. Whether you're developing new features for `babytorch` or using it in your deep learning projects, these tests provide confidence in the reliability of the library.

Feel free to contribute to `babytorch` by adding new tests or improving existing ones. Your contributions play a vital role in enhancing the robustness and performance of the framework.

Thank you for using `babytorch` and happy testing!

