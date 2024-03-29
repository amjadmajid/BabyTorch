import numpy as np
from babytorch import Tensor

def test_backward_batched_matmul():
    # Initialize inputs A and B
    A = np.random.randn(5, 3, 4)  # Batch of 5 matrices of shape (3, 4)
    B = np.random.randn(5, 4, 2)  # Batch of 5 matrices of shape (4, 2)
    
    # Simulated gradient coming from the next layer
    dY = np.random.randn(5, 3, 2)
    
    # Expected gradients
    expected_dA = np.matmul(dY, B.swapaxes(-2, -1))
    expected_dB = np.matmul(A.swapaxes(-2, -1), dY)
    
    # Using our backward method
    op = MatMulOperation()
    op.a = A
    op.b = B
    computed_dA, computed_dB = op.backward(dY)

    # Verify the computations
    assert np.allclose(expected_dA, computed_dA), f"Expected: {expected_dA}, Got: {computed_dA}"
    assert np.allclose(expected_dB, computed_dB), f"Expected: {expected_dB}, Got: {computed_dB}"

    print("All tests passed!")

test_backward_batched_matmul()
