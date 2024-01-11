import numpy as np
from babytorch import Tensor

def test_tensor_iterable():
    tensor = Tensor(np.array([1, 2, 3, 4]))
    extracted_data = [x for x in tensor]
    print(type(extracted_data[0]))
    assert extracted_data == [1, 2, 3, 4], f"Expected [1, 2, 3, 4], but got {extracted_data}"

def test_tensor_subscribable_get():
    tensor = Tensor(np.array([1, 2, 3, 4]))
    assert tensor[2] == 3, f"Expected 3, but got {tensor[2]}"

def test_tensor_subscribable_set():
    tensor = Tensor(np.array([1, 2, 3, 4]))
    tensor[2] = 5
    assert tensor[2] == 5, f"Expected 5, but got {tensor[2]}"

def test_tensor_multi_dimensional():
    tensor = Tensor(np.array([[1, 2], [3, 4]]))
    row = tensor[1]
    assert np.array_equal(row, np.array([3, 4])), f"Expected array([3, 4]), but got {row}"

    val = tensor[1, 1]
    assert val == 4, f"Expected 4, but got {val}"

def run_tests():
    test_tensor_iterable()
    test_tensor_subscribable_get()
    test_tensor_subscribable_set()
    test_tensor_multi_dimensional()
    print("All tests passed!")

# Running the tests
run_tests()
