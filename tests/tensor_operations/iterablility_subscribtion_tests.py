import unittest
from babytorch import Tensor
import torch
import cupy as cp


class TestTensorOperations(unittest.TestCase):

    # ------------------ ITERABLE ------------------ #

    def test_tensor_iterable(self):
        tensor = Tensor(cp.array([1, 2, 3, 4]))
        extracted_data = [x for x in tensor]
        self.assertEqual(extracted_data, [1, 2, 3, 4], f"Expected [1, 2, 3, 4], but got {extracted_data}")

    # ------------------ SUBSCRIPTABLE ------------------ #

    def test_tensor_subscriptable_get(self):
        tensor = Tensor(cp.array([1, 2, 3, 4]))
        self.assertEqual(tensor[2].data, 3, f"Expected 3, but got {tensor[2].data}")

    def test_tensor_subscriptable_set(self):
        tensor = Tensor(cp.array([1, 2, 3, 4]))
        tensor[2] = 5
        self.assertEqual(tensor[2].data, 5, f"Expected 5, but got {tensor[2].data}")

    # ------------------ MULTI-DIMENSIONAL ------------------ #

    def test_tensor_multi_dimensional(self):
        tensor = Tensor(cp.array([[1, 2], [3, 4]]))
        row = tensor[1]
        self.assertTrue(cp.array_equal(row.data, cp.array([3, 4])), f"Expected array([3, 4]), but got {row.data}")

        val = tensor[1, 1]
        self.assertEqual(val.data, 4, f"Expected 4, but got {val.data}")


if __name__ == '__main__':
    unittest.main()
