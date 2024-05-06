import unittest
from babytorch import Tensor
import torch
import cupy as cp


class TestTensorOperations(unittest.TestCase):

    def test_batched_matrix_multiplication_with_broadcasting(self):
        # babytorch's Tensor
        batched_matrix1 = [[[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]]]
        batched_matrix2 = [5.0, 6.0]

        # PyTorch's Tensor
        a_p = torch.tensor(batched_matrix1, requires_grad=True)
        b_p = torch.tensor(batched_matrix2, requires_grad=True)
        c_p = a_p @ b_p
        c_p.sum().backward()

        a_t = Tensor(batched_matrix1, requires_grad=True)
        b_t = Tensor(batched_matrix2, requires_grad=True)
        c_t = a_t @ b_t
        c_t.sum().backward()
        
        # exit()
        # if print_output:
        print(f" {c_t.data=},\n {a_t.grad=},\n {b_t.grad=}")
        print()
        print(f" {c_p.detach().numpy()=},\n {a_p.grad.numpy()=},\n {b_p.grad.numpy()=}")

        # Check equivalence
        self.assertTrue(cp.allclose(c_t.data, c_p.detach().numpy()))
        self.assertTrue(cp.allclose(a_t.grad, a_p.grad.numpy()))
        self.assertTrue(cp.allclose(b_t.grad, b_p.grad.numpy()))
        # print("Batched matrix multiplication with broadcasting test passed!")


if __name__ == '__main__':
    print_output = False
    unittest.main()
