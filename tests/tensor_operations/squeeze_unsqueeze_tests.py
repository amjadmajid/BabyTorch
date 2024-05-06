import unittest
from babytorch import Tensor
import torch
import cupy as cp


class TestTensorOperations(unittest.TestCase):
        
    #------------------ Squeeze ------------------#
    def test_vector_squeeze(self):
        vec = cp.array([[1.0], [-2.0], [3.0]])

        # babytorch's Tensor
        a_t = Tensor(vec, requires_grad=True)
        b_t = a_t.squeeze()
        b_t.sum().backward()

        # PyTorch's Tensor
        a_p = torch.tensor(vec, requires_grad=True)
        b_p = a_p.squeeze()
        b_p.sum().backward()

        self.assertTrue(cp.array_equal(b_t.data, b_p.detach().numpy()))
        self.assertTrue(cp.array_equal(a_t.grad, a_p.grad.numpy()))
        print("Vector Squeeze test passed!")

        if print_output:
            print("babytorch's Tensor: ", b_t.data, a_t.grad)
            print("PyTorch's Tensor:  ", b_p.detach().numpy(), a_p.grad.numpy())
            print()


    #------------------ Unsqueeze ------------------#
    def test_scalar_unsqueeze(self):
        num = 3.0

        # babytorch's Tensor
        a_t = Tensor(num, requires_grad=True)
        b_t = a_t.unsqueeze(0)
        b_t.backward(1.0)

        # PyTorch's Tensor
        a_p = torch.tensor(num, requires_grad=True)
        b_p = a_p.unsqueeze(0)
        b_p.backward()

        self.assertEqual(b_t.data, b_p.item())
        self.assertEqual(a_t.grad, a_p.grad.item())
        print("Scalar Unsqueeze test passed!")

        if print_output:
            print("babytorch's Tensor: ", b_t.data, a_t.grad)
            print("PyTorch's Tensor:  ", b_p.item(), a_p.grad.item())
            print()

    def test_vector_unsqueeze(self):
        vec = [1.0, -2.0, 3.0]

        # babytorch's Tensor
        a_t = Tensor(vec, requires_grad=True)
        b_t = a_t.unsqueeze(1)
        b_t.sum().backward()

        # PyTorch's Tensor
        a_p = torch.tensor(vec, requires_grad=True)
        b_p = a_p.unsqueeze(1)
        b_p.sum().backward()

        self.assertTrue(cp.array_equal(b_t.data, b_p.detach().numpy()))
        self.assertTrue(cp.array_equal(a_t.grad, a_p.grad.numpy()))
        print("Vector Unsqueeze test passed!")

        if print_output:
            print("babytorch's Tensor: ", b_t.data, a_t.grad)
            print("PyTorch's Tensor:  ", b_p.detach().numpy(), a_p.grad.numpy())
            print()

if __name__ == '__main__':
    print_output = True
    unittest.main()
