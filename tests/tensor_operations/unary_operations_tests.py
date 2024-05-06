import unittest
from  babytorch import Tensor
import torch
import cupy as cp


class TestTensorOperations(unittest.TestCase):

    #------------------ ReLU ------------------#
    def test_scalar_relu(self):
        num = -3.0

        # babytorch's Tensor
        a_t = Tensor(num, requires_grad=True)
        b_t = a_t.relu()
        b_t.backward()

        # PyTorch's Tensor
        a_p = torch.tensor(num, requires_grad=True)
        b_p = torch.nn.functional.relu(a_p)
        b_p.backward()

        # Check equivalence
        self.assertEqual(b_t.data, b_p.item())
        self.assertEqual(a_t.grad, a_p.grad.item())
        print("Scalar ReLU test passed!")

        if print_output:
            print("babytorch's Tensor: ",  b_t.data, a_t.grad)
            print("PyTorch's Tensor:  ",  b_p.item(), a_p.grad.item())
            print()

    def test_vector_relu(self):
        vec = [-1.0, 0.0, 2.0]

        # babytorch's Tensor
        a_t = Tensor(vec, requires_grad=True)
        b_t = a_t.relu()
        b_t.sum().backward()

        # PyTorch's Tensor
        a_p = torch.tensor(vec, requires_grad=True)
        b_p = torch.nn.functional.relu(a_p)
        b_p.sum().backward()

        # Check equivalence
        self.assertTrue(cp.array_equal(b_t.data, b_p.detach().numpy()))
        self.assertTrue(cp.array_equal(a_t.grad, a_p.grad.numpy()))
        print("Vector ReLU test passed!")

        if print_output:
            print("babytorch's Tensor: ", b_t.data, a_t.grad)
            print("PyTorch's Tensor:  ", b_p.detach().numpy(), a_p.grad.numpy())
            print()


    def test_matrix_relu(self):
        matrix = [[-1.0, 2.0], [3.0, -4.0]]

        a_t = Tensor(matrix, requires_grad=True)
        b_t = a_t.relu()
        b_t.sum().backward()
        
        a_p = torch.tensor(matrix, requires_grad=True)
        b_p = torch.nn.functional.relu(a_p)
        b_p.sum().backward()
        
        self.assertTrue(cp.array_equal(b_t.data, b_p.detach().numpy()))
        self.assertTrue(cp.array_equal(a_t.grad, a_p.grad.numpy()))
        print("Matrix ReLU test passed!")

        if print_output:
            print("babytorch's Tensor: ", b_t.data, a_t.grad)
            print("PyTorch's Tensor:  ", b_p.detach().numpy(), a_p.grad.numpy())
            print()


    #------------------ Tanh ------------------#
    def test_scalar_tanh(self):
        num = -3.0

        # babytorch's Tensor
        a_t = Tensor(num, requires_grad=True)
        b_t = a_t.tanh()
        b_t.backward()

        # PyTorch's Tensor
        a_p = torch.tensor(num, requires_grad=True)
        b_p = torch.tanh(a_p)
        b_p.backward()

        # Check equivalence
        self.assertEqual(b_t.data, b_p.item())
        self.assertAlmostEqual(a_t.grad, a_p.grad.item())
        print("Scalar Tanh test passed!")

        if print_output:
            print("babytorch's Tensor: ",  b_t.data, a_t.grad)
            print("PyTorch's Tensor:  ",  b_p.item(), a_p.grad.item())
            print()

    def test_vector_tanh(self):
        vec = [-1.0, 0.0, 2.0]

        # babytorch's Tensor
        a_t = Tensor(vec, requires_grad=True)
        b_t = a_t.tanh()
        b_t.sum().backward()

        # PyTorch's Tensor
        a_p = torch.tensor(vec, requires_grad=True)
        b_p = torch.tanh(a_p)
        b_p.sum().backward()

        # Check equivalence
        self.assertTrue(cp.array_equal(b_t.data, b_p.detach().numpy()))
        self.assertTrue(cp.array_equal(a_t.grad, a_p.grad.numpy()))
        print("Vector Tanh test passed!")

        if print_output:
            print("babytorch's Tensor: ", b_t.data, a_t.grad)
            print("PyTorch's Tensor:  ", b_p.detach().numpy(), a_p.grad.numpy())
            print()

    def test_matrix_tanh(self):
        matrix = [[-1.0, 2.0], [3.0, -4.0]]

        # babytorch's Tensor
        a_t = Tensor(matrix, requires_grad=True)
        b_t = a_t.tanh()
        b_t.sum().backward()
        
        # PyTorch's Tensor
        a_p = torch.tensor(matrix, requires_grad=True)
        b_p = torch.tanh(a_p)
        b_p.sum().backward()
        
        # Check equivalence
        self.assertTrue(cp.array_equal(b_t.data, b_p.detach().numpy()))
        self.assertTrue(cp.array_equal(a_t.grad, a_p.grad.numpy()))
        print("Matrix Tanh test passed!")

        if print_output:
            print("babytorch's Tensor: ", b_t.data, a_t.grad)
            print("PyTorch's Tensor:  ", b_p.detach().numpy(), a_p.grad.numpy())
            print()

    #------------------ Exp ------------------#
    def test_scalar_exp(self):
        num = 3.0
    
        a_t = Tensor(num, requires_grad=True)
        b_t = a_t.exp()
        b_t.backward(1.0)
        
        a_p = torch.tensor(num, requires_grad=True)
        b_p = torch.exp(a_p)
        b_p.backward(torch.tensor(1.0))
        
        self.assertEqual(b_t.data, b_p.item())
        self.assertEqual(a_t.grad, a_p.grad.item())
        print("Scalar Exp test passed!")

        if print_output:
            print("babytorch's Tensor: ", b_t.data, a_t.grad)
            print("PyTorch's Tensor:  ", b_p.item(), a_p.grad.item())
            print()

    def test_vector_exp(self):
        vec = [1.0, -2.0, 3.0]

        a_t = Tensor(vec, requires_grad=True)
        b_t = a_t.exp()
        b_t.sum().backward()
        
        a_p = torch.tensor(vec, requires_grad=True)
        b_p = torch.exp(a_p)
        b_p.sum().backward()
        
        self.assertTrue(cp.array_equal(b_t.data, b_p.detach().numpy()))
        self.assertTrue(cp.array_equal(a_t.grad, a_p.grad.numpy()))
        print("Vector Exp test passed!")

        if print_output:
            print("babytorch's Tensor: ", b_t.data, a_t.grad)
            print("PyTorch's Tensor:  ", b_p.detach().numpy(), a_p.grad.numpy())
            print()

    def test_matrix_exp(self):
        matrix = [[1.0, -2.0], [3.0, -4.0]]

        a_t = Tensor(matrix, requires_grad=True)
        b_t = a_t.exp()
        b_t.sum().backward()
        
        a_p = torch.tensor(matrix, requires_grad=True)
        b_p = torch.exp(a_p)
        b_p.sum().backward()
        
        self.assertTrue(cp.array_equal(b_t.data, b_p.detach().numpy()))
        self.assertTrue(cp.array_equal(a_t.grad, a_p.grad.numpy()))
        print("Matrix Exp test passed!")

        if print_output:
            print("babytorch's Tensor: ", b_t.data, a_t.grad)
            print("PyTorch's Tensor:  ", b_p.detach().numpy(), a_p.grad.numpy())
            print()

if __name__ == '__main__':
    print_output = False
    unittest.main()
