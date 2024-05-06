import unittest
from babytorch import Tensor
import torch
import cupy as cp


class TestTensorOperations(unittest.TestCase):

#------------------ ADDITION ------------------#
    def test_scalar_addition(self):
        num1 = 3.0
        num2 = 4.0
    
        # babytorch's Tensor
        a_t = Tensor(num1, requires_grad=True)
        b_t = Tensor(num2, requires_grad=True)
        c_t = a_t + b_t
        c_t.backward()
        
        # PyTorch's Tensor
        a_p = torch.tensor(num1, requires_grad=True)
        b_p = torch.tensor(num2, requires_grad=True)
        c_p = a_p + b_p
        c_p.backward()
                
        if print_output:
            print("babytorch's Tensor: ",  c_t.data, a_t.grad, b_t.grad)
            print("PyTorch's Tensor:  ",  c_p.item(), a_p.grad.item(), b_p.grad.item())   
            print()

        # Check equivalence
        self.assertEqual(c_t.data, c_p.item())
        self.assertEqual(a_t.grad, a_p.grad.item())
        self.assertEqual(b_t.grad, b_p.grad.item())
        print("Scalar addition test passed!")

    def test_vec_addition(self):
        # babytorch's Tensor
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [4.0, 5.0, 6.0]

        a_t = Tensor(vec1, requires_grad=True)
        b_t = Tensor(vec2, requires_grad=True)
        c_t = a_t + b_t
        c_t.sum().backward()
        
        # PyTorch's Tensor
        a_p = torch.tensor( vec1, requires_grad=True)
        b_p = torch.tensor(vec2, requires_grad=True)
        c_p = a_p + b_p
        c_p.sum().backward()

        if print_output:
            print("babytorch's Tensor: ", c_t.data, a_t.grad, b_t.grad)
            print("PyTorch's Tensor:  ", c_p.detach().numpy(), a_p.grad.numpy(), b_p.grad.numpy())
            print()
        
        # Check equivalence
        self.assertTrue(cp.array_equal(c_t.data, c_p.detach().numpy()))
        self.assertTrue(cp.array_equal(a_t.grad, a_p.grad.numpy()))
        self.assertTrue(cp.array_equal(b_t.grad, b_p.grad.numpy()))
        print("Vector addition test passed!")


    def test_matrix_addition(self):
        # babytorch's Tensor
        matrix1 = [[1.0, 2.0], [3.0, 4.0]]
        matrix2 = [[5.0, 6.0], [7.0, 8.0]]

        a_t = Tensor(matrix1, requires_grad=True)
        b_t = Tensor(matrix2, requires_grad=True)
        c_t = a_t + b_t
        c_t.sum().backward()
        
        # PyTorch's Tensor
        a_p = torch.tensor(matrix1, requires_grad=True)
        b_p = torch.tensor(matrix2, requires_grad=True)
        c_p = a_p + b_p
        c_p.sum().backward()
        
        # Check equivalence
        self.assertTrue(cp.array_equal(c_t.data, c_p.detach().numpy()))
        self.assertTrue(cp.array_equal(a_t.grad, a_p.grad.numpy()))
        self.assertTrue(cp.array_equal(b_t.grad, b_p.grad.numpy()))
        print("Matrix addition test passed!")

        if print_output:
            print("babytorch's Tensor: ", c_t.data, a_t.grad, b_t.grad)
            print("PyTorch's Tensor:  ", c_p.detach().numpy(), a_p.grad.numpy(), b_p.grad.numpy())
            print()

    def test_scalar_vector_addition(self):
        # babytorch's Tensor
        scalar = 1.0
        vector = [2.0, 3.0, 4.0]
        
        a_t = Tensor(scalar, requires_grad=True)
        b_t = Tensor(vector, requires_grad=True)
        c_t = a_t + b_t
        # print(f"{a_t.shape=}, {b_t.shape=}")
        c_t.sum().backward()
        
        # PyTorch's Tensor
        a_p = torch.tensor(scalar, requires_grad=True)
        b_p = torch.tensor(vector, requires_grad=True)
        c_p = a_p + b_p
        c_p.sum().backward()
        
        self.assertTrue(cp.array_equal(c_t.data, c_p.detach().numpy()))
        self.assertEqual(a_t.grad, a_p.grad.item())
        self.assertTrue(cp.array_equal(b_t.grad, b_p.grad.numpy()))
        print("Scalar + Vector addition test passed!")

        if print_output:
            print("babytorch's Tensor: ", c_t.data, a_t.grad, b_t.grad)
            print("PyTorch's Tensor:  ", c_p.detach().numpy(), a_p.grad.item(), b_p.grad.numpy())
            print()

    def test_scalar_matrix_addition(self):
        # babytorch's Tensor
        scalar = 1.0
        matrix = [[2.0, 3.0], [4.0, 5.0]]
        
        a_t = Tensor(scalar, requires_grad=True)
        b_t = Tensor(matrix, requires_grad=True)
        c_t = a_t + b_t
        # print(f"{a_t.shape=}, {b_t.shape=}")
        c_t.sum().backward()
        
        # PyTorch's Tensor
        a_p = torch.tensor(scalar, requires_grad=True)
        b_p = torch.tensor(matrix, requires_grad=True)
        c_p = a_p + b_p
        c_p.sum().backward()
        
        self.assertTrue(cp.array_equal(c_t.data, c_p.detach().numpy()))
        self.assertEqual(a_t.grad, a_p.grad.item())
        self.assertTrue(cp.array_equal(b_t.grad, b_p.grad.numpy()))
        print("Scalar + Matrix addition test passed!")

        if print_output:
            print("babytorch's Tensor: ", c_t.data, a_t.grad, b_t.grad)
            print("PyTorch's Tensor:  ", c_p.detach().numpy(), a_p.grad.item(), b_p.grad.numpy())
            print()

    def test_vector_matrix_addition(self):
        # babytorch's Tensor
        vector = [1.0, 2.0]
        matrix = [[2.0, 3.0], [4.0, 5.0]]
        
        a_t = Tensor(vector, requires_grad=True)
        b_t = Tensor(matrix, requires_grad=True)
        # print(f"{a_t.shape=}, {b_t.shape=}")
        c_t = a_t + b_t
        c_t.sum().backward()
        
        # PyTorch's Tensor
        a_p = torch.tensor(vector, requires_grad=True)
        b_p = torch.tensor(matrix, requires_grad=True)
        c_p = a_p + b_p
        c_p.sum().backward()

        self.assertTrue(cp.allclose(c_t.data, c_p.detach().numpy()))
        self.assertTrue(cp.allclose(a_t.grad, a_p.grad.numpy()))
        self.assertTrue(cp.allclose(b_t.grad, b_p.grad.numpy()))
        print("Vector + Matrix addition test passed!")

        if print_output:
            print("babytorch's Tensor: ", c_t.data, a_t.grad, b_t.grad)
            print("PyTorch's Tensor:  ", c_p.detach().numpy(), a_p.grad.numpy(), b_p.grad.numpy())
            print()

    def test_batched_matrix_addition(self):
        # babytorch's Tensor
        batched_matrix1 = [[[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]]]
        batched_matrix2 = [[[5.0, 6.0], [7.0, 8.0]], [[5.0, 6.0], [7.0, 8.0]]]

        a_t = Tensor(batched_matrix1, requires_grad=True)
        b_t = Tensor(batched_matrix2, requires_grad=True)
        c_t = a_t + b_t
        c_t.sum().backward()
        
        # PyTorch's Tensor
        a_p = torch.tensor(batched_matrix1, requires_grad=True)
        b_p = torch.tensor(batched_matrix2, requires_grad=True)
        c_p = a_p + b_p
        c_p.sum().backward()
        
        # Check equivalence
        self.assertTrue(cp.array_equal(c_t.data, c_p.detach().numpy()))
        self.assertTrue(cp.array_equal(a_t.grad, a_p.grad.numpy()))
        self.assertTrue(cp.array_equal(b_t.grad, b_p.grad.numpy()))
        print("Batched matrix addition test passed!")

        if print_output:
            print("babytorch's Tensor: ", c_t.data, a_t.grad, b_t.grad)
            print("PyTorch's Tensor:  ", c_p.detach().numpy(), a_p.grad.numpy(), b_p.grad.numpy())
            print()

    def test_batched_matrix_addition_with_broadcasting(self):
        # babytorch's Tensor
        batched_matrix1 = [[[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]]]
        batched_matrix2 = [5.0, 6.0]

        a_t = Tensor(batched_matrix1, requires_grad=True)
        b_t = Tensor(batched_matrix2, requires_grad=True)
        c_t = a_t + b_t
        c_t.sum().backward()
        
        # PyTorch's Tensor
        a_p = torch.tensor(batched_matrix1, requires_grad=True)
        b_p = torch.tensor(batched_matrix2, requires_grad=True)
        c_p = a_p + b_p
        c_p.sum().backward()
        
        # Check equivalence
        self.assertTrue(cp.allclose(c_t.data, c_p.detach().numpy()))
        self.assertTrue(cp.allclose(a_t.grad, a_p.grad.numpy()))
        self.assertTrue(cp.allclose(b_t.grad, b_p.grad.numpy()))
        print("Batched matrix addition with broadcasting test passed!")

        if print_output:
            print(f" {c_t.data=},\n {a_t.grad=},\n {b_t.grad=}")
            print()
            print(f" {c_p.detach().numpy()=},\n {a_p.grad.numpy()=},\n {b_p.grad.numpy()=}")

# ------------------ MULTIPLICATION ------------------#

    def test_scalar_multiplication(self):
        num1 = 3.0
        num2 = 4.0
        
        # babytorch's Tensor
        a_t = Tensor(num1, requires_grad=True)
        b_t = Tensor(num2, requires_grad=True)
        c_t = a_t * b_t
        c_t.backward()
        
        # PyTorch's Tensor
        a_p = torch.tensor(num1, requires_grad=True)
        b_p = torch.tensor(num2, requires_grad=True)
        c_p = a_p * b_p
        c_p.backward()
        
        self.assertEqual(c_t.data, c_p.item())
        self.assertEqual(a_t.grad, a_p.grad.item())
        self.assertEqual(b_t.grad, b_p.grad.item())
        print("Scalar multiplication test passed!")

        if print_output:
            print("babytorch's Tensor: ", c_t.data, a_t.grad, b_t.grad)
            print("PyTorch's Tensor:  ", c_p.item(), a_p.grad.item(), b_p.grad.item())
            print()

    def test_vec_multiplication(self):
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [4.0, 5.0, 6.0]

        # babytorch's Tensor
        a_t = Tensor(vec1, requires_grad=True)
        b_t = Tensor(vec2, requires_grad=True)
        c_t = a_t * b_t
        c_t.sum().backward()

        # PyTorch's Tensor
        a_p = torch.tensor(vec1, requires_grad=True)
        b_p = torch.tensor(vec2, requires_grad=True)
        c_p = a_p * b_p
        c_p.sum().backward()

        self.assertTrue(cp.array_equal(c_t.data, c_p.detach().numpy()))
        self.assertTrue(cp.array_equal(a_t.grad, a_p.grad.numpy()))
        self.assertTrue(cp.array_equal(b_t.grad, b_p.grad.numpy()))
        print("Vector multiplication test passed!")

        if print_output:
            print("babytorch's Tensor: ", c_t.data, a_t.grad, b_t.grad)
            print("PyTorch's Tensor:  ", c_p.detach().numpy(), a_p.grad.numpy(), b_p.grad.numpy())
            print()

    def test_matrix_multiplication(self):
        matrix1 = [[1.0, 2.0], [3.0, 4.0]]
        matrix2 = [[5.0, 6.0], [7.0, 8.0]]

        # babytorch's Tensor
        a_t = Tensor(matrix1, requires_grad=True)
        b_t = Tensor(matrix2, requires_grad=True)
        c_t = a_t * b_t
        c_t.sum().backward()

        # PyTorch's Tensor
        a_p = torch.tensor(matrix1, requires_grad=True)
        b_p = torch.tensor(matrix2, requires_grad=True)
        c_p = a_p * b_p
        c_p.sum().backward()

        self.assertTrue(cp.allclose(c_t.data, c_p.detach().numpy()))
        self.assertTrue(cp.allclose(a_t.grad, a_p.grad.numpy()))
        self.assertTrue(cp.allclose(b_t.grad, b_p.grad.numpy()))
        print("Matrix multiplication test passed!")

        if print_output:
            print("babytorch's Tensor: ", c_t.data, a_t.grad, b_t.grad)
            print("PyTorch's Tensor:  ", c_p.detach().numpy(), a_p.grad.numpy(), b_p.grad.numpy())
            print()

    def test_matrix_matrix_diff_shape_multiplication(self):
        matrix1 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        matrix2 = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

        # babytorch's Tensor
        a_t = Tensor(matrix1, requires_grad=True)
        b_t = Tensor(matrix2, requires_grad=True)
        c_t = a_t @ b_t  # Using Python's matrix multiplication operator
        
        c_t.sum().backward()

        # PyTorch's Tensor
        a_p = torch.tensor(matrix1, requires_grad=True)
        b_p = torch.tensor(matrix2, requires_grad=True)
        c_p = torch.matmul(a_p, b_p)
        c_p.sum().backward()

        self.assertTrue(cp.allclose(c_t.data, c_p.detach().numpy()))
        self.assertTrue(cp.allclose(a_t.grad, a_p.grad.numpy()))
        self.assertTrue(cp.allclose(b_t.grad, b_p.grad.numpy()))
        print("Matrix (2x3) and Matrix (3x2) multiplication test passed!")

        if print_output:
            print("babytorch's Tensor: ", c_t.data, a_t.grad, b_t.grad)
            print("PyTorch's Tensor:  ", c_p.detach().numpy(), a_p.grad.numpy(), b_p.grad.numpy())
            print()

    def test_matrix_vector_diff_shape_multiplication(self):
        matrix = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        vector = [1.0, 2.0, 3.0]

        # babytorch's Tensor
        a_t = Tensor(matrix, requires_grad=True)
        b_t = Tensor(vector, requires_grad=True)
        c_t = a_t @ b_t
        c_t.sum().backward()

        # PyTorch's Tensor
        a_p = torch.tensor(matrix, requires_grad=True)
        b_p = torch.tensor(vector, requires_grad=True)
        c_p = torch.matmul(a_p, b_p)
        c_p.sum().backward()

        self.assertTrue(cp.array_equal(c_t.data, c_p.detach().numpy()))
        self.assertTrue(cp.array_equal(a_t.grad, a_p.grad.numpy()))
        self.assertTrue(cp.array_equal(b_t.grad, b_p.grad.numpy()))
        print("Matrix (2x3) and Vector (3x1) multiplication test passed!")

        if print_output:
            print("babytorch's Tensor: ", c_t.data, a_t.grad, b_t.grad)
            print("PyTorch's Tensor:  ", c_p.detach().numpy(), a_p.grad.numpy(), b_p.grad.numpy())
            print()

    def test_batched_matrix_multiplication_with_broadcasting(self):
        # babytorch's Tensor
        batched_matrix1 = [[[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]]]
        batched_matrix2 = [5.0, 6.0]

        a_t = Tensor(batched_matrix1, requires_grad=True)
        b_t = Tensor(batched_matrix2, requires_grad=True)
        c_t = a_t @ b_t
        c_t.sum().backward()
        
        # PyTorch's Tensor
        a_p = torch.tensor(batched_matrix1, requires_grad=True)
        b_p = torch.tensor(batched_matrix2, requires_grad=True)
        c_p = a_p @ b_p
        c_p.sum().backward()
        
        # Check equivalence
        self.assertTrue(cp.allclose(c_t.data, c_p.detach().numpy()))
        self.assertTrue(cp.allclose(a_t.grad, a_p.grad.numpy()))
        self.assertTrue(cp.allclose(b_t.grad, b_p.grad.numpy()))
        print("Batched matrix multiplication with broadcasting test passed!")

        if print_output:
            print(f" {c_t.data=},\n {a_t.grad=},\n {b_t.grad=}")
            print()
            print(f" {c_p.detach().numpy()=},\n {a_p.grad.numpy()=},\n {b_p.grad.numpy()=}")

if __name__ == '__main__':
    print_output = True
    unittest.main()
