
import cupy as cp 

class Operation:
    def forward(self):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError
    
    def inputs(self):
        attributes = []
        if hasattr(self, 'a'):
            attributes.append(self.a)
        if hasattr(self, 'b'):
            attributes.append(self.b)
        return tuple(attributes)

    @staticmethod
    def broadcast_compat(shape1, shape2):
        """Check if two shapes are broadcast-compatible. For that, 
        the dimensions of the two shapes must be equal or one of them must be 1.
        
        Args:
            shape1 (tuple): Shape of the first tensor.
            shape2 (tuple): Shape of the second tensor.
        
        Returns:
            bool: True if the shapes are broadcast-compatible, False otherwise.
        
        Examples:
            >>> Operation.broadcast_compat((2, 3), (3,))
            True
            >>> Operation.broadcast_compat((2, 3), (2, 3))
            True
            >>> Operation.broadcast_compat((2, 3), (2, 4))
            False
        """
        # find the max length of the two shapes
        max_len = max(len(shape1), len(shape2))
        # pad the shapes with ones so that they have the same length
        shape1_padded = (1,) * (max_len - len(shape1)) + shape1
        shape2_padded = (1,) * (max_len - len(shape2)) + shape2
        
        for dim_shape1, dim_shape2 in zip(shape1_padded, shape2_padded):
            if dim_shape1 != 1 and dim_shape2 != 1 and dim_shape1 != dim_shape2:
                return False
        return True

    @staticmethod
    def broadcasted_axes(tensor_shape, target_shape):
        """Find the axes along which broadcasting is needed.

        Args:
            tensor_shape (tuple): Shape of the tensor.
            target_shape (tuple): Shape of the output.

        Returns:
            list: List of axes along which broadcasting is needed.

        Examples:
            >>> Operation.broadcasted_axes((2, 3), (3,))
            [0]
            >>> Operation.broadcasted_axes((2, 3), (2, 3))
            []
            >>> Operation.broadcasted_axes((2, 3), (2, 4))
            ValueError: Shapes (2, 3) and (2, 4) are not broadcast-compatible.
        """

        # Check for compatibility first
        if not Operation.broadcast_compat(tensor_shape, target_shape):
            raise ValueError(f"Shapes {tensor_shape} and {target_shape} are not broadcast-compatible.")
        
        axes = []
        tensor_shape_padded = (1,) * (len(target_shape) - len(tensor_shape)) + tensor_shape

        for i, (dim_tensor, dim_out) in enumerate(zip(tensor_shape_padded, target_shape)):
            if dim_tensor != dim_out:
                axes.append(i)
        return axes

class TransposeOperation(Operation):
    def forward(self, a):
        """Transpose operation. 
        TODO: support transposing along specified axes.
        
        Args:
            a (Tensor): Input tensor.

        Returns:
            Tensor: numpy array.
        """
        self.a = a
        return cp.transpose(a.data)

    def backward(self, grad):
        return cp.transpose(grad),

class ReshapeOperation(Operation):
    """
    A class to perform the reshape operation on a tensor.

    This class inherits from the Operation class and overrides its forward and backward methods
    to implement reshaping a tensor.

    Attributes:
    -----------
    a : Tensor
        The tensor to be reshaped.
    new_shape : tuple
        The new shape to which the tensor should be reshaped.

    Methods:
    --------
    forward() -> cp.ndarray:
        Reshapes the tensor `a` to `new_shape`.
    backward(grad: cp.ndarray) -> cp.ndarray:
        Reshapes the incoming gradient to the original shape of tensor `a`.
    """

    def __init__(self, tensor, *new_shape):
        """
        Initializes a ReshapeOperation object with the tensor to be reshaped and the new shape.

        Parameters:
        -----------
        tensor : Tensor
            The tensor that needs to be reshaped.
        new_shape : tuple
            The new shape to which the tensor should be reshaped.

        Note:
        -----
        No checks for shape compatibility are performed during initialization.
        """
        self.a = tensor
        self.new_shape = new_shape

    def forward(self) -> cp.ndarray:
        """
        Reshapes the tensor to the new shape.

        Returns:
        --------
        cp.ndarray:
            The reshaped tensor.

        Raises:
        -------
        AssertionError:
            If the tensor's data is not a numpy array.
        """
        assert isinstance(self.a.data, cp.ndarray), "Input.data must be a numpy array!"
        return self.a.data.reshape(*self.new_shape)

    def backward(self, grad: cp.ndarray) -> cp.ndarray:
        """
        Reshapes the incoming gradient back to the original shape of the tensor.

        Parameters:
        -----------
        grad : cp.ndarray
            The incoming gradient.

        Returns:
        --------
        cp.ndarray:
            The gradient reshaped to the original shape of the tensor `a`.
        """
        return grad.reshape(self.a.shape)


class MaxOperation (Operation):
    def forward(self, tensor, axis=None, keepdims=True):
        self.a = tensor
        self.axis = axis
        self.keepdims = keepdims
        self.indices = cp.argmax(tensor.data, axis=axis, keepdims=keepdims)
        return cp.max(tensor.data, axis=axis, keepdims=keepdims)

    def backward(self, grad):
        output_grad = cp.zeros_like(self.a.data)
        
        if self.axis is not None:
            grad = cp.broadcast_to(grad, self.indices.shape)
        
        # Update output_grad at the positions where the maxima were located in the forward pass
        cp.put_along_axis(output_grad, self.indices, grad, axis=self.axis)

        return output_grad

class AddOperation(Operation):
    def forward(self, a, b):
        self.a = a
        self.b = b
        result = a.data + b.data
        # print(f"AddOperation forward: a.shape={a.shape}, b.shape={b.shape}, result.shape={result.shape}")
        return result

    def backward(self, grad):
        a_axes = Operation.broadcasted_axes(self.a.shape, grad.shape)
        b_axes = Operation.broadcasted_axes(self.b.shape, grad.shape)

        a_grad = grad
        b_grad = grad

        if a_axes:
            a_grad = cp.sum(a_grad, axis=tuple(a_axes), keepdims=True)
        if b_axes:
            b_grad = cp.sum(b_grad, axis=tuple(b_axes), keepdims=True)

        # print(f"AddOperation backward: grad.shape={grad.shape}, a_grad.shape={a_grad.shape}, b_grad.shape={b_grad.shape}")
        return a_grad, b_grad

class SubOperation(Operation):
    """Subtraction operation."""
    def forward(self, a, b):
        self.a = a
        self.b = b
        return a.data - b.data

    def backward(self, grad):
        
        a_axes = Operation.broadcasted_axes(self.a.shape, grad.shape)
        b_axes = Operation.broadcasted_axes(self.b.shape, grad.shape)

        a_grad = cp.sum(grad, axis=tuple(a_axes), keepdims=True) if a_axes else grad
        # Note the negative sign for b_grad since we are computing the gradient for subtraction
        b_grad = -cp.sum(grad, axis=tuple(b_axes), keepdims=True) if b_axes else -grad

        return a_grad, b_grad

class SumOperation(Operation):
    """Sum operation along specified axes."""    
    def forward(self, a, axis=None, keepdims=False):
        self.a = a
        self.axis = axis
        return cp.sum(self.a.data, axis=self.axis, keepdims=keepdims)

    def backward(self, grad):
        return cp.broadcast_to(grad, self.a.shape)

class MulOperation(Operation):
    def forward(self, a, b):
        self.a = a
        self.b = b
        return a.data * b.data

    def backward(self, grad):
        a_axes = Operation.broadcasted_axes(self.a.shape, grad.shape)
        b_axes = Operation.broadcasted_axes(self.b.shape, grad.shape)
        
        # For multiplication, gradient wrt to a is grad * b and wrt to b is grad * a
        a_grad = cp.sum(grad * self.b.data, axis=tuple(a_axes), keepdims=True) if a_axes else grad * self.b.data
        b_grad = cp.sum(grad * self.a.data, axis=tuple(b_axes), keepdims=True) if b_axes else grad * self.a.data
        
        return a_grad, b_grad

class MatMulOperation(Operation):
    """Matrix multiplication operation."""
    def forward(self, a, b):
        self.a = a
        self.b = b
        return cp.matmul(self.a, self.b)

    def backward(self, grad):
        # Vector-Matrix multiplication
        if len(self.a.shape) == 1 and len(self.b.shape) == 2:
            a_grad = cp.matmul(grad, self.b.data.T)
            b_grad = cp.outer(self.a.data, grad)
        # Matrix-Vector multiplication
        elif len(self.a.shape) == 2 and len(self.b.shape) == 1:
            a_grad = cp.outer(grad, self.b.data)
            b_grad = cp.matmul(self.a.data.T, grad)
        # Matrix-Matrix multiplication
        elif len(self.a.shape) == 2 and len(self.b.shape) == 2:
            a_grad = cp.matmul(grad, self.b.data.T)
            b_grad = cp.matmul(self.a.data.T, grad)
        # Batched Matrix-Vector multiplication
        elif len(self.a.shape) == 3 and len(self.b.shape) == 1:
            a_grad = cp.matmul(grad[:, :, None], self.b.data[None, :])
            b_grad = cp.sum(self.a.data * grad[:, None, :], axis=0)
        # Batched Matrix-Matrix multiplication
        else:
            a_grad = cp.matmul(grad, self.b.data.swapaxes(-1, -2))
            b_grad = cp.matmul(self.a.data.swapaxes(-1, -2), grad)
        return a_grad, b_grad

class DivOperation(Operation):
    def forward(self, a, b):
        assert not cp.any(b.data == 0), "Cannot divide by zero"
        self.a = a
        self.b = b
        return a.data / b.data

    def backward(self, grad):
        a_axes = Operation.broadcasted_axes(self.a.shape, grad.shape)
        b_axes = Operation.broadcasted_axes(self.b.shape, grad.shape)
        
        # Derivative wrt to a is 1/b and wrt to b is -a/b^2
        a_grad = cp.sum(grad / self.b.data, axis=tuple(a_axes), keepdims=True) if a_axes else grad / self.b.data
        b_grad = cp.sum(grad * (-self.a.data / (self.b.data ** 2)), axis=tuple(b_axes), keepdims=True) if b_axes else grad * (-self.a.data / (self.b.data ** 2))
        
        return a_grad, b_grad

class SqueezeOperation(Operation):
    def forward(self, a, axis=None):
        self.a = a
        self.axis = axis
        self.original_shape = a.shape
        return cp.squeeze(a.data, axis=axis)

    def backward(self, grad):
        return cp.reshape(grad, self.original_shape),

class UnsqueezeOperation(Operation):
    def forward(self, a, axis):
        self.a = a
        self.axis = axis
        return cp.expand_dims(a.data, axis=axis)

    def backward(self, grad):
        return cp.squeeze(grad, axis=self.axis)
    
class ExpOperation(Operation):
    def forward(self, a):
        self.a = a
        self.__out = cp.exp(a.data)
        return self.__out

    def backward(self, grad):
        grad_input = grad * self.__out
        return grad_input

class LogOperation(Operation):
    """Logarithm operation."""
    
    def forward(self, a):
        self.a = a
        epsilon = 1e-8  # A small constant to prevent log(0)
        return cp.log(self.a.data + epsilon)
    
    def backward(self, grad):
        epsilon = 1e-8  # A small constant to prevent division by zero
        a_grad = grad / (self.a.data + epsilon)
        
        # Handle broadcasting
        a_axes = Operation.broadcasted_axes(self.a.shape, grad.shape)
        if a_axes:
            a_grad = cp.sum(a_grad, axis=tuple(a_axes), keepdims=True)
        
        return a_grad.squeeze()

class RLeUOperation(Operation):
    def __init__(self, alpha=0.):
        self.alpha = alpha

    def forward(self, a):
        self.a = a
        self.activation = cp.where(a.data > 0, a.data, self.alpha * a.data)
        return self.activation

    def backward(self, grad):
        # Reshape the grad to match the shape of self.a.data
        grad = grad.reshape(self.a.shape)
        grad_input = grad * cp.where(self.a.data > 0, 1, self.alpha)
        return grad_input,

class TanhOperation(Operation):
    def forward(self, a):
        self.a = a
        self.activation = cp.tanh(a.data)
        return self.activation

    def backward(self, grad):
        grad_input = grad * (1.0 - self.activation**2)
        return grad_input,

class SigmoidOperation(Operation):
    def forward(self, a):
        self.a = a
        self.activation = 1 / (1 + cp.exp(-a.data))
        return self.activation
    
    def backward(self, grad):
        grad_input = grad * (self.activation * (1 - self.activation))
        return grad_input,

class SliceOperation(Operation):
    def __init__(self, indices):
        self.indices = indices

    def forward(self, a):
        self.a = a
        return a.data[self.indices]
    
    def backward(self, grad):
        result = cp.zeros_like(self.a.data)
        result[self.indices] = grad
        return result,

class Conv2DOperation(Operation):
    def __init__(self, a, w, stride, padding):
        self.a = a  # Input tensor
        self.w = w  # Weight tensor (i.e., filters)
        self.stride = stride
        self.padding = padding
    
    def forward(self):
        # Get shapes
        N, C, H, W = self.a.data.shape  # Batch size, channels, height, width
        F, C, K, K = self.w.data.shape  # Number of filters, channels, kernel size

        # Output size calculation
        H_out = (H + 2 * self.padding - K) // self.stride + 1
        W_out = (W + 2 * self.padding - K) // self.stride + 1

        # Initialize output
        output = cp.zeros((N, F, H_out, W_out))

        # Padding
        if self.padding > 0:
            padded_data = cp.pad(self.a.data, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')
        else:
            padded_data = self.a.data

        # Convolution
        for n in range(N):
            for f in range(F):
                for i in range(0, H_out):
                    for j in range(0, W_out):
                        i_start = i * self.stride
                        i_end = i_start + K
                        j_start = j * self.stride
                        j_end = j_start + K

                        output[n, f, i, j] = cp.sum(
                            padded_data[n, :, i_start:i_end, j_start:j_end] * self.w.data[f]
                        )

        return output

    
    def backward(self, grad_output):
        N, C, H, W = self.a.data.shape
        F, _, K, K = self.w.data.shape
        H_out, W_out = grad_output.shape[-2], grad_output.shape[-1]

        # Initialize gradients for input and kernel with zeros
        grad_input = cp.zeros_like(self.a.data)
        grad_kernel = cp.zeros_like(self.w.data)
        
        # Padding, if needed
        if self.padding > 0:
            grad_input = cp.pad(grad_input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')
        
        # Loop over each entry of grad_output to compute grad_input and grad_kernel
        for n in range(N):
            for f in range(F):
                for i in range(0, H_out * self.stride, self.stride):
                    for j in range(0, W_out * self.stride, self.stride):
                        grad_input[n, :, i:i + K, j:j + K] += grad_output[n, f, i // self.stride, j // self.stride] * self.w.data[f]
                        grad_kernel[f] += grad_output[n, f, i // self.stride, j // self.stride] * self.a.data[n, :, i:i + K, j:j + K]
        
        # Remove padding from grad_input, if needed
        if self.padding > 0:
            grad_input = grad_input[:, :, self.padding:-self.padding, self.padding:-self.padding]
        
        return grad_input, grad_kernel

import numpy as np

class Conv2DOperationOptim(Operation):
    def __init__(self, a, w, stride, padding):
        self.a = a  # Input tensor
        self.w = w  # Weight tensor (i.e., filters)
        self.stride = stride
        self.padding = padding

    def _im2col(self, x, K, stride):
        N, C, H, W = x.shape
        H_out = (H - K) // stride + 1
        W_out = (W - K) // stride + 1

        x_col = cp.zeros((N, C, K, K, H_out, W_out))

        for i in range(K):
            for j in range(K):
                i_start = i
                i_end = i_start + H_out * stride
                j_start = j
                j_end = j_start + W_out * stride

                x_col[:, :, i, j, :, :] = x[:, :, i_start:i_end:stride, j_start:j_end:stride]

        x_col = x_col.transpose(0, 4, 5, 1, 2, 3).reshape(N * H_out * W_out, -1)
        return x_col

    def _col2im(self, col, x_shape, K, stride):
        N, C, H, W = x_shape
        H_out = (H - K) // stride + 1
        W_out = (W - K) // stride + 1
        col = col.reshape(N, H_out, W_out, C, K, K).transpose(0, 3, 4, 5, 1, 2)

        img = cp.zeros((N, C, H + 2 * self.padding, W + 2 * self.padding))
        for i in range(K):
            for j in range(K):
                img[:, :, i:i + H_out * stride:stride, j:j + W_out * stride:stride] += col[:, :, i, j, :, :]

        if self.padding > 0:
            img = img[:, :, self.padding:-self.padding, self.padding:-self.padding]

        return img

    def forward(self):
        padded_data = cp.pad(self.a.data, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')
        x_col = self._im2col(padded_data, self.w.data.shape[2], self.stride)
        w_col = self.w.data.reshape(self.w.data.shape[0], -1)
        out = cp.dot(x_col, w_col.T)

        N, C, H, W = self.a.data.shape
        F, _, K, K = self.w.data.shape
        H_out = (H + 2 * self.padding - K) // self.stride + 1
        W_out = (W + 2 * self.padding - K) // self.stride + 1

        out = out.reshape(N, H_out, W_out, F).transpose(0, 3, 1, 2)
        return out

    def backward(self, grad_output):
        F, _, K, K = self.w.data.shape
        grad_output_col = grad_output.transpose(0, 2, 3, 1).reshape(-1, F)
        w_col = self.w.data.reshape(F, -1)

        grad_col = cp.dot(grad_output_col, w_col)
        grad_input = self._col2im(grad_col, self.a.data.shape, K, self.stride)

        x_col = self._im2col(cp.pad(self.a.data, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant'), K, self.stride)
        grad_kernel = cp.dot(grad_output_col.T, x_col).reshape(F, self.a.data.shape[1], K, K)

        return grad_input, grad_kernel


class FlattenOperation(Operation):
    def forward(self, a):
        self.a = a
        self.original_shape = a.shape
        return a.data.reshape((a.shape[0], -1))

    def backward(self, grad):
        return grad.reshape(self.original_shape),


class MaxPool2DOperation(Operation):
    def __init__(self, kernel_size, stride, padding):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, a):
        self.a = a
        N, C, H, W = a.data.shape

        # Calculate output dimensions
        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1

        output = cp.zeros((N, C, H_out, W_out))
        self.indices = cp.zeros((N, C, H_out, W_out), dtype=cp.int)

        # Perform max pooling
        for i in range(0, H - self.kernel_size + 1, self.stride):
            for j in range(0, W - self.kernel_size + 1, self.stride):
                i_out = i // self.stride
                j_out = j // self.stride

                patch = a.data[:, :, i:i+self.kernel_size, j:j+self.kernel_size]
                output[:, :, i_out, j_out] = cp.max(patch, axis=(2, 3))
                self.indices[:, :, i_out, j_out] = cp.argmax(patch.reshape(N, C, -1), axis=2)

        return output

    def backward(self, grad):
        N, C, H, W = self.a.data.shape
        H_out, W_out = grad.shape[-2], grad.shape[-1]

        grad_input = cp.zeros_like(self.a.data)

        for i in range(H_out):
            for j in range(W_out):
                i_start = i * self.stride
                i_end = i_start + self.kernel_size
                j_start = j * self.stride
                j_end = j_start + self.kernel_size

                grad_patch = grad[:, :, i, j]
                grad_patch = grad_patch.reshape(N, C, 1, 1)

                indices = self.indices[:, :, i, j]
                grad_input_patch = cp.zeros((N, C, self.kernel_size, self.kernel_size), dtype=grad.dtype)

                flat_indices = cp.arange(N * C) * self.kernel_size * self.kernel_size + indices.ravel()
                grad_input_patch.ravel()[flat_indices] = grad_patch.ravel()

                grad_input[:, :, i_start:i_end, j_start:j_end] += grad_input_patch

        return grad_input

