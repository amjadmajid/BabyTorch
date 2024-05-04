import numpy as np
from babytorch.engine import Tensor
import pickle

class Module:
    def zero_grad(self):
        """Zero out the gradients for all parameters."""
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    def parameters(self):
        """Return a list of parameters (Tensors) that are trainable."""
        return []

    def named_parameters(self):
        """Yield named parameters as a generator function."""
        return ((name, param) for name, param in self.__dict__.items() if isinstance(param, Tensor))

    def save(self, filename):
        """Save the model parameters to a file."""
        with open(filename, 'wb') as f:
            states = [p.get_state() for p in self.parameters()]
            pickle.dump(states, f)

    @staticmethod
    def load(filename, model):
        """Load model parameters from a file."""
        with open(filename, 'rb') as f:
            states = pickle.load(f)
        for p, state in zip(model.parameters(), states):
            p.set_state(state)
        return model

class ReLU(Module):
    def __call__(self, x):
        return x.relu()

    def __repr__(self):
        return "ReLU"

class Tanh(Module):
    def __call__(self, x):
        return x.tanh()

    def __repr__(self):
        return "Tanh"

class Sigmoid(Module):
    def __call__(self, x):
        return x.sigmoid()

    def __repr__(self):
        return "Sigmoid"

class Linear(Module):
    def __init__(self, in_features, out_features, activation_function=None):
        self.w = Tensor(np.random.uniform(-0.1, 0.1, (in_features, out_features)), requires_grad=True)
        self.b = Tensor(np.zeros((1, out_features)), requires_grad=True)
        self.activation_function = activation_function

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        # print("Linear forward:", x.shape, self.w.shape, self.b.shape)
        out = x.__matmul__(self.w) + self.b
        if self.activation_function:
            out = self.activation_function(out)
        return out

    def parameters(self):
        return [self.w, self.b]

    def __repr__(self):
        activation_str = f", activation={self.activation_function}" if self.activation_function else ""
        return f"Linear(in_features={self.w.data.shape[0]}, out_features={self.w.data.shape[1]}{activation_str})"


class Conv2D(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.w = Tensor(np.random.uniform(-0.1, 0.1, (out_channels, in_channels, kernel_size, kernel_size)), requires_grad=True)
        self.b = Tensor(np.zeros((out_channels,)), requires_grad=True)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        conv_result = x.conv2d(self.w, self.stride, self.padding).data
        for i in range(self.out_channels):
            conv_result[:, i, :, :] += self.b.data[i]
        return Tensor(conv_result, requires_grad=x.requires_grad)

    def parameters(self):
        return [self.w, self.b]

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return (f"Conv2D(in_channels={self.in_channels}, out_channels={self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})")
    
class Flatten(Module):
    def __call__(self, x):
        return x.flatten()

    def __repr__(self):
        return "Flatten"

class Sequential(Module):
    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"Sequential({', '.join(str(layer) for layer in self.layers)})"


# import numpy as np
# from babytorch.engine import Tensor
# import pickle

# class Module:
#     def zero_grad(self):
#         for p in self.parameters():
#             p.grad = np.zeros_like(p.data)  # Reset gradient using Tensor

#     def parameters(self):
#         return []
    
#     def get_parameters(self):
#         return ( [p for p in self.parameters()])


#     def save(self, filename):
#         """
#         Save the model parameters (data, grad) to a file using serialization.
        
#         Args:
#             filename (str): The name of the file to save to.
#         """
#         with open(filename, 'wb') as f:
#             # Save just the essential attributes of each parameter.
#             states = [p.get_state() for p in self.parameters()]
#             pickle.dump(states, f)
    
#     @staticmethod
#     def load(filename, model):
#         """
#         Load the model parameters (data, grad) from a file using deserialization.
        
#         Args:
#             filename (str): The name of the file to load from.
#             model (Module): The model instance to load parameters into.
            
#         Returns:
#             Module: The model with loaded parameters.
#         """
#         with open(filename, 'rb') as f:
#             states = pickle.load(f)
        
#         # Assign the loaded states back to the parameters.
#         for p, state in zip(model.parameters(), states):
#             p.set_state(state)

#         return model

# class ReLU(Module):
#     def __call__(self, x):
#         assert isinstance(x, Tensor), "ReLU's Input must be a Tensor!"
#         return x.relu()

#     def __repr__(self):
#         return "ReLU"

# class tanh(Module):
#     def __call__(self, x):
#         assert isinstance(x, Tensor), "tanh's Input must be a Tensor!"
#         return x.tanh()

#     def __repr__(self):
#         return "tanh"

# class Linear(Module):
#     def __init__(self, in_features, num_neurons, activation_function=None):
#         # self.w and self.b will need to be transposed in the forward pass
#         # There oder in the initialization will be reversed to achieve this
#         # without explicity transposition
#         self.w = Tensor(np.random.uniform(-.1, .1, (in_features, num_neurons)), requires_grad=True)  # Shape: num_neurons x in_features
#         self.b = Tensor(np.zeros((1, num_neurons)), requires_grad=True)  # Shape: num_neurons x 1
#         self.activation_function = activation_function

#     def __call__(self, x):
#         """
#         Forward pass through the linear layer.
        
#         Parameters:
#         - x (Tensor): Input tensor.
        
#         Returns:
#         - Tensor: Output tensor after linear transformation and activation.
#         """

#         # No need for transposition in this design
#         out = x.__matmul__(self.w) + self.b 
#         # print("out:", out, out.shape)
#         # exit()
        
#         # Ensure the output is 2D
#         if len(out.data.shape) == 1:
#             out = out.reshape(-1, 1)

#         if self.activation_function:
#             # print(self.activation_function)
#             out = self.activation_function(out)
        
#         # print("Output:", out, out.shape) 
#         # exit()
#         return out

#     def parameters(self):
#         return [self.w, self.b]

#     def __repr__(self):
#         activation_str = f", {self.activation_function}" if self.activation_function else ""
#         return f"Linear[{self.w.data.shape[0]}, {self.w.data.shape[1]}{activation_str}]"

# class Sequential(Module):
#     def __init__(self, *layers):
#         self.layers = layers

#     @staticmethod
#     def tensor_preprocessing(x):
#         assert isinstance(x, Tensor), "Input must be a Tensor!"
        
#         # Handle list and tuple input data inputs
#         if isinstance(x.data, (list, tuple, float, int)):
#             x.data = np.array(x.data)

#         # Convert 1D -> 2D column vector
#         if len(x.shape) == 1:
#             x = x.reshape(1, -1)
#         return x

#     def __call__(self, x):
#         # print(">>> X:", x, x.shape)
#         x = Sequential.tensor_preprocessing(x)
#         # print(">><> X:", x, x.shape)

#         for layer in self.layers:
#             # print(f"layer: {layer}")
#             # continue
#             x = layer(x)
#             # print("x:", x, x.shape, layer)
        
#         # exit()
#         return x

#     def parameters(self):
#         return [p for layer in self.layers for p in layer.parameters()]

#     def __repr__(self):
#         return f"Sequential({', '.join(str(layer) for layer in self.layers)})"
