import numpy
from .operations import * 

NO_GRAD_CONTEXT = False

class no_grad:
    def __enter__(self):
        global NO_GRAD_CONTEXT
        self.previous_value = NO_GRAD_CONTEXT
        NO_GRAD_CONTEXT = True
        return self

    def __exit__(self, type, value, traceback):
        global NO_GRAD_CONTEXT
        NO_GRAD_CONTEXT = self.previous_value

class Tensor:
    def __init__(self, data, require_grad=False, dtype=numpy.float32, label="", _op_label=""):
        self.data = numpy.array(data, dtype=dtype)
        self.require_grad = require_grad
        self.grad = None
        self.operation = None
        self.label = label
        self._op_label = _op_label

    @property
    def shape(self):
        return self.data.shape

    def get_state(self):
            return {'data': self.data, 'grad': self.grad}
    
    def set_state(self, state):
        self.data = state['data']
        self.grad = state['grad']

    @staticmethod
    def to_tensor(np_array, require_grad=False):
        return Tensor(np_array, require_grad=require_grad)

    def _ensure_same_shape(self, other):
        assert self.data.shape == other.data.shape, "Shape mismatch. Both tensors should have the same shape."

    @property
    def T(self):
        return self.transpose()

    def reshape(self, *new_shape):
        reshape_op = ReshapeOperation(self, *new_shape)
        require_grad = self.require_grad and not NO_GRAD_CONTEXT
        output_tensor = Tensor(data=reshape_op.forward(), \
                               require_grad=require_grad, _op_label="reshape")
        if output_tensor.require_grad:
            output_tensor.operation = reshape_op
        return output_tensor

    def transpose(self):
        transpose_op = TransposeOperation()
        result = transpose_op.forward(self)
        output_tensor = Tensor(result, require_grad=self.require_grad and not NO_GRAD_CONTEXT, _op_label="T")
        if output_tensor.require_grad:
            output_tensor.operation = transpose_op
        return output_tensor
    
    def max(self, axis=None, keepdims=False):
        max_op = MaxOperation()
        result = max_op.forward(self, axis, keepdims)
        
        require_grad = self.require_grad and not NO_GRAD_CONTEXT
        output_tensor = Tensor(result, require_grad=require_grad, _op_label="max")
        
        if output_tensor.require_grad:
            output_tensor.operation = max_op
        
        return output_tensor

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        add_op = AddOperation()
        result = add_op.forward(self, other)
        require_grad = (self.require_grad or other.require_grad) and not NO_GRAD_CONTEXT
        output_tensor = Tensor(result, require_grad=require_grad, _op_label="+")
        if require_grad:
            output_tensor.operation = add_op
        return output_tensor
    
    def __radd__(self, other):
        return Tensor(other) + self

    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        subtract_op = SubOperation()
        result = subtract_op.forward(self, other)
        require_grad = (self.require_grad or other.require_grad) and not NO_GRAD_CONTEXT
        output_tensor = Tensor(result, require_grad=require_grad, _op_label="-")
        if require_grad:
            output_tensor.operation = subtract_op
        return output_tensor

    def __rsub__(self, other):
        return Tensor(other) - self
    
    def sum(self, axis=None, keepdims=False):
        sum_op = SumOperation()
        result = sum_op.forward(self, axis=axis, keepdims=keepdims)
        require_grad = self.require_grad and not NO_GRAD_CONTEXT
        output_tensor = Tensor(result, require_grad=require_grad, _op_label="sum")

        if require_grad:
            output_tensor.operation = sum_op

        return output_tensor

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        mul_op = MulOperation()
        result = mul_op.forward(self, other)
        require_grad = (self.require_grad or other.require_grad) and not NO_GRAD_CONTEXT
        output_tensor = Tensor(result, require_grad=require_grad, _op_label="*")
        if output_tensor.require_grad:
            output_tensor.operation = mul_op
        return output_tensor

    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        result_tensor = Tensor(numpy.empty_like(self.data))  
        matmul_op = MatMulOperation()
        result_tensor.data = matmul_op.forward(self, other)

        if (self.require_grad or other.require_grad) and  not NO_GRAD_CONTEXT:
            result_tensor.require_grad = True
            result_tensor.operation = matmul_op

        return result_tensor

    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        div_op = DivOperation()
        result = div_op.forward(self, other)
        require_grad = (self.require_grad or other.require_grad) and not NO_GRAD_CONTEXT
        output_tensor = Tensor(result, require_grad=require_grad, _op_label="/")
        if require_grad:
            output_tensor.operation = div_op

        return output_tensor

    def __rtruediv__(self, other):
        # Handle right division: other / self
        return Tensor(other) / self

    def relu(self, alpha=0.00):
        relu_op = RLeUOperation(alpha)
        result = relu_op.forward(self)
        output_tensor = Tensor(result, require_grad=self.require_grad and not NO_GRAD_CONTEXT, _op_label="relu")
        if output_tensor.require_grad:
            output_tensor.operation = relu_op
        return output_tensor

    def tanh(self):
        tanh_op = TanhOperation()
        result = tanh_op.forward(self)
        output_tensor = Tensor(result, require_grad=self.require_grad and not NO_GRAD_CONTEXT, _op_label="tanh")
        if output_tensor.require_grad:
            output_tensor.operation = tanh_op
        return output_tensor

    def sigmoid(self):
        sigmoid_op = SigmoidOperation()
        result = sigmoid_op.forward(self)
        output_tensor = Tensor(result, require_grad=self.require_grad and not NO_GRAD_CONTEXT, _op_label="sigmoid")
        if output_tensor.require_grad:
            output_tensor.operation = sigmoid_op
        return output_tensor

    def exp(self):
        exp_op = ExpOperation()
        result = exp_op.forward(self)
        output_tensor = Tensor(result, require_grad=self.require_grad and not NO_GRAD_CONTEXT, _op_label="exp")
        if output_tensor.require_grad:
            output_tensor.operation = exp_op
            exp_op.input = self
        return output_tensor

    def log(self):
        log_op = LogOperation()
        out = Tensor(data=log_op.forward(self), require_grad=self.require_grad, _op_label="log")
        if self.require_grad:
            out.require_grad = True
            out.operation = log_op
        return out

    def squeeze(self, axis=None):
        squeeze_op = SqueezeOperation()
        result = squeeze_op.forward(self, axis)
        output_tensor = Tensor(result, require_grad=self.require_grad and not NO_GRAD_CONTEXT)
        if output_tensor.require_grad:
            output_tensor.operation = squeeze_op
        return output_tensor

    def unsqueeze(self, axis=None):
        unsqueeze_op = UnsqueezeOperation()
        require_grad = self.require_grad and not NO_GRAD_CONTEXT
        output_tensor = Tensor(data=unsqueeze_op.forward(self, axis), \
                               require_grad=require_grad, _op_label="unsqueeze")
        if require_grad:
            output_tensor.operation = unsqueeze_op
        return output_tensor

    def conv2d(self, other, stride=1, padding=0):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        require_grad = self.require_grad or other.require_grad and not NO_GRAD_CONTEXT
        out_op = Conv2DOperationOptim(self, other, stride, padding)
        out = Tensor(data=out_op.forward(), require_grad=require_grad, _op_label="conv2d")
        if require_grad:
            out.require_grad = True
            out.operation = out_op
        return out
    
    def flatten(self):
        require_grad = self.require_grad and not NO_GRAD_CONTEXT
        out_op = FlattenOperation()
        out = Tensor(data=out_op.forward(self), require_grad=require_grad, _op_label="conv2d")
        if require_grad:
            out.require_grad = True
            out.operation = out_op
        return out
    
    def maxpool2d(self, kernel_size, stride):
        maxpool2d_op = MaxPool2DOperation(kernel_size, stride)
        require_grad = self.require_grad and not NO_GRAD_CONTEXT
        output_tensor = Tensor(data=maxpool2d_op.forward(self.data) , \
                               require_grad=require_grad, _op_label="maxpool2d")
        
        if require_grad:
            output_tensor.operation = maxpool2d_op

        return output_tensor

    def backward(self, grad=None):
        if self.grad is None:
            if grad is not None:
                assert numpy.isscalar(grad), "The gradient passed to backward() must be a scalar."
                self.grad = numpy.array(grad, dtype=self.data.dtype)
            else: 
                self.grad = numpy.ones_like(self.data)

        if not self.require_grad:
            return
        
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                if v.operation: 
                    for tensor in v.operation.inputs():
                        build_topo(tensor)
                topo.append(v)

        build_topo(self)

        for v in reversed(topo):
            if v.operation:
                grads = v.operation.backward(v.grad)
                if not isinstance(grads, tuple):
                    grads = (grads,)

                for tensor, tensor_grad in zip(v.operation.inputs(), grads):
                    if tensor.require_grad:     
                        if tensor.grad is None:
                            tensor.grad = tensor_grad
                        else:
                            if tensor.grad.flags.writeable:
                                if tensor.grad.shape != tensor_grad.shape:
                                        tensor_grad = tensor_grad.reshape(tensor.grad.shape)
                                tensor.grad += tensor_grad
                            else:
                                if tensor.grad.shape != tensor_grad.shape:
                                    tensor_grad = tensor_grad.reshape(tensor.grad.shape)
                                tensor.grad = tensor.grad + tensor_grad

    def __repr__(self):
      return f"Tensor( {str(self.data)},require_grad={self.require_grad})"

    def __iter__(self):
        # TODO: we are returning a numpy iterator here.
        return iter(self.data)  # Simply returns the iterator for the underlying numpy array.

    def __getitem__(self, indices):
        slice_op = SliceOperation(indices)
        result = slice_op.forward(self)
        output_tensor = Tensor(result, require_grad=self.require_grad and not NO_GRAD_CONTEXT, _op_label="slice")

        if output_tensor.require_grad:
            output_tensor.operation = slice_op

        return output_tensor

    def __setitem__(self, index, value):
        self.data[index] = value

    def __len__(self):
        # TODO if data is scalar this will not work!
        return len(self.data)

