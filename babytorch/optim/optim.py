# class GD:
#     # Gradient Descent with optional L2 regularization (weight decay)
#     def __init__(self, params, learning_rate=0.001, weight_decay=0.0 ):
#         self.learning_rate = learning_rate
#         self.weight_decay = weight_decay
#         self.params = params

#     def update(self):
#         if self.weight_decay > 0:
#             self._update_with_weight_decay()
#         else:
#             self._update()

#     def _update(self):
#         for p in self.params:
#             p.data -= self.learning_rate * p.grad

#     def _update_with_weight_decay(self):
#         """L2 regularization (weight decay)"""
#         for p in self.params:
#             p.grad.flags.writeable = True
#             p.grad += self.weight_decay * p.data
#             p.data -= self.learning_rate * p.grad
            

class SGD:
    """
    Stochastic Gradient Descent optimizer with optional L2 regularization (weight decay).
    """
    def __init__(self, params, learning_rate=0.001, weight_decay=0.0):
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive.")
        
        if weight_decay < 0:
            raise ValueError("Weight decay cannot be negative.")
        
        if not hasattr(params, '__iter__'):
            raise ValueError("params must be an iterable collection.")

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.params = params

    def step(self):
        if self.weight_decay > 0:
            self._update_with_weight_decay()
        else:
            self._update()

    def _update(self):
        for p in self.params:
            if not hasattr(p, 'data') or not hasattr(p, 'grad'):
                raise AttributeError("Each parameter must have 'data' and 'grad' attributes.")
            
            if p.grad is not None:
                p.data -= self.learning_rate * p.grad

    def _update_with_weight_decay(self):
        for p in self.params:
            if not hasattr(p, 'data'):
                continue  # Skip if parameter does not have data attribute
            
            if p.grad is None:
                print(f"Warning: The gradient for parameter {p} is None. Skipping update.")
                continue  # Skip the update for this parameter if its gradient is None
                
            original_grad = p.grad.copy()  # Preserve the original gradient
            p.grad += self.weight_decay * p.data
            p.data -= self.learning_rate * p.grad
            p.grad = original_grad  # Restore the original gradient

