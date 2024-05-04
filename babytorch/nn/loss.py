from babytorch.engine import Tensor

class MSELoss:
    def __call__(self, predictions, targets):
        return self.forward(predictions, targets)
    
    def forward(self, predictions, targets):
        assert isinstance(predictions, Tensor), "predictions must be a Tensor"
        assert isinstance(targets, Tensor), "targets must be a Tensor"

        diff = predictions - targets

        squared_diff = diff * diff
        # sum the squared differences and then divide by the number of elements to compute the mean
        loss = squared_diff.sum() / predictions.data.size

        return loss
    
    def backward(self, grad=None):
        # Assuming we call backward on the loss Tensor, this will compute gradients
        # We don't implement the backward pass here since our framework will handle that.
        pass


class CrossEntropyLoss:
    def __call__(self, predictions, labels):
        return self.forward(predictions, labels)

    def forward(self, predictions, labels):
        # Softmax function to convert raw scores to probabilities
        # using the max trick to avoid numerical overflow errors (to stablize the computation of the exponentials)
        
        labels_len = len(labels)

        # print("predictions: ", predictions)
        # exit()
        epsilon = Tensor(1e-8, requires_grad=True)

        max_vals = predictions.max(axis=1, keepdims=True)
        # print("max_vals: ", max_vals)
        exps = (predictions - max_vals).exp()
        # print("exps: ", exps)
        softmax = exps / exps.sum(axis=1, keepdims=True)

        # print("softmax: ", softmax)
        # exit()

        selected_probs = softmax[range(labels_len), labels.data] # we cannot index Tensor. This has to be fixed

        # TODO: This is a workaround. Tensor supports indexing with a list of indices, 
        # but not with a Tensor of indices. This must be fixed
        # selected_probs = Tensor(selected_probs, requires_grad=True)

        # print("selected_probs: ", selected_probs)
        # print("selected_probs: ", selected_probs + epsilon)
        # exit()
        # log_likelihood = (-selected_probs + epsilon).log()
        neg_tensor = Tensor(-1, requires_grad=True)
        log_likelihood = (selected_probs + epsilon).log() * neg_tensor
        # print("log_likelihood: ", neg_tensor * log_likelihood)
        # exit()

        # average the loss over the batch size (this is consistent with most deep learning frameworks)
        loss = log_likelihood.sum() / labels_len

        # print("loss: ", loss.operation.inputs()[0].operation.inputs()[0].operation.inputs()[0].operation.inputs()[0].operation )
        # exit()
        return loss

    def backward(self, grad=None):
        # Assuming we call backward on the loss Tensor, this will compute gradients
        # We don't implement the backward pass here since our framework will handle that.
        pass



