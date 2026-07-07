"""Loss functions: how wrong is the model, as a single number?

A loss function compares the model's predictions with the true targets
and boils the mismatch down to one scalar.  That scalar is the tensor we
call ``.backward()`` on -- everything the model learns starts here.

Both losses below are built entirely out of Tensor operations, so their
backward passes come for free from the autodiff engine.
"""

from ..backend import xp
from ..engine import Tensor


class MSELoss:
    """Mean Squared Error -- the go-to loss for regression.

    ``loss = mean( (prediction - target)^2 )``

    Squaring does two jobs: errors in both directions count as positive,
    and large errors are punished much more than small ones.
    """

    def __call__(self, predictions, targets):
        return self.forward(predictions, targets)

    def forward(self, predictions, targets):
        assert isinstance(predictions, Tensor), "predictions must be a Tensor"
        if not isinstance(targets, Tensor):
            targets = Tensor(targets)
        diff = predictions - targets
        return (diff * diff).mean()


class CrossEntropyLoss:
    """Cross-entropy -- the go-to loss for classification (and LLMs!).

    The model outputs one raw score ("logit") per class.  Cross-entropy

    1. turns the scores into probabilities with softmax, then
    2. looks at the probability given to the *correct* class, and
    3. charges ``-log`` of it.

    ``-log(p)`` is tiny when the model was confident and right
    (p close to 1), and huge when it was confident and wrong (p close to 0).

    We compute steps 1-2 together with ``log_softmax`` (the log-sum-exp
    trick) instead of an actual softmax followed by ``log`` -- same math,
    but immune to overflow/underflow.

    Shapes: ``predictions`` is ``(batch, num_classes)`` and ``targets``
    holds one integer class id per row, e.g. ``[2, 0, 1, ...]``.
    A language model predicting the next token is exactly this with
    ``num_classes = vocabulary size``.
    """

    def __call__(self, predictions, targets):
        return self.forward(predictions, targets)

    def forward(self, predictions, targets):
        assert isinstance(predictions, Tensor), "predictions must be a Tensor"

        # Targets may arrive as a Tensor, a list, or an array; index arrays
        # must be integers.
        if isinstance(targets, Tensor):
            targets = targets.data
        targets = xp.asarray(targets).astype(xp.int64)
        assert targets.ndim == 1, (
            f"targets must be a 1-D array of class ids, got shape {targets.shape}")
        n = targets.shape[0]

        log_probs = predictions.log_softmax(axis=-1)      # (n, num_classes)
        # Pick out, for every row, the log-probability of its true class.
        picked = log_probs[xp.arange(n), targets]          # (n,)
        return -picked.mean()
