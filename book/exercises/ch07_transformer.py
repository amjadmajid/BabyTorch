"""Build-it exercises for Chapter 7 -- The Transformer.

First prove you know where every parameter lives; then perform real
model surgery. Grade yourself:

    pytest book/exercises/test_ch07_transformer.py -v
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "tutorials", "llm"))

import babytorch
import babytorch.nn as nn
from babytorch import Tensor
from babytorch.backend import xp
from model import GPT


def count_gpt_parameters(vocab_size, block_size, n_embd, n_head, n_layer):
    """Predict ``GPT(...).num_parameters()`` in closed form -- no model!

    Write the arithmetic yourself, component by component (chapter 7's
    "where the parameters live" table is the map; the layer definitions
    in babytorch/nn/nn.py tell you each piece's exact shapes -- don't
    forget biases, and don't forget LayerNorm has TWO vectors).

    Per block: qkv Linear, projection Linear, two LayerNorms, and the
    MLP's two Linears. Outside: two embeddings, the final LayerNorm,
    and the output head. Return a plain int.

    (``n_head`` doesn't appear in your formula? Notice that -- and make
    sure you understand why.)
    """
    raise NotImplementedError("your code here")


class TiedGPT(GPT):
    """CHALLENGE (*): weight tying, the classic GPT-2 trick.

    The token embedding (vocab, C) turns ids INTO vectors; the output
    head's weight (C, vocab) turns vectors back into id-scores. GPT-2
    makes them the SAME matrix, saving vocab*C parameters and often
    improving the model.

    Two steps:

    * ``__init__``: call ``super().__init__(...)``, then ``del
      self.head`` -- removing the attribute hides the head's weights
      from ``Module.parameters()``.
    * ``forward``: identical to ``GPT.forward`` except the last line
      computes logits from the token-embedding matrix, transposed
      (``.T`` is a tensor operation, so gradients still reach the
      embedding). No bias.

    Most of ``forward`` is copied for you; finish the last line.
    """

    def __init__(self, vocab_size, block_size, n_embd=64, n_head=4,
                 n_layer=4, dropout=0.0):
        super().__init__(vocab_size, block_size, n_embd, n_head,
                         n_layer, dropout)
        del self.head              # its job is taken over by the embedding

    def forward(self, idx):
        if isinstance(idx, Tensor):
            idx = idx.data
        idx = xp.asarray(idx).astype(xp.int64)
        B, T = idx.shape

        tok = self.token_embedding(idx)
        pos = self.position_embedding(xp.arange(T))
        x = self.drop(tok + pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        # your code here: logits from the tied embedding matrix
        raise NotImplementedError("your code here")
