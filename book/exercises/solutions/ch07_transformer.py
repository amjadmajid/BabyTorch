"""Solutions for chapter 7. Struggle first -- that's where the learning is."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "..", "tutorials", "llm"))

from babytorch import Tensor
from babytorch.backend import xp
from model import GPT


def count_gpt_parameters(vocab_size, block_size, n_embd, n_head, n_layer):
    C, V = n_embd, vocab_size
    embeddings = V * C + block_size * C
    per_block = (
        (C * 3 * C + 3 * C)        # qkv Linear: weight + bias
        + (C * C + C)              # output projection
        + 2 * (2 * C)              # two LayerNorms, gamma + beta each
        + (C * 4 * C + 4 * C)      # MLP up
        + (4 * C * C + C)          # MLP down
    )
    final_norm = 2 * C
    head = C * V + V
    # n_head never appears: splitting C into heads reorganizes the same
    # numbers, it doesn't add any.
    return embeddings + n_layer * per_block + final_norm + head


class TiedGPT(GPT):
    def __init__(self, vocab_size, block_size, n_embd=64, n_head=4,
                 n_layer=4, dropout=0.0):
        super().__init__(vocab_size, block_size, n_embd, n_head,
                         n_layer, dropout)
        del self.head

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
        # The tied read-out: the same matrix that embedded the tokens,
        # transposed. .T is a tensor op, so backward() reaches the
        # embedding through BOTH of its jobs.
        return x @ self.token_embedding.weight.T
