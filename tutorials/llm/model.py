"""BabyGPT: a tiny GPT-style language model built with BabyTorch.

This is a complete, decoder-only Transformer -- the same architecture (in
miniature) behind GPT-2, GPT-4, LLaMA and friends.  It reads a sequence of
token ids and predicts the next token at every position.

The pieces, from the inside out::

    CausalSelfAttention   tokens look at earlier tokens and mix information
    MLP                   a per-token 2-layer network that "thinks"
    Block                 LayerNorm + Attention + MLP, wired with residuals
    GPT                   embeddings + a stack of Blocks + an output head

Everything is written with ordinary BabyTorch tensor operations, so the
autograd engine differentiates the whole thing automatically -- there is
no special "transformer gradient" code anywhere.

The file is meant to be *read*.  Follow the shapes in the comments (B =
batch, T = time/sequence length, C = channels/embedding size) and the data
flow will tell you the whole story.
"""

import math

import babytorch
import babytorch.nn as nn
from babytorch import Tensor
from babytorch.backend import xp


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal mask.

    *Self-attention* lets every position build a query ("what am I looking
    for?"), a key ("what do I offer?") and a value ("what will I pass on?").
    Each position compares its query against all keys to decide how much to
    attend to every other position, then reads out a weighted mix of values.

    *Causal* means a position may only attend to itself and earlier
    positions -- never the future.  That is what makes the model a valid
    next-token predictor: position t must not peek at token t+1.

    *Multi-head* means we do all of the above in ``n_head`` independent
    subspaces at once, letting different heads specialize (one might track
    syntax, another long-range references), then concatenate the results.
    """

    def __init__(self, n_embd, n_head, block_size, dropout=0.0):
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        self.head_size = n_embd // n_head

        # One linear layer produces query, key and value together (3 * n_embd),
        # which we split apart after -- cheaper than three separate layers.
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)  # mixes the heads back together
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # A constant (T, T) matrix: 0 where attention is allowed, a large
        # negative number where it is forbidden (the future).  Added to the
        # scores before softmax, the -1e9 entries become ~0 probability.
        mask = xp.triu(xp.full((block_size, block_size), -1e9), k=1)
        self.mask = Tensor(mask)  # not a parameter: requires_grad is False

    def forward(self, x):
        B, T, C = x.shape

        # Project to queries, keys, values, then split: each is (B, T, C).
        qkv = self.qkv(x)
        q = qkv[:, :, :C]
        k = qkv[:, :, C:2 * C]
        v = qkv[:, :, 2 * C:]

        # Split the C channels into n_head separate heads and move the head
        # axis next to the batch axis: (B, T, C) -> (B, n_head, T, head_size).
        q = q.reshape(B, T, self.n_head, self.head_size).transpose((0, 2, 1, 3))
        k = k.reshape(B, T, self.n_head, self.head_size).transpose((0, 2, 1, 3))
        v = v.reshape(B, T, self.n_head, self.head_size).transpose((0, 2, 1, 3))

        # Attention scores: how much should query i attend to key j?
        # (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)
        # Scale by 1/sqrt(head_size) so the scores don't grow with head_size
        # (large scores would make softmax nearly one-hot and kill gradients).
        att = (q @ k.transpose((0, 1, 3, 2))) * (1.0 / math.sqrt(self.head_size))

        # Forbid attending to the future, then normalize into probabilities.
        att = att + self.mask[:T, :T]
        att = att.softmax(axis=-1)
        att = self.attn_dropout(att)

        # Use the weights to read a mix of values, then reassemble the heads.
        y = att @ v                                   # (B, nh, T, hs)
        y = y.transpose((0, 2, 1, 3)).reshape(B, T, C)  # back to (B, T, C)

        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    """The per-position feed-forward network ("what to think about a token").

    After attention has *gathered* information from other positions, this
    little 2-layer network *processes* it, independently at each position.
    It widens to ``4 * n_embd`` (the standard expansion factor), applies a
    GELU non-linearity, then projects back down.
    """

    def __init__(self, n_embd, dropout=0.0):
        self.fc = nn.Linear(n_embd, 4 * n_embd)
        self.gelu = nn.GELU()
        self.proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.proj(self.gelu(self.fc(x))))


class Block(nn.Module):
    """One Transformer block: communicate (attention), then compute (MLP).

    Two design details make deep stacks trainable:

    * **Residual connections** (``x = x + sublayer(x)``): the input is added
      back to each sublayer's output, so gradients have a direct highway to
      the early layers and the block only has to learn a *change*.
    * **Pre-LayerNorm** (normalize *before* each sublayer): keeps the scale
      of activations steady all the way down the stack.
    """

    def __init__(self, n_embd, n_head, block_size, dropout=0.0):
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # communicate
        x = x + self.mlp(self.ln2(x))    # compute
        return x


class GPT(nn.Module):
    """The full model: turn a sequence of token ids into next-token logits.

    Forward pass:

    1. Look up a vector for each token (token embedding) and for each
       position (position embedding) and add them -- now every token knows
       *what* it is and *where* it is.
    2. Run the sum through a stack of Transformer blocks.
    3. A final LayerNorm and a linear "head" produce, for every position, a
       score for every token in the vocabulary -- the prediction of what
       comes next.
    """

    def __init__(self, vocab_size, block_size, n_embd=64, n_head=4,
                 n_layer=4, dropout=0.0):
        self.block_size = block_size
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = [Block(n_embd, n_head, block_size, dropout)
                       for _ in range(n_layer)]
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        """idx: integer array (B, T) of token ids -> logits (B, T, vocab_size)."""
        if isinstance(idx, Tensor):
            idx = idx.data
        idx = xp.asarray(idx).astype(xp.int64)
        B, T = idx.shape
        assert T <= self.block_size, (
            f"sequence length {T} exceeds block size {self.block_size}")

        tok = self.token_embedding(idx)                 # (B, T, C)
        pos = self.position_embedding(xp.arange(T))     # (T, C)
        x = self.drop(tok + pos)                        # broadcast add

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)                             # (B, T, vocab_size)

    def loss(self, idx, targets):
        """Cross-entropy between predictions and the shifted targets.

        ``targets`` is ``(B, T)``: for each position, the id of the token
        that actually came next.  We flatten batch and time together so the
        loss is one big classification over all positions at once.
        """
        logits = self.forward(idx)                      # (B, T, vocab)
        B, T, V = logits.shape
        logits = logits.reshape(B * T, V)
        if isinstance(targets, Tensor):
            targets = targets.data
        targets = xp.asarray(targets).astype(xp.int64).reshape(B * T)
        return nn.CrossEntropyLoss()(logits, targets)

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressively extend ``idx`` by ``max_new_tokens`` tokens.

        The loop is the essence of "inference" in a language model:

            forward the context -> look at the last position's logits ->
            turn them into probabilities -> sample one token ->
            append it -> repeat.

        * ``temperature`` reshapes the distribution before sampling: <1
          makes the model bolder (sharper, more repetitive), >1 makes it
          more random/creative.
        * ``top_k`` restricts sampling to the k most likely tokens, which
          cuts off the unlikely "tail" and keeps output coherent.
        """
        idx = xp.asarray(idx).astype(xp.int64)
        if idx.ndim == 1:
            idx = idx[None, :]

        self.eval()
        with babytorch.no_grad():
            for _ in range(max_new_tokens):
                # Never feed more than block_size tokens of context.
                idx_cond = idx[:, -self.block_size:]
                logits = self.forward(idx_cond).data      # (B, T, vocab)
                logits = logits[:, -1, :] / temperature   # last step: (B, vocab)

                if top_k is not None:
                    k = min(top_k, logits.shape[-1])
                    # Zero out everything below the k-th largest logit.
                    kth = xp.sort(logits, axis=-1)[:, -k][:, None]
                    logits = xp.where(logits < kth, -xp.inf, logits)

                # Softmax -> probabilities, then sample one token per row.
                probs = _softmax_np(logits)
                next_id = _sample(probs)                  # (B, 1)
                idx = xp.concatenate([idx, next_id], axis=1)
        return idx


# --- small numpy/cupy helpers for sampling (no autograd needed) -----------

def _softmax_np(logits):
    logits = logits - xp.max(logits, axis=-1, keepdims=True)
    exps = xp.exp(logits)
    return exps / xp.sum(exps, axis=-1, keepdims=True)


def _sample(probs):
    """Draw one index per row from a batch of probability vectors."""
    import numpy as np
    from babytorch.backend import to_numpy
    probs = to_numpy(probs)
    out = np.array([np.random.choice(len(p), p=p) for p in probs],
                   dtype=np.int64)
    return xp.asarray(out)[:, None]
