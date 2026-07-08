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

        # Purely for inspection (see tutorials/llm/attention_viz.py): set
        # store_attention = True and forward() will keep its most recent
        # post-softmax weights, so you can *look at* what a head attends to.
        self.store_attention = False
        self.last_attention = None

    def forward(self, x, kv_cache=None):
        """x: (B, T, C).  With a ``kv_cache`` (generation only -- see
        ``GPT.generate``), ``x`` holds just the newest token(s); the keys
        and values of every earlier position come from the cache."""
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

        # KV cache: put the keys/values remembered from earlier calls in
        # front of this call's, then remember the extended arrays for the
        # next call.  Pure inference bookkeeping on raw arrays -- no
        # gradients flow through the cache.
        if kv_cache is not None:
            if kv_cache["k"] is not None:
                k = Tensor(xp.concatenate([kv_cache["k"], k.data], axis=2))
                v = Tensor(xp.concatenate([kv_cache["v"], v.data], axis=2))
            kv_cache["k"], kv_cache["v"] = k.data, v.data

        # Attention scores: how much should query i attend to key j?
        # (B, nh, T, hs) @ (B, nh, hs, T_total) -> (B, nh, T, T_total)
        # Scale by 1/sqrt(head_size) so the scores don't grow with head_size
        # (large scores would make softmax nearly one-hot and kill gradients).
        att = (q @ k.transpose((0, 1, 3, 2))) * (1.0 / math.sqrt(self.head_size))

        # Forbid attending to the future, then normalize into probabilities.
        # The T queries sit at the *last* T of the T_total key positions, so
        # take the matching rows of the mask.  Without a cache T_total == T
        # and this is simply mask[:T, :T].
        T_total = k.shape[2]
        att = att + self.mask[T_total - T:T_total, :T_total]
        att = att.softmax(axis=-1)
        if self.store_attention:
            self.last_attention = att.data    # (B, n_head, T, T_total)
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

    def forward(self, x, kv_cache=None):
        x = x + self.attn(self.ln1(x), kv_cache)   # communicate
        x = x + self.mlp(self.ln2(x))              # compute
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

    def forward(self, idx, kv_caches=None):
        """idx: integer array (B, T) of token ids -> logits (B, T, vocab_size).

        ``kv_caches`` (generation only): one cache per block, carrying the
        keys/values of already-processed positions, so ``idx`` needs to
        hold only the tokens that come after them.  See ``generate``.
        """
        if isinstance(idx, Tensor):
            idx = idx.data
        idx = xp.asarray(idx).astype(xp.int64)
        B, T = idx.shape

        # With a cache, this call's tokens sit *after* the cached positions,
        # so their position ids start where the cache ends.
        past = 0
        if kv_caches is not None and kv_caches[0]["k"] is not None:
            past = kv_caches[0]["k"].shape[2]
        assert past + T <= self.block_size, (
            f"sequence length {past + T} exceeds block size {self.block_size}")

        tok = self.token_embedding(idx)                       # (B, T, C)
        pos = self.position_embedding(xp.arange(past, past + T))  # (T, C)
        x = self.drop(tok + pos)                              # broadcast add

        for i, block in enumerate(self.blocks):
            x = block(x, kv_caches[i] if kv_caches is not None else None)
        x = self.ln_f(x)
        return self.head(x)                                   # (B, T, vocab_size)

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

    def empty_kv_caches(self):
        """A fresh key/value cache per block, for ``generate``'s fast path."""
        return [{"k": None, "v": None} for _ in self.blocks]

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                 use_cache=True):
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
        * ``use_cache`` enables the KV cache.  Naively, lap t re-runs the
          whole t-token context just to read one new row of logits.  The
          cache keeps every block's keys and values, so after the first
          lap the model forwards *only the newest token* and attends to
          the stored past: the same logits for a fraction of the work.
        """
        idx = xp.asarray(idx).astype(xp.int64)
        if idx.ndim == 1:
            idx = idx[None, :]

        self.eval()
        caches = self.empty_kv_caches() if use_cache else None
        with babytorch.no_grad():
            for _ in range(max_new_tokens):
                # Never use more than block_size tokens of context.
                idx_cond = idx[:, -self.block_size:]

                if caches is None:
                    # No cache: forward the whole context, every lap.
                    logits = self.forward(idx_cond).data
                elif caches[0]["k"] is None:
                    # First lap ("prefill"): run the whole prompt once and
                    # let every block store its keys and values.
                    logits = self.forward(idx_cond, caches).data
                elif caches[0]["k"].shape[2] < self.block_size:
                    # Steady state: the past is cached -- forward only the
                    # token sampled on the previous lap.
                    logits = self.forward(idx[:, -1:], caches).data
                else:
                    # The context window is full and now slides one step
                    # per token.  Our position embeddings are absolute, so
                    # every cached key belongs to a position id that just
                    # shifted -- the cache is stale; rebuild it.
                    caches = self.empty_kv_caches()
                    logits = self.forward(idx_cond, caches).data

                logits = logits[:, -1, :] / temperature   # last position: (B, vocab)

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
