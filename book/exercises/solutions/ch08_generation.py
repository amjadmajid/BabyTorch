"""Solutions for chapter 8. Struggle first -- that's where the learning is."""

import babytorch
from babytorch.backend import xp


def top_p_filter(probs, p):
    order = xp.argsort(probs)[::-1]           # most probable first
    sorted_probs = probs[order]
    cumulative = xp.cumsum(sorted_probs)
    # keep the smallest prefix reaching p (the first index where the
    # running total is already >= p is still inside the kept set)
    cutoff = int(xp.searchsorted(cumulative, p)) + 1
    keep = order[:max(1, cutoff)]

    out = xp.zeros_like(probs)
    out[keep] = probs[keep]
    return out / out.sum()


def generate_greedy(model, ids, max_new_tokens):
    ids = xp.asarray(ids).astype(xp.int64)
    model.eval()
    with babytorch.no_grad():
        for _ in range(max_new_tokens):
            context = ids[None, -model.block_size:]      # (1, T), cropped
            logits = model.forward(context).data
            next_id = int(xp.argmax(logits[0, -1]))      # no dice: argmax
            ids = xp.concatenate([ids, xp.asarray([next_id])])
    return ids
