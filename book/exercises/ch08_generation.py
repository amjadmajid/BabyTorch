"""Build-it exercises for Chapter 8 -- Training a GPT.

Two pieces of the sampling toolbox that ``GPT.generate`` doesn't have.
Grade yourself:

    pytest book/exercises/test_ch08_generation.py -v
"""

import babytorch
from babytorch.backend import xp


def top_p_filter(probs, p):
    """Nucleus (top-p) sampling's core: keep the smallest set of tokens
    whose combined probability reaches ``p``; zero out the rest;
    renormalize.

    ``probs`` is a 1-D array of probabilities summing to 1 (raw ``xp``
    array, not a Tensor -- sampling happens outside the graph, exactly
    like in ``GPT.generate``). Return a same-shaped array where:

    * tokens are considered from most to least probable;
    * the kept set is the SMALLEST prefix (in that order) whose
      cumulative probability is >= ``p`` (always keep at least one);
    * kept tokens stay at their original positions, everything else
      is 0, and the result sums to 1 again.

    Where top-k keeps a fixed COUNT, top-p keeps a fixed amount of
    PROBABILITY -- adaptive: confident distributions keep few tokens,
    flat ones keep many. Hints: ``xp.argsort``, ``xp.cumsum``.
    """
    raise NotImplementedError("your code here")


def generate_greedy(model, ids, max_new_tokens):
    """CHALLENGE (*): greedy decoding -- generation with the dice removed.

    Mirror ``GPT.generate`` (tutorials/llm/model.py), but instead of
    sampling, always take the single most likely next token
    (``argmax``). Contract:

    * ``ids`` is a 1-D ``xp`` array of token ids; return a 1-D array of
      length ``len(ids) + max_new_tokens``;
    * feed the model at most ``model.block_size`` trailing tokens each
      lap (the model's forward expects a (B, T) batch -- use
      ``ids[None, :]``, and its output is a Tensor: take ``.data``);
    * run under ``babytorch.no_grad()`` -- no one will call backward.

    Try your finished function on a trained checkpoint and watch it
    loop: greedy text repeats itself. That experience is WHY chapter 8
    samples instead.
    """
    raise NotImplementedError("your code here")
