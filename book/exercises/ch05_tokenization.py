"""Build-it exercises for Chapter 5 -- Tokenization.

Pure Python, no tensors: tokenizers are string plumbing with big
consequences. Grade yourself:

    pytest book/exercises/test_ch05_tokenization.py -v

``babytorch/text/tokenizers.py`` is your reference for the API style.
"""

from collections import Counter


class WordTokenizer:
    """One token per WORD -- and a lesson about unknown words.

    The chapter says word-level tokenizers go blind on words they never
    saw. Feel that yourself by building one, with the standard escape
    hatch: a special ``<unk>`` token.

    Contract:

    * ``fit(text)`` -- vocabulary = ``"<unk>"`` with id **0**, then the
      unique whitespace-separated words of ``text``, sorted, with ids
      1, 2, 3, ... Returns ``self``.
    * ``vocab_size`` (property) -- number of tokens including ``<unk>``.
    * ``encode(text)`` -- list of ids; any word not in the vocabulary
      becomes id 0.
    * ``decode(ids)`` -- the words joined by single spaces (id 0 turns
      back into the literal string ``"<unk>"``).
    """

    def __init__(self, text=""):
        if text:
            self.fit(text)

    def fit(self, text):
        raise NotImplementedError("your code here")

    @property
    def vocab_size(self):
        raise NotImplementedError("your code here")

    def encode(self, text):
        raise NotImplementedError("your code here")

    def decode(self, ids):
        raise NotImplementedError("your code here")


def most_frequent_pair(words):
    """CHALLENGE (*): one turn of the BPE crank.

    ``words`` is a list of strings. Count every ADJACENT pair of
    characters inside each word -- e.g. "low" contributes ('l', 'o')
    and ('o', 'w') -- and return the most frequent pair as a tuple
    ``(a, b)``. Any winner is fine in case of a tie.

    This is steps 2-3 of the BPE algorithm in the chapter, isolated.
    (``collections.Counter`` is your friend; compare
    ``BPETokenizer._get_pair_counts`` afterwards.)
    """
    raise NotImplementedError("your code here")
