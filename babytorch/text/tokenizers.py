"""Tokenizers: turn text into integer ids and back.

A language model cannot read letters; it reads *token ids* -- small
integers that index into an embedding table.  A tokenizer defines the
vocabulary (the set of tokens) and the two-way mapping:

    text  --encode-->  [ids]  --model-->  [ids]  --decode-->  text

Two tokenizers are provided, from simplest to most realistic:

* :class:`CharTokenizer` -- one token per character.  Trivial to
  understand; the vocabulary is just the set of characters that appear.
* :class:`BPETokenizer`  -- a compact, word-boundary BPE for learning the
  merge algorithm. Production GPT tokenizers use a byte-level variant with
  additional normalization and pre-tokenization rules.
"""

import json
from collections import Counter


class CharTokenizer:
    """Character-level tokenizer: every distinct character is one token.

    Example::

        tok = CharTokenizer("hello world")
        tok.encode("low")   # -> [4, 6, 10]  (ids depend on the vocabulary)
        tok.decode([4, 6, 10])  # -> "low"
    """

    def __init__(self, text=""):
        if text:
            self.fit(text)
        else:
            self.chars = []
            self.stoi = {}
            self.itos = {}

    def fit(self, text):
        """Build the vocabulary from all characters seen in ``text``."""
        self.chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}   # string -> int
        self.itos = {i: ch for i, ch in enumerate(self.chars)}   # int -> string
        return self

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, text):
        """Text -> list of integer ids."""
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        """List of ids -> text."""
        return ''.join(self.itos[int(i)] for i in ids)

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'chars': self.chars}, f)

    @classmethod
    def load(cls, path):
        tok = cls()
        with open(path, 'r', encoding='utf-8') as f:
            tok.chars = json.load(f)['chars']
        tok.stoi = {ch: i for i, ch in enumerate(tok.chars)}
        tok.itos = {i: ch for i, ch in enumerate(tok.chars)}
        return tok


class BPETokenizer:
    """A minimal Byte Pair Encoding tokenizer (the GPT family's approach).

    Training, in plain words:

    1. Start with the vocabulary = all individual characters.
    2. Count every adjacent pair of tokens in the corpus.
    3. Merge the most frequent pair into a single new token.
    4. Repeat 2-3 until the vocabulary reaches the target size.

    Each merge is remembered as a rule; encoding new text just replays the
    rules in the order they were learned. This exposes the core merge
    algorithm, kept small and readable rather than production-complete or fast.
    """

    def __init__(self):
        self.merges = {}     # (token_a, token_b) -> merged_token  (rank order)
        self.vocab = {}      # token string -> id
        self.inverse_vocab = {}

    @staticmethod
    def _get_pair_counts(sequences):
        """Count adjacent token pairs across all word sequences."""
        counts = Counter()
        for seq, freq in sequences.items():
            symbols = seq.split()
            for a, b in zip(symbols[:-1], symbols[1:]):
                counts[(a, b)] += freq
        return counts

    @staticmethod
    def _merge_pair(pair, sequences):
        """Replace every occurrence of ``pair`` with the merged symbol."""
        merged = ''.join(pair)
        out = Counter()
        for seq, freq in sequences.items():
            symbols = seq.split()
            rewritten = []
            i = 0
            while i < len(symbols):
                if (i + 1 < len(symbols)
                        and symbols[i] == pair[0]
                        and symbols[i + 1] == pair[1]):
                    rewritten.append(merged)
                    i += 2
                else:
                    rewritten.append(symbols[i])
                    i += 1
            out[' '.join(rewritten)] += freq
        return out

    def fit(self, text, vocab_size=512, verbose=False):
        """Learn merge rules from ``text`` up to ``vocab_size`` tokens.

        A special end-of-word marker ``</w>`` is appended to each word so
        the model can tell where words end (and doesn't merge across
        spaces).
        """
        if not isinstance(vocab_size, int) or isinstance(vocab_size, bool):
            raise TypeError("vocab_size must be an integer.")
        self.merges = {}

        # Represent each unique word as space-separated characters + </w>.
        words = text.split()
        sequences = Counter(' '.join(list(w) + ['</w>']) for w in words)

        # Seed the vocabulary with the base characters.
        base = set()
        for seq in sequences:
            base.update(seq.split())
        vocab = sorted(base)
        if vocab_size < len(vocab):
            raise ValueError(
                f"vocab_size={vocab_size} is smaller than the base vocabulary "
                f"of {len(vocab)} symbols.")

        while len(vocab) < vocab_size:
            pair_counts = self._get_pair_counts(sequences)
            if not pair_counts:
                break
            best = max(pair_counts, key=pair_counts.get)
            sequences = self._merge_pair(best, sequences)
            self.merges[best] = ''.join(best)
            merged = ''.join(best)
            if merged not in vocab:
                vocab.append(merged)
            if verbose and len(vocab) % 50 == 0:
                print(f"vocab size {len(vocab)}: merged {best}")

        self.vocab = {tok: i for i, tok in enumerate(vocab)}
        self.inverse_vocab = {i: tok for tok, i in self.vocab.items()}
        return self

    @property
    def vocab_size(self):
        return len(self.vocab)

    def _tokenize_word(self, word):
        """Apply the learned merges to one word, return its sub-tokens."""
        symbols = list(word) + ['</w>']
        # Replay merges in the order they were learned (dict preserves it).
        for pair, merged in self.merges.items():
            i = 0
            while i < len(symbols) - 1:
                if symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                    symbols[i:i + 2] = [merged]
                else:
                    i += 1
        return symbols

    def encode(self, text, errors="strict"):
        """Text -> list of ids.

        Because this teaching tokenizer starts from corpus characters rather
        than all 256 bytes, new characters may be out of vocabulary. The
        default ``errors="strict"`` raises instead of silently deleting text;
        pass ``errors="ignore"`` only when that loss is intentional.
        """
        if errors not in ("strict", "ignore"):
            raise ValueError("errors must be 'strict' or 'ignore'.")
        ids = []
        for word in text.split():
            for tok in self._tokenize_word(word):
                if tok in self.vocab:
                    ids.append(self.vocab[tok])
                elif errors == "strict":
                    raise ValueError(
                        f"Token {tok!r} is outside this BPE vocabulary; fit on "
                        "representative text or call encode(..., errors='ignore').")
        return ids

    def decode(self, ids):
        """List of ids -> text (the </w> marker becomes a space)."""
        tokens = [self.inverse_vocab[int(i)] for i in ids]
        return ''.join(tokens).replace('</w>', ' ').strip()

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'merges': [[list(k), v] for k, v in self.merges.items()],
                'vocab': self.vocab,
            }, f)

    @classmethod
    def load(cls, path):
        tok = cls()
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        tok.merges = {tuple(k): v for k, v in data['merges']}
        tok.vocab = data['vocab']
        tok.inverse_vocab = {i: t for t, i in tok.vocab.items()}
        return tok
