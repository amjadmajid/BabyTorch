"""Solutions for chapter 5. Struggle first -- that's where the learning is."""

from collections import Counter


class WordTokenizer:
    UNK = "<unk>"

    def __init__(self, text=""):
        if text:
            self.fit(text)

    def fit(self, text):
        self.words = [self.UNK] + sorted(set(text.split()))
        self.stoi = {w: i for i, w in enumerate(self.words)}
        self.itos = {i: w for i, w in enumerate(self.words)}
        return self

    @property
    def vocab_size(self):
        return len(self.words)

    def encode(self, text):
        # dict.get with a default: the entire <unk> mechanism.
        return [self.stoi.get(w, 0) for w in text.split()]

    def decode(self, ids):
        return " ".join(self.itos[int(i)] for i in ids)


def most_frequent_pair(words):
    counts = Counter()
    for w in words:
        for a, b in zip(w[:-1], w[1:]):
            counts[(a, b)] += 1
    return max(counts, key=counts.get)
