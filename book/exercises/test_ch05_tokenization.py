"""Grader for chapter 5."""

from collections import Counter

from grading import load, exercise

impl = load("ch05_tokenization")

CORPUS = "the cat sat on the mat"


@exercise
def test_vocabulary_layout():
    tok = impl.WordTokenizer(CORPUS)
    assert tok.vocab_size == 6, "5 unique words + <unk>"
    assert tok.encode("cat") == [1], \
        "<unk> takes id 0; real words are sorted and numbered from 1"
    assert tok.encode("the") == [5]


@exercise
def test_round_trip():
    tok = impl.WordTokenizer(CORPUS)
    assert tok.decode(tok.encode("the cat sat")) == "the cat sat"


@exercise
def test_unknown_words_become_unk():
    tok = impl.WordTokenizer(CORPUS)
    ids = tok.encode("the dog sat")
    assert ids[1] == 0, "an unseen word must map to id 0"
    assert tok.decode(ids) == "the <unk> sat", \
        "decoding id 0 yields the literal string '<unk>'"


@exercise
def test_most_frequent_pair_is_a_maximum():
    words = ["low", "lower", "lowest", "slow"]
    counts = Counter()
    for w in words:
        for a, b in zip(w[:-1], w[1:]):
            counts[(a, b)] += 1
    got = impl.most_frequent_pair(words)
    assert tuple(got) in counts, f"{got!r} never occurs in the corpus"
    assert counts[tuple(got)] == max(counts.values()), \
        "return a pair with the highest count"


@exercise
def test_most_frequent_pair_clear_winner():
    assert tuple(impl.most_frequent_pair(["aab", "aac", "aad"])) == ("a", "a")
