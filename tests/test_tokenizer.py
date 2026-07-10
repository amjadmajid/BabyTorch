"""Tests for the character-level and BPE tokenizers."""

import pytest

from babytorch.text import CharTokenizer, BPETokenizer


def test_char_roundtrip():
    tok = CharTokenizer("hello world")
    text = "hello world"
    assert tok.decode(tok.encode(text)) == text


def test_char_vocab_is_unique_sorted():
    tok = CharTokenizer("banana")
    assert tok.chars == ['a', 'b', 'n']
    assert tok.vocab_size == 3


def test_char_save_load(tmp_path):
    tok = CharTokenizer("the quick brown fox")
    path = str(tmp_path / "vocab.json")
    tok.save(path)
    reloaded = CharTokenizer.load(path)
    assert reloaded.chars == tok.chars
    assert reloaded.encode("fox") == tok.encode("fox")


def test_bpe_learns_merges():
    text = "low low low lower lowest newest newest widest " * 20
    tok = BPETokenizer().fit(text, vocab_size=60)
    # merges were learned and the vocab grew beyond the base characters
    assert len(tok.merges) > 0
    assert tok.vocab_size <= 60


def test_bpe_roundtrip_words():
    text = "the cat sat on the mat the cat ran " * 30
    tok = BPETokenizer().fit(text, vocab_size=80)
    decoded = tok.decode(tok.encode("the cat"))
    assert "the" in decoded and "cat" in decoded


def test_bpe_save_load(tmp_path):
    text = "aaa bbb aaa bbb ccc aaa " * 20
    tok = BPETokenizer().fit(text, vocab_size=40)
    path = str(tmp_path / "bpe.json")
    tok.save(path)
    reloaded = BPETokenizer.load(path)
    assert reloaded.encode("aaa") == tok.encode("aaa")


def test_bpe_merge_matches_whole_symbols_not_substrings():
    sequences = {"aa b": 2, "a b": 3}
    merged = BPETokenizer._merge_pair(("a", "b"), sequences)
    assert merged["aa b"] == 2
    assert merged["ab"] == 3


def test_bpe_refit_discards_old_merge_rules():
    tok = BPETokenizer().fit("low low lower", vocab_size=20)
    old_merges = dict(tok.merges)
    tok.fit("cat cat car", vocab_size=20)
    assert tok.merges != old_merges
    assert tok.decode(tok.encode("cat")) == "cat"


def test_bpe_unknown_characters_raise_instead_of_disappearing():
    tok = BPETokenizer().fit("cat cat", vocab_size=20)
    with pytest.raises(ValueError, match="outside this BPE vocabulary"):
        tok.encode("cat!")
    assert tok.decode(tok.encode("cat!", errors="ignore")) == "cat"
