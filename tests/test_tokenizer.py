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
