"""Text corpora for language modelling."""

import os
import urllib.request

TINY_SHAKESPEARE_URL = ("https://raw.githubusercontent.com/karpathy/"
                        "char-rnn/master/data/tinyshakespeare/input.txt")


def load_tiny_shakespeare(root='./data', download=True):
    """Return the Tiny Shakespeare corpus as one big string (~1.1 MB).

    40,000 lines of Shakespeare's plays, concatenated -- the "hello world"
    dataset of character-level language modelling.  Downloaded once and
    cached in ``root``.
    """
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, 'tinyshakespeare.txt')

    if not os.path.exists(path):
        if not download:
            raise FileNotFoundError(
                f"{path} not found. Pass download=True to fetch it.")
        print(f"Downloading Tiny Shakespeare to {path} ...")
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, path)

    with open(path, 'r', encoding='utf-8') as f:
        return f.read()
