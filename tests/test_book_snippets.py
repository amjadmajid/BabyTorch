"""The book's collapsible implementation snippets must match the source.

Every ``<details>`` block in ``book/*.md`` whose ``<summary>`` names a
file in ``<code>...</code>`` promises a *verbatim* excerpt of that file.
This test re-checks the promise, so the book cannot silently drift away
from the implementation it describes.

Inside a snippet, a line consisting solely of ``# ...`` marks an
elision: the pieces around it must each still match the source exactly.
"""

import glob
import os
import re

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DETAILS = re.compile(r"<details>(.*?)</details>", re.S)
SOURCE = re.compile(r"<summary>.*?<code>([^<]+)</code>.*?</summary>", re.S)
FENCE = re.compile(r"```python\n(.*?)```", re.S)
ELISION = re.compile(r"^[ \t]*# \.\.\.[ \t]*\n", re.M)


def _snippets():
    for md in sorted(glob.glob(os.path.join(REPO, "book", "*.md"))):
        with open(md, encoding="utf-8") as f:
            text = f.read()
        for block in DETAILS.findall(text):
            m = SOURCE.search(block)
            if not m:
                continue
            for i, code in enumerate(FENCE.findall(block)):
                yield os.path.basename(md), m.group(1), i, code


CASES = list(_snippets())


def test_the_book_has_snippet_boxes():
    assert len(CASES) >= 10, "expected the book to contain snippet boxes"


@pytest.mark.parametrize(
    "md,src,i,code", CASES,
    ids=[f"{m}:{s}:{i}" for m, s, i, _ in CASES])
def test_snippet_matches_source(md, src, i, code):
    path = os.path.join(REPO, src)
    assert os.path.exists(path), f"{md} names a file that does not exist: {src}"
    with open(path, encoding="utf-8") as f:
        source = f.read()

    for chunk in ELISION.split(code):
        chunk = chunk.strip("\n")
        if not chunk:
            continue
        assert chunk in source, (
            f"{md}: a snippet no longer matches {src} verbatim.\n"
            f"The stale chunk begins with:\n  {chunk.splitlines()[0]!r}\n"
            f"Update the book excerpt to match the current code.")
