#!/usr/bin/env python3
"""Preprocess a BabyTorch book chapter for pandoc -> PDF.

The book is GitHub-flavored Markdown with HTML ``<details>`` boxes that have no
equivalent in print. This rewrites those into plain Markdown pandoc renders
well, and drops the per-chapter navigation footer (redundant in one PDF). It
operates on a *copy* in the build directory -- the real ``book/*.md`` sources
are never touched, so the snippet drift-guard is unaffected.

Two box shapes are handled (their exact spec lives in
``tests/test_book_snippets.py``):

* "How it's implemented" source boxes -- the ``<summary>`` (with its ``<b>`` and
  ``<code>path</code>``) becomes a bold caption line, followed by the code
  block unchanged.
* exercise ``<summary>Answer</summary>`` boxes -- become a bold "Answer."
  callout followed by the answer prose.

Usage::

    python preprocess.py chapter.md > chapter.pre.md
"""

import re
import sys

DETAILS = re.compile(r"<details>\s*(.*?)\s*</details>", re.S)
SUMMARY = re.compile(r"<summary>(.*?)</summary>", re.S)


def _inline_html_to_md(s):
    """<b>x</b> -> **x**, <code>y</code> -> `y` (the only inline HTML used)."""
    s = re.sub(r"<b>(.*?)</b>", r"**\1**", s, flags=re.S)
    s = re.sub(r"<code>(.*?)</code>", r"`\1`", s, flags=re.S)
    return s


def _convert_details(match):
    block = match.group(1)
    sm = SUMMARY.search(block)
    if not sm:
        return block
    raw_summary = sm.group(1)
    body = block[sm.end():].strip("\n")
    summary_md = _inline_html_to_md(raw_summary).strip()

    if "<code>" in raw_summary:
        # Source box: the summary already carries its own bold via <b>, so
        # emit it as a caption line (not a heading -- keep it out of the TOC),
        # then the verbatim code body.
        return f"\n{summary_md}\n\n{body}\n"
    # Answer box: bolded label + the answer prose.
    return f"\n**{summary_md}.**\n\n{body}\n"


def _strip_nav_footer(text):
    """Drop the trailing nav line (``… | [Contents](README.md) | …``)."""
    lines = text.rstrip("\n").split("\n")
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip():
            if "[Contents](README.md)" in lines[i]:
                del lines[i]
                # also drop a now-dangling separator rule right above it
                while lines and not lines[-1].strip():
                    lines.pop()
                if lines and lines[-1].strip() == "---":
                    lines.pop()
            break
    return "\n".join(lines).rstrip("\n") + "\n"


def main():
    with open(sys.argv[1], encoding="utf-8") as f:
        text = f.read()
    text = DETAILS.sub(_convert_details, text)
    text = _strip_nav_footer(text)
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
