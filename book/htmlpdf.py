#!/usr/bin/env python3
"""Render the BabyTorch book to PDF via Markdown -> HTML -> WeasyPrint.

Used for the Arabic (RTL) edition, where a browser-grade layout engine handles
right-to-left text, left-to-right code islands, and SVG figures far more
robustly than the LaTeX bidi stack. RTL is a few lines of CSS here:

    body { direction: rtl }
    pre, code { direction: ltr; unicode-bidi: isolate }   # code stays LTR

Usage:  python htmlpdf.py <src-dir> <out.pdf> [--rtl] [--title T] [--subtitle S]

The <details> boxes (which don't print) are rewritten to always-visible styled
callouts; the code inside them is untouched. Figures resolve relative to
<src-dir>. Requires: weasyprint, markdown  (pip install weasyprint markdown).
"""

import argparse
import os
import re
import sys

import markdown
from weasyprint import HTML

DETAILS = re.compile(r"<details>\s*(.*?)\s*</details>", re.S)
SUMMARY = re.compile(r"<summary>(.*?)</summary>", re.S)
NAV = re.compile(r"^.*\[.*?\]\(README\.md\).*$", re.M)   # per-chapter nav footer


def _inline_html_to_md(s):
    s = re.sub(r"<b>(.*?)</b>", r"**\1**", s, flags=re.S)
    s = re.sub(r"<code>(.*?)</code>", r"`\1`", s, flags=re.S)
    return s


def _boxes(md_text):
    """Rewrite <details> source/answer boxes into markdown="1" divs."""
    def repl(m):
        block = m.group(1)
        sm = SUMMARY.search(block)
        if not sm:
            return m.group(0)
        raw_summary = sm.group(1)
        body = block[sm.end():].strip("\n")
        title = _inline_html_to_md(raw_summary).strip()
        if "<code>" in raw_summary:          # source box (title already bold)
            cls, heading = "sourcebox", title
        else:                                # answer box
            cls, heading = "answerbox", f"**{title}**"
        return (f'\n<div class="{cls}" markdown="1">\n\n'
                f'{heading}\n\n{body}\n\n</div>\n')
    return DETAILS.sub(repl, md_text)


CSS = """
@page {{ size: A4; margin: 2cm 2.2cm; }}
@page {{ @bottom-center {{ content: counter(page); color:#999; font-size:9pt; }} }}
html {{ font-size: 12pt; }}
body {{ direction: {dir}; font-family: 'Amiri','Noto Naskh Arabic',serif;
        line-height: 1.75; color:#111; text-align: {align}; }}
h1,h2,h3,h4 {{ font-family:'Amiri','Noto Naskh Arabic',serif; line-height:1.35;
               color:#111; }}
h1 {{ font-size: 1.9rem; margin-top: 0; page-break-before: always; }}
h1:first-of-type {{ page-break-before: avoid; }}
h2 {{ font-size: 1.35rem; border-bottom:1px solid #eee; padding-bottom:2px;
      margin-top: 1.4em; }}
h3 {{ font-size: 1.1rem; }}
/* Code stays left-to-right inside RTL text. */
pre, code, kbd, samp {{ direction: ltr; unicode-bidi: isolate;
    font-family:'Menlo','DejaVu Sans Mono',monospace; }}
pre {{ background:#f6f8fa; border:1px solid #eee; border-radius:6px;
       padding:10px 12px; overflow-x:auto; font-size:9pt; line-height:1.45;
       text-align:left; white-space:pre-wrap; word-wrap:break-word; }}
code {{ background:#f0f1f2; padding:1px 4px; border-radius:3px; font-size:0.88em; }}
pre code {{ background:none; padding:0; }}
.sourcebox, .answerbox {{ border:1px solid #e2e2e2; border-radius:8px;
    padding:2px 14px; margin:12px 0; background:#fafafa;
    break-inside: avoid-page; }}
.answerbox {{ background:#f2f7ff; border-color:#dbe6fb; }}
img {{ max-width:100%; height:auto; display:block; margin:10px auto; }}
a {{ color:#2a5db0; text-decoration:none; }}
table {{ border-collapse:collapse; margin:10px 0; font-size:0.95em; }}
th,td {{ border:1px solid #ccc; padding:5px 9px; text-align:{align}; }}
th {{ background:#f4f4f4; }}
blockquote {{ border-inline-start:3px solid #ddd; margin-inline-start:0;
              padding-inline-start:12px; color:#444; }}
.title-page {{ text-align:center; margin-top:35%; page-break-after:always; }}
.title-page h1 {{ page-break-before: avoid; font-size:2.6rem; border:none; }}
.title-page .sub {{ color:#555; font-size:1.1rem; margin-top:0.5em; }}
.title-page .author {{ margin-top:2em; font-size:1.1rem; }}
.title-page .email {{ margin-top:0.3em; font-size:0.95rem; color:#666;
                      direction:ltr; }}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("srcdir")
    ap.add_argument("out")
    ap.add_argument("--rtl", action="store_true")
    ap.add_argument("--title", default="BabyTorch")
    ap.add_argument("--subtitle", default="")
    ap.add_argument("--author", default="")
    ap.add_argument("--email", default="")
    args = ap.parse_args()

    chapters = sorted(f for f in os.listdir(args.srcdir)
                      if re.match(r"\d\d-.*\.md$", f))
    md = markdown.Markdown(extensions=["fenced_code", "tables", "md_in_html",
                                       "attr_list", "sane_lists"])

    parts = [f'<div class="title-page"><h1>{args.title}</h1>'
             f'<div class="sub">{args.subtitle}</div>'
             f'<div class="author">{args.author}</div>'
             f'<div class="email">{args.email}</div></div>']
    for name in chapters:
        with open(os.path.join(args.srcdir, name), encoding="utf-8") as f:
            text = f.read()
        text = NAV.sub("", text)          # drop per-chapter nav footers
        text = _boxes(text)
        md.reset()
        parts.append(md.convert(text))

    direction = "rtl" if args.rtl else "ltr"
    align = "right" if args.rtl else "left"
    css = CSS.format(dir=direction, align=align)
    html = (f'<!doctype html><html><head><meta charset="utf-8">'
            f'<style>{css}</style></head><body>{"".join(parts)}</body></html>')

    # base_url = srcdir so figures/fig-*.svg resolve.
    HTML(string=html, base_url=args.srcdir + os.sep).write_pdf(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
