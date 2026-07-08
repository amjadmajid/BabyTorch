# Building the book as a PDF

The book renders to PDF in two editions:

| Edition | Command | Output | Toolchain |
|---------|---------|--------|-----------|
| English | `./build.sh en` | `build/book-en.pdf` | pandoc + XeLaTeX |
| Arabic (RTL) | `./build.sh ar` | `build/book-ar.pdf` | Markdown → HTML → WeasyPrint |

Both read the same `book/*.md` (and `book/ar/*.md`) sources — nothing is
duplicated. Outputs land in `book/build/` (gitignored). The Arabic edition
reuses the English figures (English labels); its RTL text and left-to-right
code are handled by the layout engine, not by hand.

## Prerequisites

**English** (`./build.sh en`):
- `pandoc` — `brew install pandoc`
- A LaTeX distribution with **xelatex** — MacTeX / TeX Live
- `rsvg-convert` (to embed the SVG figures) — `brew install librsvg`

**Arabic** (`./build.sh ar`):
- Python 3 with **weasyprint** and **markdown** — `pip install weasyprint markdown`
  (WeasyPrint's native deps — pango, cairo, harfbuzz — come via
  `brew install pango`, usually already present.)
- A book-grade Arabic font — `brew install --cask font-amiri`
  (Noto Naskh Arabic is a fine alternative; the macOS system fonts also work).

If WeasyPrint lives in a virtualenv, point the build at it:

```bash
PYTHON=/path/to/venv/bin/python ./build.sh ar
```

## Build

```bash
cd book
./build.sh en      # -> build/book-en.pdf
./build.sh ar      # -> build/book-ar.pdf
```

## How it fits together

- **`build.sh`** — the entry point; dispatches to the right toolchain per edition.
- **`pandoc/preprocess.py`** — rewrites the `<details>` boxes and drops the
  per-chapter nav footer for the English (pandoc) build. Operates on copies in
  `build/`, never the sources.
- **`pandoc/preamble.tex`, `pandoc/metadata-en.yaml`** — LaTeX preamble and
  metadata for the English edition.
- **`htmlpdf.py`** — the Arabic (and any RTL) build: Markdown → HTML with a
  print stylesheet (`direction: rtl`, code kept `direction: ltr`), then
  WeasyPrint. Title page, table styling, figure embedding, page numbers.

The code shown in the book stays honest: `tests/test_book_snippets.py` checks
every `<details>` source box — in **both** the English and Arabic editions —
against the real source files, character for character.
