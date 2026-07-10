# Building the book as a PDF

The book renders to PDF in two editions:

| Edition | Command | Output | Toolchain |
|---------|---------|--------|-----------|
| English | `./build.sh en` | `output/pdf/BabyTorch-Book-English.pdf` | pandoc + XeLaTeX |
| Arabic (RTL) | `./build.sh ar` | `output/pdf/BabyTorch-Book-Arabic.pdf` | Markdown → HTML → WeasyPrint |

Both read the same `book/*.md` (and `book/ar/*.md`) sources — nothing is
duplicated. Final PDFs land in `book/output/pdf/` and temporary conversion
files in `book/build/` (both gitignored). The Arabic edition
reuses the English figures (English labels); its RTL text and left-to-right
code are handled by the layout engine, not by hand.

## Prerequisites

**English** (`./build.sh en`):
- `pandoc` — `brew install pandoc`
- A LaTeX distribution with **xelatex** — MacTeX / TeX Live
- `rsvg-convert` or Inkscape (to embed the SVG figures) —
  `brew install librsvg` or `brew install --cask inkscape`

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
./build.sh en      # -> output/pdf/BabyTorch-Book-English.pdf
./build.sh ar      # -> output/pdf/BabyTorch-Book-Arabic.pdf
```

## How it fits together

- **`build.sh`** — the entry point; dispatches to the right toolchain per edition.
- **`pandoc/preprocess.py`** — rewrites the `<details>` boxes and drops the
  per-chapter nav footer for the English build, and points figure links to the
  generated PDF artwork. Operates on copies in `build/`, never the sources.
- **`pandoc/preamble.tex`, `pandoc/metadata-en.yaml`** — LaTeX preamble and
  metadata for the 6×9-inch, two-sided English print edition. Part dividers,
  front matter, running heads, embedded fonts, and print margins are generated
  by the same reproducible build.
- **`htmlpdf.py`** — the Arabic (and any RTL) build: Markdown → HTML with a
  6×9-inch print stylesheet (`direction: rtl`, code kept `direction: ltr`),
  then WeasyPrint. It creates the title/copyright pages, grouped contents,
  part dividers, running heads, figure embedding, and page numbers.

## Print specification

- Trim size: **6 × 9 inches**, no bleed required (all artwork stays inside the
  text block).
- English interior: two-sided, chapters open on right-hand pages, mirrored
  inner/outer margins with binding allowance.
- Arabic interior: right-to-left text with isolated left-to-right code blocks.
- Fonts are embedded in both PDFs. For the best Arabic typography, install
  Amiri; the build falls back to an available Arabic-capable system font.

Before publishing a new edition, run `pytest -q`, build both languages, inspect
`pdfinfo`/`pdffonts`, and render representative pages with `pdftoppm`.

The code shown in the book stays honest: `tests/test_book_snippets.py` checks
every `<details>` source box — in **both** the English and Arabic editions —
against the real source files, character for character.
