#!/usr/bin/env bash
#
# Build the BabyTorch book as a PDF.
#
#   ./build.sh en     # English -> output/pdf/BabyTorch-Book-English.pdf
#   ./build.sh ar     # Arabic  -> output/pdf/BabyTorch-Book-Arabic.pdf
#
# English uses pandoc + XeLaTeX (best typography). Arabic uses a Markdown ->
# HTML -> WeasyPrint path (htmlpdf.py): a browser-grade engine handles RTL
# text, left-to-right code islands, and SVG figures far more robustly than the
# LaTeX bidi stack. Override the interpreter with PYTHON=... for the Arabic
# build (it needs: pip install weasyprint markdown).
#
# Requires: pandoc, xelatex (TeX Live), rsvg-convert  [English];
#           python3 with weasyprint + markdown         [Arabic].
# Written for macOS's bash 3.2.

set -euo pipefail

lang="${1:-en}"
here="$(cd "$(dirname "$0")" && pwd)"
build="$here/build"
output="$here/output/pdf"
mkdir -p "$build"
mkdir -p "$output"
mkdir -p "$build/home" "$build/config" "$build/cache"

# --- Arabic (RTL): Markdown -> HTML -> WeasyPrint ---------------------------
if [ "$lang" = "ar" ]; then
  PY="${PYTHON:-python3}"
  out="$output/BabyTorch-Book-Arabic.pdf"
  echo "weasyprint: building $out ..."
  "$PY" "$here/htmlpdf.py" "$here/ar" "$out" --rtl \
    --title "BabyTorch" \
    --subtitle "كيف تعمل أُطُر التعلّم العميق — وكيف يُبنى نموذج GPT" \
    --author "د. امجد يوسف مجيد" \
    --email "amjad@slimx.ai"
  exit 0
fi

# --- English: pandoc + XeLaTeX ---------------------------------------------
if [ "$lang" != "en" ]; then
  echo "usage: $0 en|ar" >&2
  exit 1
fi

srcdir="$here"
meta="$here/pandoc/metadata-en.yaml"
out="$output/BabyTorch-Book-English.pdf"
mkdir -p "$build/en"

# XeLaTeX cannot embed SVG directly. Convert figures with either common tool;
# fail clearly instead of silently producing a book with missing diagrams.
mkdir -p "$build/figures"
for svg in "$srcdir"/figures/*.svg; do
  name="$(basename "$svg" .svg)"
  pdf="$build/figures/$name.pdf"
  if command -v rsvg-convert >/dev/null 2>&1; then
    rsvg-convert -f pdf -o "$pdf" "$svg"
  elif command -v inkscape >/dev/null 2>&1; then
    HOME="$build/home" XDG_CONFIG_HOME="$build/config" \
      XDG_CACHE_HOME="$build/cache" \
      inkscape "$svg" --export-type=pdf --export-filename="$pdf" \
      >/dev/null 2>&1
  else
    echo "error: SVG conversion needs rsvg-convert or inkscape" >&2
    exit 1
  fi
done

# Preprocess each chapter (NN-name.md, sorted) into build/en/, then hand the
# copies to pandoc.  The glob expands in lexicographic (= chapter) order.
preprocess() {
  f="$1"
  b="$(basename "$f")"
  python3 "$here/pandoc/preprocess.py" "$f" > "$build/en/$b"
  printf '%s' "$build/en/$b"
}

preface="$(preprocess "$srcdir/preface.md")"
appendix="$(preprocess "$srcdir/appendix-a-pytorch.md")"
glossary="$(preprocess "$srcdir/glossary.md")"
references="$(preprocess "$srcdir/references.md")"

chapters=""
for f in "$srcdir"/[0-9][0-9]-*.md; do
  chapters="$chapters $(preprocess "$f")"
done
[ -n "$chapters" ] || { echo "no chapters found in $srcdir" >&2; exit 1; }

echo "pandoc: building $out ..."
# shellcheck disable=SC2086  (pre_files is space-joined on purpose)
pandoc \
  --metadata-file="$meta" \
  --from=gfm+raw_attribute \
  --pdf-engine=xelatex \
  --include-in-header="$here/pandoc/preamble.tex" \
  --top-level-division=chapter \
  --resource-path="$build:$srcdir" \
  "$here/pandoc/frontmatter-en.md" \
  "$preface" \
  "$here/pandoc/part-i-en.md" \
  "$build/en/01-tensors.md" "$build/en/02-autograd.md" \
  "$build/en/03-neural-networks.md" "$build/en/04-training.md" \
  "$here/pandoc/part-ii-en.md" \
  "$build/en/05-tokenization.md" "$build/en/06-attention.md" \
  "$build/en/07-transformer.md" "$build/en/08-training-a-gpt.md" \
  "$here/pandoc/part-iii-en.md" \
  "$build/en/09-tabular-methods.md" "$build/en/10-policy-gradients.md" \
  "$build/en/11-deep-q-learning.md" \
  "$here/pandoc/part-iv-en.md" \
  "$build/en/12-diffusion.md" "$build/en/13-image-diffusion.md" \
  "$here/pandoc/appendix-en.md" "$appendix" \
  "$here/pandoc/backmatter-en.md" "$glossary" "$references" \
  -o "$out"

echo "wrote $out"
