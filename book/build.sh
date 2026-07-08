#!/usr/bin/env bash
#
# Build the BabyTorch book as a PDF.
#
#   ./build.sh en     # English -> build/book-en.pdf   (pandoc + xelatex)
#   ./build.sh ar     # Arabic  -> build/book-ar.pdf   (WeasyPrint, RTL)
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
mkdir -p "$build"

# --- Arabic (RTL): Markdown -> HTML -> WeasyPrint ---------------------------
if [ "$lang" = "ar" ]; then
  PY="${PYTHON:-python3}"
  echo "weasyprint: building $build/book-ar.pdf ..."
  "$PY" "$here/htmlpdf.py" "$here/ar" "$build/book-ar.pdf" --rtl \
    --title "BabyTorch" \
    --subtitle "كيف تعمل أُطُر التعلّم العميق — وكيف يُبنى نموذج GPT" \
    --author "أمجد ماجد"
  exit 0
fi

# --- English: pandoc + XeLaTeX ---------------------------------------------
if [ "$lang" != "en" ]; then
  echo "usage: $0 en|ar" >&2
  exit 1
fi

srcdir="$here"
meta="$here/pandoc/metadata-en.yaml"
out="$build/book-en.pdf"
mkdir -p "$build/en"

# Preprocess each chapter (NN-name.md, sorted) into build/en/, then hand the
# copies to pandoc.  The glob expands in lexicographic (= chapter) order.
pre_files=""
for f in "$srcdir"/[0-9][0-9]-*.md; do
  b="$(basename "$f")"
  python3 "$here/pandoc/preprocess.py" "$f" > "$build/en/$b"
  pre_files="$pre_files $build/en/$b"
done
[ -n "$pre_files" ] || { echo "no chapters found in $srcdir" >&2; exit 1; }

echo "pandoc: building $out ..."
# shellcheck disable=SC2086  (pre_files is space-joined on purpose)
pandoc \
  --metadata-file="$meta" \
  --from=gfm \
  --pdf-engine=xelatex \
  --include-in-header="$here/pandoc/preamble.tex" \
  --toc --toc-depth=1 \
  --resource-path="$srcdir" \
  $pre_files \
  -o "$out"

echo "wrote $out"
