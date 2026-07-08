#!/usr/bin/env bash
#
# Build the BabyTorch book as a PDF.
#
#   ./build.sh en     # English -> build/book-en.pdf
#   ./build.sh ar     # Arabic  -> build/book-ar.pdf   (RTL)
#
# Requires: pandoc, xelatex (TeX Live), rsvg-convert.  See BUILD.md.
# Written for macOS's bash 3.2 -- no mapfile / associative arrays.

set -euo pipefail

lang="${1:-en}"
here="$(cd "$(dirname "$0")" && pwd)"
build="$here/build"
pre="$here/pandoc/preprocess.py"

case "$lang" in
  en)
    srcdir="$here"
    meta="$here/pandoc/metadata-en.yaml"
    out="$build/book-en.pdf"
    engine="xelatex"
    extra=""
    ;;
  ar)
    srcdir="$here/ar"
    meta="$here/pandoc/metadata-ar.yaml"
    out="$build/book-ar.pdf"
    # xelatex (TeX Live) finds the Arabic font in ~/Library/Fonts and drives
    # polyglossia RTL fine; we avoid pandoc's babel path via preamble-ar.tex.
    engine="xelatex"
    # RTL preamble (polyglossia). NOTE: the ltr-code.lua filter (to keep code
    # left-to-right) is temporarily disabled — its \LR wrapping clashes with
    # polyglossia+unicode-math and aborts the build. Consequence: code blocks
    # currently render RTL-reversed in book-ar.pdf. See ltr-code.lua. TODO.
    extra="--include-in-header $here/pandoc/preamble-ar.tex"
    ;;
  *)
    echo "usage: $0 en|ar" >&2
    exit 1
    ;;
esac

mkdir -p "$build/$lang"

# Preprocess each chapter (NN-name.md, sorted) into build/<lang>/, then hand
# the copies to pandoc.  The glob expands in lexicographic (= chapter) order.
pre_files=""
for f in "$srcdir"/[0-9][0-9]-*.md; do
  b="$(basename "$f")"
  python3 "$pre" "$f" > "$build/$lang/$b"
  pre_files="$pre_files $build/$lang/$b"
done
[ -n "$pre_files" ] || { echo "no chapters found in $srcdir" >&2; exit 1; }

echo "pandoc: building $out ..."
# shellcheck disable=SC2086  (pre_files/extra are space-joined on purpose)
pandoc \
  --metadata-file="$meta" \
  --from=gfm \
  --pdf-engine="$engine" \
  --include-in-header="$here/pandoc/preamble.tex" \
  --toc --toc-depth=1 \
  --resource-path="$srcdir" \
  $extra \
  $pre_files \
  -o "$out"

echo "wrote $out"
