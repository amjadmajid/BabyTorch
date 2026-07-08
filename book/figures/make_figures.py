"""Generate the SVG figures for the BabyTorch book.

Every figure in book/figures/*.svg is produced by this script, so the
diagrams stay consistent and anyone can tweak and regenerate them:

    python book/figures/make_figures.py

The figures share one visual language (inspired by Sebastian Raschka's
LLM architecture gallery): rounded blocks on a light card, one color per
component *kind*, used identically in every figure --

    blue   embeddings / inputs        violet  LayerNorm
    orange attention                  yellow  output head / logits
    aqua   MLP / feed-forward         red     loss & gradients
    gray   plain data (tensors, ids)

No external dependencies; plain hand-measured SVG.
"""

import os

# ---------------------------------------------------------------------------
# Palette (validated for color-vision-deficiency separation)
# ---------------------------------------------------------------------------

INK = "#0b0b0b"          # primary text
INK2 = "#52514e"         # secondary text
MUTED = "#898781"        # captions
CARD = "#fcfcfb"         # figure background
BORDER = "rgba(11,11,11,0.10)"

BLUE = ("#ddeafa", "#2a78d6")     # (fill, stroke): embeddings / inputs
ORANGE = ("#fce3d9", "#eb6834")   # attention
AQUA = ("#d9f3e8", "#1baf7a")     # MLP / feed-forward
VIOLET = ("#e5e2f6", "#4a3aa7")   # LayerNorm
YELLOW = ("#fcedcc", "#eda100")   # output head / logits
RED = ("#fadcdc", "#e34948")      # loss / gradients / backward
GRAY = ("#f0efec", "#c3c2b7")     # plain data
CONTAINER = ("#f7f6f3", "#d8d6cf")  # outer grouping boxes

FONT = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, 'SF Mono', Menlo, Consolas, monospace"

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Tiny SVG helpers
# ---------------------------------------------------------------------------

def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def text(x, y, s, size=12, fill=INK, weight="normal", anchor="middle",
         font=FONT, spacing=None, rotate=None):
    extra = f" letter-spacing='{spacing}'" if spacing else ""
    if rotate is not None:
        extra += f" transform='rotate({rotate} {x} {y})'"
    return (f"<text x='{x}' y='{y}' font-family=\"{font}\" font-size='{size}' "
            f"fill='{fill}' font-weight='{weight}' text-anchor='{anchor}'"
            f"{extra}>{esc(s)}</text>")


def rect(x, y, w, h, fill, stroke, rx=9, sw=1.5, dash=None):
    d = f" stroke-dasharray='{dash}'" if dash else ""
    return (f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='{rx}' "
            f"fill='{fill}' stroke='{stroke}' stroke-width='{sw}'{d}/>")


def block(cx, y, w, h, color, label, sub=None, label_size=13, sub_size=10.5):
    """A rounded component block centered horizontally on cx."""
    fill, stroke = color
    out = [rect(cx - w / 2, y, w, h, fill, stroke)]
    if sub:
        out.append(text(cx, y + h / 2 - 3, label, label_size, INK, "600"))
        out.append(text(cx, y + h / 2 + 12, sub, sub_size, INK2))
    else:
        out.append(text(cx, y + h / 2 + label_size * 0.36, label,
                        label_size, INK, "600"))
    return "\n".join(out)


def pill(cx, cy, w, label, size=12, color=GRAY, h=30, mono=False):
    fill, stroke = color
    out = [f"<rect x='{cx - w / 2}' y='{cy - h / 2}' width='{w}' height='{h}' "
           f"rx='{h / 2}' fill='{fill}' stroke='{stroke}' stroke-width='1.3'/>"]
    out.append(text(cx, cy + size * 0.36, label, size, INK2, "500",
                    font=MONO if mono else FONT))
    return "\n".join(out)


def line(x1, y1, x2, y2, stroke=INK2, sw=1.6, dash=None, marker=True,
         marker_id="arr"):
    d = f" stroke-dasharray='{dash}'" if dash else ""
    m = f" marker-end='url(#{marker_id})'" if marker else ""
    return (f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{stroke}' "
            f"stroke-width='{sw}'{d}{m}/>")


def path(d, stroke=INK2, sw=1.6, dash=None, marker=True, marker_id="arr"):
    dd = f" stroke-dasharray='{dash}'" if dash else ""
    m = f" marker-end='url(#{marker_id})'" if marker else ""
    return (f"<path d='{d}' fill='none' stroke='{stroke}' "
            f"stroke-width='{sw}'{dd}{m}/>")


def plus(cx, cy, r=12):
    return (f"<circle cx='{cx}' cy='{cy}' r='{r}' fill='#ffffff' "
            f"stroke='{INK2}' stroke-width='1.6'/>"
            f"<line x1='{cx - r + 5}' y1='{cy}' x2='{cx + r - 5}' y2='{cy}' "
            f"stroke='{INK2}' stroke-width='1.8'/>"
            f"<line x1='{cx}' y1='{cy - r + 5}' x2='{cx}' y2='{cy + r - 5}' "
            f"stroke='{INK2}' stroke-width='1.8'/>")


def title_block(t, sub=None, x=32):
    out = [text(x, 42, t, 17, INK, "700", "start")]
    if sub:
        out.append(text(x, 63, sub, 12, INK2, "normal", "start"))
    return "\n".join(out)


def legend(x, y, items, size=10.5):
    """Small color-code legend: items = [(color, label), ...]."""
    out = []
    for color, lab in items:
        fill, stroke = color
        out.append(f"<rect x='{x}' y='{y - 9}' width='14' height='12' rx='3' "
                   f"fill='{fill}' stroke='{stroke}' stroke-width='1.2'/>")
        out.append(text(x + 20, y + 1, lab, size, INK2, "normal", "start"))
        x += 24 + 7.2 * len(lab) + 16
    return "\n".join(out)


def svg(w, h, body, markers_color=INK2):
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" font-family="{FONT}">
<defs>
<marker id="arr" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
  <path d="M 0 1 L 9 5 L 0 9 z" fill="{markers_color}"/>
</marker>
<marker id="arr-red" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
  <path d="M 0 1 L 9 5 L 0 9 z" fill="{RED[1]}"/>
</marker>
<marker id="arr-blue" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
  <path d="M 0 1 L 9 5 L 0 9 z" fill="{BLUE[1]}"/>
</marker>
</defs>
<rect x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" rx="14" fill="{CARD}" stroke="{BORDER}"/>
{body}
</svg>
"""


def save(name, content):
    p = os.path.join(HERE, name)
    with open(p, "w") as f:
        f.write(content)
    print("wrote", p)


# ---------------------------------------------------------------------------
# Figure: BabyGPT full architecture (chapter 7 flagship)
# ---------------------------------------------------------------------------

def fig_babygpt():
    W, H = 880, 1120
    CX = 280          # main stack center
    b = []

    b.append(title_block("BabyGPT — a decoder-only Transformer",
                         "tutorials/llm/model.py · default configuration from train.py"))

    # ---- bottom of the stack: inputs ----------------------------------
    b.append(pill(CX, 985, 236, "token ids — shape (B, T)", 12))
    b.append(text(CX + 130, 989, '"To be…" → [44, 53, 1, …]', 10.5, MUTED,
                  anchor="start"))

    # two embeddings side by side
    b.append(line(CX, 968, CX, 940))
    b.append(path(f"M {CX} 952 L {CX - 88} 952 L {CX - 88} 938", marker=True))
    b.append(path(f"M {CX} 952 L {CX + 88} 952 L {CX + 88} 938", marker=True))
    b.append(block(CX - 88, 884, 158, 52, BLUE, "Token embedding", "65 × 192 lookup", 12))
    b.append(block(CX + 88, 884, 158, 52, BLUE, "Position embedding", "128 × 192 lookup", 12))
    b.append(text(CX + 182, 918, "positions 0…T−1", 10, MUTED, anchor="start"))

    # join at +, then dropout
    b.append(path(f"M {CX - 88} 884 L {CX - 88} 862 L {CX - 16} 862", marker=True))
    b.append(path(f"M {CX + 88} 884 L {CX + 88} 862 L {CX + 16} 862", marker=True))
    b.append(plus(CX, 862))
    b.append(text(CX - 26, 845, "what it is + where it sits", 10, MUTED, anchor="end"))
    b.append(line(CX, 849, CX, 828))
    b.append(block(CX, 794, 130, 32, GRAY, "Dropout 0.1", None, 11.5))
    b.append(line(CX, 794, CX, 768))
    b.append(text(CX + 12, 780, "x — (B, T, 192)", 10.5, MUTED, anchor="start",))

    # ---- the repeated Transformer block --------------------------------
    top, bot = 300, 762   # container bounds
    b.append(rect(90, top, 380, bot - top - 6, CONTAINER[0], CONTAINER[1],
                  rx=12, sw=1.4))
    b.append(text(104, top + 24, "TRANSFORMER BLOCK", 11, MUTED, "600",
                  "start", spacing="0.08em"))

    # x6 bracket on the right of the container
    bx = 486
    b.append(path(f"M {bx} {top + 6} L {bx + 10} {top + 6} L {bx + 10} {bot - 12} L {bx} {bot - 12}",
                  stroke=INK2, sw=1.4, marker=False))
    b.append(text(bx + 24, (top + bot) / 2 - 2, "× 6", 15, INK, "700", "start"))
    b.append(text(bx + 24, (top + bot) / 2 + 16, "identical blocks,", 10.5, INK2, anchor="start"))
    b.append(text(bx + 24, (top + bot) / 2 + 30, "stacked", 10.5, INK2, anchor="start"))

    ib = 276  # inner block width

    # -- sublayer 1: attention with pre-LN and residual
    b.append(line(CX, 762, CX, 726))                       # into LN1
    b.append(block(CX, 690, ib, 36, VIOLET, "LayerNorm", None, 12.5))
    b.append(line(CX, 690, CX, 662))
    b.append(block(CX, 606, ib, 56, ORANGE, "Causal self-attention",
                   "6 heads · head_size 32 · no peeking ahead", 13))
    b.append(line(CX, 606, CX, 566))
    b.append(plus(CX, 552))
    # residual skip around attention
    b.append(path(f"M {CX} 744 L 118 744 L 118 552 L {CX - 16} 552"))
    b.append(text(109, 646, "residual", 10, MUTED, anchor="middle",
                  rotate=-90))
    b.append(text(CX + 24, 540, "communicate", 10.5, MUTED, anchor="start"))

    # -- sublayer 2: MLP with pre-LN and residual
    b.append(line(CX, 539, CX, 516))
    b.append(block(CX, 480, ib, 36, VIOLET, "LayerNorm", None, 12.5))
    b.append(line(CX, 480, CX, 452))
    b.append(block(CX, 396, ib, 56, AQUA, "MLP (feed-forward)",
                   "Linear 192→768 · GELU · Linear 768→192", 13))
    b.append(line(CX, 396, CX, 356))
    b.append(plus(CX, 342))
    b.append(path(f"M {CX} 530 L 118 530 L 118 342 L {CX - 16} 342"))
    b.append(text(CX + 24, 330, "compute", 10.5, MUTED, anchor="start"))
    b.append(line(CX, 329, CX, 300 - 6))

    # ---- top of the stack: final norm, head, logits ---------------------
    b.append(line(CX, 294, CX, 262))
    b.append(block(CX, 226, ib, 36, VIOLET, "Final LayerNorm", None, 12.5))
    b.append(line(CX, 226, CX, 198))
    b.append(block(CX, 142, ib, 56, YELLOW, "Output head",
                   "Linear 192 → 65: one score per token", 13))
    b.append(line(CX, 142, CX, 116))
    b.append(pill(CX, 98, 260, "logits — shape (B, T, 65)", 12, YELLOW))
    b.append(text(CX + 140, 92, "a next-token guess,", 10.5, MUTED, anchor="start"))
    b.append(text(CX + 140, 106, "at every position", 10.5, MUTED, anchor="start"))

    # ---- right-hand model card ------------------------------------------
    px, pw = 588, 260
    py, ph = 96, 336
    b.append(rect(px, py, pw, ph, "#ffffff", CONTAINER[1], rx=12, sw=1.3))
    b.append(text(px + 18, py + 30, "Model card", 13, INK, "700", "start"))
    b.append(text(px + 18, py + 48, "the train.py defaults", 10.5, MUTED,
                  anchor="start"))
    rows = [
        ("n_embd", "192  (width of every token vector)"),
        ("n_head", "6  → head_size 32"),
        ("n_layer", "6  Transformer blocks"),
        ("block_size", "128 tokens of context"),
        ("vocab_size", "65 characters (Shakespeare)"),
        ("dropout", "0.1"),
        ("parameters", "2,719,169  ≈ 2.7 M"),
        ("optimizer", "AdamW · cosine warmup"),
    ]
    ry = py + 76
    for k, v in rows:
        b.append(text(px + 18, ry, k, 11, INK, "600", "start", font=MONO))
        b.append(text(px + 108, ry, v, 10.5, INK2, "normal", "start"))
        ry += 26
    b.append(text(px + 18, ry + 6, "≈ 98 % of the parameters live in the", 10.5,
                  MUTED, anchor="start"))
    b.append(text(px + 18, ry + 21, "six blocks — nearly all serving  @", 10.5,
                  MUTED, anchor="start"))

    # scale-up note
    ny = py + ph + 40
    b.append(text(px + 18, ny, "Same skeleton, more floors:", 11.5, INK, "600",
                  anchor="start"))
    for i, ln in enumerate([
            "GPT-2:  n_embd 768 · 12 layers · 124 M",
            "GPT-3:  n_embd 12288 · 96 layers · 175 B"]):
        b.append(text(px + 18, ny + 22 + i * 18, ln, 10.5, INK2, "normal",
                      "start", font=MONO))

    # reading direction hint + legend
    b.append(text(32, H - 62, "Read bottom-up: ids enter at the bottom, logits leave at the top.",
                  11, MUTED, "normal", "start"))
    b.append(legend(32, H - 30, [
        (BLUE, "embedding"), (VIOLET, "LayerNorm"), (ORANGE, "attention"),
        (AQUA, "MLP"), (YELLOW, "head / logits"), (GRAY, "data"),
    ]))

    save("fig-babygpt.svg", svg(W, H, "\n".join(b)))


# ---------------------------------------------------------------------------
# Figure: tensors & broadcasting (chapter 1)
# ---------------------------------------------------------------------------

def cells(x, y, cols, rows, cs, fill, stroke, sw=1.1, rx=2):
    """A grid of little square cells (one 'tensor')."""
    out = []
    for r in range(rows):
        for c in range(cols):
            out.append(rect(x + c * cs, y + r * cs, cs - 2, cs - 2,
                            fill, stroke, rx=rx, sw=sw))
    return "\n".join(out)


def fig_tensors():
    W, H = 880, 400
    b = [title_block("Tensors — blocks of numbers with a shape",
                     "and broadcasting, which lets a small tensor serve a whole batch")]

    # ---- left panel: ranks ---------------------------------------------
    b.append(text(40, 110, "One structure, any number of dimensions", 12.5,
                  INK, "600", "start"))
    cs = 22
    y0 = 140
    # scalar
    b.append(cells(60, y0, 1, 1, cs, GRAY[0], GRAY[1]))
    b.append(text(70, y0 + 62, "scalar", 11, INK2))
    b.append(text(70, y0 + 78, "( )", 10.5, MUTED, font=MONO))
    # vector
    b.append(cells(150, y0, 4, 1, cs, GRAY[0], GRAY[1]))
    b.append(text(192, y0 + 62, "vector", 11, INK2))
    b.append(text(192, y0 + 78, "(4,)", 10.5, MUTED, font=MONO))
    # matrix
    b.append(cells(280, y0, 3, 2, cs, GRAY[0], GRAY[1]))
    b.append(text(312, y0 + 62, "matrix", 11, INK2))
    b.append(text(312, y0 + 78, "(2, 3)", 10.5, MUTED, font=MONO))
    # 3-D: two offset matrices
    b.append(cells(408 + 10, y0 - 10, 3, 2, cs, "#e7e6e1", "#cfcdc6"))
    b.append(cells(400, y0, 3, 2, cs, GRAY[0], GRAY[1]))
    b.append(text(438, y0 + 62, "3-D tensor", 11, INK2))
    b.append(text(438, y0 + 78, "(2, 2, 3)", 10.5, MUTED, font=MONO))

    b.append(text(40, 290, "The shape is the bookkeeping: a batch of 32 sentences,", 11, INK2, "normal", "start"))
    b.append(text(40, 307, "128 tokens each, 192 numbers per token, is one tensor", 11, INK2, "normal", "start"))
    b.append(text(40, 324, "of shape (32, 128, 192).", 11, INK2, "normal", "start"))

    # ---- right panel: broadcasting --------------------------------------
    px = 540
    b.append(text(px, 110, "Broadcasting: stretch size-1 dimensions to fit", 12.5,
                  INK, "600", "start"))
    cs2 = 17
    gx, gy = px, 132
    # x: (6-row stand-in for 32) x 8
    b.append(cells(gx, gy, 8, 6, cs2, BLUE[0], BLUE[1]))
    b.append(text(gx + 4 * cs2, gy + 6 * cs2 + 16, "x — (32, 10)", 10.5, INK2, font=MONO))
    # plus
    b.append(text(gx + 8 * cs2 + 16, gy + 3 * cs2 + 4, "+", 20, INK, "600"))
    # b: one real row + ghost copies
    bx0 = gx + 8 * cs2 + 36
    b.append(cells(bx0, gy, 8, 1, cs2, YELLOW[0], YELLOW[1]))
    for r in range(1, 6):
        b.append(cells(bx0, gy + r * cs2, 8, 1, cs2, "#ffffff", "#dad8d0", sw=1.0))
    b.append(text(bx0 + 4 * cs2, gy + 6 * cs2 + 16, "b — (1, 10)", 10.5, INK2, font=MONO))
    b.append(text(bx0 + 4 * cs2, gy + 6 * cs2 + 33,
                  "one real row, virtually copied 32×", 10, MUTED))

    b.append(text(px, 290, "One bias row serves the whole batch. Remember the", 11, INK2, "normal", "start"))
    b.append(text(px, 307, "debt for chapter 2: whatever was copied forward must", 11, INK2, "normal", "start"))
    b.append(text(px, 324, "be summed on the way back.", 11, INK2, "normal", "start"))

    b.append(legend(40, H - 34, [(BLUE, "a batch of data"),
                                 (YELLOW, "one small tensor"),
                                 (GRAY, "any tensor")]))
    save("fig-tensors.svg", svg(W, H, "\n".join(b)))


# ---------------------------------------------------------------------------
# Figure: the computation graph, forward and backward (chapter 2)
# ---------------------------------------------------------------------------

def op_chip(cx, cy, label, r=17):
    return (f"<circle cx='{cx}' cy='{cy}' r='{r}' fill='#ffffff' "
            f"stroke='{INK2}' stroke-width='1.6'/>" +
            text(cx, cy + 5, label, 14, INK, "700"))


def fig_autograd():
    W, H = 880, 470
    b = [title_block("The computation graph of  y = x² + 3x   (at x = 2)",
                     "forward: compute and record · backward: replay in reverse, multiplying local derivatives")]

    yA, yB, yMid = 170, 330, 250   # rows: a-path, b-path, x/y row

    # nodes
    b.append(block(120, yMid - 26, 130, 52, BLUE, "x = 2.0",
                   "requires_grad", 13))
    b.append(op_chip(315, yA, "×"))
    b.append(op_chip(315, yB, "×"))
    b.append(pill(150, yB + 58, 120, "3.0 — constant", 10.5))
    b.append(block(475, yA - 22, 130, 44, GRAY, "a = x·x = 4.0", None, 12))
    b.append(block(475, yB - 22, 130, 44, GRAY, "b = 3·x = 6.0", None, 12))
    b.append(op_chip(640, yMid, "+"))
    b.append(block(770, yMid - 22, 130, 44, YELLOW, "y = 10.0", None, 13))

    # forward edges (solid)
    b.append(path(f"M 165 {yMid - 26} C 220 {yA + 20} 240 {yA + 8} 296 {yA + 6}"))
    b.append(path(f"M 175 {yMid - 4} C 230 {yA + 40} 245 {yA + 22} 297 {yA + 13}"))
    b.append(path(f"M 165 {yMid + 26} C 220 {yB - 20} 240 {yB - 8} 296 {yB - 6}"))
    b.append(path(f"M 210 {yB + 58} C 250 {yB + 50} 270 {yB + 30} 300 {yB + 16}"))
    b.append(line(333, yA, 409, yA))
    b.append(line(333, yB, 409, yB))
    b.append(path(f"M 541 {yA} C 590 {yA} 600 {yMid - 30} 626 {yMid - 12}"))
    b.append(path(f"M 541 {yB} C 590 {yB} 600 {yMid + 30} 626 {yMid + 12}"))
    b.append(line(658, yMid, 704, yMid))

    # backward edges (red, dashed, right-to-left, drawn offset)
    RS, RM = RED[1], "arr-red"
    def back(x1, y1, x2, y2, curve=0):
        if curve:
            return path(f"M {x1} {y1} Q {(x1 + x2) / 2} {y1 + curve} {x2} {y2}",
                        stroke=RS, sw=1.5, dash="5 4", marker_id=RM)
        return line(x1, y1, x2, y2, stroke=RS, sw=1.5, dash="5 4",
                    marker_id=RM)

    b.append(back(750, yMid + 34, 668, yMid + 22, 16))
    b.append(text(716, yMid + 58, "dy/dy = 1", 10, RS, font=MONO))
    b.append(back(614, yMid - 24, 548, yA + 12, -6))
    b.append(back(614, yMid + 24, 548, yB - 2, 6))
    b.append(text(596, yA + 34, "1", 10.5, RS, font=MONO))
    b.append(text(596, yB - 26, "1", 10.5, RS, font=MONO))
    b.append(back(409, yA + 14, 341, yA + 14))
    b.append(back(409, yB - 14, 341, yB - 14))
    b.append(back(292, yA + 22, 176, yMid - 14, 30))
    b.append(text(300, yA - 48, "d(x·x)/dx = 2x = 4.0", 10, RS, font=MONO))
    b.append(back(292, yB - 22, 176, yMid + 14, -30))
    b.append(text(315, yB + 52, "d(3x)/dx = 3.0", 10, RS, font=MONO))

    # the accumulated gradient badge
    b.append(rect(58, yMid + 44, 190, 34, RED[0], RED[1], rx=8, sw=1.4))
    b.append(text(153, yMid + 65, "x.grad = 4.0 + 3.0 = 7.0", 11.5, INK, "600",
                  font=MONO))
    b.append(text(153, yMid + 96, "two paths meet: gradients add", 10, MUTED))

    # legend
    b.append(line(40, H - 36, 88, H - 36))
    b.append(text(96, H - 32, "forward — compute values, record the graph", 10.5,
                  INK2, "normal", "start"))
    b.append(line(420, H - 36, 468, H - 36, stroke=RS, dash="5 4",
                  marker_id=RM))
    b.append(text(476, H - 32, "backward — chain rule, one operation at a time",
                  10.5, INK2, "normal", "start"))

    save("fig-autograd.svg", svg(W, H, "\n".join(b)))


# ---------------------------------------------------------------------------
# Figure: a model is layers + loss (chapter 3)
# ---------------------------------------------------------------------------

def fig_mlp():
    W, H = 880, 352
    b = [title_block("A model with knobs — and the loss that grades it",
                     "nn.Sequential(nn.Linear(2, 8, nn.Tanh()), nn.Linear(8, 1, nn.Sigmoid()))  ·  33 parameters")]

    yC = 165
    b.append(pill(95, yC, 110, "x — (4, 2)", 11.5))
    b.append(line(150, yC, 186, yC))
    b.append(block(268, yC - 30, 160, 60, AQUA, "Linear 2 → 8",
                   "w (2, 8) · b (1, 8)", 12.5))
    b.append(line(348, yC, 378, yC))
    b.append(block(414, yC - 20, 70, 40, GRAY, "Tanh", None, 11.5))
    b.append(text(414, yC - 52, "the bend", 10, MUTED))
    b.append(line(449, yC, 479, yC))
    b.append(block(561, yC - 30, 160, 60, AQUA, "Linear 8 → 1",
                   "w (8, 1) · b (1, 1)", 12.5))
    b.append(line(641, yC, 671, yC))
    b.append(block(707, yC - 20, 70, 40, GRAY, "Sigmoid", None, 11))
    b.append(path(f"M 742 {yC} L 790 {yC} L 790 {yC + 60} L 700 {yC + 60}"))
    b.append(pill(628, yC + 60, 140, "prediction (4, 1)", 11))
    b.append(line(558, yC + 60, 500, yC + 60))
    b.append(block(430, yC + 38, 130, 44, RED, "MSELoss", "(pred − y)²  mean", 12))
    b.append(pill(430, yC + 118, 130, "targets y (4, 1)", 11))
    b.append(line(430, yC + 103, 430, yC + 86))
    b.append(line(365, yC + 60, 300, yC + 60))
    b.append(pill(232, yC + 60, 130, "loss — one number", 10.5, RED))
    b.append(text(232, yC + 92, "the tensor you call .backward() on", 10, MUTED))

    b.append(text(268, yC - 52, "8 little detectors", 10, MUTED))
    b.append(text(561, yC - 52, "combine them", 10, MUTED))

    b.append(legend(40, H - 30, [(AQUA, "layers with parameters"),
                                 (GRAY, "parameter-free"),
                                 (RED, "loss")]))
    save("fig-mlp.svg", svg(W, H, "\n".join(b)))


# ---------------------------------------------------------------------------
# Figure: the training loop (chapter 4)
# ---------------------------------------------------------------------------

def fig_training_loop():
    W, H = 880, 470
    b = [title_block("The four-step loop that trains everything",
                     "from the two-line regression to BabyGPT — only the model and the data change")]

    bw, bh = 250, 74
    xL, xR = 200, 660
    yT, yB = 160, 330

    b.append(block(xL, yT - bh / 2, bw, bh, BLUE, "1 · forward",
                   "predictions = model(x)", 14, 11))
    b.append(block(xR, yT - bh / 2, bw, bh, RED, "2 · loss",
                   "loss = criterion(predictions, y)", 14, 11))
    b.append(block(xR, yB - bh / 2, bw, bh, RED, "3 · backward",
                   "loss.backward()", 14, 11))
    b.append(block(xL, yB - bh / 2, bw, bh, YELLOW, "4 · update",
                   "optimizer.step()", 14, 11))

    # clockwise arrows
    b.append(line(xL + bw / 2 + 8, yT, xR - bw / 2 - 10, yT))
    b.append(text((xL + xR) / 2, yT - 12, "how wrong?", 10.5, MUTED))
    b.append(line(xR, yT + bh / 2 + 6, xR, yB - bh / 2 - 10))
    b.append(text(xR + 20, (yT + yB) / 2 + 4, "a gradient for", 10.5, MUTED, anchor="start"))
    b.append(text(xR + 20, (yT + yB) / 2 + 19, "every parameter", 10.5, MUTED, anchor="start"))
    b.append(line(xR - bw / 2 - 8, yB, xL + bw / 2 + 10, yB))
    b.append(text((xL + xR) / 2, yB + 26, "p = p − learning_rate · grad", 10.5, MUTED, font=MONO))
    b.append(path(f"M {xL} {yB - bh / 2 - 6} L {xL} {yT + bh / 2 + 34} L {xL} {yT + bh / 2 + 10}"))
    b.append(block(xL, (yT + yB) / 2 - 17, 172, 34, GRAY,
                   "optimizer.zero_grad()", None, 10.5))
    b.append(text(xL - 100, (yT + yB) / 2 + 3, "forget old", 10.5, MUTED, anchor="end"))
    b.append(text(xL - 100, (yT + yB) / 2 + 18, "gradients first!", 10.5, MUTED, anchor="end"))

    # center
    b.append(text((xL + xR) / 2, (yT + yB) / 2 - 2, "repeat", 13, INK, "700"))
    b.append(text((xL + xR) / 2, (yT + yB) / 2 + 16, "until the loss stops falling", 10.5, INK2))

    b.append(text(40, H - 56, "The classic silent bug: skip zero_grad() and every batch trains on stale gradients from the last one —",
                  10.5, INK2, "normal", "start"))
    b.append(text(40, H - 40, "backward() accumulates by design, because a tensor used in several places must sum its contributions.",
                  10.5, INK2, "normal", "start"))
    save("fig-training-loop.svg", svg(W, H, "\n".join(b)))


# ---------------------------------------------------------------------------
# Figure: tokenization and self-supervised pairs (chapter 5)
# ---------------------------------------------------------------------------

def fig_tokenization():
    W, H = 880, 500
    b = [title_block("Text becomes numbers — and labels come free",
                     "a tokenizer maps text to ids; shifting a window by one token creates the training pairs")]

    # ---- panel A: the round trip ----------------------------------------
    yA = 140
    b.append(text(40, 108, "The round trip", 12.5, INK, "600", "start"))
    b.append(pill(105, yA, 110, '"To be"', 12))
    b.append(line(162, yA, 196, yA))
    b.append(block(258, yA - 20, 120, 40, BLUE, "encode", None, 12.5))
    b.append(line(318, yA, 352, yA))
    b.append(pill(448, yA, 180, "[44, 53, 1, 40, 43]", 11.5, GRAY, mono=True))
    b.append(line(540, yA, 574, yA))
    b.append(block(636, yA - 20, 120, 40, BLUE, "decode", None, 12.5))
    b.append(line(696, yA, 730, yA))
    b.append(pill(785, yA, 100, '"To be"', 12))
    b.append(text(448, yA + 36, "the model only ever sees these ids", 10, MUTED))
    b.append(text(258, yA - 36, "CharTokenizer / BPE", 10, MUTED))

    # ---- panel B: x/y windows -------------------------------------------
    yB = 268
    b.append(text(40, yB - 42, "Where the training pairs come from", 12.5, INK,
                  "600", "start"))
    corpus = list("First Citizen:")
    cs = 40
    x0 = 120
    for i, ch in enumerate(corpus):
        shown = "␣" if ch == " " else ch
        b.append(rect(x0 + i * cs, yB, cs - 4, cs - 4, GRAY[0], GRAY[1],
                      rx=6, sw=1.2))
        b.append(text(x0 + i * cs + (cs - 4) / 2, yB + 24, shown, 14, INK,
                      "500", font=MONO))
    b.append(text(x0 - 16, yB + 24, "…", 13, MUTED, anchor="end"))
    b.append(text(x0 + len(corpus) * cs + 4, yB + 24, "…", 13, MUTED,
                  anchor="start"))

    # x window above (chars 0..7), y window below (chars 1..8)
    T = 8
    bx = x0 - 5
    b.append(rect(bx, yB - 34, T * cs - 2, 24, BLUE[0], BLUE[1], rx=7, sw=1.4))
    b.append(text(bx + T * cs / 2, yB - 18, "x — the input window (T = 8)",
                  10.5, INK, "600"))
    yb2 = yB + cs + 8
    b.append(rect(bx + cs, yb2, T * cs - 2, 24, YELLOW[0], YELLOW[1], rx=7,
                  sw=1.4))
    b.append(text(bx + cs + T * cs / 2, yb2 + 16,
                  "y — the same window, one step right", 10.5, INK, "600"))

    b.append(text(120, yb2 + 62, "y[t] is the token that really followed x[…t] — every position is one",
                  11, INK2, "normal", "start"))
    b.append(text(120, yb2 + 79, "next-token exercise, and the raw text itself is the answer key.",
                  11, INK2, "normal", "start"))
    b.append(text(120, yb2 + 105, "get_batch() stacks batch_size random windows into x, y of shape (B, T).",
                  10.5, MUTED, "normal", "start"))

    save("fig-tokenization.svg", svg(W, H, "\n".join(b)))


# ---------------------------------------------------------------------------
# Figure: causal self-attention, one head (chapter 6)
# ---------------------------------------------------------------------------

def fig_attention():
    W, H = 880, 596
    b = [title_block("One head of causal self-attention",
                     "every position asks a question (q), offers an answer (k), and carries a payload (v)")]

    # ---- left: pipeline ---------------------------------------------------
    cx = 170
    b.append(pill(cx, 118, 170, "x — (B, T, C)", 11.5))
    b.append(line(cx, 133, cx, 158))
    b.append(block(cx, 158, 190, 44, ORANGE, "Linear C → 3C", "one matmul makes q, k, v", 12))
    # split into q k v
    for dx, lab in [(-70, "q"), (0, "k"), (70, "v")]:
        b.append(path(f"M {cx} 202 L {cx + dx} 216 L {cx + dx} 228"))
        b.append(block(cx + dx, 230, 54, 36, ORANGE, lab, None, 13))
    b.append(text(cx, 286, "each (T, hs)", 10, MUTED))

    # scores = q k^T
    b.append(path(f"M {cx - 70} 266 L {cx - 70} 282 L {cx - 18} 296"))
    b.append(path(f"M {cx} 266 L {cx} 282 L {cx - 6} 288"))
    b.append(block(cx - 12, 296, 200, 40, GRAY, "scores = q @ kᵀ / √hs",
                   None, 12))
    b.append(text(cx - 12, 352, "÷ √hs keeps softmax soft —", 10, MUTED))
    b.append(text(cx - 12, 366, "big dots would kill the gradient", 10, MUTED))

    # ---- center: the masked score matrix ---------------------------------
    gx, gy, cs, T = 430, 150, 34, 6
    b.append(text(gx + T * cs / 2, gy - 26, "attention weights (T × T), after mask + softmax",
                  11.5, INK, "600"))
    hi = 4  # highlighted query row
    hi_weights = {0: 0.15, 1: 0.80, 2: 0.20, 3: 0.15, 4: 0.55}
    for i in range(T):
        for j in range(T):
            if j > i:      # the future: masked
                b.append(rect(gx + j * cs, gy + i * cs, cs - 3, cs - 3,
                              "#f3f2ef", "#d8d6cf", rx=4, sw=1.0))
                b.append(text(gx + j * cs + (cs - 3) / 2,
                              gy + i * cs + (cs - 3) / 2 + 3.5, "−∞", 8.5,
                              "#b6b4ac"))
            elif i == hi:  # the highlighted query row: real-looking weights
                op = 0.18 + hi_weights[j]
                b.append(f"<rect x='{gx + j * cs}' y='{gy + i * cs}' "
                         f"width='{cs - 3}' height='{cs - 3}' rx='4' "
                         f"fill='{ORANGE[1]}' fill-opacity='{op:.2f}' "
                         f"stroke='{ORANGE[1]}' stroke-width='1.2'/>")
            else:          # allowed, not highlighted
                b.append(rect(gx + j * cs, gy + i * cs, cs - 3, cs - 3,
                              ORANGE[0], ORANGE[1], rx=4, sw=0.9))
    # axes
    b.append(text(gx - 12, gy + hi * cs + 20, "query i=4", 10.5, INK, "600",
                  anchor="end"))
    b.append(text(gx - 12, gy + hi * cs + 34, "attends to 0…4", 9.5, MUTED,
                  anchor="end"))
    b.append(text(gx + T * cs / 2, gy + T * cs + 18, "keys j (positions offering answers)",
                  10, MUTED))
    b.append(text(gx - 44, gy + T * cs / 2, "queries i", 10, MUTED,
                  rotate=-90))
    b.append(path(f"M {gx + 2.4 * cs} {gy - 8} L {gx + 4.6 * cs} {gy + 1.4 * cs}",
                  stroke=MUTED, sw=1.1, dash="3 3", marker=False))
    b.append(text(gx + 4.9 * cs, gy - 12, "upper triangle = the future,", 10,
                  MUTED, anchor="start"))
    b.append(text(gx + 4.9 * cs, gy + 2, "−∞ before softmax → weight 0", 10,
                  MUTED, anchor="start"))

    # each row sums to 1
    b.append(text(gx + T * cs + 14, gy + hi * cs + 18, "each row sums to 1",
                  10, MUTED, anchor="start"))

    # ---- bottom: read the values ------------------------------------------
    yv = 476
    mx = gx + T * cs / 2
    b.append(path(f"M {mx} {gy + T * cs + 26} L {mx} {yv - 42}"))
    b.append(block(mx, yv - 40, 210, 40, ORANGE,
                   "out = weights @ v", "a learned mix of payloads", 12))
    # v's route from the projection to the read-out
    b.append(path(f"M 267 248 L 310 248 L 310 {yv - 20} L {mx - 107} {yv - 20}"))
    b.append(text(316, yv - 48, "v — the payloads,", 9.5, MUTED, anchor="start"))
    b.append(text(316, yv - 35, "saved for the read-out", 9.5, MUTED, anchor="start"))
    b.append(line(mx, yv, mx, yv + 22))
    b.append(pill(mx, yv + 40, 190, "fresh vector per position", 10.5))

    b.append(text(40, H - 30, "Everything here is Part I machinery — Linear, reshape, @, softmax — so backward() differentiates it automatically.",
                  10.5, INK2, "normal", "start"))
    save("fig-attention.svg", svg(W, H, "\n".join(b)))


# ---------------------------------------------------------------------------
# Figure: multi-head = the same trick in parallel subspaces (chapter 6)
# ---------------------------------------------------------------------------

def fig_multihead():
    W, H = 880, 396
    b = [title_block("Multi-head: six attentions in parallel, then mix",
                     "(B, T, C) → reshape → (B, n_head, T, head_size) → attention per head → concat → project")]

    # channel bar split into 6 heads
    bx, by, bw, bh = 120, 120, 640, 34
    b.append(text(bx + bw / 2, by - 14, "C = 192 channels of every token vector",
                  11, INK2))
    for hh in range(6):
        seg = bw / 6
        b.append(rect(bx + hh * seg, by, seg - 4, bh, ORANGE[0], ORANGE[1],
                      rx=6, sw=1.2))
        b.append(text(bx + hh * seg + seg / 2 - 2, by + 22, f"head {hh + 1}",
                      10.5, INK, "600"))
    b.append(text(bx + bw + 14, by + 22, "32 each", 10, MUTED, anchor="start"))

    # three example head lanes + ellipsis
    lanes = [bx + bw / 6 * 0.5, bx + bw / 6 * 2.5, bx + bw / 6 * 5.5]
    for lx in lanes:
        b.append(line(lx - 2, by + bh, lx - 2, by + bh + 26))
        b.append(block(lx - 2, by + bh + 28, 92, 40, ORANGE, "attention",
                       None, 10.5))
        b.append(line(lx - 2, by + bh + 68, lx - 2, by + bh + 92))
    b.append(text(bx + bw / 6 * 4, by + bh + 52, "…", 16, MUTED))
    b.append(text(lanes[0] - 60, by + bh + 52, "own q·kᵀ table,", 9.5, MUTED, anchor="end"))
    b.append(text(lanes[0] - 60, by + bh + 65, "own specialty", 9.5, MUTED, anchor="end"))

    # concat bar
    cy = by + bh + 94
    b.append(rect(bx, cy, bw - 4, 30, ORANGE[0], ORANGE[1], rx=6, sw=1.2))
    b.append(text(bx + bw / 2, cy + 19, "concatenate heads — back to (B, T, C)",
                  11, INK, "600"))
    b.append(line(bx + bw / 2, cy + 30, bx + bw / 2, cy + 54))
    b.append(block(bx + bw / 2, cy + 56, 220, 40, ORANGE,
                   "projection — Linear C → C", "lets the heads exchange notes", 11.5))

    b.append(text(40, H - 34, "No new machinery: the split is a reshape + transpose, so all six heads run in one batched matmul.",
                  10.5, INK2, "normal", "start"))
    save("fig-multihead.svg", svg(W, H, "\n".join(b)))


# ---------------------------------------------------------------------------
# Figure: autoregressive generation (chapter 8)
# ---------------------------------------------------------------------------

def fig_generation():
    W, H = 880, 420
    b = [title_block("Generation: a classifier in a loop",
                     "forward the context, shape the distribution, sample one token, append, repeat")]

    yC = 190
    b.append(pill(130, yC, 190, 'context  "ROMEO:"', 11.5, BLUE))
    b.append(text(130, yC + 28, "cropped to the last", 9.5, MUTED))
    b.append(text(130, yC + 41, "block_size tokens", 9.5, MUTED))
    b.append(line(226, yC, 262, yC))

    # mini GPT stack icon
    gx = 318
    for i, (col, lab) in enumerate([(BLUE, ""), (ORANGE, ""), (AQUA, "")]):
        b.append(rect(gx - 52, yC - 34 + i * 24, 104, 20, col[0], col[1],
                      rx=5, sw=1.2))
    b.append(text(gx, yC + 52, "GPT (frozen,", 10.5, INK2))
    b.append(text(gx, yC + 66, "no_grad + eval)", 10.5, INK2))
    b.append(line(gx + 54, yC, gx + 88, yC))

    b.append(pill(gx + 148, yC, 116, "logits (65,)", 11, YELLOW))
    b.append(line(gx + 208, yC, gx + 240, yC))
    b.append(block(gx + 302, yC - 22, 120, 44, GRAY, "÷ temperature",
                   "sharpen or flatten", 11))
    b.append(line(gx + 362, yC, gx + 392, yC))
    b.append(block(gx + 448, yC - 22, 106, 44, GRAY, "top-k",
                   "drop the junk tail", 11))

    # softmax + sample downward
    sx = gx + 448
    b.append(line(sx, yC + 22, sx, yC + 50))
    b.append(block(sx, yC + 52, 106, 36, GRAY, "softmax → sample", None, 9.5))

    # the sampled token + append loop
    b.append(pill(sx, yC + 122, 106, 'next: "e"', 11, YELLOW))
    b.append(line(sx, yC + 90, sx, yC + 106))
    b.append(path(f"M {sx - 53} {yC + 122} L 80 {yC + 122} L 80 {yC + 18}"))
    b.append(text((sx + 130) / 2, yC + 138, "append and go around again — one token per lap",
                  10.5, MUTED))

    b.append(text(40, H - 58, "temperature < 1 sharpens (safe, repetitive) · = 1 is the honest learned distribution · > 1 flattens toward keysmash.",
                  10.5, INK2, "normal", "start"))
    b.append(text(40, H - 40, "top-k keeps only the k most likely tokens: individually-unlikely junk can add up to a big probability without it.",
                  10.5, INK2, "normal", "start"))
    save("fig-generation.svg", svg(W, H, "\n".join(b)))


# ---------------------------------------------------------------------------
# Part III figures: reinforcement learning
# ---------------------------------------------------------------------------

def fig_rl_loop():
    W, H = 880, 400
    b = [title_block("Reinforcement learning: the agent–environment loop",
                     "no dataset — the agent acts, the game answers with a reward and a new state, and it learns to score higher")]

    ax, ex, cy = 250, 630, 205
    bw, bh = 214, 96
    b.append(block(ax, cy - bh / 2, bw, bh, ORANGE, "Agent",
                   "policy  π(a | s)", 15, 12))
    b.append(block(ex, cy - bh / 2, bw, bh, GRAY, "Environment",
                   "GridWorld · Snake", 15, 12))

    # top: the action flows from agent to environment
    b.append(path(f"M {ax + 50} {cy - bh / 2} L {ax + 50} 133 L {ex} 133 L {ex} {cy - bh / 2}",
                  marker=True))
    b.append(pill((ax + ex) / 2 + 30, 133, 120, "action  aₜ", 12, YELLOW))
    # bottom: reward + next state flow back from environment to agent
    b.append(path(f"M {ex - 50} {cy + bh / 2} L {ex - 50} 277 L {ax} 277 L {ax} {cy + bh / 2}",
                  marker=True, marker_id="arr-blue", stroke=BLUE[1]))
    b.append(pill((ax + ex) / 2 - 30, 277, 244, "reward  rₜ   +   next state  sₜ₊₁", 12, BLUE))

    b.append(text(W / 2, 335,
                  "The goal: choose actions to maximise the return  G = r₀ + γ r₁ + γ² r₂ + …  (γ < 1 values sooner rewards more).",
                  11, INK2))
    b.append(legend(40, H - 26, [(ORANGE, "agent / policy"), (GRAY, "environment"),
                                 (YELLOW, "action"), (BLUE, "reward & state")]))
    save("fig-rl-loop.svg", svg(W, H, "\n".join(b)))


def fig_policy_gradient():
    W, H = 880, 486
    b = [title_block("Policy gradient: turn one played episode into a better policy",
                     "REINFORCE → Actor-Critic → PPO differ only in the weight they put on each action")]

    y0 = 132
    b.append(text(40, y0 - 20, "One episode, start to finish", 12, INK, "600", "start"))
    xs = [110, 250, 390, 520]
    steps = [("s₀", "a₀", "r₀"), ("s₁", "a₁", "r₁"), ("s₂", "a₂", "r₂"), ("…", "", "")]
    for i, x in enumerate(xs):
        s, a, r = steps[i]
        b.append(block(x, y0, 86, 38, BLUE, s, None, 13))
        if a:
            b.append(text(x, y0 + 60, f"action {a}", 9.5, MUTED))
            b.append(text(x, y0 + 74, f"reward {r}", 9.5, MUTED))
        if i < len(xs) - 1:
            b.append(line(x + 43, y0 + 19, xs[i + 1] - 43, y0 + 19))

    b.append(path(f"M 300 {y0 + 78} L 300 {y0 + 104}", marker=True))
    b.append(block(300, y0 + 106, 380, 38, GRAY,
                   "return  Gₜ = rₜ + γ rₜ₊₁ + γ² rₜ₊₂ + …", None, 12))
    b.append(path(f"M 300 {y0 + 144} L 300 {y0 + 170}", marker=True))
    b.append(block(300, y0 + 172, 380, 46, RED,
                   "loss = − log π(aₜ | sₜ) · Aₜ", "gradient ascent on reward", 13))
    b.append(text(300, y0 + 240, "make high-advantage actions more likely", 10.5, MUTED))

    px = 600
    b.append(rect(px, 106, 248, 264, "#ffffff", CONTAINER[1], rx=12, sw=1.3))
    b.append(text(px + 18, 132, "What is the advantage Aₜ?", 12.5, INK, "700", "start"))
    rows = [
        ("REINFORCE", "Gₜ − baseline (mean)", ORANGE),
        ("Actor-Critic", "Gₜ − V(sₜ),  V learned", AQUA),
        ("PPO", "clip the ratio · Aₜ", VIOLET),
    ]
    ry = 170
    for name, formula, col in rows:
        b.append(rect(px + 18, ry - 13, 14, 14, col[0], col[1], rx=3, sw=1.2))
        b.append(text(px + 40, ry, name, 11.5, INK, "600", "start"))
        b.append(text(px + 40, ry + 17, formula, 10, INK2, "normal", "start", font=MONO))
        ry += 58
    b.append(text(px + 18, ry + 2, "Same gradient, better baseline:", 10, MUTED, anchor="start"))
    b.append(text(px + 18, ry + 16, "less noise, faster learning.", 10, MUTED, anchor="start"))

    b.append(legend(40, H - 24, [(BLUE, "state"), (GRAY, "return"),
                                 (RED, "loss / gradient")]))
    save("fig-policy-gradient.svg", svg(W, H, "\n".join(b)))


def fig_dqn():
    W, H = 880, 452
    b = [title_block("DQN: learn Q(s, a), and two tricks that keep it stable",
                     "replay breaks the correlation between steps; a slow target network gives the Bellman update something steady to aim at")]

    # -- left column: environment -> transitions -> replay buffer ----------
    b.append(block(150, 110, 176, 50, GRAY, "Environment", "play & collect", 12.5))
    b.append(path("M 150 160 L 150 184", marker=True))
    b.append(pill(150, 200, 190, "(s, a, r, s′)", 11, GRAY, mono=True))
    b.append(path("M 150 216 L 150 250", marker=True))
    b.append(block(150, 252, 190, 52, GRAY, "Replay buffer", "a ring of past steps", 12))

    # random batch feeds both networks (a little bus at x = 340)
    b.append(path("M 245 278 L 340 278", marker=False))
    b.append(text(300, 268, "random batch", 9.5, MUTED))
    b.append(f"<line x1='340' y1='163' x2='340' y2='325' stroke='{INK2}' stroke-width='1.6'/>")
    b.append(path("M 340 163 L 366 163", marker=True))          # -> Q-network
    b.append(path("M 340 325 L 366 325", marker=True))          # -> target network

    # -- middle: the two networks ------------------------------------------
    b.append(block(470, 138, 196, 50, ORANGE, "Q-network", "Q(s, a) — trained", 12.5))
    b.append(block(470, 300, 196, 50, AQUA, "Target network", "slow copy of Q", 12))
    # soft update: the Q-network slowly trails into the target
    b.append(path("M 470 188 L 470 298", stroke=MUTED, sw=1.3, dash="4 3", marker=True))
    b.append(text(478, 246, "soft update", 9, MUTED, anchor="start"))

    # -- right: the loss (top) compares Q(s,a) with the Bellman target ------
    b.append(block(730, 138, 250, 50, RED, "Huber loss", "(Q(s,a) − target)²", 12.5))
    b.append(block(730, 300, 250, 50, GRAY, "r + γ maxₐ Q_target(s′, a)",
                   "the Bellman target", 11.5))
    b.append(path("M 568 163 L 603 163", marker=True))          # Q(s,a) -> loss
    b.append(path("M 568 325 L 603 325", marker=True))          # target net -> Bellman target
    b.append(path("M 730 300 L 730 190", marker=True))          # Bellman target -> loss
    # gradient from the loss back into the Q-network
    b.append(path("M 730 138 L 730 110 L 470 110 L 470 136",
                  stroke=RED[1], sw=1.5, dash="5 4", marker=True, marker_id="arr-red"))
    b.append(text(600, 102, "gradient → Q-network", 9.5, RED[1]))

    b.append(text(40, H - 44,
                  "Act ε-greedily: a random action with probability ε (high at first, then decaying), the argmax of Q otherwise.",
                  10.5, INK2, "normal", "start"))
    b.append(legend(40, H - 22, [(GRAY, "environment / data"), (ORANGE, "Q-network"),
                                 (AQUA, "target network"), (RED, "loss")]))
    save("fig-dqn.svg", svg(W, H, "\n".join(b)))


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    fig_babygpt()
    fig_tensors()
    fig_autograd()
    fig_mlp()
    fig_training_loop()
    fig_tokenization()
    fig_attention()
    fig_multihead()
    fig_generation()
    fig_rl_loop()
    fig_policy_gradient()
    fig_dqn()
