"""make_card.py: draws the Kaggle writeup's card image as a book-style cover.

Run from anywhere: python paper/submission_v3/card/make_card.py

Kaggle cuts two crops from one upload: a 2:1 "cover" (560 x 280) and a square "thumbnail". Everything that
matters sits in the centre square (x 140-420 at 560 x 280), so the thumbnail is a complete cover on its own.
The sides are only texture: dim tool calls of an agent at work, and one bright row where the same call
repeats from edge to edge (the loop of Fig 1a).

Outputs: card.png (560 x 280), card@2x.png (1120 x 560, upload this one) and thumb_preview.png (the centre
square, to check the thumbnail crop).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

OUT = Path(__file__).resolve().parent

W, H = 560, 280            # card size in px; the axes use the same units, y up
CX = W / 2                 # centre of the thumbnail square
SQ = H                     # thumbnail square side

NAVY = "#0b2c57"
DIM_TILE, DIM_TEXT = "#16406f", "#3a6496"
GRAY_TILE, GRAY_TEXT = "#2a4a70", "#9db3cf"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
WHITE, PALE, SOFT = "#ffffff", "#c9dcf5", "#86b6ef"
SANS = ["Segoe UI", "Arial", "DejaVu Sans"]
MONO = ["Consolas", "DejaVu Sans Mono"]

TW, TH, GAP = 44, 22, 6    # wallpaper tile size and spacing
ROW_Y = [252, 222, 192, 162, 132, 102, 72, 42, 12]   # tile bottoms, top row first
LOOP_ROW = 102             # the bright row
CALLS = ["read", "search", "edit", "grep", "read", "submit", "ls", "edit", "search", "read", "grep", "edit"]


def tile(ax, x, y, w, h, text, fc, tc, fs=6.2, weight="normal", alpha=1.0, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=4", fc=fc, ec="none",
                                alpha=alpha, zorder=z))
    ax.text(x + w / 2, y + h / 2 - 0.4, text, ha="center", va="center", color=tc, fontsize=fs,
            family=MONO, weight=weight, alpha=alpha, zorder=z + 1)


SQ_L, SQ_R = CX - SQ / 2, CX + SQ / 2   # thumbnail square edges
CLEAR = 8                                # wallpaper keeps this far outside the square


def wallpaper(ax):
    """Dim calls on both sides only; none enters the thumbnail square. Darker nearer the square."""
    for r, y in enumerate(ROW_Y):
        if y == LOOP_ROW:
            continue
        offset = (r % 2) * (TW + GAP) / 2 - 20
        x, i = offset, r * 3
        while x < W:
            if x + TW <= SQ_L - CLEAR:
                d = SQ_L - CLEAR - (x + TW)
            elif x >= SQ_R + CLEAR:
                d = x - (SQ_R + CLEAR)
            else:
                d = None
            if d is not None:
                tile(ax, x, y, TW, TH, CALLS[i % len(CALLS)], DIM_TILE, DIM_TEXT,
                     alpha=0.45 + 0.55 * min(1.0, d / 90), z=1)
            x += TW + GAP
            i += 1


def loop_row(ax):
    """Two ordinary calls on the far left, then the same call again and again, through the cover.
    Tiles are centred on CX so the square cuts the row symmetrically."""
    pitch = TW + GAP
    for k in range(-5, 6):
        x = CX + k * pitch - TW / 2
        if k < -3:
            tile(ax, x, LOOP_ROW, TW, TH, ["read", "edit"][k + 5], GRAY_TILE, GRAY_TEXT, z=4)
        else:
            tile(ax, x, LOOP_ROW, TW, TH, "grep", BLUE, WHITE, weight="bold", z=4)


def fit_text(ax, fig, x, y, s, max_w, size, **kw):
    """Place text, shrinking it until it is at most max_w data units wide."""
    renderer = fig.canvas.get_renderer()
    t = ax.text(x, y, s, fontsize=size, **kw)
    while True:
        bb = t.get_window_extent(renderer).transformed(ax.transData.inverted())
        if bb.width <= max_w or size < 6:
            return t
        size -= 0.5
        t.set_fontsize(size)


def draw():
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_facecolor(NAVY)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.set_facecolor(NAVY)

    wallpaper(ax)
    loop_row(ax)

    # cover text, all inside the centre square
    left = CX - SQ / 2 + 26
    width = SQ - 52
    ax.text(left, 252, "GEMMA 4 31B  ·  CODING AGENT", color=SOFT, fontsize=6.6, family=SANS,
            weight="bold", va="center", zorder=5)
    ax.plot([left, left + 22], [240, 240], color=ORANGE, lw=2.2, solid_capstyle="butt", zorder=5)
    fit_text(ax, fig, left - 1, 210, "When Coding", width, 25, color=WHITE, family=SANS, weight="black",
             va="center", zorder=5)
    fit_text(ax, fig, left - 1, 178, "Agents Repeat", width, 25, color=WHITE, family=SANS, weight="black",
             va="center", zorder=5)
    ax.text(left, 145, "the same tool call, again and again", color=PALE, fontsize=7.4, family=SANS,
            style="italic", va="center", zorder=5)

    ax.text(left, 62, "Persistence, Selection, and the\nLimits of Loop Breaking", color=PALE, fontsize=8.4,
            family=SANS, weight="semibold", va="center", linespacing=1.3, zorder=5)
    ax.text(left, 24, "Jiarui Zha", color=SOFT, fontsize=7.2, family=SANS, va="center", zorder=5)

    for scale, suffix in ((1, ""), (2, "@2x")):
        fig.savefig(OUT / f"card{suffix}.png", dpi=100 * scale, facecolor=NAVY)
    plt.close(fig)

    big = Image.open(OUT / "card@2x.png")
    s = big.height
    x0 = (big.width - s) // 2
    big.crop((x0, 0, x0 + s, s)).save(OUT / "thumb_preview.png")


if __name__ == "__main__":
    draw()
