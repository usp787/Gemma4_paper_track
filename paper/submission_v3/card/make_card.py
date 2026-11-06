"""make_card.py: draws the Kaggle writeup's card and thumbnail image (560 x 280, plus a 2x copy).

Run from anywhere: python paper/submission_v3/card/make_card.py
Palette and fonts match paper/figures/make_figures.py; the tile strip is Fig 1a's loop motif.
Numbers come from the v3 refined submission: dedup proxy continuation 0.78 (k = 2) -> 0.99 (k >= 10);
loop task runs resolve 20% (11/54) against 37% (61/164) for the rest.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent

SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
TILE = "#efeee9"
SANS = ["Segoe UI", "Arial", "DejaVu Sans"]
MONO = ["Consolas", "DejaVu Sans Mono"]

W, H = 560, 280  # card size in px; axes use the same units, y up


def tile(ax, x, y, w, h, text, fc, tc, ec="none", ls="-", fs=9, family=MONO, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=5", fc=fc, ec=ec,
                                lw=1.0, ls=ls, zorder=2))
    ax.text(x + w / 2, y + h / 2 - 0.5, text, ha="center", va="center", color=tc, fontsize=fs,
            family=family, weight=weight, zorder=3)


def draw():
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    L = 28  # side margin

    # kicker and title
    ax.text(L, 246, "GEMMA 4 31B  ·  CODING AGENT", color=MUTED, fontsize=7.2, family=SANS, weight="bold",
            va="center")
    ax.text(L - 1, 218, "When Coding Agents Repeat", color=INK, fontsize=21, family=SANS, weight="bold",
            va="center")

    # tile strip: two ordinary calls, then the same call again and again
    tw, th, gap, y0 = 54, 34, 6, 118
    labels = ["read", "edit"] + ["grep"] * 6
    x = L
    blue_x = []
    for i, lab in enumerate(labels):
        if i < 2:
            tile(ax, x, y0, tw, th, lab, TILE, INK2)
        else:
            tile(ax, x, y0, tw, th, lab, BLUE, "white", weight="bold")
            blue_x.append(x)
        x += tw + gap
    ax.text(x + 4, y0 + th / 2, "…", color=BLUE, fontsize=16, family=SANS, va="center", weight="bold")

    # bracket over the repeated calls
    bx0, bx1, by = blue_x[0], blue_x[-1] + tw, y0 + th + 9
    ax.plot([bx0, bx0, bx1, bx1], [by - 4, by, by, by - 4], color=INK2, lw=1.0, solid_capstyle="butt")
    ax.text((bx0 + bx1) / 2, by + 9, "the same tool call, again and again", color=INK2, fontsize=8.2,
            family=SANS, ha="center", va="center")

    # two headline numbers
    ax.plot([L, W - L], [96, 96], color=GRID, lw=0.8)
    sy = 66
    ax.text(L, sy, "78% → 99%", color=BLUE, fontsize=19, family=SANS, weight="bold", va="center")
    ax.text(L, sy - 27, "chance of one more identical call, after 2 vs 10+\nin a row, with the request held fixed",
            color=INK2, fontsize=7.6, family=SANS, va="center", linespacing=1.25)

    rx = 300
    ax.text(rx, sy, "20% vs 37%", color=ORANGE, fontsize=19, family=SANS, weight="bold", va="center")
    ax.text(rx, sy - 27, "task runs resolved: runs that loop\nagainst all other runs",
            color=INK2, fontsize=7.6, family=SANS, va="center", linespacing=1.25)

    for scale, suffix in ((1, ""), (2, "@2x")):
        fig.savefig(OUT / f"card{suffix}.png", dpi=100 * scale, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    draw()
