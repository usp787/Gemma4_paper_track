"""make_figures.py: draws the overview's figures from fig_data.py into this folder, as PNG (300 dpi) and SVG.

Run from anywhere: python paper/figures/make_figures.py
Written 2026-10-24 for paper/overview.html. Colours are the dataviz skill's reference palette (slots 1-3,
validated all-pairs in light mode; the aqua slot sits below 3:1 on the surface, so every chart that uses it
has a legend and a data table in the overview). Fig 6 is a model sketch, not a fit: its two parameters are set
from two observed points (see MODEL below).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from scipy.stats import beta as beta_dist

import fig_data as D

OUT = Path(__file__).resolve().parent

SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
WARN = "#fab219"  # status "warning", only with its icon and label
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # blue 250 / 400 / 550 / 700, ordinal

plt.rcParams.update({
    "font.family": ["Segoe UI", "Arial", "DejaVu Sans"],
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "axes.titlelocation": "left",
    "axes.labelcolor": INK2,
    "axes.edgecolor": BASE,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor": SURFACE,
    "axes.axisbelow": True,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": BASE,
    "ytick.color": BASE,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "svg.fonttype": "none",
})


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def wilson(r, n, z=1.96):
    """Binomial 95% interval. Ignores clustering by task, so it is narrower than the records' task resamples."""
    centre = (r + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(r * (1 - r) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def dot(ax, x, y, color, filled=True, size=5.5, **kw):
    ax.plot(x, y, "o", ms=size, color=color, mfc=color if filled else SURFACE, mec=color if not filled else SURFACE,
            mew=1.3 if not filled else 1.1, zorder=4, **kw)


# ---------------------------------------------------------------- Fig 1a: the loop and the two arms (drawing)
def box(ax, x, y, w, h, text, fc, tc=INK, ec="none", ls="-", fs=7.5, family=None, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08", fc=fc, ec=ec, lw=0.8,
                                ls=ls, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc, fontsize=fs, family=family,
            weight=weight, zorder=3)


def fig1a():
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.set_xlim(0, 14.4)
    ax.set_ylim(0, 7.8)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(0.1, 7.55, "A tool-call loop", fontsize=9.5, weight="bold", color=INK, va="center")
    ax.text(0.1, 7.05, "synthetic example:", fontsize=7.8, color=MUTED, va="center")
    ax.text(2.9, 7.05, 'c = grep -rn "parse_range" src/   →   (no output)', fontsize=7.6, color=INK2,
            va="center", family="Consolas")

    tiles = ["read", "grep", "read"] + ["c"] * 7 + ["submit"]
    w, step, y0, h = 0.98, 1.12, 5.35, 0.7
    for i, t in enumerate(tiles):
        x = 0.1 + i * step
        if t == "c":
            box(ax, x, y0, w, h, "c", BLUE, tc="#ffffff", fs=8.5, weight="bold", family="Consolas")
            ax.text(x + w / 2, y0 - 0.25, f"k = {i - 2}", ha="center", va="center", fontsize=7, color=INK2)
        else:
            box(ax, x, y0, w, h, t, "#ecebe6", tc=INK2)
    # brackets and labels
    xa, xb = 0.1 + 3 * step, 0.1 + 9 * step + w
    ax.plot([xa, xa, xb, xb], [6.2, 6.35, 6.35, 6.2], color=INK2, lw=0.8)
    ax.text((xa + xb) / 2, 6.55, "lock-in: the same call, again and again", ha="center", fontsize=8, color=INK)
    ax.annotate("onset", xy=(xa + w / 2, y0 - 0.45), xytext=(xa + w / 2 - 1.6, y0 - 0.95), fontsize=7.5,
                color=INK, ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7, shrinkA=0, shrinkB=2))
    xw = 0.1 + 10 * step - 0.07
    ax.add_patch(plt.Circle((xw, 4.45), 0.17, color=WARN, zorder=3))
    ax.text(xw, 4.45, "!", ha="center", va="center", fontsize=8, weight="bold", color=INK, zorder=4)
    ax.plot([xw, xw], [4.62, y0 - 0.05], color=INK2, lw=0.7)
    ax.text(xw + 0.3, 4.45, "budget warning in the result:\nthe run ends from outside", fontsize=7.2, color=INK,
            va="center")

    # what the model is shown
    ax.text(0.1, 3.45, "What the model is shown before its next call, at k = 4", fontsize=9, weight="bold",
            color=INK, va="center")
    rows = [(2.45, "Control", "copies in view", 4, "0.96"), (1.25, "Fixed request", "dedup proxy", 1, "0.88")]
    for y, name, sub, ncopies, p in rows:
        ax.text(0.1, y + 0.45, name, fontsize=8.2, weight="bold", color=INK, va="center")
        ax.text(0.1, y + 0.1, sub, fontsize=7.5, color=MUTED, va="center")
        box(ax, 2.2, y, 2.3, 0.62, "task + earlier calls", "#ecebe6", tc=INK2)
        for j in range(4):
            x = 4.65 + j * 1.18
            if j < ncopies:
                box(ax, x, y, 1.08, 0.62, "c\nno output", BLUE, tc="#ffffff", fs=6.3)
            else:
                box(ax, x, y, 1.08, 0.62, "removed", SURFACE, tc=MUTED, ec=BASE, ls=(0, (2, 2)), fs=7)
        ax.annotate("", xy=(9.75, y + 0.31), xytext=(9.4, y + 0.31),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8))
        ax.text(9.9, y + 0.31, "next call?", fontsize=7.8, color=INK2, va="center")
        ax.text(11.35, y + 0.31, f"P(c again) = {p}", fontsize=8, color=INK, va="center", weight="bold")
    ax.text(11.35, 2.45 - 0.18, "copies seal the escapes", fontsize=7, color=MUTED, va="center")
    ax.text(11.35, 1.25 - 0.18, "same request every step", fontsize=7, color=MUTED, va="center")
    ax.text(0.1, 0.25, "With the request fixed, each step is a fresh draw from the same prompt: the chance to continue can "
            "rise with k only\nthrough which states keep looping (selection). The gap to the control is what the copies "
            "add (reinforcement).", fontsize=7.2, color=INK2, va="center")
    save(fig, "fig1a_design")


# ---------------------------------------------------------------- Fig 1b: the lock-in hazard
SERIES = [
    ("ctrl_31b", "31B, copies in view (control)", BLUE, -0.09),
    ("dedup_31b", "31B, request held fixed (dedup)", ORANGE, 0.0),
    ("ctrl_12b", "12B, copies in view (control)", AQUA, 0.09),
]


def fig1b():
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [1, 3.4], "wspace": 0.3})
    for key, label, color, dx in SERIES:
        r, n = D.HAZARD[key][0]
        lo, hi = wilson(r, n)
        a0.errorbar(dx, r, yerr=[[r - lo], [hi - r]], fmt="none", ecolor=color, elinewidth=1.1, zorder=3)
        dot(a0, dx, r, color)
    a0.set_xlim(-0.3, 0.3)
    a0.set_xticks([0], ["k = 1"])
    a0.set_ylim(0, 0.08)
    a0.set_yticks([0, 0.02, 0.04, 0.06, 0.08])
    a0.grid(axis="x", visible=False)
    a0.set_title("Onset")
    a0.set_ylabel("P(one more identical call)")
    a0.annotate("12B: 2.4×", xy=(0.09, 0.058), xytext=(0.0, 0.072), fontsize=7.5, color=INK, ha="center",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))

    x = np.arange(4)
    for key, label, color, dx in SERIES:
        rs = np.array([r for r, _ in D.HAZARD[key][1:]])
        ci = np.array([wilson(r, n) for r, n in D.HAZARD[key][1:]])
        a1.plot(x + dx, rs, color=color, lw=1.6, solid_capstyle="round", solid_joinstyle="round", zorder=3)
        a1.errorbar(x + dx, rs, yerr=[rs - ci[:, 0], ci[:, 1] - rs], fmt="none", ecolor=color, elinewidth=1.0,
                    zorder=3)
        dot(a1, x + dx, rs, color, label=label)
    a1.set_xticks(x, ["k = 2", "3–4", "5–9", "10+"])
    a1.set_ylim(0.6, 1.02)
    a1.grid(axis="x", visible=False)
    a1.set_title("Lock-in, once copying has started")
    h, l = a1.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.07), handletextpad=0.3,
               columnspacing=1.6)
    a1.annotate("request held fixed, still 0.79 → 0.99:\nmost of the height is selection", xy=(0.03, 0.79),
                xytext=(1.25, 0.68), fontsize=7.5, color=INK, va="center",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6, shrinkB=4))
    a1.annotate("", xy=(2.0, 0.937), xytext=(2.0, 0.997),
                arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.7, shrinkA=1, shrinkB=1))
    a1.text(2.1, 0.895, "copies close\nthe last escapes", fontsize=7.5, color=INK, ha="left", va="center")
    save(fig, "fig1b_hazard")


# ---------------------------------------------------------------- Fig 2: the dose test
def fig2():
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    xs = np.array(D.DOSE_COPIES)
    for key, label, color in [("identical", "identical copies of c", BLUE),
                              ("respelled", "c, then re-spellings of it", ORANGE)]:
        ys = D.DOSE[key]
        ax.plot(xs, ys, color=color, lw=1.6, solid_capstyle="round", zorder=3)
        dot(ax, xs, ys, color, label=label)
    for xv, yv in [(2, D.DOSE_CLOSED_LOOP["after 2 identical"]), (4, D.DOSE_CLOSED_LOOP["at 3+"])]:
        ax.plot(xv, yv, "D", ms=5, mfc=SURFACE, mec=INK2, mew=1.1, zorder=4)
    ax.plot([], [], "D", ms=5, mfc=SURFACE, mec=INK2, mew=1.1, label="closed loop, for comparison")
    ax.annotate("", xy=(4.18, 0.434), xytext=(4.18, 0.038),
                arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.7, shrinkA=1, shrinkB=1))
    ax.text(4.28, 0.236, "+0.38\n(0.26 to 0.50)", fontsize=7.5, color=INK, va="center")
    ax.text(2.08, 0.775, "0.78 after 2", fontsize=7.2, color=INK2, va="center")
    ax.text(3.9, 0.957, "0.96 at 3+", fontsize=7.2, color=INK2, va="center", ha="right")
    ax.set_xticks([1, 2, 4], ["1", "2", "4"])
    ax.set_xlim(0.7, 5.0)
    ax.set_ylim(0, 1.02)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("times the empty search c is in the context")
    ax.set_ylabel("P(next call = c)")
    ax.set_title("Copies reinforce only as exact copies")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.70), handletextpad=0.3)
    save(fig, "fig2_dose")


# ---------------------------------------------------------------- Fig 3: the levers, loops against outcome
def fig3():
    rows = [("T = 1", 3.0, BLUE), ("always-block", 2.0, BLUE), ("dedup", 1.0, BLUE), ("12B", -0.4, MUTED)]
    names = {"T = 1": "T = 1 (Google's setting)", "always-block": "always-block template",
             "dedup": "dedup proxy", "12B": "12B instead of 31B\n(size, for scale)"}
    panels = [("stale", "Stale share of calls", "points", (-28, 24)),
              ("run_length", "Length of runs of 3+", "calls", (-14, 6)),
              ("resolved", "Resolved task runs", "points", (-32, 12))]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, (metric, title, unit, lim) in zip(axes, panels):
        ax.axvline(0, color=INK2, lw=0.8, zorder=1)
        for name, y, color in rows:
            est, lo, hi = D.LEVERS[name][metric]
            ax.plot([lo, hi], [y, y], color=color, lw=1.4, solid_capstyle="round", zorder=2)
            dot(ax, est, y, color, filled=(lo > 0 or hi < 0))
        ax.axhline(0.3, color=GRID, lw=0.8)
        ax.set_xlim(*lim)
        ax.grid(axis="y", visible=False)
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel(f"lever − control ({unit})")
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks([r[1] for r in rows], [names[r[0]] for r in rows])
    axes[0].set_ylim(-1.0, 3.6)
    axes[2].text(-31, 3.45, "the outcome", fontsize=7.2, color=MUTED)
    axes[0].text(-27, 3.45, "the loops", fontsize=7.2, color=MUTED)
    save(fig, "fig3_levers")


# ---------------------------------------------------------------- Fig 4: how runs end
def fig4():
    cats = ["model's own exit", "budget warning", "compaction", "task end"]
    colors = [BLUE, INK2, MUTED, BASE]
    ink_on = ["#ffffff", "#ffffff", "#ffffff", INK]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True, gridspec_kw={"wspace": 0.2})
    for ax, data, title in [(axes[0], D.EXITS_3PLUS, "All runs of 3+ identical calls"),
                            (axes[1], D.EXITS_10PLUS, "Runs of 10+ identical calls")]:
        for i, arm in enumerate(D.EXIT_ARMS):
            counts = np.array(data[arm], float)
            shares = counts / counts.sum()
            left = 0.0
            for j, s in enumerate(shares):
                ax.barh(i, s, left=left, height=0.58, color=colors[j], edgecolor=SURFACE, linewidth=1.2,
                        label=cats[j] if i == 0 else None)
                if s >= 0.1:
                    ax.text(left + s / 2, i, f"{s:.0%}", ha="center", va="center", fontsize=7, color=ink_on[j])
                left += s
            ax.text(1.02, i, f"{int(counts.sum())}" + (" runs" if i == 0 else ""), va="center", fontsize=7,
                    color=MUTED)
        ax.set_xlim(0, 1.0)
        ax.set_xticks([0, 0.5, 1.0], ["0", "50%", "100%"])
        ax.grid(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_title(title, fontsize=8.5)
    axes[0].set_yticks(range(len(D.EXIT_ARMS)), D.EXIT_ARMS)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06), handlelength=1.0,
               handletextpad=0.4, columnspacing=1.4)
    save(fig, "fig4_exits")


# ---------------------------------------------------------------- Fig 5: T-open
def fig5():
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    groups = [("skipped", "where the model skipped the thought\n(114 states, 36 tasks)"),
              ("thought", "where it chose to think\n(99 states, 34 tasks)")]
    wbar = 0.3
    for gi, (key, _) in enumerate(groups):
        for k, (arm, color, dx) in enumerate([("T-think", MUTED, -0.17), ("T-open", BLUE, 0.17)]):
            e, n = D.TOPEN[key][arm]
            ax.bar(gi + dx, e / n, width=wbar, color=color,
                   label=(["free to think (T-think)", "thought forced open (T-open)"][k] if gi == 0 else None))
            ax.text(gi + dx, e / n + 0.02, f"{e / n:.0%}", ha="center", va="bottom", fontsize=7.5, color=INK)
    d, lo, hi = D.TOPEN["skipped"]["diff"]
    ax.annotate("", xy=(0.17, 0.66), xytext=(-0.17, 0.66),
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
    ax.text(0.0, 0.69, f"+{d:.0f} points ({lo:.0f} to {hi:.0f})", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks([0, 1], [g[1] for g in groups])
    ax.set_ylim(0, 0.95)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8], ["0", "20%", "40%", "60%", "80%"])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("next call escapes the loop")
    ax.set_title("A forced thought helps where the model skipped one")
    ax.legend(loc="upper right", handlelength=1.0)
    ax.tick_params(axis="x", length=0)
    save(fig, "fig5_topen")


# ---------------------------------------------------------------- Fig 6: the formal model, as a sketch
def model_params():
    """Beta(a, b) mixing distribution set from the dedup hazard at k = 1 and k = 2: h(k) = (a+k-1)/(a+b+k-1)."""
    h1, h2 = D.HAZARD["dedup_31b"][0][0], D.HAZARD["dedup_31b"][1][0]
    s = (1 - h2) / (h2 - h1)  # a + b, from h1 = a/s and h2 = (a+1)/(s+1)
    return h1 * s, s - h1 * s


def model_hazard(k, a, b):
    return (a + k - 1) / (a + b + k - 1)


def model_bins(a, b, kmax=60):
    """Bin averages of h(k), weighted by the states at risk at k, S(k) = E[p^(k-1)] (censoring ignored)."""
    ks = np.arange(1, kmax + 1)
    s = np.cumprod(np.r_[1.0, model_hazard(ks[:-1], a, b)])
    h = model_hazard(ks, a, b)
    out = {}
    for name, lo, hi in [("1", 1, 1), ("2", 2, 2), ("3–4", 3, 4), ("5–9", 5, 9), ("10+", 10, kmax)]:
        m = (ks >= lo) & (ks <= hi)
        out[name] = float((h[m] * s[m]).sum() / s[m].sum())
    return out


def fig6():
    a, b = model_params()
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.32})
    ks = [1, 2, 3, 5, 10, 30]
    edges = [0, 0.5, 0.9, 0.99, 1.0]
    bands = ["p < 0.5", "0.5–0.9", "0.9–0.99", "p ≥ 0.99"]
    for i, k in enumerate(ks):
        cdf = beta_dist.cdf(edges, a + k - 1, b)
        shares = np.diff(cdf)
        bottom = 0.0
        for j in range(3, -1, -1):  # the near-certain states at the bottom
            a0.bar(i, shares[j], bottom=bottom, width=0.62, color=RAMP[j], edgecolor=SURFACE, linewidth=1.0,
                   label=bands[j] if i == 0 else None)
            bottom += shares[j]
    a0.set_xticks(range(len(ks)), [f"k = {ks[0]}"] + [str(k) for k in ks[1:]])
    a0.set_ylim(0, 1)
    a0.set_yticks([0, 0.5, 1], ["0", "50%", "100%"])
    a0.grid(False)
    a0.set_title("Who is still copying at k", fontsize=8.5)
    a0.set_ylabel("share of states still in the run")
    h, l = a0.get_legend_handles_labels()
    a0.legend(h[::-1], l[::-1], title="each state's chance to draw the same call again", title_fontsize=7,
              loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=7, handlelength=0.9,
              columnspacing=0.8, handletextpad=0.3)
    a0.tick_params(axis="x", length=0)

    kk = np.arange(1, 61)
    a1.plot(kk, model_hazard(kk, a, b), color=INK2, lw=1.4, zorder=2, label="sketch: Beta-geometric")
    spans = [(1, 1), (2, 2), (3, 4), (5, 9), (10, 60)]
    for key, color, label in [("dedup_31b", ORANGE, "observed, request held fixed"),
                              ("ctrl_31b", BLUE, "observed, copies in view")]:
        for (lo, hi), (r, _) in zip(spans, D.HAZARD[key]):
            if hi > lo:
                a1.plot([lo, hi], [r, r], color=color, lw=1.6, solid_capstyle="round", zorder=3)
            dot(a1, np.sqrt(lo * hi), r, color, size=5)
        a1.plot([], [], "o-", color=color, ms=5, mec=SURFACE, lw=1.6, label=label)
    a1.set_xscale("log")
    a1.set_xticks([1, 2, 5, 10, 30, 60], ["1", "2", "5", "10", "30", "60"])
    a1.minorticks_off()
    a1.set_ylim(0, 1.04)
    a1.set_xlabel("k, identical calls so far")
    a1.set_ylabel("P(one more identical call)")
    a1.set_title("The hazard one mixing distribution implies", fontsize=8.5)
    a1.legend(loc="lower right", fontsize=7, handlelength=1.4)
    a1.text(2.6, 0.5, f"a = {a:.3f}, b = {b:.2f}, set from k = 1 and 2\nb < 1: no finite mean run length,\n"
            "so a run ends from outside", fontsize=7, color=INK2, va="center")
    save(fig, "fig6_model")
    return a, b


if __name__ == "__main__":
    fig1a()
    fig1b()
    fig2()
    fig3()
    fig4()
    fig5()
    a, b = fig6()
    print(f"model: a = {a:.5f}, b = {b:.5f}")
    for name, v in model_bins(a, b).items():
        print(f"  bin {name}: model {v:.3f}")
    for k in (1, 2, 3, 5, 10, 30):
        cdf = beta_dist.cdf([0.5, 0.9, 0.99], a + k - 1, b)
        print(f"  k={k}: p<0.5 {cdf[0]:.3f}  0.5-0.9 {cdf[1]-cdf[0]:.3f}  0.9-0.99 {cdf[2]-cdf[1]:.3f}  "
              f">=0.99 {1-cdf[2]:.3f}")
    print("wrote", sorted(p.name for p in OUT.glob("fig*.png")))
