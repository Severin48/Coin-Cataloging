"""Collection analysis: value distribution, what drives value, melt vs Numista (offline).

    uv run analyze.py [--run results/...]

Uses the same per-coin values as build_table.py (build_rows) and writes everything to
<run>/analysis/: report.md (key numbers + all plots), PNG plots, coins_analysis.csv
(one flat row per coin) and correlations.csv.

"Reliable" values come from a trusted Numista price, melt value or a manual override.
LLM guesses are shown separately: they cover the coins Numista hasn't priced yet and are
rough. Numista was queried in priority order, so the reliable subset leans towards the
coins that looked valuable; keep that in mind when comparing the two groups.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import build_table as B  # noqa: E402
from coins_common import DEFAULT_RUN, Item  # noqa: E402

# Validated categorical slots (light surface), in fixed order; gray for "other".
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY = "#9a9994"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
METAL_COLORS = {"gold": ORANGE, "silver": BLUE, "base": AQUA}
SOURCE_GROUPS = [("Numista", BLUE), ("Melt", ORANGE), ("LLM guess", GRAY)]

ERAS: List[Tuple[str, int, int]] = [
    ("before 1800", -9999, 1799),
    ("1800–1870", 1800, 1870),
    ("1871–1918", 1871, 1918),
    ("1919–1945", 1919, 1945),
    ("1946–1970", 1946, 1970),
    ("1971–2000", 1971, 2000),
    ("2001+", 2001, 9999),
]

COUNTRY_ALIAS = {
    "Bavaria": "Germany",
    "Hanover": "Germany",
    "Saxony": "Germany",
    "Prussia": "Germany",
    "Württemberg": "Germany",
    "Austria-Hungary": "Austria",
    "British India": "India",
    "United Arab Republic": "Egypt",
    "Russian Empire": "Russia / USSR",
    "Russian SFSR": "Russia / USSR",
    "Soviet Union": "Russia / USSR",
}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "font.size": 10,
        "legend.frameon": False,
        "lines.linewidth": 2,
    }
)
EUR_FMT = FuncFormatter(lambda v, _: f"{v:,.0f} €" if v >= 1 else f"{v:.2g} €")


# ------------------------------------------------------------------------- data


def parse_year(value: object) -> Optional[int]:
    m = re.search(r"\d{3,4}", str(value or ""))
    year = int(m.group()) if m else None
    return year if year and 500 <= year <= datetime.now().year else None


def country_group(country: Optional[str]) -> str:
    if not country:
        return "Unknown"
    base = re.split(r"\s*\(|\s+[-–]\s+", country)[0].strip()
    return COUNTRY_ALIAS.get(base, base)


def era_of(year: Optional[int]) -> Optional[str]:
    if year is None:
        return None
    return next(name for name, lo, hi in ERAS if lo <= year <= hi)


def source_group(source: Optional[str]) -> Optional[str]:
    if source is None:
        return None
    if source.startswith("numista_") or source == "override":
        return "Numista"
    return "Melt" if source == "melt" else "LLM guess"


def load(run: str) -> List[Item]:
    rows = B.build_rows(run, with_thumbs=False)
    out = []
    for r in rows:
        ident, match = r["_ident"], r["_match"]
        if "confidence" not in ident or ident.get("is_coin") is False:
            continue
        if ident.get("exclude") or r["est_value_eur"] is None:
            continue
        trusted = B.trusted_match(ident, match)
        year = parse_year(r["year"])
        mintage = match.get("mintage")
        out.append(
            {
                "coin_id": r["coin_id"],
                "country": r["country"],
                "country_group": country_group(r["country"]),
                "year": year,
                "era": era_of(year),
                "denomination": r["denomination"],
                "coin_type": r["coin_type"] or "unknown",
                "metal": B.metal(r["composition"]) or "unknown",
                "grade_guess": r["grade_guess"],
                "value": r["est_value_eur"],
                "value_source": r["value_source"],
                "source_group": source_group(r["value_source"]),
                "reliable": r["value_source"] not in (None, "llm_guess"),
                "melt": r["melt_eur"],
                "numista": r["price_at_grade"],
                "numista_grade": r["price_grade"],
                "llm_guess": r["llm_guess"],
                "silver_g": r["silver_g"],
                "gold_g": r["gold_g"],
                "weight_g": r["weight_g"],
                "mintage": mintage if isinstance(mintage, (int, float)) else None,
                "confidence": r["confidence"],
                "prices": (match.get("prices") or {}) if trusted else {},
            }
        )
    return out


# ------------------------------------------------------------------- utilities


def save(fig: plt.Figure, out_dir: str, name: str, saved: List[str]) -> None:
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, name), dpi=150)
    plt.close(fig)
    saved.append(name)


def eur(v: Optional[float]) -> str:
    return "–" if v is None else f"{v:,.2f} €"


def log_bins(values: Sequence[float], per_decade: int = 6) -> np.ndarray:
    lo = np.floor(np.log10(max(min(values), 0.01)))
    hi = np.ceil(np.log10(max(values)))
    return np.logspace(lo, hi, int((hi - lo) * per_decade) + 1)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Tuple[Optional[float], int]:
    if len(xs) < 10:
        return None, len(xs)
    rho = spearmanr(xs, ys).statistic
    return float(rho), len(xs)


def pairs(
    rows: List[Item], x: str, y: str, keep: Callable[[Item], bool] = lambda r: True
) -> Tuple[List[float], List[float]]:
    sel = [r for r in rows if r[x] is not None and r[y] is not None and keep(r)]
    return [r[x] for r in sel], [r[y] for r in sel]


# ------------------------------------------------------------------------ plots


def plot_distribution(rows: List[Item], out: str, saved: List[str]) -> Dict:
    values = [r["value"] for r in rows if r["value"] > 0]
    bins = log_bins(values)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    stacks = [
        [r["value"] for r in rows if r["value"] > 0 and r["source_group"] == g]
        for g, _ in SOURCE_GROUPS
    ]
    ax.hist(
        stacks,
        bins=bins,
        stacked=True,
        color=[c for _, c in SOURCE_GROUPS],
        label=[f"{g} ({len(s)})" for (g, _), s in zip(SOURCE_GROUPS, stacks)],
        edgecolor=SURFACE,
        linewidth=1,
    )
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(EUR_FMT)
    med = float(np.median(values))
    ax.axvline(med, color=INK_2, linewidth=1, linestyle="--")
    ax.annotate(
        f"median {eur(med)}",
        (med, ax.get_ylim()[1] * 0.95),
        xytext=(4, 0),
        textcoords="offset points",
        color=INK_2,
        va="top",
    )
    ax.set_title("How much is each coin worth?")
    ax.set_xlabel("Estimated value per coin (log scale)")
    ax.set_ylabel("Coins")
    ax.legend(title="Value from")
    save(fig, out, "01_value_distribution.png", saved)
    q = np.percentile(values, [25, 50, 75, 90, 99])
    return {
        "n": len(values),
        "total": sum(values),
        "mean": float(np.mean(values)),
        "quantiles": dict(zip(["p25", "p50", "p75", "p90", "p99"], q.tolist())),
        "below_1": sum(v < 1 for v in values),
        "1_to_10": sum(1 <= v < 10 for v in values),
        "10_to_100": sum(10 <= v < 100 for v in values),
        "100_plus": sum(v >= 100 for v in values),
    }


def plot_concentration(rows: List[Item], out: str, saved: List[str]) -> Dict:
    values = np.sort([r["value"] for r in rows])[::-1]
    cum = np.cumsum(values) / values.sum() * 100
    share = np.arange(1, len(values) + 1) / len(values) * 100
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(share, cum, color=BLUE)
    marks = {}
    for pct in (1, 5, 10, 20):
        k = max(1, int(round(len(values) * pct / 100)))
        marks[pct] = float(cum[k - 1])
        ax.plot(pct, cum[k - 1], "o", color=BLUE, markersize=6, mec=SURFACE, mew=2)
        ax.annotate(
            f"top {pct}% → {cum[k - 1]:.0f}% of value",
            (pct, cum[k - 1]),
            xytext=(8, -12),
            textcoords="offset points",
            color=INK_2,
        )
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 102)
    ax.set_title("A few coins carry most of the value")
    ax.set_xlabel("% of coins (most valuable first)")
    ax.set_ylabel("% of total value")
    save(fig, out, "02_value_concentration.png", saved)
    return marks


def plot_countries(rows: List[Item], out: str, saved: List[str]) -> List[Tuple]:
    total: Dict[str, Dict[bool, float]] = defaultdict(lambda: {True: 0.0, False: 0.0})
    count: Counter = Counter()
    for r in rows:
        total[r["country_group"]][r["reliable"]] += r["value"]
        count[r["country_group"]] += 1
    top = sorted(total, key=lambda c: -sum(total[c].values()))[:20][::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    rel = [total[c][True] for c in top]
    guess = [total[c][False] for c in top]
    ax.barh(top, rel, color=BLUE, label="Numista / melt", height=0.7)
    ax.barh(
        top,
        guess,
        left=rel,
        color=GRAY,
        label="LLM guess",
        height=0.7,
        edgecolor=SURFACE,
    )
    for i, c in enumerate(top):
        ax.annotate(
            f"{eur(rel[i] + guess[i])} · {count[c]} coins",
            (rel[i] + guess[i], i),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            color=INK_2,
            fontsize=8.5,
        )
    ax.xaxis.set_major_formatter(EUR_FMT)
    ax.set_xlim(0, max(a + b for a, b in zip(rel, guess)) * 1.35)
    ax.grid(axis="y", visible=False)
    ax.set_title("Total value by country (top 20)")
    ax.legend(loc="lower right")
    save(fig, out, "03_value_by_country.png", saved)

    # value per coin, countries with enough coins
    groups = defaultdict(list)
    for r in rows:
        groups[r["country_group"]].append(r["value"])
    big = [c for c, v in groups.items() if len(v) >= 10 and c != "Unknown"]
    big.sort(key=lambda c: np.median(groups[c]))
    fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(big) + 1)))
    ax.boxplot(
        [groups[c] for c in big],
        orientation="horizontal",
        tick_labels=[f"{c} ({len(groups[c])})" for c in big],
        widths=0.6,
        showfliers=True,
        flierprops={
            "marker": "o",
            "markersize": 3,
            "markerfacecolor": GRAY,
            "markeredgecolor": "none",
        },
        medianprops={"color": ORANGE, "linewidth": 2},
        boxprops={"color": BLUE},
        whiskerprops={"color": BLUE},
        capprops={"color": BLUE},
    )
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(EUR_FMT)
    ax.grid(axis="y", visible=False)
    ax.set_title("Value per coin by country (≥ 10 coins, sorted by median)")
    ax.set_xlabel("Estimated value per coin (log scale)")
    save(fig, out, "04_value_per_coin_by_country.png", saved)
    return [
        (c, count[c], sum(total[c].values()), float(np.median(groups[c])))
        for c in top[::-1]
    ]


def plot_year(rows: List[Item], out: str, saved: List[str]) -> None:
    sel = [r for r in rows if r["year"] and r["reliable"] and r["value"] > 0]
    fig, ax = plt.subplots(figsize=(10, 5))
    for metal_name in ("base", "silver", "gold"):
        pts = [r for r in sel if r["metal"] == metal_name]
        if pts:
            ax.scatter(
                [r["year"] for r in pts],
                [r["value"] for r in pts],
                s=18,
                color=METAL_COLORS[metal_name],
                alpha=0.65,
                edgecolors=SURFACE,
                linewidths=0.6,
                label=f"{metal_name} ({len(pts)})",
            )
    by_decade = defaultdict(list)
    for r in sel:
        by_decade[r["year"] // 10 * 10].append(r["value"])
    decades = sorted(d for d, v in by_decade.items() if len(v) >= 5)
    ax.plot(
        [d + 5 for d in decades],
        [np.median(by_decade[d]) for d in decades],
        color=INK,
        linewidth=2,
        label="median per decade (≥ 5 coins)",
    )
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(EUR_FMT)
    ax.set_title("Value vs year (Numista / melt values only)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Value per coin (log scale)")
    ax.legend(loc="lower left")
    save(fig, out, "05_value_vs_year.png", saved)


def plot_eras(rows: List[Item], out: str, saved: List[str]) -> List[Tuple]:
    names = [e[0] for e in ERAS]
    groups = {n: [r for r in rows if r["era"] == n] for n in names}
    names = [n for n in names if groups[n]]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    y = np.arange(len(names))
    panels = [
        ("Coins", [len(groups[n]) for n in names], "{:,.0f}"),
        ("Total value", [sum(r["value"] for r in groups[n]) for n in names], "{:,.0f} €"),
        ("Median value per coin", [float(np.median([r["value"] for r in groups[n]])) for n in names], "{:,.2f} €"),
    ]  # fmt: skip
    for ax, (title, vals, fmt) in zip(axes, panels):
        ax.barh(y, vals, color=BLUE, height=0.65)
        for i, v in enumerate(vals):
            ax.annotate(
                fmt.format(v),
                (v, i),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=8.5,
                color=INK_2,
            )
        ax.set_xlim(0, max(vals) * 1.35)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
    axes[0].set_yticks(y, names)
    axes[0].invert_yaxis()
    fig.suptitle(
        "By era (year as identified; all value sources)", x=0.01, ha="left", color=INK_2
    )
    save(fig, out, "06_value_by_era.png", saved)
    return [
        (n, len(groups[n]), sum(r["value"] for r in groups[n]), float(np.median([r["value"] for r in groups[n]])))
        for n in names
    ]  # fmt: skip


SHORT = {
    "circulation": "circul.",
    "commemorative": "commem.",
    "token_or_medal": "token/medal",
    "extremely_fine": "extr. fine",
    "uncirculated": "uncirc.",
    "circulated": "circulated",
}


def boxes(ax: plt.Axes, groups: Dict[str, List[float]], title: str) -> None:
    labels = [k for k in groups if groups[k]]
    ax.boxplot(
        [groups[k] for k in labels],
        tick_labels=[f"{SHORT.get(k, k)}\n({len(groups[k])})" for k in labels],
        widths=0.55,
        flierprops={"marker": "o", "markersize": 3, "markerfacecolor": GRAY, "markeredgecolor": "none"},
        medianprops={"color": ORANGE, "linewidth": 2},
        boxprops={"color": BLUE},
        whiskerprops={"color": BLUE},
        capprops={"color": BLUE},
    )  # fmt: skip
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(EUR_FMT)
    ax.grid(axis="x", visible=False)
    ax.set_title(title)


def plot_metal_type(rows: List[Item], out: str, saved: List[str]) -> Dict:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharey=True)
    pos = [r for r in rows if r["value"] > 0]
    metal_groups = {
        m: [r["value"] for r in pos if r["metal"] == m]
        for m in ("base", "silver", "gold", "unknown")
    }
    type_groups = {
        t: [r["value"] for r in pos if r["coin_type"] == t]
        for t in ("circulation", "commemorative", "bullion", "token_or_medal")
    }
    grade_groups = {
        g: [r["value"] for r in pos if r["grade_guess"] == g]
        for g in ("poor", "circulated", "extremely_fine", "uncirculated", "proof")
    }
    boxes(axes[0], metal_groups, "By metal")
    boxes(axes[1], type_groups, "By coin type")
    boxes(axes[2], grade_groups, "By condition (LLM's guess)")
    axes[2].tick_params(axis="x", labelsize=8)
    axes[0].set_ylabel("Value per coin (log scale)")
    save(fig, out, "07_value_by_metal_type_grade.png", saved)  # fmt: skip
    return {
        k: (len(v), float(np.median(v)), sum(v))
        for k, v in {**metal_groups, **type_groups}.items()
        if v
    }


def plot_melt_vs_numista(rows: List[Item], out: str, saved: List[str]) -> Dict:
    sel = [r for r in rows if r["melt"] and r["numista"]]
    if len(sel) < 3:
        return {"n": len(sel)}
    melt = np.array([r["melt"] for r in sel])
    num = np.array([r["numista"] for r in sel])
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for metal_name in ("silver", "gold"):
        idx = [i for i, r in enumerate(sel) if r["metal"] == metal_name]
        if idx:
            ax.scatter(
                melt[idx], num[idx], s=20, color=METAL_COLORS[metal_name], alpha=0.7,
                edgecolors=SURFACE, linewidths=0.6, label=f"{metal_name} ({len(idx)})",
            )  # fmt: skip
    lo, hi = min(melt.min(), num.min()) * 0.8, max(melt.max(), num.max()) * 1.25
    ax.plot([lo, hi], [lo, hi], color=INK_2, linewidth=1, linestyle="--")
    ax.annotate("Numista = melt", (hi / 4, hi / 4), xytext=(6, -10), textcoords="offset points", ha="left", color=INK_2)  # fmt: skip
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.xaxis.set_major_formatter(EUR_FMT)
    ax.yaxis.set_major_formatter(EUR_FMT)
    ax.set_title("Melt value vs Numista price (coins with both)")
    ax.set_xlabel("Melt value (today's metal price)")
    ax.set_ylabel(
        f"Numista price at assumed grade ({os.environ.get('ASSUMED_GRADE', 'xf')})"
    )
    ax.legend(loc="upper left")
    save(fig, out, "08_melt_vs_numista.png", saved)

    ratio = num / melt
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    bins = np.logspace(
        np.floor(np.log10(ratio.min())), np.ceil(np.log10(ratio.max())), 30
    )
    below, above = ratio[ratio < 1], ratio[ratio >= 1]
    ax.hist([below, above], bins=bins, stacked=True, color=[ORANGE, BLUE], edgecolor=SURFACE,
            label=[f"Numista below melt ({len(below)}) → melt value used", f"Numista at/above melt ({len(above)})"])  # fmt: skip
    ax.axvline(1, color=INK_2, linewidth=1, linestyle="--")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.set_title("Collector premium: Numista price ÷ melt value")
    ax.set_xlabel("Ratio (log scale)")
    ax.set_ylabel("Coins")
    ax.legend()
    save(fig, out, "09_numista_premium.png", saved)

    diff = num - melt
    return {
        "n": len(sel),
        "mean_melt": float(melt.mean()),
        "mean_numista": float(num.mean()),
        "mean_diff": float(diff.mean()),
        "median_diff": float(np.median(diff)),
        "median_ratio": float(np.median(ratio)),
        "geo_mean_ratio": float(np.exp(np.mean(np.log(ratio)))),
        "below_melt": int((ratio < 1).sum()),
        "above_2x": int((ratio >= 2).sum()),
        "sum_melt": float(melt.sum()),
        "sum_numista": float(num.sum()),
    }


GRADE_FROM_GUESS = {
    "poor": "vg",
    "circulated": "f",
    "extremely_fine": "xf",
    "uncirculated": "unc",
    "proof": "unc",
}


def plot_grade_sensitivity(rows: List[Item], out: str, saved: List[str]) -> Dict:
    """Total value of the Numista-priced coins if every coin were in grade g."""
    sel = [r for r in rows if r["prices"] and r["source_group"] != "LLM guess"]
    if not sel:
        return {}

    def value(r: Item, grade: str) -> float:
        price, _ = B.price_at(r["prices"], grade)
        return max(price or 0, r["melt"] or 0)

    totals = {g: sum(value(r, g) for r in sel) for g in B.GRADES}
    auto = sum(value(r, GRADE_FROM_GUESS.get(r["grade_guess"], "f")) for r in sel)
    current = os.environ.get("ASSUMED_GRADE", "xf").strip().lower() or "xf"
    labels = [g.upper() for g in B.GRADES] + ["per coin\n(LLM grade)"]
    vals = list(totals.values()) + [auto]
    colors = [ORANGE if g == current else BLUE for g in B.GRADES] + [AQUA]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(labels, vals, color=colors, width=0.65)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:,.0f} €", (i, v), xytext=(0, 4), textcoords="offset points", ha="center", color=INK_2, fontsize=9)  # fmt: skip
    ax.yaxis.set_major_formatter(EUR_FMT)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, max(vals) * 1.12)
    ax.set_title("Total value at each assumed grade")
    ax.set_xlabel(
        f"{len(sel)} Numista-priced coins · orange = current setting · green = each coin at the LLM's grade guess"
    )
    ax.set_ylabel("Total value (max of Numista price and melt)")
    save(fig, out, "10_grade_sensitivity.png", saved)  # fmt: skip
    return {"n": len(sel), "totals": totals, "auto": auto, "current": current}


def plot_llm_vs_reference(rows: List[Item], out: str, saved: List[str]) -> Dict:
    sel = [r for r in rows if r["reliable"] and r["llm_guess"] and r["value"] > 0 and r["value_source"] != "override"]  # fmt: skip
    if len(sel) < 3:
        return {"n": len(sel)}
    ref = np.array([r["value"] for r in sel])
    guess = np.array([r["llm_guess"] for r in sel])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        ref, guess, s=16, color=BLUE, alpha=0.55, edgecolors=SURFACE, linewidths=0.5
    )
    lo, hi = min(ref.min(), guess.min()) * 0.8, max(ref.max(), guess.max()) * 1.25
    ax.plot([lo, hi], [lo, hi], color=INK_2, linewidth=1, linestyle="--")
    for f in (0.5, 2):
        ax.plot([lo, hi], [lo * f, hi * f], color=GRID, linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.xaxis.set_major_formatter(EUR_FMT)
    ax.yaxis.set_major_formatter(EUR_FMT)
    ratio = guess / ref
    within = float(np.mean((ratio >= 0.5) & (ratio <= 2)) * 100)
    ax.set_title(f"How good are the LLM's value guesses? ({within:.0f}% within 2×)")
    ax.set_xlabel("Numista / melt value")
    ax.set_ylabel("LLM guess")
    save(fig, out, "11_llm_guess_vs_reference.png", saved)
    return {
        "n": len(sel),
        "median_ratio": float(np.median(ratio)),
        "within_2x": within,
        "sum_guess": float(guess.sum()),
        "sum_ref": float(ref.sum()),
    }


def plot_mintage(rows: List[Item], out: str, saved: List[str]) -> None:
    sel = [r for r in rows if r["mintage"] and r["mintage"] > 0 and r["numista"]]
    if len(sel) < 10:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for metal_name in ("base", "silver", "gold"):
        pts = [r for r in sel if r["metal"] == metal_name]
        if pts:
            ax.scatter([r["mintage"] for r in pts], [r["numista"] for r in pts], s=18, color=METAL_COLORS[metal_name],
                       alpha=0.65, edgecolors=SURFACE, linewidths=0.6, label=f"{metal_name} ({len(pts)})")  # fmt: skip
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(EUR_FMT)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_title("Rarity: mintage vs Numista price")
    ax.set_xlabel("Mintage (log scale)")
    ax.set_ylabel("Numista price (log scale)")
    ax.legend()
    save(fig, out, "12_mintage_vs_price.png", saved)


def correlations(rows: List[Item]) -> List[Tuple[str, str, Optional[float], int]]:
    """Spearman rank correlation of value with numeric factors (sign = direction)."""
    out = []
    for label, keep in (
        ("reliable", lambda r: r["reliable"]),
        ("all", lambda r: True),
    ):
        for factor, key in (
            ("year (newer →)", "year"),
            ("mintage", "mintage"),
            ("silver content g", "silver_g"),
            ("weight g", "weight_g"),
            ("LLM confidence", "confidence"),
        ):
            xs, ys = pairs(rows, key, "value", keep)
            rho, n = spearman(xs, ys)
            out.append((label, factor, rho, n))
    return out


# ----------------------------------------------------------------------- report


def write_csv(rows: List[Item], path: str) -> None:
    keys = [k for k in rows[0] if k != "prices"]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    parser.add_argument("--run", default=DEFAULT_RUN)
    args = parser.parse_args()
    out = os.path.join(args.run, "analysis")
    os.makedirs(out, exist_ok=True)

    rows = load(args.run)
    if not rows:
        sys.exit("No valued coins found - run build_table.py's inputs first.")
    saved: List[str] = []
    dist = plot_distribution(rows, out, saved)
    conc = plot_concentration(rows, out, saved)
    countries = plot_countries(rows, out, saved)
    plot_year(rows, out, saved)
    eras = plot_eras(rows, out, saved)
    groups = plot_metal_type(rows, out, saved)
    mvn = plot_melt_vs_numista(rows, out, saved)
    grade = plot_grade_sensitivity(rows, out, saved)
    llm = plot_llm_vs_reference(rows, out, saved)
    plot_mintage(rows, out, saved)
    corr = correlations(rows)

    write_csv(rows, os.path.join(out, "coins_analysis.csv"))
    with open(
        os.path.join(out, "correlations.csv"), "w", encoding="utf-8", newline=""
    ) as f:
        w = csv.writer(f)
        w.writerow(["subset", "factor", "spearman_rho", "n"])
        w.writerows(
            [(s, fa, None if r is None else round(r, 3), n) for s, fa, r, n in corr]
        )

    src = Counter(r["source_group"] for r in rows)
    grade_env = os.environ.get("ASSUMED_GRADE", "xf")
    q = dist["quantiles"]
    L = [
        "# Collection analysis",
        "",
        f"Built {datetime.now():%Y-%m-%d %H:%M} from `{args.run}` · silver "
        f"{B.env_float('SILVER_EUR_PER_G')} €/g · gold {B.env_float('GOLD_EUR_PER_G')} €/g · "
        f"assumed grade **{grade_env}**",
        "",
        f"{len(rows)} valued coins. Value from Numista: {src['Numista']}, melt: {src['Melt']}, "
        f"LLM guess: {src['LLM guess']} (not yet priced by Numista; rough). The Numista-priced "
        "coins were looked up in priority order, so they lean towards the more valuable ones.",
        "",
        "## Value distribution",
        "",
        f"- Total **{eur(dist['total'])}**, mean {eur(dist['mean'])}, median **{eur(q['p50'])}** "
        f"(quartiles {eur(q['p25'])} – {eur(q['p75'])}; 90th percentile {eur(q['p90'])}; "
        f"99th {eur(q['p99'])}).",
        f"- Under 1 €: {dist['below_1']} coins · 1–10 €: {dist['1_to_10']} · 10–100 €: "
        f"{dist['10_to_100']} · 100 € and up: {dist['100_plus']}.",
        f"- Concentration: the top 1% of coins hold {conc[1]:.0f}% of the value, top 5% "
        f"{conc[5]:.0f}%, top 10% {conc[10]:.0f}%, top 20% {conc[20]:.0f}%.",
        "",
        "![](01_value_distribution.png)",
        "",
        "![](02_value_concentration.png)",
        "",
        "## Melt vs Numista",
        "",
    ]  # fmt: skip
    if mvn.get("n", 0) >= 3:
        L += [
            f"{mvn['n']} precious-metal coins have both a melt value and a Numista price "
            f"(at grade {grade_env}).",
            "",
            f"- Average melt {eur(mvn['mean_melt'])} vs average Numista {eur(mvn['mean_numista'])}: "
            f"Numista is on average **{eur(mvn['mean_diff'])}** higher (median difference "
            f"{eur(mvn['median_diff'])}).",
            f"- Typical premium over melt: median ratio **{mvn['median_ratio']:.2f}×** "
            f"(geometric mean {mvn['geo_mean_ratio']:.2f}×). {mvn['above_2x']} coins are worth at "
            "least twice their metal.",
            f"- {mvn['below_melt']} coins have a Numista price *below* melt: Numista prices lag the "
            "current silver price, and the sheet already uses melt for those.",
            f"- Together: melt {eur(mvn['sum_melt'])} vs Numista {eur(mvn['sum_numista'])}.",
            "",
            "![](08_melt_vs_numista.png)",
            "",
            "![](09_numista_premium.png)",
            "",
        ]  # fmt: skip
    else:
        L += ["Too few coins with both a melt value and a Numista price.", ""]
    if grade:
        t = grade["totals"]
        cur = grade["current"]
        L += [
            "## Grade assumption",
            "",
            f"For the {grade['n']} Numista-priced coins, total value if every coin were in grade:",
            "",
            "| " + " | ".join(g.upper() for g in t) + " | per coin (LLM grade) |",
            "|" + "---|" * (len(t) + 1),
            "| " + " | ".join(f"{v:,.0f} €" for v in t.values()) + f" | {grade['auto']:,.0f} € |",
            "",
            f"The current setting ({cur.upper()}) gives {t.get(cur, 0):,.0f} €; F would give "
            f"{t['f']:,.0f} € ({(t['f'] / t[cur] - 1) * 100:+.0f}%) and VF {t['vf']:,.0f} € "
            f"({(t['vf'] / t[cur] - 1) * 100:+.0f}%). Melt is a floor for silver and gold, so the "
            "gap is mostly in the collector premium.",
            "",
            "![](10_grade_sensitivity.png)",
            "",
        ]  # fmt: skip
    L += [
        "## What goes with higher value",
        "",
        "Spearman rank correlation with value per coin (+1: always higher, 0: no relation). "
        "*reliable* = Numista/melt values only.",
        "",
        "| Factor | reliable | n | all | n |",
        "|---|---|---|---|---|",
    ]
    by = {(s, fa): (r, n) for s, fa, r, n in corr}
    for fa in dict.fromkeys(fa for _, fa, _, _ in corr):
        cells = []
        for s in ("reliable", "all"):
            r, n = by[(s, fa)]
            cells += ["–" if r is None else f"{r:+.2f}", str(n)]
        L.append(f"| {fa} | " + " | ".join(cells) + " |")
    L += [
        "",
        "| Group | Coins | Median per coin | Total |",
        "|---|---|---|---|",
    ]
    L += [
        f"| {k} | {n} | {eur(med)} | {eur(tot)} |"
        for k, (n, med, tot) in groups.items()
    ]
    L += [
        "",
        "![](07_value_by_metal_type_grade.png)",
        "",
        "![](05_value_vs_year.png)",
        "",
        "### By era",
        "",
        "| Era | Coins | Total | Median per coin |",
        "|---|---|---|---|",
    ]
    L += [f"| {n} | {c} | {eur(t)} | {eur(m)} |" for n, c, t, m in eras]
    L += [
        "",
        "![](06_value_by_era.png)",
        "",
        "### By country",
        "",
        "| Country | Coins | Total | Median per coin |",
        "|---|---|---|---|",
    ]
    L += [f"| {c} | {n} | {eur(t)} | {eur(m)} |" for c, n, t, m in countries]
    L += [
        "",
        "![](03_value_by_country.png)",
        "",
        "![](04_value_per_coin_by_country.png)",
        "",
    ]
    if "12_mintage_vs_price.png" in saved:
        L += ["![](12_mintage_vs_price.png)", ""]
    if llm.get("n", 0) >= 3:
        L += [
            "## LLM guesses vs real prices",
            "",
            f"On {llm['n']} coins with a Numista or melt value, the LLM's guess was within a "
            f"factor of 2 for {llm['within_2x']:.0f}% (median guess/real ratio "
            f"{llm['median_ratio']:.2f}; sums {eur(llm['sum_guess'])} vs {eur(llm['sum_ref'])}). "
            "Use this to judge how far to trust the LLM-guess part of the total.",
            "",
            "![](11_llm_guess_vs_reference.png)",
            "",
        ]
    L += ["Data: `coins_analysis.csv` (one row per coin), `correlations.csv`.", ""]
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(
        f"Wrote {out}/report.md, {len(saved)} plots, coins_analysis.csv, correlations.csv"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
