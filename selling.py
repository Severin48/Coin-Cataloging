"""Selling plan: which coins to sell one by one, and which in bulk or by metal weight.

    uv run selling.py [--run results/...] [--lang EN|DE]

Writes <run>/selling/ (offline, uses the same values as build_table.py):
  selling_guide.md   the plan, steps, silver by fineness, lots, the singles with pictures
  selling_plan.xlsx  sheets Overview, Singles, Precious metal, Lots: picture, coin ID, where
                     to find it (page, row, position + a marked page thumbnail for singles),
                     value, melt value and expected proceeds. Language: SUMMARY_LANGUAGE.
  selling_plan.csv   the route of every coin
  tradeoff.png       expected proceeds against the number of coins sold individually
  locate/            page thumbnails with the single coins marked

Every coin gets the cheapest route that realizes most of its value:

- gold    -> precious-metal dealer, priced by gold content
- silver  -> precious-metal dealer, priced by fine silver weight
- lot     -> base-metal coins worth >= SELL_LOT_MIN_EUR, sold in themed lots (by country)
- bulk    -> everything else, sold by the kilo / as a mixed lot
- single  -> sold individually (eBay, dealer, auction), only when that earns at least
             SELL_SINGLE_MIN_GAIN_EUR more than the coin's bulk route; at most
             SELL_MAX_SINGLES coins, best first

The rates are the share of the value you can expect to actually receive (after fees,
shipping, dealer margin, real grade below the assumed one). All of them can be set in .env;
the defaults are deliberately conservative.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter

import analyze as A
import build_table as B
import i18n
from coins_common import DEFAULT_RUN, IMG_DIR, Item, load_coins
from i18n import localize, tr

# name -> (default, meaning); every value can be overridden in .env
SETTINGS: Dict[str, Tuple[float, str]] = {
    "SELL_MAX_SINGLES": (50, "max. coins to sell individually"),
    "SELL_SINGLE_MIN_GAIN_EUR": (10, "a single listing must earn at least this much more than bulk"),
    "SELL_SINGLE_RATE": (0.70, "share of Numista value realized when sold individually"),
    "SELL_SILVER_RATE": (0.85, "share of melt value a precious-metal dealer pays for silver"),
    "SELL_GOLD_RATE": (0.95, "share of melt value a dealer pays for gold"),
    "SELL_LOT_MIN_EUR": (3, "base-metal coins worth at least this go into themed lots"),
    "SELL_LOT_RATE": (0.40, "share of value realized in themed lots"),
    "SELL_BULK_RATE": (0.15, "share of value realized for bulk / per-kilo coins"),
    "SELL_AUCTION_MIN_EUR": (150, "singles worth more than this: dealer or auction house"),
}  # fmt: skip

ROUTES = {
    "single": "Sell individually",
    "gold": "Gold: precious-metal dealer",
    "silver": "Silver: precious-metal dealer by weight",
    "lot": "Themed lots (base metal)",
    "bulk": "Bulk / by the kilo",
}
OUTLIER_NOTES = "outlier_notes.csv"
THUMB = 80  # px per coin side in the workbook
LOCATE_W, LOCATE_H = 220, 150  # px box the page thumbnail fits into


# ----------------------------------------------------------------------- plan


def settings() -> Dict[str, float]:
    out = {}
    for name, (default, _) in SETTINGS.items():
        out[name] = B.env_float(name) if os.environ.get(name, "").strip() else default
    return out


def outlier_ids() -> set:
    if not os.path.exists(OUTLIER_NOTES):
        return set()
    with open(OUTLIER_NOTES, encoding="utf-8", newline="") as f:
        return {row["coin_id"] for row in csv.DictReader(f)}


def verify_reasons(r: Item, outliers: set) -> str:
    reasons = []
    if not r["reliable"]:
        reasons.append("value is an LLM guess")
    review = r.get("needs_review") or ""
    for key, text in (
        ("ambiguous", "Numista match ambiguous"),
        ("disagree", "duplicates identified differently"),
        ("metal differs", "metal differs"),
        ("≫", "price ≫ LLM estimate"),
        ("low confidence", "low LLM confidence"),
    ):
        if key in review:
            reasons.append(text)
    if r["coin_id"] in outliers:
        reasons.append("see outlier_review.xlsx")
    return "; ".join(reasons)


def fallback(r: Item, s: Dict[str, float]) -> Tuple[str, float]:
    """Route and expected proceeds if the coin is NOT sold individually."""
    if r["metal"] == "gold":
        return "gold", (r["melt"] or r["value"]) * s["SELL_GOLD_RATE"]
    if r["metal"] == "silver":
        # without a known weight the dealer weighs it; the LLM guess stands in for melt
        return "silver", (r["melt"] or r["value"]) * s["SELL_SILVER_RATE"]
    if r["value"] >= s["SELL_LOT_MIN_EUR"]:
        return "lot", r["value"] * s["SELL_LOT_RATE"]
    return "bulk", r["value"] * s["SELL_BULK_RATE"]


def plan(rows: List[Item]) -> Tuple[List[Item], Dict[str, float]]:
    """All coins, ranked by how much selling them individually gains, with a route each."""
    s = settings()
    outliers = outlier_ids()
    out = []
    for r in rows:
        route, base = fallback(r, s)
        single = r["value"] * s["SELL_SINGLE_RATE"]
        out.append(
            {
                **r,
                "fallback_route": route,
                "fallback_eur": base,
                "single_eur": single,
                "gain_eur": single - base,
                "verify": verify_reasons(r, outliers),
                "silver_fineness": B.fineness(r.get("composition"), "silver")
                if r["silver_g"]
                else None,
            }
        )
    ranked = sorted(out, key=lambda p: -p["gain_eur"])
    n_single = 0
    for p in ranked:
        is_single = (
            n_single < s["SELL_MAX_SINGLES"]
            and p["gain_eur"] >= s["SELL_SINGLE_MIN_GAIN_EUR"]
        )
        n_single += is_single
        p["route"] = "single" if is_single else p["fallback_route"]
        p["expected_eur"] = p["single_eur"] if is_single else p["fallback_eur"]
        p["where"] = (
            ("dealer / auction" if p["value"] >= s["SELL_AUCTION_MIN_EUR"] else "eBay")
            if is_single
            else None
        )
    return ranked, s


def lot_groups(plans: List[Item], min_coins: int = 10) -> Dict[str, List[Item]]:
    """Themed lots by country; small countries go into one 'world mix' lot."""
    by_country = defaultdict(list)
    for p in plans:
        if p["route"] == "lot":
            by_country[p["country_group"]].append(p)
    lots: Dict[str, List[Item]] = defaultdict(list)
    for country, members in by_country.items():
        key = country if len(members) >= min_coins else "World mix"
        lots[key].extend(members)
    lots = dict(sorted(lots.items(), key=lambda kv: -sum(p["value"] for p in kv[1])))
    for name, members in lots.items():
        for p in members:
            p["lot"] = name
    return lots


def by_route(plans: List[Item]) -> Dict[str, List[Item]]:
    out: Dict[str, List[Item]] = defaultdict(list)
    for p in plans:
        out[p["route"]].append(p)
    return out


def summary(plans: List[Item], lots: Dict[str, List[Item]]) -> Dict[str, Any]:
    routes = by_route(plans)
    dealer = int(bool(routes["silver"] or routes["gold"]))
    bulk = int(bool(routes["bulk"]))
    sales = {
        "single": len(routes["single"]),
        "silver": dealer,
        "gold": 0 if routes["silver"] else dealer,
        "lot": len(lots),
        "bulk": bulk,
    }
    return {
        "routes": routes,
        "sales": sales,
        "effort": sum(sales.values()),
        "total": sum(p["expected_eur"] for p in plans),
        "catalog": sum(p["value"] for p in plans),
        "no_singles": sum(p["fallback_eur"] for p in plans),
        "no_singles_sales": dealer + len(lots) + bulk,
        "all_singles": sum(p["single_eur"] for p in plans),
    }


# ------------------------------------------------------------------ locations


def add_locations(run: str, plans: List[Item]) -> None:
    """Page (scan name), row from the top and position from the left of every coin.

    Coins are numbered row by row, left to right on the front scan, so a new row starts
    wherever the x position jumps back to the left.
    """
    coins = load_coins(run)
    by_page: Dict[str, List[Item]] = defaultdict(list)
    for c in coins:
        by_page[c["pair_key"]].append(c)
    where: Dict[str, Tuple[int, int, Item]] = {}
    for page, members in by_page.items():
        members.sort(key=lambda c: c["coin_number"])
        row, pos, last_x = 1, 0, None
        for c in members:
            x = c["front_rect"]["x"]
            if last_x is not None and x < last_x:
                row, pos = row + 1, 0
            pos += 1
            last_x = x
            where[c["coin_id"]] = (row, pos, c)
    for p in plans:
        row, pos, coin = where[p["coin_id"]]
        p.update(
            page=coin["pair_key"],
            coin_number=coin["coin_number"],
            row=row,
            pos=pos,
            front_scan=coin["front_image"],
            _coin=coin,
        )


def write_locators(plans: List[Item], out_dir: str) -> None:
    """Page thumbnail with the coin marked, for the singles."""
    os.makedirs(out_dir, exist_ok=True)
    by_scan: Dict[str, List[Item]] = defaultdict(list)
    for p in plans:
        by_scan[p["front_scan"]].append(p)
    for scan, members in by_scan.items():
        img = cv2.imread(os.path.join(IMG_DIR, scan))
        if img is None:
            continue
        scale = min(LOCATE_W / img.shape[1], LOCATE_H / img.shape[0])
        small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        for p in members:
            r = p["_coin"]["front_rect"]
            pic = small.copy()
            x, y = int(r["x"] * scale), int(r["y"] * scale)
            w, h = int(r["w"] * scale), int(r["h"] * scale)
            cv2.rectangle(pic, (x - 2, y - 2), (x + w + 2, y + h + 2), (0, 0, 255), 3)
            path = os.path.join(out_dir, f"{p['coin_id']}.jpg")
            cv2.imwrite(path, pic, [cv2.IMWRITE_JPEG_QUALITY, 85])
            p["locator"] = path


def thumb(run: str, coin_id: str) -> Optional[np.ndarray]:
    sides = []
    for side in ("front", "back"):
        img = cv2.imread(os.path.join(run, "crops", f"{coin_id}_{side}.jpg"))
        if img is None:
            return None
        sides.append(cv2.resize(img, (THUMB, THUMB), interpolation=cv2.INTER_AREA))
    return np.hstack(sides)


# --------------------------------------------------------------------- outputs


def write_plan_csv(plans: List[Item], path: str) -> None:
    order = list(ROUTES)
    keys = [
        "route", "coin_id", "page", "row", "pos", "country", "denomination", "year",
        "title", "value", "value_source", "melt", "silver_g", "silver_fineness", "gold_g",
        "expected_eur", "gain_eur", "where", "lot", "verify",
    ]  # fmt: skip
    rows = sorted(plans, key=lambda p: (order.index(p["route"]), -p["expected_eur"]))
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for p in rows:
            w.writerow({**p, "expected_eur": round(p["expected_eur"], 2), "gain_eur": round(p["gain_eur"], 2)})  # fmt: skip


def plot_tradeoff(plans: List[Item], s: Dict[str, float], path: str) -> None:
    """Expected proceeds against the number of coins sold individually."""
    gains = np.array([max(p["gain_eur"], 0) for p in plans])  # plans are ranked by gain
    base = sum(p["fallback_eur"] for p in plans)
    n_max = int(min(len(gains), max(200, s["SELL_MAX_SINGLES"] * 3)))
    xs = np.arange(0, n_max + 1)
    ys = base + np.concatenate([[0], np.cumsum(gains[:n_max])])
    chosen = sum(p["route"] == "single" for p in plans)
    nxt = gains[chosen] if chosen < len(gains) else 0
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(xs, ys, color=A.BLUE)
    ax.plot(chosen, ys[chosen], "o", color=A.ORANGE, markersize=8, mec=A.SURFACE, mew=2)  # fmt: skip
    ax.annotate(
        f"plan: {chosen} singles → {ys[chosen]:,.0f} €\n"
        f"(next coin would add {nxt:,.0f} € before effort)",
        (chosen, ys[chosen]),
        xytext=(12, -34),
        textcoords="offset points",
        color=A.INK_2,
    )
    ax.axhline(base, color=A.INK_2, linewidth=1, linestyle="--")
    ax.annotate(f"no singles: {base:,.0f} €", (n_max, base), xytext=(0, 4), textcoords="offset points", ha="right", color=A.INK_2)  # fmt: skip
    ax.set_xlim(0, n_max)
    ax.set_ylim(base * 0.9, ys[-1] * 1.05)
    ax.yaxis.set_major_formatter(A.EUR_FMT)
    ax.set_title("Each extra single listing earns less")
    ax.set_xlabel("Coins sold individually (best first)")
    ax.set_ylabel("Expected proceeds, whole collection")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def eur(v: float) -> str:
    return f"{v:,.0f} €"


def describe(p: Item) -> str:
    return f"{p['country'] or '?'} · {p['title'] or p['denomination'] or '?'} · {p['year'] or '?'}"


def location(p: Item) -> str:
    return f"page {p['page']}, row {p['row']}, pos. {p['pos']}"


def guide_lines(plans: List[Item], s: Dict[str, float], lots: Dict[str, List[Item]], run: str) -> List[str]:  # fmt: skip
    sm = summary(plans, lots)
    routes, singles, total = sm["routes"], sm["routes"]["single"], sm["total"]
    silver_price = B.env_float("SILVER_EUR_PER_G") or 0
    gold_price = B.env_float("GOLD_EUR_PER_G") or 0
    single_share = sum(p["expected_eur"] for p in singles) / total * 100 if total else 0
    extra_sales = len(plans) - sm["effort"]

    L = [
        "# Selling guide",
        "",
        f"Built {datetime.now():%Y-%m-%d %H:%M} from `{run}` · silver {silver_price} €/g · "
        f"gold {gold_price} €/g · assumed grade {os.environ.get('ASSUMED_GRADE', 'xf')}. "
        "Pictures and where to find each coin: `selling_plan.xlsx`.",
        "",
        f"**Plan: sell {len(singles)} coins individually and the other {len(plans) - len(singles)} "
        f"by weight or in lots. Expected proceeds about {eur(total)}** "
        f"(catalog value {eur(sm['catalog'])}), with about {sm['effort']} sales instead of "
        f"{len(plans)}. The singles bring in {single_share:.0f}% of that.",
        "",
        "| Route | Coins | Catalog value | Expected proceeds | Sales | How |",
        "|---|---|---|---|---|---|",
    ]
    how = {
        "single": "eBay (auction or fixed price)"
        + ("; coin dealer or auction house for the top ones" if any(p["where"] != "eBay" for p in singles) else ""),
        "silver": f"one dealer, by fine silver weight; aim for ≥ {silver_price * s['SELL_SILVER_RATE']:.2f} €/g fine",
        "gold": f"precious-metal dealer, by gold content; aim for ≥ {gold_price * s['SELL_GOLD_RATE']:.0f} €/g fine",
        "lot": f"{len(lots)} lots by country on eBay (\"Konvolut\")",
        "bulk": "one or a few mixed lots, sold by the kilo",
    }  # fmt: skip
    for route, label in ROUTES.items():
        members = routes[route]
        if members:
            L.append(
                f"| {label} | {len(members)} | {eur(sum(p['value'] for p in members))} | "
                f"**{eur(sum(p['expected_eur'] for p in members))}** | {sm['sales'][route] or '(same trip)'} | {how[route]} |"
            )  # fmt: skip
    L += [
        "",
        "Effort against return:",
        "",
        "| Scenario | Sales | Expected proceeds |",
        "|---|---|---|",
        f"| Everything to the dealer / in bulk lots, no singles | {sm['no_singles_sales']} | {eur(sm['no_singles'])} |",
        f"| **This plan** | **{sm['effort']}** | **{eur(total)}** |",
        f"| Every coin listed individually | {len(plans)} | {eur(sm['all_singles'])} |",
        "",
        f"Listing every coin individually would change the proceeds by {sm['all_singles'] - total:+,.0f} € "
        f"for {extra_sales} extra sales ({(sm['all_singles'] - total) / max(extra_sales, 1):+.2f} € per "
        "sale, before the time for photos, listing and shipping). Silver sells better by weight "
        "than one by one, and most base-metal coins are worth too little to list.",
        "",
        "![](tradeoff.png)",
        "",
        "## Steps",
        "",
        "1. **Don't clean anything.** Cleaned coins lose most of their collector value.",
        "2. **Pull the coins out** with `selling_plan.xlsx`: every sheet gives the page (scan "
        "name without `_v`/`_r`), the row from the top and the position from the left. The "
        "*Singles* sheet also shows the page with the coin marked.",
        "3. **Check the singles first** (*Check* column). Confirm year, mint mark and variant "
        "against Numista, and compare recent sold prices with the eBay link. A wrong "
        "identification matters most here.",
        f"4. **Sell the singles.** Under {eur(s['SELL_AUCTION_MIN_EUR'])}: eBay with your own "
        "photos of both sides, as an auction from a low start or at a fixed price near the Numista "
        f"VF price. Above {eur(s['SELL_AUCTION_MIN_EUR'])}: get an offer from a coin dealer or "
        "consign to a numismatic auction, and compare that with eBay.",
        "5. **Sort the silver by fineness** (table below, *Precious metal* sheet) and weigh each "
        "group. Get offers from two or three precious-metal dealers per gram of *fine* silver "
        "on the same day, and compare them with the spot price. Bring the gold on the same trip.",
        "6. **Sell the base-metal coins worth a few euros as themed lots** (*Lots* sheet; one "
        "country per lot, with a photo of all coins and a list of the better ones).",
        "7. **Sell the rest by the kilo** as a mixed lot of world coins: everything not on "
        "one of the sheets. Don't spend time sorting it.",
        "",
    ]  # fmt: skip

    if routes["silver"]:
        fine: Dict[Optional[float], List[Item]] = defaultdict(list)
        for p in routes["silver"]:
            fine[p["silver_fineness"]].append(p)
        L += [
            "## Silver for the dealer, by fineness",
            "",
            "| Fineness | Coins | Fine silver g | Melt value | Expected offer |",
            "|---|---|---|---|---|",
        ]
        for f in sorted(fine, key=lambda x: -(x or 0)):
            members = fine[f]
            grams = sum(p["silver_g"] or 0 for p in members)
            melt = sum(p["melt"] or 0 for p in members)
            label = f"{f:.3f}".lstrip("0") if f else "weight unknown (weigh these)"
            L.append(
                f"| {label} | {len(members)} | {f'{grams:,.0f}' if grams else '–'} | "
                f"{eur(melt) if melt else '–'} | {eur(sum(p['expected_eur'] for p in members))} |"
            )
        L.append("")

    if lots:
        L += [
            "## Themed lots",
            "",
            "| Lot | Coins | Catalog value | Expected |",
            "|---|---|---|---|",
        ]
        for name, members in lots.items():
            L.append(
                f"| {name} | {len(members)} | {eur(sum(p['value'] for p in members))} | "
                f"{eur(sum(p['expected_eur'] for p in members))} |"
            )
        L.append("")

    if singles:
        L += [
            f"## The {len(singles)} coins to sell individually",
            "",
            "*Extra* = how much more selling the coin individually should bring than its bulk "
            "route. Sorted by that.",
            "",
            "| # | Picture | Coin | Where to find | Value | Melt | Expected | Extra | Sell via | Check |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]  # fmt: skip
        for i, p in enumerate(singles, start=1):
            cid = p["coin_id"]
            pics = (
                f'<img src="../crops/{cid}_front.jpg" width="56"> '
                f'<img src="../crops/{cid}_back.jpg" width="56">'
            )
            L.append(
                f"| {i} | {pics} | `{cid}` {describe(p)} | "
                f'<img src="locate/{cid}.jpg" width="110"><br>{location(p)} | '
                f"{eur(p['value'])} | "
                f"{eur(p['melt']) if p['melt'] else '–'} | {eur(p['expected_eur'])} | "
                f"+{eur(p['gain_eur'])} | {p['where']} | {p['verify'] or '–'} |"
            )
        L.append("")

    unverified = sum(1 for p in singles if p["verify"])
    L += [
        f"**Caveats.** {unverified} of the singles have something to check first. "
        f"{sum(1 for p in plans if not p['reliable'])} coins are valued only by LLM guess. "
        "The guesses tend to come out low (see *LLM guesses vs real prices* in "
        "`analysis/report.md`), so some coins in the bulk lots may be worth more. They are "
        "re-ranked once Numista has priced them. Metal prices are from `.env`; update them "
        "before selling.",
        "",
        "Settings (override any of them in `.env`; the rates are what you actually receive, "
        "after fees and margin):",
        "",
        "| Setting | Value | Meaning |",
        "|---|---|---|",
    ]
    L += [f"| `{k}` | {s[k]:g} | {SETTINGS[k][1]} |" for k in SETTINGS]
    L += ["", "The route of every coin: `selling_plan.csv`.", ""]
    return L


# -------------------------------------------------------------------- workbook

EUR_FORMAT = '#,##0.00 "€"'


def header(ws, columns: List[Tuple[str, str, float]]) -> None:
    for col, (_, title, width) in enumerate(columns, start=1):
        cell = ws.cell(row=1, column=col, value=tr(title))
        cell.font = Font(bold=True)
        cell.fill = B.HEADER_FILL
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        ws.column_dimensions[get_column_letter(col)].width = width
    ws.freeze_panes = "D2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(columns))}1"


def add_image(ws, img: np.ndarray, anchor: str) -> None:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if ok:
        ws.add_image(XLImage(BytesIO(buf.tobytes())), anchor)


def cell_value(key: str, p: Item, idx: int) -> Any:
    if key == "rank":
        return idx
    if key == "where":
        return tr(p["where"]) if p.get("where") else None
    if key == "verify":
        return (
            "; ".join(tr(x) for x in p["verify"].split("; ")) if p["verify"] else None
        )
    if key == "metal":
        return tr(p["metal"])
    if key == "lot":
        return localize("country", tr(p["lot"]))
    if key == "silver_fineness":
        return p["silver_fineness"] or (
            tr("weight unknown") if p["metal"] == "silver" else None
        )
    if key in ("expected_eur", "gain_eur", "value", "melt"):
        return round(p[key], 2) if p.get(key) is not None else None
    return localize(key, p.get(key))


def coin_sheet(ws, run: str, members: List[Item], columns: List[Tuple[str, str, float]]) -> None:  # fmt: skip
    header(ws, columns)
    for i, p in enumerate(members, start=1):
        r = i + 1
        height = THUMB
        for col, (key, _, _) in enumerate(columns, start=1):
            anchor = f"{get_column_letter(col)}{r}"
            if key == "picture":
                pic = thumb(run, p["coin_id"])
                if pic is not None:
                    add_image(ws, pic, anchor)
                continue
            if key == "locator":
                pic = cv2.imread(p.get("locator") or "")
                if pic is not None:
                    add_image(ws, pic, anchor)
                    height = max(height, pic.shape[0])
                continue
            cell = ws.cell(row=r, column=col)
            if key in ("numista_url", "ebay_sold"):
                if p.get(key):
                    cell.value, cell.hyperlink, cell.font = (
                        tr("open"),
                        p[key],
                        B.LINK_FONT,
                    )
                    cell.alignment = Alignment(vertical="center")
                continue
            cell.value = cell_value(key, p, i)
            if key in ("expected_eur", "gain_eur", "value", "melt"):
                cell.number_format = EUR_FORMAT
            cell.alignment = Alignment(
                vertical="center", wrap_text=key in ("title", "verify")
            )
        ws.row_dimensions[r].height = height * 0.75 + 4
    ws.auto_filter.ref = f"A1:{get_column_letter(len(columns))}{len(members) + 1}"


PIC_W = THUMB * 2 / 7 + 1
LOCATION_COLUMNS = [
    ("coin_id", "Coin ID", 17),
    ("page", "Page", 10),
    ("row", "Row", 6),
    ("pos", "Pos.", 6),
]
SINGLE_COLUMNS = [
    ("rank", "#", 4),
    ("picture", "Picture", PIC_W),
    ("locator", "Where on the page", LOCATE_W / 7 + 1),
    *LOCATION_COLUMNS,
    ("country", "Country", 18),
    ("title", "Coin", 30),
    ("year", "Year", 7),
    ("value", "Value €", 10),
    ("melt", "Melt €", 10),
    ("expected_eur", "Expected €", 11),
    ("gain_eur", "Extra vs bulk €", 11),
    ("where", "Sell via", 12),
    ("verify", "Check before selling", 30),
    ("numista_url", "Numista", 9),
    ("ebay_sold", "eBay sold", 9),
]
METAL_COLUMNS = [
    ("picture", "Picture", PIC_W),
    *LOCATION_COLUMNS,
    ("metal", "Metal", 8),
    ("silver_fineness", "Fineness", 10),
    ("silver_g", "Silver g", 9),
    ("gold_g", "Gold g", 8),
    ("country", "Country", 18),
    ("title", "Coin", 30),
    ("year", "Year", 7),
    ("melt", "Melt €", 10),
    ("value", "Value €", 10),
    ("expected_eur", "Expected €", 11),
]
LOT_COLUMNS = [
    ("picture", "Picture", PIC_W),
    *LOCATION_COLUMNS,
    ("lot", "Lot", 14),
    ("country", "Country", 18),
    ("title", "Coin", 30),
    ("year", "Year", 7),
    ("value", "Value €", 10),
    ("expected_eur", "Expected €", 11),
]


def page_order(p: Item) -> Tuple[str, int]:
    return p["page"], p["coin_number"]


def write_workbook(run: str, plans: List[Item], s: Dict[str, float], lots: Dict[str, List[Item]], path: str) -> None:  # fmt: skip
    sm = summary(plans, lots)
    routes = sm["routes"]
    wb = Workbook()
    ws = wb.active
    ws.title = tr("Overview")
    lines: List[Tuple[Any, ...]] = [
        (tr("Selling plan"),),
        (tr("Built"), datetime.now().strftime("%Y-%m-%d %H:%M")),
        (tr("Silver € per g (pure)"), B.env_float("SILVER_EUR_PER_G")),
        (tr("Gold € per g (pure)"), B.env_float("GOLD_EUR_PER_G")),
        (),
        (
            tr("Route"),
            tr("Coins"),
            tr("Catalog value €"),
            tr("Expected €"),
            tr("Sales"),
        ),
    ]
    for route, label in ROUTES.items():
        members = routes[route]
        if members:
            lines.append((tr(label), len(members), round(sum(p["value"] for p in members), 2),
                          round(sum(p["expected_eur"] for p in members), 2), sm["sales"][route] or None))  # fmt: skip
    lines += [
        (
            tr("Total"),
            len(plans),
            round(sm["catalog"], 2),
            round(sm["total"], 2),
            sm["effort"],
        ),
        (),
        (tr("location_help"),),
        (tr("bulk_help"),),
        (tr("details_help"),),
    ]
    for r, line in enumerate(lines, start=1):
        for c, value in enumerate(line, start=1):
            cell = ws.cell(row=r, column=c, value=value)
            if c in (3, 4) and isinstance(value, float) and r > 5:
                cell.number_format = EUR_FORMAT
    ws["A1"].font = Font(bold=True, size=14)
    for c in range(1, 6):
        ws.cell(row=6, column=c).font = Font(bold=True)
        ws.cell(row=6, column=c).fill = B.HEADER_FILL
        ws.cell(row=len(lines) - 4, column=c).font = Font(bold=True)
    for col, width in zip("ABCDE", (44, 10, 16, 16, 10)):
        ws.column_dimensions[col].width = width

    coin_sheet(wb.create_sheet(tr("Singles")), run, routes["single"], SINGLE_COLUMNS)
    metal = sorted(routes["gold"] + routes["silver"], key=page_order)
    coin_sheet(wb.create_sheet(tr("Precious metal")), run, metal, METAL_COLUMNS)
    lot_coins = sorted(routes["lot"], key=lambda p: (list(lots).index(p["lot"]), *page_order(p)))  # fmt: skip
    coin_sheet(wb.create_sheet(tr("Lots")), run, lot_coins, LOT_COLUMNS)
    try:
        wb.save(path)
    except PermissionError:
        sys.exit(f"{path} is open in Excel - close it and run again.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--lang", choices=["EN", "DE", "en", "de"], help="workbook language (default: SUMMARY_LANGUAGE)")  # fmt: skip
    args = parser.parse_args()
    if args.lang:
        i18n.set_language(args.lang)
    out = os.path.join(args.run, "selling")
    os.makedirs(out, exist_ok=True)

    rows = A.load(args.run)
    if not rows:
        sys.exit("No valued coins found.")
    plans, s = plan(rows)
    lots = lot_groups(plans)
    add_locations(args.run, plans)
    write_locators([p for p in plans if p["route"] == "single"], os.path.join(out, "locate"))  # fmt: skip

    plot_tradeoff(plans, s, os.path.join(out, "tradeoff.png"))
    write_plan_csv(plans, os.path.join(out, "selling_plan.csv"))
    with open(os.path.join(out, "selling_guide.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(guide_lines(plans, s, lots, args.run)))
    write_workbook(args.run, plans, s, lots, os.path.join(out, "selling_plan.xlsx"))
    sm = summary(plans, lots)
    print(
        f"Wrote {out}/: selling_guide.md, selling_plan.xlsx ({i18n.LANG}), selling_plan.csv. "
        f"{len(sm['routes']['single'])} singles, expected {sm['total']:,.0f} € in {sm['effort']} sales."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
