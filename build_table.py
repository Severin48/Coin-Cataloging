"""Stage 5: merge everything into <run>/coins.xlsx (offline, no API calls).

    uv run build_table.py [--run results/...] [--no-thumbs]

Reads extraction_results(_with_groups).json, identifications.jsonl, numista_matches.jsonl
and overrides.csv. Writes per-coin crops to <run>/crops/ (linked from the sheet),
coins.xlsx (sheets Summary, Highlights, Coins, Types, Top, Review), highlights.md and
valuation.json (the metal prices and grade the values are based on).

Value = max(Numista price at ASSUMED_GRADE, melt value); melt needs SILVER_EUR_PER_G /
GOLD_EUR_PER_G in .env. Without a Numista price the LLM guess is used and marked as such.
Melt, value and totals are Excel formulas on the Summary metal prices: change the prices
there and everything recalculates without re-running this script.

--share writes coins_share.xlsx for sending to others: fixed values instead of formulas
(email and phone previews don't calculate formulas), no links to files on this PC,
bigger pictures.

SUMMARY_LANGUAGE=DE in .env (or --lang DE) writes the workbook in German: sheet names,
headers, labels, countries and grades (s/ss/vz/st). Internal keys stay English.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import cv2
import numpy as np
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

import i18n
from coins_common import (
    DEFAULT_RUN,
    Item,
    apply_overrides,
    crop_side,
    load_coins,
    read_jsonl,
)
from i18n import localize, sheet_ref, source_label, tr

GRADES = ["g", "vg", "f", "vf", "xf", "au", "unc"]
REVIEW_VALUE_EUR = 20.0  # verify everything worth at least this before selling
THUMB_PX = 64
# --share: fixed values instead of formulas (email/phone previews don't calculate),
# no links to local files, bigger pictures
SHARE = False
HIGHLIGHT_MIN_CONFIDENCE = 0.8


def silver_cell() -> str:
    """Input cells on the Summary sheet, referenced by the formulas."""
    return f"{sheet_ref('Summary')}!$B$3"


def gold_cell() -> str:
    return f"{sheet_ref('Summary')}!$B$4"


def env_float(name: str) -> Optional[float]:
    value = os.environ.get(name, "").strip()
    return float(value) if value else None


def fineness(composition: Optional[str], metal: str) -> Optional[float]:
    """'Silver (.700) (Copper .300)' -> 0.7, 'silver .925' -> 0.925, '90% silver' -> 0.9."""
    if not composition or metal not in composition.lower():
        return None
    text = composition.lower()
    m = re.search(metal + r"[^,;/]*?\(?\s*0?\.(\d{3,4})", text)
    if m:
        return float("0." + m.group(1))
    m = re.search(r"(\d{2,3}(?:\.\d+)?)\s*%\s*" + metal, text) or re.search(
        metal + r"[^,;/]*?(\d{2,3}(?:\.\d+)?)\s*%", text
    )
    if m:
        return float(m.group(1)) / 100
    m = re.search(metal + r"[^,;/]*?\b(\d{3})\b", text)  # "silver 925"
    if m and 300 <= int(m.group(1)) <= 999:
        return int(m.group(1)) / 1000
    return None


def price_at(
    prices: Dict[str, float], grade: str
) -> Tuple[Optional[float], Optional[str]]:
    if not prices:
        return None, None
    if grade in prices:
        return prices[grade], grade
    target = GRADES.index(grade)
    nearest = min(
        prices, key=lambda g: abs(GRADES.index(g) - target) if g in GRADES else 99
    )
    return prices[nearest], nearest


def ebay_sold_url(ident: Item) -> Optional[str]:
    parts = [
        ident.get("country"),
        ident.get("denomination_title"),
        ident.get("year"),
        ident.get("subject"),
    ]
    query = " ".join(str(p) for p in parts if p)
    if not query:
        return None
    return f"https://www.ebay.de/sch/i.html?_nkw={quote_plus(query[:80])}&LH_Sold=1&LH_Complete=1"


def save_crops(
    run: str, coin: Item, need_thumb: bool = True
) -> Tuple[str, str, Optional[np.ndarray]]:
    """Writes <run>/crops/<coin_id>_front|back.jpg once; returns relative paths + thumb."""
    crop_dir = os.path.join(run, "crops")
    os.makedirs(crop_dir, exist_ok=True)
    paths, thumbs = [], []
    for side in ("front", "back"):
        rel = f"crops/{coin['coin_id']}_{side}.jpg"
        path = os.path.join(run, rel)
        if os.path.exists(path):
            if not need_thumb:
                paths.append(rel)
                continue
            img = cv2.imread(path)
        else:
            img = crop_side(coin, side, max_size=768)
            cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        paths.append(rel)
        thumbs.append(
            cv2.resize(img, (THUMB_PX, THUMB_PX), interpolation=cv2.INTER_AREA)
        )
    thumb = np.hstack(thumbs) if need_thumb and len(thumbs) == 2 else None
    return paths[0], paths[1], thumb


def metal(composition: Optional[str]) -> Optional[str]:
    if not composition:
        return None
    text = composition.lower()
    for name in ("gold", "silver", "platinum", "palladium"):
        if name in text and "plated" not in text and "clad" not in text:
            return name
    return "base"


def metal_mismatch(ident: Item, match: Item) -> bool:
    """The LLM saw silver but Numista says gold (or base metal), etc."""
    seen, listed = (
        metal(ident.get("composition_guess")),
        metal(match.get("composition")),
    )
    return bool(seen and listed and seen != listed)


def trusted_match(ident: Item, match: Item) -> bool:
    """Numista type is reliable enough to take its prices and metal content."""
    override = ident.get("numista_type_id")
    if "numista_type_id" in (ident.get("overridden") or []):
        # trusted once numista.py has fetched the overridden type
        return match.get("type_id") == override
    return (
        match.get("status") in ("matched", "matched_no_issue")
        and (match.get("match_margin") or 0) >= 0.5
        and (ident.get("confidence") or 0) >= 0.6
        and not metal_mismatch(ident, match)
    )


def price_suspicious(r: Item) -> bool:
    """Numista price far above the LLM's estimate: rare variant or wrong issue?"""
    price, guess = r.get("price_at_grade"), r.get("llm_guess")
    return bool(price and guess and price > 30 and price > 4 * guess)


def build_rows(run: str, with_thumbs: bool) -> List[Item]:
    coins = load_coins(run)
    idents = {
        r["coin_id"]: r for r in read_jsonl(os.path.join(run, "identifications.jsonl"))
    }
    for cid in {c["coin_id"] for c in coins} - idents.keys():
        idents[cid] = {"coin_id": cid}
    apply_overrides(idents)
    matches = {
        r["coin_id"]: r for r in read_jsonl(os.path.join(run, "numista_matches.jsonl"))
    }

    silver_eur = env_float("SILVER_EUR_PER_G")
    gold_eur = env_float("GOLD_EUR_PER_G")
    grade = os.environ.get("ASSUMED_GRADE", "xf").strip().lower() or "xf"

    rows: List[Item] = []
    for coin in coins:
        cid = coin["coin_id"]
        ident = idents[cid]
        match = matches.get(cid, {})
        front, back, thumb = save_crops(run, coin, need_thumb=with_thumbs)
        # An ambiguous match may be the wrong type (10 Pfennig vs gold 10 Mark):
        # show its prices, but value the coin only from a trusted match.
        trusted = trusted_match(ident, match)
        composition = (match.get("composition") if trusted else None) or ident.get(
            "composition_guess"
        )
        weight = match.get("weight_g") if trusted else None
        silver = fineness(composition, "silver")
        gold = fineness(composition, "gold")
        silver_g = round(weight * silver, 2) if weight and silver else None
        gold_g = round(weight * gold, 2) if weight and gold else None
        melt = None
        if silver_g and silver_eur:
            melt = silver_g * silver_eur
        if gold_g and gold_eur:
            melt = (melt or 0) + gold_g * gold_eur
        prices = match.get("prices") or {}
        price, price_grade = price_at(prices, grade) if trusted else (None, None)

        if ident.get("est_value_eur") is not None:  # manual override
            value, source = ident["est_value_eur"], "override"
        elif price is not None or melt is not None:
            if (melt or 0) > (price or 0):
                value, source = melt, "melt"
            else:
                value, source = price, f"numista_{price_grade}"
        else:
            value, source = ident.get("est_value_eur_guess"), "llm_guess"

        rows.append(
            {
                "coin_id": cid,
                "group_id": coin.get("group_id"),
                "thumb": thumb if with_thumbs else None,
                "front_crop": front,
                "back_crop": back,
                "front_scan": coin["front_image"],
                "back_scan": coin["back_image"],
                "country": ident.get("country"),
                "denomination": ident.get("denomination_title"),
                "denomination_value": ident.get("denomination_value"),
                "currency": match.get("currency") or ident.get("currency"),
                "year": ident.get("year"),
                "mint_mark": ident.get("mint_mark"),
                "subject": ident.get("subject"),
                "coin_type": ident.get("coin_type"),
                "grade_guess": ident.get("grade_guess"),
                "composition": composition,
                "weight_g": weight,
                "silver_g": silver_g,
                "gold_g": gold_g,
                "melt_eur": round(melt, 2) if melt is not None else None,
                "numista_title": match.get("title"),
                "numista_id": match.get("type_id"),
                "numista_url": match.get("url"),
                "km": match.get("km") or ident.get("km_number"),
                "mintage": match.get("mintage"),
                **{f"price_{g}": prices.get(g) for g in ("f", "vf", "xf", "unc")},
                "est_value_eur": round(value, 2) if value is not None else None,
                "value_source": source if value is not None else None,
                "price_at_grade": price,
                "price_grade": price_grade,
                "llm_guess": ident.get("est_value_eur_guess"),
                "metal_mismatch": metal_mismatch(ident, match),
                "override_eur": ident.get("est_value_eur"),
                "ebay_sold": ebay_sold_url(ident),
                "confidence": ident.get("confidence"),
                "match_status": match.get("status"),
                "match_margin": match.get("match_margin"),
                "notes": ident.get("notes"),
                "overridden": ", ".join(ident.get("overridden", [])) or None,
                "_ident": ident,
                "_match": match,
            }
        )
    flag_reviews(rows)
    return rows


def flag_reviews(rows: List[Item]) -> None:
    by_group: Dict[Any, set] = defaultdict(set)
    for r in rows:
        if r["country"] or r["denomination_value"] is not None:
            by_group[r["group_id"]].add(
                (r["country"], r["denomination_value"], r["year"])
            )
    for r in rows:
        ident = r["_ident"]
        reasons = []
        if "confidence" not in ident:
            reasons.append("not identified")
        elif ident.get("is_coin") is False or ident.get("exclude"):
            reasons.append("not a coin / excluded")
        else:
            if (r["confidence"] or 0) < 0.6:
                reasons.append("low confidence")
            if r["match_status"] in ("no_candidates", "type_not_found"):
                reasons.append("no Numista match")
            elif r["match_status"] == "matched" and (r["match_margin"] or 0) < 0.5:
                reasons.append("ambiguous Numista match")
            if r["group_id"] is not None and len(by_group[r["group_id"]]) > 1:
                reasons.append("cluster members disagree")
            if r["metal_mismatch"]:
                reasons.append("metal differs from Numista type")
            if price_suspicious(r):
                reasons.append("Numista price ≫ LLM estimate: check issue/variant")
            if (r["est_value_eur"] or 0) >= REVIEW_VALUE_EUR:
                reasons.append(f"worth ≥ {REVIEW_VALUE_EUR:.0f} €: verify")
        if "numista_type_id" in (ident.get("overridden") or []):
            reasons = [x for x in reasons if "Numista" not in x]
        r["needs_review"] = "; ".join(reasons) or None


# ----------------------------------------------------------------------- writing

COLUMNS = [
    # key, header, width, kind
    ("thumb", "Picture", 18, "thumb"),
    ("coin_id", "Coin ID", 16, None),
    ("country", "Country", 18, None),
    ("denomination", "Denomination", 14, None),
    ("year", "Year", 7, None),
    ("subject", "Subject", 28, None),
    ("est_value_eur", "Est. value €", 11, "eur"),
    ("value_source", "Value source", 12, None),
    ("melt_eur", "Melt €", 9, "eur"),
    ("price_at_grade", "Numista € @grade", 10, "eur"),
    ("price_grade", "Grade", 6, None),
    ("llm_guess", "LLM guess €", 9, "eur"),
    ("override_eur", "Override €", 9, "eur"),
    ("price_f", "F €", 8, "eur"),
    ("price_vf", "VF €", 8, "eur"),
    ("price_xf", "XF €", 8, "eur"),
    ("price_unc", "UNC €", 8, "eur"),
    ("composition", "Composition", 22, None),
    ("weight_g", "Weight g", 9, None),
    ("silver_g", "Silver g", 9, None),
    ("gold_g", "Gold g", 8, None),
    ("numista_title", "Numista type", 30, None),
    ("numista_url", "Numista", 10, "link"),
    ("ebay_sold", "eBay sold", 10, "link"),
    ("km", "KM#", 10, None),
    ("mintage", "Mintage", 11, None),
    ("currency", "Currency", 16, None),
    ("denomination_value", "Face value", 9, None),
    ("mint_mark", "Mint", 6, None),
    ("coin_type", "Type", 13, None),
    ("grade_guess", "Grade (guess)", 13, None),
    ("confidence", "LLM conf.", 9, None),
    ("match_status", "Match", 12, None),
    ("needs_review", "Needs review", 30, None),
    ("group_id", "Cluster", 8, None),
    ("front_crop", "Front", 9, "file"),
    ("back_crop", "Back", 9, "file"),
    ("front_scan", "Front scan", 16, None),
    ("back_scan", "Back scan", 16, None),
    ("numista_id", "Numista ID", 10, None),
    ("overridden", "Overridden", 14, None),
    ("notes", "Notes", 40, None),
]
HEADER_FILL = PatternFill("solid", fgColor="DDE4EE")
LINK_FONT = Font(color="1F5FBF", underline="single")


def value_formulas(letter: Dict[str, str], r: int) -> Dict[str, str]:
    """Live Excel formulas for melt, value and value source in row r."""
    ag, au = f"{letter['silver_g']}{r}", f"{letter['gold_g']}{r}"
    melt, num = f"{letter['melt_eur']}{r}", f"{letter['price_at_grade']}{r}"
    llm, ovr = f"{letter['llm_guess']}{r}", f"{letter['override_eur']}{r}"
    grade = f"{letter['price_grade']}{r}"
    s_ovr, s_llm, s_melt = (source_label(s) for s in ("override", "llm_guess", "melt"))
    s_num = "Numista " if i18n.is_de() else "numista_"
    return {
        "melt_eur": f'=IF(AND({ag}="",{au}=""),"",N({ag})*{silver_cell()}+N({au})*{gold_cell()})',
        "est_value_eur": f'=IF({ovr}<>"",{ovr},IF(AND({melt}="",{num}=""),'
        f'IF({llm}="","",{llm}),MAX(N({melt}),N({num}))))',
        "value_source": f'=IF({ovr}<>"","{s_ovr}",IF(AND({melt}="",{num}=""),'
        f'IF({llm}="","","{s_llm}"),IF(N({melt})>N({num}),"{s_melt}","{s_num}"&{grade})))',
    }


def sheet_columns(with_thumbs: bool) -> List[Tuple[str, str, int, Optional[str]]]:
    return [
        c
        for c in COLUMNS
        if (with_thumbs or c[3] != "thumb") and not (SHARE and c[3] == "file")
    ]


def write_coin_sheet(ws, rows: List[Item], with_thumbs: bool) -> None:
    columns = sheet_columns(with_thumbs)
    letter = {key: get_column_letter(i) for i, (key, *_) in enumerate(columns, start=1)}
    for col, (_, header, width, _) in enumerate(columns, start=1):
        cell = ws.cell(row=1, column=col, value=tr(header))
        cell.font = Font(bold=True)
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        ws.column_dimensions[get_column_letter(col)].width = (
            THUMB_PX * 2 / 7 + 2 if kind_is_thumb(columns[col - 1]) else width
        )
    for r_idx, row in enumerate(rows, start=2):
        if with_thumbs:
            ws.row_dimensions[r_idx].height = THUMB_PX * 0.75 + 4
        formulas = {} if SHARE else value_formulas(letter, r_idx)
        for col, (key, _, _, kind) in enumerate(columns, start=1):
            value = formulas.get(key, localize(key, row.get(key)))
            if kind == "thumb":
                if value is not None:
                    ok, buf = cv2.imencode(
                        ".jpg", value, [cv2.IMWRITE_JPEG_QUALITY, 70 if SHARE else 80]
                    )
                    img = XLImage(BytesIO(buf.tobytes()))
                    ws.add_image(img, f"{get_column_letter(col)}{r_idx}")
                continue
            cell = ws.cell(row=r_idx, column=col)
            if kind in ("link", "file") and value:
                cell.value = tr("open") if kind == "link" else os.path.basename(value)
                cell.hyperlink = value
                cell.font = LINK_FONT
            else:
                cell.value = value
            if kind == "eur":
                cell.number_format = '#,##0.00 "€"'
            cell.alignment = Alignment(
                vertical="center", wrap_text=key in ("subject", "needs_review", "notes")
            )
    ws.freeze_panes = "C2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(columns))}{len(rows) + 1}"


def kind_is_thumb(column: Tuple[str, str, int, Optional[str]]) -> bool:
    return column[3] == "thumb"


def type_rows(rows: List[Item], coin_row: Dict[str, int], est_col: str) -> List[Item]:
    groups: Dict[Any, List[Item]] = defaultdict(list)
    for r in rows:
        if r["needs_review"] and "not identified" in r["needs_review"]:
            continue
        key = r["numista_id"] or (
            r["country"],
            r["denomination"],
            r["year"],
            r["subject"],
        )
        groups[key].append(r)
    out = []
    for members in groups.values():
        first = members[0]
        each = first["est_value_eur"]
        ref = f"{sheet_ref('Coins')}!${est_col}${coin_row[first['coin_id']]}"
        out.append(
            {
                "numista_title": first["numista_title"],
                "country": first["country"],
                "denomination": first["denomination"],
                "year": first["year"],
                "subject": first["subject"],
                "count": len(members),
                "est_value_eur": each if SHARE else f'=IF({ref}="","",{ref})',
                "total_eur": (
                    (round(each * len(members), 2) if each is not None else None)
                    if SHARE
                    else f'=IF({ref}="","",{ref}*{len(members)})'
                ),
                "_sort": (each or 0) * len(members),
                "value_source": first["value_source"],
                "silver_g_total": round(sum(m["silver_g"] or 0 for m in members), 1)
                or None,
                "numista_url": first["numista_url"],
                "coin_ids": ", ".join(m["coin_id"] for m in members),
            }
        )
    return sorted(out, key=lambda t: -t["_sort"])


def write_simple_sheet(
    ws, rows: List[Item], columns: List[Tuple[str, str, int, Optional[str]]]
) -> None:
    for col, (_, header, width, _) in enumerate(columns, start=1):
        cell = ws.cell(row=1, column=col, value=tr(header))
        cell.font = Font(bold=True)
        cell.fill = HEADER_FILL
        ws.column_dimensions[get_column_letter(col)].width = width
    for r_idx, row in enumerate(rows, start=2):
        for col, (key, _, _, kind) in enumerate(columns, start=1):
            cell = ws.cell(row=r_idx, column=col)
            value = localize(key, row.get(key))
            if kind == "link" and value:
                cell.value, cell.hyperlink, cell.font = tr("open"), value, LINK_FONT
            else:
                cell.value = value
            if kind == "eur":
                cell.number_format = '#,##0.00 "€"'
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(columns))}{len(rows) + 1}"


def write_summary(
    ws, rows: List[Item], types: List[Item], letter: Dict[str, str]
) -> None:
    n = len(rows) + 1

    def col(key: str) -> str:
        return f"{sheet_ref('Coins')}!{letter[key]}2:{letter[key]}{n}"

    identified = [r for r in rows if r["confidence"] is not None]
    matched = [r for r in rows if r["match_status"] == "matched"]
    grade = os.environ.get("ASSUMED_GRADE", "xf").strip().lower() or "xf"

    def total(key: str, only: Optional[str] = None) -> Any:
        if not SHARE:
            if only:
                return (
                    f'=SUMIF({col("value_source")},"{source_label(only)}",{col(key)})'
                )
            return f"=SUM({col(key)})"
        return round(
            sum(r[key] or 0 for r in rows if only is None or r["value_source"] == only),
            2,
        )

    lines = [
        (
            "Valuation inputs",
            tr("fixed values at these prices" if SHARE else "edit B3/B4: melt and values recalculate"),
            True,
        ),
        ("Prices taken from .env on", datetime.now().strftime("%Y-%m-%d %H:%M"), False),
        ("Silver € per g (pure)", env_float("SILVER_EUR_PER_G") or 0, False),
        ("Gold € per g (pure)", env_float("GOLD_EUR_PER_G") or 0, False),
        ("Assumed grade (Numista price)", grade_text(grade), False),
        ("Numista prices fetched", numista_fetch_month(), False),
        ("", "", False),
        ("Collection", "", True),
        ("Coins", len(rows), False),
        ("Identified by LLM", len(identified), False),
        ("Matched to Numista", len(matched), False),
        ("Distinct types", len(types), False),
        ("Need review", sum(1 for r in rows if r["needs_review"]), False),
        ("Highlights (confidently valuable)", sum(1 for r in rows if is_highlight(r)), False),
        ("Total estimated value €", total("est_value_eur"), False),
        ("  of which from LLM guesses €", total("est_value_eur", "llm_guess"), False),
        ("Total melt value €", total("melt_eur"), False),
        ("Total silver g", total("silver_g"), False),
        ("Total gold g", total("gold_g"), False),
    ]  # fmt: skip
    if i18n.is_de():
        lines += [
            ("", "", False),
            ("How to read this", "", True),
            (tr("grade_legend"), "", False),
            (tr("source_legend"), "", False),
        ]
    ws.column_dimensions["A"].width = 40
    ws.column_dimensions["B"].width = 44
    for i, (label, value, heading) in enumerate(lines, start=1):
        ws.cell(row=i, column=1, value=tr(label)).font = Font(
            bold=True, size=12 if heading else 11
        )
        cell = ws.cell(row=i, column=2, value=value)
        if "€" in label and not heading:
            cell.number_format = '#,##0.00 "€"'
    for ref in ("B3", "B4"):
        ws[ref].fill = PatternFill("solid", fgColor="FFF2CC")


def grade_text(grade: str) -> str:
    shown = i18n.grade_label(grade) or grade
    if shown != grade:
        shown = f"{shown} ({grade.upper()})"
    return f"{shown}  {tr('(change in .env + re-run)')}"


def numista_fetch_month() -> str:
    from numista import load_usage

    months = sorted(load_usage())
    return ", ".join(months) if months else "-"


def write_valuation_json(run: str, rows: List[Item]) -> None:
    info = {
        "built": datetime.now().isoformat(timespec="seconds"),
        "silver_eur_per_g": env_float("SILVER_EUR_PER_G"),
        "gold_eur_per_g": env_float("GOLD_EUR_PER_G"),
        "assumed_grade": os.environ.get("ASSUMED_GRADE", "xf"),
        "numista_prices_fetched": numista_fetch_month(),
        "total_est_value_eur": round(sum(r["est_value_eur"] or 0 for r in rows), 2),
        "total_silver_g": round(sum(r["silver_g"] or 0 for r in rows), 1),
        "total_gold_g": round(sum(r["gold_g"] or 0 for r in rows), 2),
        "how_to_update": "Edit SILVER_EUR_PER_G / GOLD_EUR_PER_G in .env and run "
        "'uv run build_table.py', or change Summary!B3/B4 directly in coins.xlsx.",
    }
    with open(os.path.join(run, "valuation.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


# -------------------------------------------------------------------- highlights


def is_highlight(r: Item) -> bool:
    """Valuable AND every check passed: confident ID, clear Numista match, real price."""
    ident = r["_ident"]
    clear_match = (r["match_margin"] or 0) >= 0.5 or "numista_type_id" in (
        ident.get("overridden") or []
    )
    return (
        (r["confidence"] or 0) >= HIGHLIGHT_MIN_CONFIDENCE
        and r["match_status"] in ("matched", "matched_no_issue")
        and clear_match
        and r["value_source"] not in (None, "llm_guess")
        and trusted_match(ident, r["_match"])
        and not price_suspicious(r)
        and ident.get("is_coin", True)
        and not ident.get("exclude")
    )


def write_highlights_md(run: str, rows: List[Item], top_n: int) -> None:
    def fmt(value: Optional[float]) -> str:
        return f"{value:,.2f} €" if value is not None else "–"

    def describe(r: Item) -> str:
        title = (
            r["numista_title"] or f"{r['denomination'] or '?'} ({r['subject'] or '-'})"
        )
        return f"{r['country'] or '?'} · {title} · {r['year'] or '?'}"

    highlights = sorted(
        (r for r in rows if is_highlight(r)), key=lambda r: -r["est_value_eur"]
    )
    promising = sorted(
        (
            r
            for r in rows
            if not is_highlight(r) and (r["est_value_eur"] or 0) >= REVIEW_VALUE_EUR
        ),
        key=lambda r: -r["est_value_eur"],
    )
    total = sum(r["est_value_eur"] for r in highlights)
    lines = [
        "# Coin highlights",
        "",
        f"Built {datetime.now():%Y-%m-%d %H:%M} · silver {env_float('SILVER_EUR_PER_G')} €/g · "
        f"gold {env_float('GOLD_EUR_PER_G')} €/g · grade {os.environ.get('ASSUMED_GRADE', 'xf')} · "
        f"Numista prices fetched {numista_fetch_month()}",
        "",
        f"**{len(highlights)} confidently valuable coins** (LLM confidence ≥ "
        f"{HIGHLIGHT_MIN_CONFIDENCE}, unambiguous Numista match, value from Numista price "
        f"or melt), worth **{fmt(total)}** together.",
        "",
        "| # | Coin | Value | Basis | Melt | Silver g | Conf. | Links |",
        "|---|---|---|---|---|---|---|---|",
    ]  # fmt: skip
    for i, r in enumerate(highlights[:top_n], start=1):
        links = (
            f"[Numista]({r['numista_url']}) · [eBay sold]({r['ebay_sold']}) · "
            f"[front]({r['front_crop']}) · [back]({r['back_crop']})"
        )
        lines.append(
            f"| {i} | `{r['coin_id']}` {describe(r)} | **{fmt(r['est_value_eur'])}** | "
            f"{r['value_source']} | {fmt(r['melt_eur'])} | {r['silver_g'] or '–'} | "
            f"{r['confidence']:.2f} | {links} |"
        )
    lines += [
        "",
        f"## Possibly valuable, not yet confirmed ({len(promising)})",
        "",
        f"Worth ≥ {REVIEW_VALUE_EUR:.0f} € by some source, but a check failed. "
        "Review these first (`/identify-coins`).",
        "",
        "| Coin | Value | Basis | Why not confirmed |",
        "|---|---|---|---|",
    ]
    for r in promising[:top_n]:
        lines.append(
            f"| `{r['coin_id']}` {describe(r)} | {fmt(r['est_value_eur'])} | "
            f"{r['value_source']} | {r['needs_review'] or 'below confidence/match threshold'} |"
        )
    with open(os.path.join(run, "highlights.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--no-thumbs", action="store_true", help="faster, smaller file")
    parser.add_argument("--top", type=int, default=100, help="rows on the Top sheet")
    parser.add_argument(
        "--share",
        action="store_true",
        help="write coins_share.xlsx to send to others (see above)",
    )
    parser.add_argument(
        "--lang",
        choices=["EN", "DE", "en", "de"],
        help="workbook language (default: SUMMARY_LANGUAGE in .env, else EN)",
    )
    args = parser.parse_args()  # fmt: skip
    with_thumbs = not args.no_thumbs
    if args.lang:
        i18n.set_language(args.lang)
    global SHARE, THUMB_PX
    if args.share:
        SHARE, THUMB_PX = True, 110

    rows = build_rows(args.run, with_thumbs)
    rows.sort(
        key=lambda r: (
            r["coin_id"].split("_coin")[0],
            int(r["coin_id"].split("_coin")[1]),
        )
    )
    columns = sheet_columns(with_thumbs)
    letter = {key: get_column_letter(i) for i, (key, *_) in enumerate(columns, start=1)}
    coin_row = {r["coin_id"]: i for i, r in enumerate(rows, start=2)}
    types = type_rows(rows, coin_row, letter["est_value_eur"])
    highlights = sorted(
        (r for r in rows if is_highlight(r)), key=lambda r: -r["est_value_eur"]
    )

    wb = Workbook()
    write_summary(wb.active, rows, types, letter)
    wb.active.title = tr("Summary")
    write_coin_sheet(wb.create_sheet(tr("Highlights")), highlights, with_thumbs)
    write_coin_sheet(wb.create_sheet(tr("Coins")), rows, with_thumbs)
    write_simple_sheet(
        wb.create_sheet(tr("Types")),
        types,
        [
            ("numista_title", "Numista type", 34, None),
            ("country", "Country", 18, None),
            ("denomination", "Denomination", 14, None),
            ("year", "Year", 7, None),
            ("subject", "Subject", 28, None),
            ("count", "Count", 7, None),
            ("est_value_eur", "Value each €", 11, "eur"),
            ("total_eur", "Total €", 11, "eur"),
            ("value_source", "Value source", 12, None),
            ("silver_g_total", "Silver g total", 11, None),
            ("numista_url", "Numista", 10, "link"),
            ("coin_ids", "Coin IDs", 60, None),
        ],
    )
    top = sorted(
        (r for r in rows if r["est_value_eur"] is not None),
        key=lambda r: -r["est_value_eur"],
    )[: args.top]
    write_coin_sheet(wb.create_sheet(tr("Top")), top, with_thumbs)
    if not SHARE:  # the review queue is a to-do list for us, not for recipients
        write_coin_sheet(
            wb.create_sheet(tr("Review")),
            [r for r in rows if r["needs_review"]],
            with_thumbs,
        )

    out = os.path.join(args.run, "coins_share.xlsx" if SHARE else "coins.xlsx")
    try:
        wb.save(out)
    except PermissionError:
        sys.exit(f"{out} is open in Excel - close it and run again.")
    write_valuation_json(args.run, rows)
    write_highlights_md(args.run, rows, top_n=40)
    n_review = sum(1 for r in rows if r["needs_review"])
    print(
        f"Wrote {out} ({i18n.LANG}): {len(rows)} coins, {len(types)} types, {len(highlights)} highlights, "
        f"{n_review} to review. Also highlights.md and valuation.json."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
