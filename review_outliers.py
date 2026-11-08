"""Manual review sheet for high-stakes outliers: <run>/outlier_review.xlsx (offline).

    uv run review_outliers.py [--run results/...]

A coin is listed when any of these holds:
- an earlier match (numista_matches_previous.jsonl) pointed to a type worth >= 50 €
  and the current match differs (rejected match),
- the current match is not trusted (ambiguous, low confidence, metal mismatch)
  but its type is worth >= 50 € in some grade,
- the Numista price is far above the LLM estimate,
- the LLM estimates >= 50 € but there is no trusted match,
- or outlier_notes.csv has a hand-written assessment for it.

outlier_notes.csv (repo root, tracked) holds the assessment per coin: priority,
verdict, likely coin and value, why, what to look for. Coins without notes get an
automatic reason. The last two columns are for your own verdict and notes.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from io import BytesIO
from typing import Dict, List

import cv2
import numpy as np
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

import build_table as B
from coins_common import DEFAULT_RUN, Item, read_jsonl

NOTES_CSV = "outlier_notes.csv"
STAKE_EUR = 50.0
PX = 150  # per side

COLUMNS = [
    # key, header, width
    ("priority", "Prio", 5),
    ("picture", "Picture (front | back)", PX * 2 / 7 + 2),
    ("coin_id", "Coin ID", 17),
    ("verdict", "Our verdict", 26),
    ("likely_coin", "Most likely coin", 30),
    ("likely_value_eur", "Likely value €", 14),
    ("suspect", "Valuable match that was suspected", 30),
    ("suspect_price", "Its highest Numista price €", 12),
    ("why", "Why we think so", 55),
    ("look_for", "What to look for", 55),
    ("ai_reading", "What the AI read", 34),
    ("legends", "Legends read by the AI", 40),
    ("ai_notes", "AI notes", 40),
    ("confidence", "AI conf.", 8),
    ("current", "Current Numista match", 30),
    ("suspect_url", "Suspected type", 10),
    ("current_url", "Current type", 10),
    ("ebay", "eBay sold", 10),
    ("front", "Front", 9),
    ("back", "Back", 9),
    ("your_verdict", "YOUR verdict", 16),
    ("your_notes", "YOUR notes", 40),
]
LINKS = {"suspect_url", "current_url", "ebay", "front", "back"}
WRAP = {"verdict", "likely_coin", "suspect", "why", "look_for", "ai_reading", "legends", "ai_notes", "current"}  # fmt: skip
VERDICTS = [
    "wrong match - confirmed",
    "really valuable",
    "unsure - ask expert",
    "fine as is",
]


def max_price(match: Item) -> float:
    return max((match.get("prices") or {}).values(), default=0.0)


def numista_url(match: Item) -> str | None:
    if not match.get("type_id"):
        return None
    return (
        match.get("url")
        or f"https://en.numista.com/catalogue/pieces{match['type_id']}.html"
    )


def auto_reason(row: Item, previous: Item) -> str | None:
    ident, match = row["_ident"], row["_match"]
    if "confidence" not in ident:
        return None
    if previous.get("type_id") and previous["type_id"] != match.get("type_id"):
        if max_price(previous) >= STAKE_EUR:
            return "Earlier match was replaced by the improved matching."
    if match.get("type_id") and not B.trusted_match(ident, match):
        if max_price(match) >= STAKE_EUR:
            if row["metal_mismatch"]:
                return "Metal seen by the AI differs from the Numista type."
            return "Numista match is ambiguous or the AI was unsure."
    if B.price_suspicious(row):
        return "Numista price is far above the AI's estimate."
    if (ident.get("est_value_eur_guess") or 0) >= STAKE_EUR and not B.trusted_match(
        ident, match
    ):
        return "AI estimates it as valuable, but there is no reliable Numista match."
    return None


def picture(run: str, row: Item):
    images = [
        cv2.imread(os.path.join(run, row[k])) for k in ("front_crop", "back_crop")
    ]
    images = [cv2.resize(i, (PX, PX), interpolation=cv2.INTER_AREA) for i in images]
    gap = np.full((PX, 4, 3), 255, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", np.hstack([images[0], gap, images[1]]))
    return XLImage(BytesIO(buf.tobytes()))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", default=DEFAULT_RUN)
    args = parser.parse_args()  # fmt: skip

    rows = {r["coin_id"]: r for r in B.build_rows(args.run, with_thumbs=False)}
    previous = {
        r["coin_id"]: r
        for r in read_jsonl(os.path.join(args.run, "numista_matches_previous.jsonl"))
    }
    notes: Dict[str, Dict[str, str]] = {}
    if os.path.exists(NOTES_CSV):
        with open(NOTES_CSV, encoding="utf-8", newline="") as f:
            notes = {n["coin_id"]: n for n in csv.DictReader(f)}

    cases: List[Item] = []
    for cid, row in rows.items():
        prev = previous.get(cid, {})
        reason = auto_reason(row, prev)
        note = notes.get(cid)
        if not reason and not note:
            continue
        ident, match = row["_ident"], row["_match"]
        # the suspected valuable type: the replaced one, else the current one
        suspect = prev if prev.get("type_id") and prev["type_id"] != match.get("type_id") else match  # fmt: skip
        note = dict(note or {})
        xf = (suspect.get("prices") or {}).get("xf")
        if not note and xf is not None and xf < 10 and max_price(suspect) >= STAKE_EUR:
            grade = (ident.get("grade_guess") or "unknown").replace("_", " ")
            note = {
                "verdict": "Low stakes: only top-grade pieces are worth more",
                "likely_value_eur": f"about {xf:.2f} (XF)",
                "why": f"{reason} The high Numista price is for uncirculated pieces only; "
                f"in XF the type is worth {xf:.2f} €.",
                "look_for": f"Only worth more with no wear at all (AI grade guess: {grade}).",
            }
        cases.append(
            {
                "priority": int(note.get("priority") or 3),
                "coin_id": cid,
                "verdict": note.get("verdict") or "Not assessed yet",
                "likely_coin": note.get("likely_coin"),
                "likely_value_eur": note.get("likely_value_eur"),
                "suspect": suspect.get("title"),
                "suspect_price": round(max_price(suspect), 2) or None,
                "why": note.get("why") or reason,
                "look_for": note.get("look_for")
                or "Compare the photo with the Numista pictures of both types.",
                "ai_reading": " · ".join(
                    str(x)
                    for x in (
                        ident.get("country"),
                        ident.get("denomination_title"),
                        ident.get("year"),
                        ident.get("mint_mark") and f"mint {ident['mint_mark']}",
                        ident.get("subject"),
                    )
                    if x
                ),
                "legends": ident.get("legends"),
                "ai_notes": ident.get("notes"),
                "confidence": ident.get("confidence"),
                "current": match.get("title") or match.get("status"),
                "suspect_url": numista_url(suspect),
                "current_url": numista_url(match) if match is not suspect else None,
                "ebay": row["ebay_sold"],
                "front": row["front_crop"],
                "back": row["back_crop"],
                "_row": row,
            }
        )
    cases.sort(key=lambda c: (c["priority"], -(c["suspect_price"] or 0)))

    wb = Workbook()
    ws = wb.active
    ws.title = "Outliers"
    for col, (_, header, width) in enumerate(COLUMNS, start=1):
        cell = ws.cell(row=1, column=col, value=header)
        cell.font = Font(bold=True)
        cell.fill = PatternFill("solid", fgColor="DDE4EE")
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        ws.column_dimensions[get_column_letter(col)].width = width
    for key in ("your_verdict", "your_notes"):
        idx = [c[0] for c in COLUMNS].index(key) + 1
        ws.cell(row=1, column=idx).fill = PatternFill("solid", fgColor="FFF2CC")
    verdict_col = get_column_letter([c[0] for c in COLUMNS].index("your_verdict") + 1)
    validation = DataValidation(type="list", formula1=f'"{",".join(VERDICTS)}"', allow_blank=True)  # fmt: skip
    ws.add_data_validation(validation)

    for r_idx, case in enumerate(cases, start=2):
        ws.row_dimensions[r_idx].height = PX * 0.75 + 6
        for col, (key, _, _) in enumerate(COLUMNS, start=1):
            if key == "picture":
                ws.add_image(
                    picture(args.run, case["_row"]), f"{get_column_letter(col)}{r_idx}"
                )
                continue
            value = case.get(key)
            cell = ws.cell(row=r_idx, column=col)
            if key in LINKS and value:
                cell.value = (
                    "open" if key not in ("front", "back") else os.path.basename(value)
                )
                cell.hyperlink = value
                cell.font = Font(color="1F5FBF", underline="single")
            else:
                cell.value = value
            cell.alignment = Alignment(vertical="top", wrap_text=key in WRAP)
            if key == "priority":
                cell.font = Font(bold=True)
        validation.add(f"{verdict_col}{r_idx}")
        for key in ("your_verdict", "your_notes"):
            idx = [c[0] for c in COLUMNS].index(key) + 1
            ws.cell(row=r_idx, column=idx).fill = PatternFill("solid", fgColor="FFFBEA")
    ws.freeze_panes = "D2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(COLUMNS))}{len(cases) + 1}"

    legend = wb.create_sheet("How to use")
    text = [
        "Outlier review",
        "",
        "Coins where a very valuable Numista type was suspected, or where the match or price looks doubtful.",
        "Prio 1 = real money at stake or clearly worth checking; 2 = worth a look; 3 = low stakes.",
        "",
        "For each coin: read 'Why we think so', check 'What to look for' on the real coin (magnifier, scale),",
        "then pick YOUR verdict from the dropdown and add notes. Links: Numista = catalogue page,",
        "eBay sold = prices people actually paid, Front/Back = the photo crops on this PC.",
        "",
        "If a coin is really something else, add a correction to overrides.csv (coin_id,field,value,source,note),",
        "e.g. numista_type_id, and re-run numista.py and build_table.py.",
    ]
    legend.column_dimensions["A"].width = 120
    for i, line in enumerate(text, start=1):
        legend.cell(row=i, column=1, value=line).font = Font(
            bold=i == 1, size=13 if i == 1 else 11
        )

    out = os.path.join(args.run, "outlier_review.xlsx")
    try:
        wb.save(out)
    except PermissionError:
        sys.exit(f"{out} is open in Excel - close it and run again.")
    by_prio = {p: sum(1 for c in cases if c["priority"] == p) for p in (1, 2, 3)}
    print(
        f"Wrote {out}: {len(cases)} coins (priority 1/2/3: {by_prio[1]}/{by_prio[2]}/{by_prio[3]})."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
