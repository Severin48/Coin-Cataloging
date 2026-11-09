"""Open coins: Excel worksheet for your own observations, then Claude identifies again.

    uv run open_coins.py                    # write <run>/open_coins.xlsx (keeps notes already typed)
    uv run open_coins.py --all              # also the coins that only wait for Numista quota
    uv run open_coins.py --list             # free: which coins have new notes + cost estimate
    uv run open_coins.py --submit           # send ONLY coins with a new/changed note (Batch API)
    uv run open_coins.py --collect [--wait] # fetch the results into identifications.jsonl
    # then: uv run numista.py && uv run build_table.py && uv run open_coins.py

The worksheet has one row per open coin ("identify manually" in build_table.open_status),
most valuable first: picture, where it lies, the LLM's identification and value, links to
both crops and a yellow column "My observations". Coins you wrote a note for stay on the
sheet after they're resolved, with the new result in "Status".

Notes are saved to coin_notes.csv (repo root, tracked) every time the script runs, so
rebuilding the sheet never loses them; deleting a note in Excel deletes it there too.
A Numista link or "N# 12345" in a note becomes a numista_type_id override in overrides.csv;
coins with such an override (from a note or set by hand) are never sent to Claude.

Each request sends both crops plus your note and the earlier identification; the answer
is appended to identifications.jsonl (the last row per coin wins), with your note in
"user_notes". A note is sent again only after you change it.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional

import anthropic
import cv2
import numpy as np
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from openpyxl import Workbook, load_workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

import build_table as B
import i18n
import identify as ID
import selling
from coins_common import (
    DEFAULT_RUN,
    OVERRIDES_CSV,
    Item,
    append_jsonl,
    crop_side,
    iter_overrides,
    load_coins,
    read_jsonl,
)
from i18n import localize, tr

NOTES_CSV = "coin_notes.csv"
THUMB = 100  # px per side in the sheet
NOTE_FILL = PatternFill("solid", fgColor="FFF2CC")
EFFORT = "medium"  # few, hard coins: worth more thinking than the bulk run (low)
EST_OUTPUT_TOKENS = 2000

NOTES_PROMPT = """The owner examined the physical coin and wrote these observations
(possibly in German):
<owner_notes>
{note}
</owner_notes>

Earlier identification from the photos alone (may be wrong):
{previous}

Identify the coin again. The owner holds the coin, so prefer their readings (legends, date,
mint mark, metal, weight, diameter) over what the photos suggest. If a note clearly
contradicts the photos, follow the note and say so in `notes`. If the owner names a
catalogue type, use it. Answer in English as before; `notes` should say briefly what
changed compared with the earlier identification."""

COLUMNS = [
    # key, header, width
    ("picture", "Picture", THUMB * 2 / 7 + 2),
    ("coin_id", "Coin ID", 17),
    ("where", "Where on the page", 14),
    ("title", "Estimated coin", 34),
    ("composition", "Composition", 14),
    ("value", "Estimated value €", 11),
    ("open_reason", "Open: why", 18),
    ("numista_title", "Numista suggestion", 26),
    ("front_crop", "Front", 9),
    ("back_crop", "Back", 9),
    ("note", "My observations", 60),
    ("status", "Status", 30),
]

HELP_LINES = [
    "help_title",
    "help_what",
    "help_examples",
    "help_numista",
    "help_next",
    "help_only",
]


# ------------------------------------------------------------------------ notes


def load_notes() -> Dict[str, str]:
    if not os.path.exists(NOTES_CSV):
        return {}
    with open(NOTES_CSV, encoding="utf-8", newline="") as f:
        return {r["coin_id"]: r["note"] for r in csv.DictReader(f) if r["note"].strip()}


def save_notes(notes: Dict[str, str], old: Dict[str, str]) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d")
    prev: Dict[str, str] = {}
    if os.path.exists(NOTES_CSV):
        with open(NOTES_CSV, encoding="utf-8", newline="") as f:
            prev = {r["coin_id"]: r["updated"] for r in csv.DictReader(f)}
    with open(NOTES_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["coin_id", "note", "updated"])
        for cid in sorted(notes):
            changed = notes[cid] != old.get(cid)
            w.writerow([cid, notes[cid], stamp if changed else prev.get(cid, stamp)])


def sheet_notes(path: str) -> Optional[Dict[str, str]]:
    """coin_id -> note as typed in the worksheet ('' = cleared), None if no sheet."""
    if not os.path.exists(path):
        return None
    try:
        wb = load_workbook(path, read_only=True)
    except PermissionError:
        sys.exit(f"{path} is open in Excel - save it, close it and run again.")
    ws = wb.worksheets[0]
    header = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
    wanted = (
        {"Coin ID", i18n.DE["Coin ID"]},
        {"My observations", i18n.DE["My observations"]},
    )
    try:
        id_col = next(i for i, h in enumerate(header) if h in wanted[0])
        note_col = next(i for i, h in enumerate(header) if h in wanted[1])
    except StopIteration:
        sys.exit(f"{path}: columns 'Coin ID' / 'My observations' not found.")
    out = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[id_col]:
            out[str(row[id_col])] = str(row[note_col] or "").strip()
    wb.close()
    return out


def sync_notes(run: str) -> Dict[str, str]:
    """Merge the worksheet into coin_notes.csv (the sheet wins for coins it shows)."""
    old = load_notes()
    notes = dict(old)
    typed = sheet_notes(os.path.join(run, "open_coins.xlsx"))
    for cid, note in (typed or {}).items():
        if note:
            notes[cid] = note
        else:
            notes.pop(cid, None)
    if notes != old:
        save_notes(notes, old)
        print(f"{NOTES_CSV}: {len(notes)} notes ({len(notes) - len(old):+d}).")
    add_numista_overrides(notes)
    return notes


# only unambiguous forms: "laut Numista 1950 geprägt" must not become type 1950
NUMISTA_ID = re.compile(r"(?:numista\.com/catalogue/pieces|\bN#\s*)(\d{1,7})", re.I)


def add_numista_overrides(notes: Dict[str, str]) -> None:
    """A Numista link / N# in a note is as good as a manual numista_type_id override."""
    current: Dict[str, str] = {}
    for row in iter_overrides():
        if row["field"] == "numista_type_id":
            current[row["coin_id"]] = row["value"]
    new = []
    for cid, note in sorted(notes.items()):
        m = NUMISTA_ID.search(note)
        if m and current.get(cid) != m.group(1):
            new.append([cid, "numista_type_id", m.group(1), NOTES_CSV, "from my notes"])
    if new:
        with open(OVERRIDES_CSV, "rb") as f:  # don't glue the new row onto the last one
            f.seek(-1, os.SEEK_END)
            needs_newline = f.read(1) not in (b"\n", b"\r")
        with open(OVERRIDES_CSV, "a", encoding="utf-8", newline="") as f:
            if needs_newline:
                f.write("\r\n")
            csv.writer(f).writerows(new)
        print(f"{OVERRIDES_CSV}: {len(new)} Numista types taken from notes.")


# ------------------------------------------------------------------------ sheet


def describe(r: Item) -> str:
    parts = [
        localize("country", r["country"]) or "?",
        r["denomination"] or "?",
        r["year"] or "?",
    ]
    if r["subject"]:
        parts.append(r["subject"])
    return " · ".join(parts)


def picture(run: str, coin_id: str) -> Optional[np.ndarray]:
    sides = []
    for side in ("front", "back"):
        img = cv2.imread(os.path.join(run, "crops", f"{coin_id}_{side}.jpg"))
        if img is None:
            return None
        sides.append(cv2.resize(img, (THUMB, THUMB), interpolation=cv2.INTER_AREA))
    return np.hstack(sides)


def manual_types() -> Dict[str, str]:
    """coin_id -> Numista type set by hand: such coins need no re-identification."""
    return {
        row["coin_id"]: row["value"]
        for row in iter_overrides()
        if row["field"] == "numista_type_id" and row["value"]
    }


def status(r: Item, note: str, pending: set) -> str:
    ident = r["_ident"]
    if r["coin_id"] in pending:
        return tr("sent to Claude, waiting for the result")
    if note and r["coin_id"] in manual_types():
        done = f"{tr('Numista type set by hand')}: N# {manual_types()[r['coin_id']]}"
        if r["open_reason"]:
            return f"{done}, {localize('open_reason', r['open_reason'])}"
        value = (
            f"{r['est_value_eur']:,.2f} €" if r["est_value_eur"] is not None else "?"
        )
        return f"{done}, {value} ({localize('value_source', r['value_source'])})"
    if note and ident.get("user_notes") == note:
        if r["open_reason"]:
            return (
                f"{tr('identified again')}: {localize('open_reason', r['open_reason'])}"
            )
        value = (
            f"{r['est_value_eur']:,.2f} €" if r["est_value_eur"] is not None else "?"
        )
        return f"{tr('identified again')}: {value} ({localize('value_source', r['value_source'])})"
    if note:
        return tr("new note: will be sent with --submit")
    return ""


def pending_ids(run: str) -> set:
    return {
        cid
        for b in ID.load_batches(batches_path(run))
        if not b.get("collected")
        for cid in b["coin_ids"]
    }


def write_sheet(
    run: str, rows: List[Item], notes: Dict[str, str], everything: bool
) -> str:
    wanted = {B.MANUAL, B.WAIT} if everything else {B.MANUAL}
    chosen = [r for r in rows if r["open_action"] in wanted or r["coin_id"] in notes]
    chosen.sort(key=lambda r: -(r["est_value_eur"] or r["llm_guess"] or 0))
    selling.add_locations(run, chosen)
    pending = pending_ids(run)

    wb = Workbook()
    ws = wb.active
    ws.title = tr("Open coins")
    for col, (_, header, width) in enumerate(COLUMNS, start=1):
        cell = ws.cell(row=1, column=col, value=tr(header))
        cell.font = Font(bold=True)
        cell.fill = NOTE_FILL if header == "My observations" else B.HEADER_FILL
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        ws.column_dimensions[get_column_letter(col)].width = width
    for i, r in enumerate(chosen, start=2):
        ws.row_dimensions[i].height = THUMB * 0.75 + 4
        note = notes.get(r["coin_id"], "")
        values = {
            "coin_id": r["coin_id"],
            "where": f"{r['page']} · {tr('Row')} {r['row']} · {tr('Pos.')} {r['pos']}",
            "title": describe(r),
            "composition": r["composition"],
            "value": r["est_value_eur"]
            if r["est_value_eur"] is not None
            else r["llm_guess"],
            "open_reason": localize("open_reason", r["open_reason"]),
            "numista_title": r["numista_title"],
            "note": note or None,
            "status": status(r, note, pending) or None,
        }
        for col, (key, _, _) in enumerate(COLUMNS, start=1):
            cell = ws.cell(row=i, column=col)
            if key == "picture":
                pic = picture(run, r["coin_id"])
                if pic is not None:
                    ok, buf = cv2.imencode(".jpg", pic, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    ws.add_image(XLImage(BytesIO(buf.tobytes())), cell.coordinate)
                continue
            if key in ("front_crop", "back_crop"):
                cell.value, cell.hyperlink, cell.font = tr("open"), r[key], B.LINK_FONT
            elif key == "numista_title" and r["numista_url"]:
                cell.value = r["numista_title"]
                cell.hyperlink, cell.font = r["numista_url"], B.LINK_FONT
            else:
                cell.value = values[key]
            if key == "value":
                cell.number_format = '#,##0.00 "€"'
            if key == "note":
                cell.fill = NOTE_FILL
            cell.alignment = Alignment(
                vertical="center",
                wrap_text=key
                in ("title", "open_reason", "numista_title", "note", "status"),
            )
    ws.freeze_panes = "C2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(COLUMNS))}{len(chosen) + 1}"

    help_ws = wb.create_sheet(tr("How to"))
    help_ws.column_dimensions["A"].width = 110
    for i, key in enumerate(HELP_LINES, start=1):
        cell = help_ws.cell(row=i, column=1, value=tr(key))
        cell.alignment = Alignment(wrap_text=True, vertical="top")
        cell.font = Font(bold=i == 1, size=12 if i == 1 else 11)

    out = os.path.join(run, "open_coins.xlsx")
    try:
        wb.save(out)
    except PermissionError:
        sys.exit(f"{out} is open in Excel - close it and run again.")
    n_notes = sum(1 for r in chosen if notes.get(r["coin_id"]))
    print(f"Wrote {out} ({i18n.LANG}): {len(chosen)} coins, {n_notes} with a note.")
    return out


# ------------------------------------------------------------ re-identification


def batches_path(run: str) -> str:
    return os.path.join(run, "reidentify_batches.json")


def todo(run: str, notes: Dict[str, str]) -> List[Item]:
    """Coins with a note that Claude hasn't seen yet (new or changed)."""
    idents = {
        r["coin_id"]: r for r in read_jsonl(os.path.join(run, "identifications.jsonl"))
    }
    pending = pending_ids(run)
    typed = manual_types()
    coins = {c["coin_id"]: c for c in load_coins(run)}
    out = []
    for cid, note in sorted(notes.items()):
        if cid not in coins or cid in pending or cid in typed:
            continue
        if idents.get(cid, {}).get("user_notes") == note:
            continue
        out.append({**coins[cid], "_note": note, "_ident": idents.get(cid, {})})
    return out


def previous_text(ident: Item) -> str:
    keys = (
        "country", "denomination_title", "year", "mint_mark", "subject",
        "composition_guess", "km_number", "est_value_eur_guess", "confidence", "notes",
    )  # fmt: skip
    if not ident:
        return "none"
    return json.dumps({k: ident.get(k) for k in keys}, ensure_ascii=False)


def build_params(item: Item, model: str) -> Dict[str, Any]:
    content: List[Dict[str, Any]] = []
    for label, side in (("A", "front"), ("B", "back")):
        content.append({"type": "text", "text": f"Image {label}:"})
        image = ID.upscale(crop_side(item, side, ID.MAX_SIDE_PX))
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": ID.encode_jpeg(image),
                },
            }
        )
    text = NOTES_PROMPT.format(
        note=item["_note"], previous=previous_text(item["_ident"])
    )
    content.append({"type": "text", "text": text})
    return {
        "model": model,
        "max_tokens": 16000,
        "system": ID.SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": content}],
        "output_config": {
            "format": {"type": "json_schema", "schema": ID.SCHEMA},
            "effort": EFFORT,
        },
    }


def cmd_list(
    client: anthropic.Anthropic, run: str, notes: Dict[str, str], model: str
) -> None:
    items = todo(run, notes)
    if not items:
        print("No new or changed notes - nothing to send.")
        return
    for it in items:
        print(f"  {it['coin_id']:<18} {it['_note'][:90]}")
    params = build_params(items[0], model)
    params.pop("max_tokens")
    n_in = client.messages.count_tokens(**params).input_tokens
    price_in, price_out = ID.PRICES[model]
    per_coin = (n_in * price_in + EST_OUTPUT_TOKENS * price_out) / 1e6 * 0.5
    print(
        f"{len(items)} coins with new notes, ~{n_in} input tokens each ({model}, Batch API): "
        f"about ${per_coin * len(items):.2f}. Send them with --submit."
    )


def cmd_submit(
    client: anthropic.Anthropic, run: str, notes: Dict[str, str], model: str
) -> None:
    items = todo(run, notes)
    if not items:
        print("No new or changed notes - nothing to send.")
        return
    requests = [
        Request(
            custom_id=it["coin_id"],
            params=MessageCreateParamsNonStreaming(**build_params(it, model)),
        )
        for it in items
    ]
    batch = client.messages.batches.create(requests=requests)
    batches = ID.load_batches(batches_path(run))
    batches.append(
        {
            "id": batch.id,
            "model": model,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "coin_ids": [it["coin_id"] for it in items],
            "notes": {it["coin_id"]: it["_note"] for it in items},
            "collected": False,
        }
    )
    ID.save_batches(batches_path(run), batches)
    print(f"Submitted batch {batch.id} with {len(items)} coins.")
    print("Collect with: uv run open_coins.py --collect --wait")


def cmd_collect(client: anthropic.Anthropic, run: str, wait: bool) -> None:
    out_path, _, err_path = ID.paths(run)
    batches = ID.load_batches(batches_path(run))
    open_batches = [b for b in batches if not b.get("collected")]
    if not open_batches:
        print("No open batches.")
        return
    infos: List[Item] = []
    while open_batches:
        for b in list(open_batches):
            state = client.messages.batches.retrieve(b["id"])
            if state.processing_status != "ended":
                print(
                    f"{b['id']}: {state.processing_status} ({state.request_counts.processing} processing)"
                )
                continue
            rows, errors = [], []
            for result in client.messages.batches.results(b["id"]):
                cid = result.custom_id
                if result.result.type != "succeeded":
                    errors.append(
                        {
                            "coin_id": cid,
                            "error": result.result.type,
                            "stage": "reidentify",
                        }
                    )
                    continue
                row, info = ID.parse_message(cid, result.result.message, b["model"])
                infos.append(info)
                if row is None:
                    errors.append({**info, "stage": "reidentify"})
                    continue
                row["user_notes"] = b["notes"][cid]
                row["reidentified"] = time.strftime("%Y-%m-%d")
                rows.append(row)
                print(
                    f"  {cid:<18} {row['country'] or '?'} | {row['denomination_title'] or '?'} | "
                    f"{row['year'] or '?'} | ~{row['est_value_eur_guess']} € | conf {row['confidence']:.2f}"
                )
            append_jsonl(out_path, rows)
            append_jsonl(err_path, errors)
            b["collected"] = True
            ID.save_batches(batches_path(run), batches)
            open_batches.remove(b)
            print(
                f"{b['id']}: {len(rows)} coins identified again, {len(errors)} failed."
            )
        if open_batches and not wait:
            print("Not finished yet; run again later (or add --wait).")
            break
        if open_batches:
            time.sleep(30)
    if infos:
        print(f"Cost: ${ID.cost_usd(infos, batches[0]['model'], batch=True):.2f}")
        print(
            "Next: uv run numista.py (new lookups need Numista quota), "
            "uv run build_table.py, uv run open_coins.py"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--model", default=ID.DEFAULT_MODEL, choices=sorted(ID.PRICES))
    parser.add_argument(
        "--all", action="store_true", help="also coins waiting for Numista"
    )
    parser.add_argument("--lang", choices=["EN", "DE", "en", "de"])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--list", action="store_true", help="free: coins with new notes + cost"
    )
    mode.add_argument(
        "--submit", action="store_true", help="send them to Claude (Batch API)"
    )
    mode.add_argument("--collect", action="store_true", help="fetch the batch results")
    parser.add_argument(
        "--wait", action="store_true", help="with --collect: poll until done"
    )
    args = parser.parse_args()  # fmt: skip
    if args.lang:
        i18n.set_language(args.lang)

    notes = sync_notes(args.run)
    if args.list or args.submit or args.collect:
        client = anthropic.Anthropic()
        if args.list:
            cmd_list(client, args.run, notes, args.model)
        elif args.submit:
            cmd_submit(client, args.run, notes, args.model)
        else:
            cmd_collect(client, args.run, args.wait)
        return 0
    write_sheet(args.run, B.build_rows(args.run, with_thumbs=False), notes, args.all)
    return 0


if __name__ == "__main__":
    sys.exit(main())
