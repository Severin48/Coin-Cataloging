"""Stage 3: identify every coin with Claude vision (both sides per request).

    uv run identify.py --estimate          # free: count input tokens of one request
    uv run identify.py --sample 30         # ~$1: synchronous test, prints measured cost + projection
    uv run identify.py --batch             # submit all remaining coins via the Batch API (50 % off)
    uv run identify.py --collect [--wait]  # fetch finished batch results into identifications.jsonl

Output: <run>/identifications.jsonl (one JSON object per coin, see docs/PLAN.md).
Already identified coins and coins in pending batches are always skipped.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

import anthropic
import cv2
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from coins_common import (
    DEFAULT_RUN,
    Item,
    append_jsonl,
    crop_side,
    load_coins,
    read_jsonl,
)

DEFAULT_MODEL = "claude-opus-5-5"
# USD per million tokens (input, output), standard API; the Batch API costs half
PRICES = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}
MAX_SIDE_PX = 768
MIN_SIDE_PX = 512  # small crops are upscaled: dates and legends read more reliably
BATCH_CHUNK = 400  # requests per batch; keeps each batch well below the 256 MB limit

SYSTEM_PROMPT = """You are an expert numismatist cataloguing a private coin collection.

Each request shows two photos of ONE coin: image A and image B. They are the two sides of
the same coin, but the order is unknown (either may be the obverse) and the rotation is
arbitrary. Some coins are inside plastic capsules.

Identify the coin from what you can actually read and see: legends, the denomination
numeral and unit, the date, mint marks, portrait or design. The same design often exists
in several denominations, so read the numeral carefully. If something is not legible,
use null rather than guessing; lower the confidence instead.

Field guidance:
- country: issuing country/entity at the time of minting, in English
  (e.g. "Czechoslovakia", "Germany (Federal Republic)", "United States", "Israel").
- denomination_value: the face value as a number in the coin's own unit (½ dollar -> 0.5).
- denomination_title: the denomination the way an English catalogue titles it,
  e.g. "100 Korun", "50 Schilling", "½ Dollar", "5 Francs", "25 Pence".
- currency: currency name, e.g. "Czechoslovak koruna".
- year: Gregorian year as a string (convert Hebrew, Islamic, era dates); null if none.
- year_as_written: the date exactly as on the coin if it is not a plain Gregorian year.
- subject: commemorated topic or person, null for regular circulation coins.
- composition_guess: e.g. "silver .925", "copper-nickel", "aluminium"; null if unsure.
- km_number: Krause (KM#) number if you know it reliably, else null.
- numista_query: a short English search query that would find this type in the Numista
  catalogue: denomination title plus 1-3 distinctive keywords (no country, no year).
- front_is_obverse: true if image A is the obverse (state side), false if image B is.
- grade_guess: rough condition from the photos.
- est_value_eur_guess: typical current retail value in EUR for this type in the
  observed condition, including precious-metal value; null if no idea.
- confidence: probability (0-1) that country + denomination + year + subject are all correct.
- notes: anything a human reviewer should know (damage, doubts, possible variants)."""

SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "is_coin": {"type": "boolean"},
        "country": {"type": ["string", "null"]},
        "denomination_value": {"type": ["number", "null"]},
        "denomination_title": {"type": ["string", "null"]},
        "currency": {"type": ["string", "null"]},
        "year": {"type": ["string", "null"]},
        "year_as_written": {"type": ["string", "null"]},
        "mint_mark": {"type": ["string", "null"]},
        "subject": {"type": ["string", "null"]},
        "coin_type": {
            "type": "string",
            "enum": ["circulation", "commemorative", "bullion", "token_or_medal", "unknown"],
        },
        "composition_guess": {"type": ["string", "null"]},
        "km_number": {"type": ["string", "null"]},
        "numista_query": {"type": ["string", "null"]},
        "legends": {"type": ["string", "null"]},
        "front_is_obverse": {"type": ["boolean", "null"]},
        "grade_guess": {
            "type": "string",
            "enum": ["poor", "circulated", "extremely_fine", "uncirculated", "proof", "unknown"],
        },
        "est_value_eur_guess": {"type": ["number", "null"]},
        "confidence": {"type": "number"},
        "notes": {"type": ["string", "null"]},
    },
    "required": [
        "is_coin", "country", "denomination_value", "denomination_title", "currency",
        "year", "year_as_written", "mint_mark", "subject", "coin_type",
        "composition_guess", "km_number", "numista_query", "legends",
        "front_is_obverse", "grade_guess", "est_value_eur_guess", "confidence", "notes",
    ],
    "additionalProperties": False,
}  # fmt: skip


def upscale(img):
    if max(img.shape[:2]) >= MIN_SIDE_PX:
        return img
    scale = MIN_SIDE_PX / max(img.shape[:2])
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def encode_jpeg(img) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return base64.standard_b64encode(buf.tobytes()).decode("ascii")


def build_params(item: Item, model: str) -> Dict[str, Any]:
    content: List[Dict[str, Any]] = []
    for label, side in (("A", "front"), ("B", "back")):
        content.append({"type": "text", "text": f"Image {label}:"})
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": encode_jpeg(upscale(crop_side(item, side, MAX_SIDE_PX))),
                },
            }
        )
    content.append({"type": "text", "text": "Identify this coin."})
    output_config: Dict[str, Any] = {
        "format": {"type": "json_schema", "schema": SCHEMA}
    }
    if not model.startswith("claude-haiku"):
        output_config["effort"] = "low"
    return {
        "model": model,
        "max_tokens": 8000,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": content}],
        "output_config": output_config,
    }


def parse_message(cid: str, message: Any, model: str) -> Tuple[Item | None, Item]:
    """Returns (identification row or None, usage/error info)."""
    usage = {
        "input_tokens": message.usage.input_tokens,
        "output_tokens": message.usage.output_tokens,
    }
    info: Item = {"coin_id": cid, "stop_reason": message.stop_reason, **usage}
    if message.stop_reason in ("refusal", "max_tokens"):
        return None, info
    text = next((b.text for b in message.content if b.type == "text"), "")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        info["error"] = "invalid JSON"
        return None, info
    return {"coin_id": cid, **data, "model": model, "usage": usage}, info


def cost_usd(rows: List[Item], model: str, batch: bool) -> float:
    price_in, price_out = PRICES[model]
    total = sum(
        r["input_tokens"] * price_in + r["output_tokens"] * price_out for r in rows
    )
    return total / 1e6 * (0.5 if batch else 1.0)


def paths(run: str) -> Tuple[str, str, str]:
    return (
        os.path.join(run, "identifications.jsonl"),
        os.path.join(run, "identify_batches.json"),
        os.path.join(run, "identify_errors.jsonl"),
    )


def load_batches(path: str) -> List[Item]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_batches(path: str, batches: List[Item]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(batches, f, indent=2)


def todo_coins(run: str) -> List[Item]:
    out_path, batches_path, _ = paths(run)
    done = {r["coin_id"] for r in read_jsonl(out_path)}
    pending = {
        cid
        for b in load_batches(batches_path)
        if not b.get("collected")
        for cid in b["coin_ids"]
    }
    return [c for c in load_coins(run) if c["coin_id"] not in done | pending]


def cmd_estimate(client: anthropic.Anthropic, run: str, model: str) -> None:
    coins = todo_coins(run)
    if not coins:
        print("Nothing left to identify.")
        return
    params = build_params(coins[0], model)
    params.pop("max_tokens")
    n_in = client.messages.count_tokens(**params).input_tokens
    price_in, price_out = PRICES[model]
    print(f"{len(coins)} coins to identify, {n_in} input tokens per coin ({model}).")
    for n_out in (400, 1000, 2000):
        per_coin = (n_in * price_in + n_out * price_out) / 1e6 * 0.5
        print(
            f"  if ~{n_out} output tokens/coin: batch total ≈ ${per_coin * len(coins):.2f}"
        )
    print("Run --sample 30 to measure the real output tokens.")


def cmd_sample(client: anthropic.Anthropic, run: str, model: str, n: int) -> None:
    coins = todo_coins(run)
    step = max(1, len(coins) // n)  # spread over all scans, not just the first one
    sample = coins[::step][:n]
    out_path, _, err_path = paths(run)

    use_fallback = [model in ("claude-opus-5-5", "claude-sonnet-5-5")]

    def call(item: Item) -> Tuple[Item | None, Item]:
        params = build_params(item, model)
        if use_fallback[0]:
            # Server-side refusal fallback (not available on the Batch API)
            try:
                message = client.beta.messages.create(
                    **params,
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",
                )
                return parse_message(item["coin_id"], message, model)
            except anthropic.BadRequestError as e:
                if "fallback" not in str(e).lower():
                    raise
                use_fallback[0] = False
        message = client.messages.create(**params)
        return parse_message(item["coin_id"], message, model)

    print(f"Identifying {len(sample)} coins synchronously with {model} ...")
    infos: List[Item] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for row, info in pool.map(call, sample):
            infos.append(info)
            if row is None:
                append_jsonl(err_path, [info])
                print(f"  {info['coin_id']}: FAILED ({info})")
                continue
            append_jsonl(out_path, [row])
            print(
                f"  {row['coin_id']:<16} {row['country'] or '?'} | "
                f"{row['denomination_title'] or '?'} | {row['year'] or '?'} | "
                f"{row['subject'] or '-'} | {row['composition_guess'] or '?'} | "
                f"~{row['est_value_eur_guess']} € | conf {row['confidence']:.2f}"
            )

    spent = cost_usd(infos, model, batch=False)
    per_coin_batch = cost_usd(infos, model, batch=True) / len(infos)
    remaining = len(todo_coins(run))
    avg_out = sum(i["output_tokens"] for i in infos) / len(infos)
    avg_in = sum(i["input_tokens"] for i in infos) / len(infos)
    print(
        f"\nThis sample cost ${spent:.2f} ({avg_in:.0f} in / {avg_out:.0f} out tokens per coin)."
    )
    print(
        f"Projected Batch API cost for the remaining {remaining} coins: "
        f"${per_coin_batch * remaining:.2f}"
    )


def cmd_batch(client: anthropic.Anthropic, run: str, model: str, limit: int) -> None:
    coins = todo_coins(run)
    if limit:
        coins = coins[:limit]
    if not coins:
        print("Nothing left to identify.")
        return
    _, batches_path, _ = paths(run)
    batches = load_batches(batches_path)
    for start in range(0, len(coins), BATCH_CHUNK):
        chunk = coins[start : start + BATCH_CHUNK]
        requests = [
            Request(
                custom_id=c["coin_id"],
                params=MessageCreateParamsNonStreaming(**build_params(c, model)),
            )
            for c in chunk
        ]
        batch = client.messages.batches.create(requests=requests)
        batches.append(
            {
                "id": batch.id,
                "model": model,
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "coin_ids": [c["coin_id"] for c in chunk],
                "collected": False,
            }
        )
        save_batches(batches_path, batches)  # save after every batch: crash-safe
        print(f"Submitted batch {batch.id} with {len(chunk)} coins.")
    print("Collect the results with: uv run identify.py --collect --wait")


def cmd_collect(client: anthropic.Anthropic, run: str, wait: bool) -> None:
    out_path, batches_path, err_path = paths(run)
    batches = load_batches(batches_path)
    open_batches = [b for b in batches if not b.get("collected")]
    if not open_batches:
        print("No open batches.")
        return
    all_infos: List[Item] = []
    while open_batches:
        for b in list(open_batches):
            status = client.messages.batches.retrieve(b["id"])
            if status.processing_status != "ended":
                counts = status.request_counts
                print(
                    f"{b['id']}: {status.processing_status} "
                    f"({counts.succeeded} done, {counts.processing} processing)"
                )
                continue
            rows, errors = [], []
            for result in client.messages.batches.results(b["id"]):
                if result.result.type == "succeeded":
                    row, info = parse_message(
                        result.custom_id, result.result.message, b["model"]
                    )
                    all_infos.append(info)
                    (rows if row else errors).append(row or info)
                else:
                    errors.append(
                        {"coin_id": result.custom_id, "error": result.result.type}
                    )
            append_jsonl(out_path, rows)
            append_jsonl(err_path, errors)
            b["collected"] = True
            save_batches(batches_path, batches)
            open_batches.remove(b)
            print(f"{b['id']}: collected {len(rows)} coins, {len(errors)} failed.")
        if open_batches and not wait:
            print("Not finished yet; run again later (or add --wait).")
            break
        if open_batches:
            time.sleep(60)
    if all_infos:
        model = batches[0]["model"]
        print(
            f"Cost of collected results: ${cost_usd(all_infos, model, batch=True):.2f}"
        )
    done = {r["coin_id"] for r in read_jsonl(out_path)}
    failed = {e["coin_id"] for e in read_jsonl(err_path)} - done
    if failed:
        print(
            f"{len(failed)} failed coins are logged in {err_path}; "
            "they are retried by the next --batch."
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(PRICES))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--estimate", action="store_true")
    mode.add_argument("--sample", type=int, metavar="N")
    mode.add_argument("--batch", action="store_true")
    mode.add_argument("--collect", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="max coins for --batch")
    parser.add_argument("--wait", action="store_true", help="poll until batches end")
    args = parser.parse_args()

    client = anthropic.Anthropic()
    if args.estimate:
        cmd_estimate(client, args.run, args.model)
    elif args.sample:
        cmd_sample(client, args.run, args.model, args.sample)
    elif args.batch:
        cmd_batch(client, args.run, args.model, args.limit)
    else:
        cmd_collect(client, args.run, args.wait)
    return 0


if __name__ == "__main__":
    sys.exit(main())
