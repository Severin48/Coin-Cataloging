---
name: coin-pipeline
description: Run, debug or extend the coin cataloguing pipeline in this repo (extract crops → cluster → LLM identify → Numista match → coins.xlsx). Use when the user asks to run a stage, re-run on new images, build or rebuild the spreadsheet, check API/quota cost, or implement the next phase from docs/PLAN.md.
---

# Coin pipeline

The source of truth for *what* to build is `docs/PLAN.md`. This skill covers *how*
to run each stage and the conventions that stop the stages from drifting apart.

## Stages and files

| # | Stage | Script | Reads | Writes (in `results/<run>/`) | Cost |
|---|---|---|---|---|---|
| 1 | Extract coins | `main.py` → `extract_coin_images.detect_coins()` | `images/*_v.jpg`, `*_r.jpg` | `extraction_results.json` (+ `extracted/`, `debug/` if enabled) | free |
| 2 | Cluster | `clustering.py`, review with `check_clusters.py` | extraction JSON + `images/` | `extraction_results_with_groups.json`, `cluster_review_output/` | free (GPU) |
| 3 | Identify | `identify.py` | extraction JSON + `images/` | `identifications.jsonl` | **paid (Claude API)** |
| 4 | Catalogue and prices | `numista.py` | `identifications.jsonl` | `numista_matches.jsonl`, `numista_cache/`, `numista_usage.json` | free quota, 2,000 req/month |
| 5 | Table | `build_table.py` | all of the above + `overrides.csv` + metal prices in `.env` | `coins.xlsx`, `highlights.md`, `valuation.json`, `crops/` | free |

Shared helpers (default run, `coin_id`, cropping, JSONL, overrides) are in `coins_common.py`.
Numista cache and monthly counter are global (`results/numista_cache/`, `results/numista_usage.json`),
because the quota belongs to the API key, not to a run.

## Conventions (keep these when you change any stage)

- **One canonical run.** Every stage reads and writes the same `results/<run>/` folder.
  Coin numbering changes when extraction changes, so never mix files from two runs.
  Canonical run: `results/2025_11_16_15_40_15` (see PLAN Phase 0).
- **Coin ID** = `f"{pair_key}_coin{coin_number}"`. That is the key in every JSONL,
  the CSV and the xlsx, and it matches the extracted PNG stem (`<coin_id>_front.png`).
- **Front/back are paired correctly but not labelled reliably.** `front` is just the
  `_v` scan. Never assume it is the obverse. Send both sides to anything that identifies the coin.
- Crop from the original scans using `front_rect` / `back_rect`. The saved PNGs are optional.
- `images/`, `results/`, `bag_images/` are gitignored user data. Never delete them.
  `overrides.csv` (manual corrections, `coin_id,field,value`) lives in the repo root and is tracked.
- Secrets come from `.env` (`ANTHROPIC_API_KEY`, `NUMISTA_API_KEY`). Never write them
  into code, logs or chat output. `.env` already holds the Numista key.
- Every stage is **resumable and idempotent**. It skips coin IDs or URLs it already has.
- Tooling: `uv run <script>`. After edits, run `uv run ruff check --fix` and `uv run ruff format`.
  The code base is typed; keep the type hints.

## Paid stages: rules

- **Identify:** always run `--sample 30` first and show the user the measured cost per coin
  and the projected total. Start the full `--batch` only after the user agrees.
  Budget is about €20 total.
- **Numista:** check `uv run numista.py --status` before a run. Stop at about 1,900 requests per
  calendar month. Process in priority order. Serve repeats from `numista_cache/`.
- Never re-run a paid stage just to rebuild the table. `build_table.py` is offline.

## Answering "what are my most valuable coins?"

Run `uv run build_table.py`, then read `<run>/highlights.md`. A coin counts as a highlight only if
LLM confidence ≥ 0.8, the Numista match is unambiguous (margin ≥ 0.5 or a `numista_type_id`
override), and the value comes from a Numista price or melt, not from the LLM guess
(`is_highlight()` in `build_table.py`). The second list in that file, "possibly valuable, not
confirmed", is the review queue for `identify-coins`. Always state the metal prices and the
grade the values assume (`valuation.json`).

## Quick checks

```bash
# How many coins / clusters in the canonical run
uv run python -c "import json,collections as c;d=json.load(open('results/2025_11_16_15_40_15/extraction_results.json'));print(len(d))"
# Numista quota used this month (no request)
uv run numista.py --status
```

Related skills: `tune-clustering` (stage 2), `identify-coins` (manual or assisted identification and review).
