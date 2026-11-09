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
| 6 | Analysis | `analyze.py` | same inputs, via `build_table.build_rows` | `analysis/` (`report.md`, PNG plots, `coins_analysis.csv`, `correlations.csv`) | free |
| 7 | Selling plan | `selling.py` | `analyze.load` + `outlier_notes.csv`, `SELL_*` in `.env` | `selling/` (`selling_guide.md`, `selling_plan.xlsx`, `selling_plan.csv`, `tradeoff.png`, `locate/`) | free |

Workbook language: `SUMMARY_LANGUAGE=EN|DE` in `.env` (or `--lang`), handled by `i18n.py`. Only
written cell text is translated (`tr()`, `localize()`); keep internal keys (`value_source`,
review reasons, grades) English, because formulas, highlights and `review_outliers.py` compare them.

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

## Open coins (value still an LLM guess)

`build_table.open_status()` gives every coin without a Numista price, melt value or override a
reason and a next step. *wait for Numista quota* clears on the next `numista.py` run. *identify
manually* (not identified, not found on Numista, uncertain match, no price) never clears by itself:
work through it with `identify-coins` and record `numista_type_id` or `est_value_eur` in
`overrides.csv`. These coins are on the sheet **Open**, in the last section of `highlights.md` and
counted on the Summary sheet.

For the owner's own checks, `open_coins.py` writes `<run>/open_coins.xlsx` (only the manual ones,
`--all` adds the waiting ones) with a column *My observations*. Notes are synced to `coin_notes.csv`
(repo root, tracked) on every run; a Numista link or `N# 12345` in a note becomes a
`numista_type_id` override. `--list` (free) / `--submit` / `--collect --wait` re-identify **only
coins whose note is new or changed** (Batch API, paid: show `--list` cost first), appending to
`identifications.jsonl` with `user_notes`. Afterwards run `numista.py`, `build_table.py`, `open_coins.py`.

## Outliers (suspected rare types, doubtful prices)

`uv run review_outliers.py` writes `<run>/outlier_review.xlsx`: coins where a valuable type
was suspected (also from `numista_matches_previous.jsonl`), the match is untrusted, or the price
looks implausible. Hand-written assessments live in `outlier_notes.csv` (repo root, tracked:
priority, verdict, likely coin and value, why, what to look for). When you assess a new
outlier, look at the crops first (`crop_pair.py`) and add a row there.
Known limitation: if the mint mark wasn't readable, the coin is priced as the common mint
issue of that year, so key-date mints (e.g. Morgan CC) only show up through review.

## Quick checks

```bash
# How many coins / clusters in the canonical run
uv run python -c "import json,collections as c;d=json.load(open('results/2025_11_16_15_40_15/extraction_results.json'));print(len(d))"
# Numista quota used this month (no request)
uv run numista.py --status
```

Related skills: `tune-clustering` (stage 2), `identify-coins` (manual or assisted identification and review).
