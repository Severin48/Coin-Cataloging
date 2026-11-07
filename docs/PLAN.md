# Plan: from coin crops to a priced spreadsheet

Goal: one `coins.xlsx` with every coin (front/back image paths), what it is
(country, denomination, currency, year, mint mark, subject, metal), what it is
worth (Numista price by grade, metal melt value, links as proof), and a sorted
view of the most valuable coins to sell.

Status (2026-11-07): all phases built and run. 1,665 coins identified ($13.22 Batch API). November Numista quota used (1,950 requests): 665 coins resolved, including all silver/gold ones. 449 highlights worth ~€9,450, collection ~€13,100. **Next: run `uv run numista.py` on December 2 or later** (schedule at the end of this file).

---

## Key decisions

### 1. Identify every coin with a vision LLM, not with image search

- **Numista search-by-image** costs €100 activation + €100/month minimum. Too expensive.
- **Google** retired its public image-search API. Lens has no official API.
- **A vision LLM reads the coin.** It reads the legend, the numeral and the year,
  and it knows the common catalogue (KM) numbers. Our crops are sharp, so this works well.
  It answers with structured JSON that we can feed straight into Numista's *text* search.
- The **free Numista plan** (2,000 requests/month) includes text search, type
  details (weight, composition), issues and **price estimates by grade**. Only
  image search is paid. → Use Numista for the catalogue and prices, not for identification.

### 2. Stop tuning the clustering; let identification decide what is "the same coin"

The 25-vs-50-Schilling problem is a limit of image embeddings. The design is identical
and only the numeral differs. No `eps` value fixes that.

Instead:

- Run the LLM on **every** coin (1,665 coins is affordable; see costs below).
- Deduplicate Numista lookups by the **identity key** the LLM returns
  (`issuer + denomination + currency + year + subject`), not by cluster.
- Keep the clusters as a **consistency check**. If members of one cluster get
  different identities, flag them for review. The difference is either a real
  denomination difference (fine) or an LLM misread (fix it).

A side effect: a slightly *loose* `eps` is now harmless and makes the check more
useful. See `.claude/skills/tune-clustering` if you still want better clusters.

### 3. Value = max(Numista price at an assumed grade, melt value), proven by links

Many of these coins are silver commemoratives (ČSSR 50/100 Kčs, Israel, Canada,
Kennedy halves, UK crowns). For those, the silver content sets a floor price.
Numista gives `weight` and `composition`, so `melt_eur = weight × fineness × silver €/g`.
Each row gets a Numista link plus an eBay **sold listings** search link, so you can
check any coin with one click. No eBay API is needed for that.

### 4. No GUI

The Streamlit labelling tool sketched in `clustering.py` is the most work for the
least benefit. Replace it with Excel plus an `overrides.csv` that the table builder
applies, and the `identify-coins` Claude Code skill for the review queue.

---

## Costs

### LLM identification (one call per coin, both sides, Batch API = 50 % off)

Per coin: 2 crops at about 560 px (~420 image tokens each), a ~700-token prompt, and
~400–1,000 output tokens (JSON + thinking).

| Model (Batch API) | Price in/out per MTok | ~1,665 coins |
|---|---|---|
| Claude Opus 5.5 (effort `low`) | $2 / $10 | **~$12–20** |
| Claude Sonnet 5.5 | $1 / $5 | **~$6–10** |
| Claude Haiku 4.5 | $0.50 / $2.50 | ~$2–3 (weaker coin knowledge) |

Output tokens are the biggest unknown. **Step 1 runs a 30-coin sample first**
(< $1) and prints the measured cost per coin before you start the full batch.

Free alternatives:

- **Local VLM** on the 5070 Ti (e.g. Qwen-VL 7–8B via Ollama / LM Studio). It reads
  legends and years fine but knows catalogues less well. Expect more review work.
- **Claude Code itself** (the `identify-coins` skill) on your subscription. This is
  good for the review queue and the top-value coins. It is too slow for all 1,665.

### Numista (free, 2,000 requests/month)

Requests: 1 search per **(country, year)**, shared by every coin of that country
and year (one search returns all types of that year), plus type details, issues and prices
per Numista type (3). Measured: **23 requests for 6 types** (~3.8 per type). That covers
about 500 types per month.
We probably have 600–900 unique types, so:

- Cache every response on disk. No request is ever repeated.
- Process in **priority order** (precious metal and high LLM value guess first),
  with a hard stop at ~1,900 requests. The rest runs after the quota resets next month.
- Skip prices for coins the LLM rates as base metal and < €1. That saves many requests.
- `GET /types/{id}/sales_records` (real auction results) only for the top ~50.

Total: roughly **€6–20 in API costs**, depending on the model.

---

## Phases

Each phase is about one short session. Run every stage on one **canonical
results run** so coin IDs stay stable.

### Phase 0: Housekeeping (~30 min)

- [x] Pick the canonical run: `results/2025_11_16_15_40_15` (1,665 coins; the two
      later Nov runs are identical). The cluster `group_id`s live in the **August** run
      (1,687 coins), whose coin numbers do **not** line up with the November run
      (only 1,472 of 1,665 `pair_key`+`coin_number`+rect match).
      → Re-ran `clustering.py` on the canonical run: 1,346 clusters (done).
- [x] Make `JSON_IN` etc. CLI arguments (or one `config.py`) instead of hardcoded paths.
- [x] Stable coin ID everywhere: `coin_id = f"{pair_key}_coin{coin_number}"`
      (the same stem as the extracted PNGs).
- [x] Add the missing deps to `pyproject.toml`: `torch`, `torchvision` (CUDA index),
      `scikit-learn`, plus `anthropic`, `requests`, `openpyxl`, `python-dotenv`.
- [x] Secrets go in `.env` (gitignored): `ANTHROPIC_API_KEY`, `NUMISTA_API_KEY`.

### Phase 1: `identify.py`: LLM identification

- Input: the canonical `extraction_results.json`. Crop both sides from `images/`
  using the rects. Don't depend on saved PNGs. Downscale to ≤ 768 px.
- Prompt: "two sides of the same coin, order and orientation unknown". Ask for
  structured output (JSON schema, see below). Return `null` instead of guessing, plus
  a `confidence`.
- `--sample 30` runs synchronously, prints the answers and the **measured** cost
  per coin with a projection for all coins.
- `--batch` uses the Message Batches API. It is resumable: skip coin IDs already in the output.
- Output: `results/<run>/identifications.jsonl`, one line per coin.

```json
{
  "coin_id": "b2_s1_coin3",
  "country": "Czechoslovakia",
  "denomination_value": 100,
  "denomination_text": "100 Korun",
  "currency": "Koruna",
  "year": "1978",
  "mint_mark": null,
  "subject": "Charles IV (Karel IV), 1346-1978",
  "composition_guess": "silver",
  "km_number": null,
  "numista_query": "100 korun 1978 Charles IV",
  "front_is_obverse": false,
  "est_value_eur_guess": 12,
  "confidence": 0.85,
  "notes": ""
}
```

### Phase 2: `numista.py`: catalogue match and prices

1. Group identifications by identity key. Order the groups by priority.
2. `GET /types?q=<numista_query>&issuer=<code>&year=<year>&object_type=147`.
   Get the issuer codes from `GET /issuers` once and cache them. They are French-based,
   e.g. `allemagne`. Fuzzy-match the LLM's `country` against the issuer names
   (Czechoslovakia → the ČSSR issuer, not modern Czechia/Slovakia).
3. Rank candidates by year range, denomination in title, and **visual similarity**
   between our crops and the candidates' `obverse_thumbnail`/`reverse_thumbnail`.
   The thumbnails come with the search result. Use the same embedding model as the
   clustering; there's no extra API cost.
4. `GET /types/{id}` → `value`, `composition`, `weight`, `references` (KM#), `url`.
5. `GET /types/{id}/issues` → pick the issue by year and mint letter.
6. `GET /types/{id}/issues/{issue_id}/prices?currency=EUR` → `g … unc`.
7. Cache: `results/<run>/numista_cache/<sha1(url)>.json`. Request counter in
   `numista_usage.json` per calendar month.

Output: `results/<run>/numista_matches.jsonl` (identity key → type ID, issue ID,
prices, match score).

### Phase 3: `build_table.py`: the spreadsheet

Merge extraction, clusters, identifications, Numista matches and `overrides.csv`
into `results/<run>/coins.xlsx`. No API calls, so you can re-run it any time after a correction.

| Sheet | Content |
|---|---|
| `Coins` | one row per coin: `coin_id`, `group_id`, front/back paths (hyperlinks), small embedded thumbnails (optional), country, denomination, currency, year, mint mark, subject, composition, weight_g, silver_g, melt_eur, Numista title/ID/link, price_vf/xf/unc, `est_value_eur`, eBay sold link, `needs_review` + reason |
| `Types` | one row per identity: count of coins, the same value columns, total value |
| `Top` | sorted by `est_value_eur × count` |
| `Review` | low confidence, no Numista match, cluster disagreement, or high value |

`est_value_eur = max(melt_eur, numista price at ASSUMED_GRADE)`, with default grade `xf`.
`SILVER_EUR_PER_G` / `GOLD_EUR_PER_G` go in `.env`. Update them by hand before you sell.

eBay sold-listings link:
`https://www.ebay.de/sch/i.html?_nkw=<query>&LH_Sold=1&LH_Complete=1`

### Phase 4: Review loop

- Work through the `Review` sheet. For each coin either fix it in `overrides.csv`
  (`coin_id,field,value`) or ask Claude Code: *"/identify-coins the review queue"*.
  The skill looks at the crops, checks Numista/the web, and writes overrides.
- Re-run `numista.py` (only new identities cost requests) → `build_table.py`.

### Phase 5: Sell

- For the top ~50, call `sales_records` on Numista, open the eBay sold links, and check grade by eye.
- Silver coins below ~1.5× melt: sell as a lot by silver weight.
- Real collector pieces: list them one by one.

---

## Optional improvements (only if needed)

- Better clusters: DINOv2 embeddings, swap-invariant pair distance (both sides),
  rotation averaging, average-linkage instead of DBSCAN. Details in the `tune-clustering` skill.
- eBay Browse API (free developer account) for *active* listing prices.
  Sold prices need the restricted Marketplace Insights API, so use the links instead.

---

## Numista quota schedule (free plan: 2,000 requests per calendar month)

`numista.py` counts every request in `results/numista_usage.json` and stops at **1,950 per month**.
It also stops on a 429 from Numista. Everything it fetched is cached forever.
Only run it from **one machine**: the counter is local. A second laptop would not know what
this one already used.

Measured on the full collection (2026-11-07): 1,665 coins identified for **$13.22**
(Batch API, Opus 5.5), 140 countries, 881 distinct (country, year) searches, ~1,390 types
(~410 of them silver/gold). Base-metal types skip the details request (2 instead of 3),
so the whole collection needs **~4,000 requests**, about 2 months of quota.

| When | Command | Numista requests |
|---|---|---|
| **November 7** ✅ | `uv run numista.py` (done, 1,950 used) | ≤ 1,923: all silver/gold coins (~1,500) + the most valuable base-metal ones |
| November, afterwards | review + `overrides.csv` + `uv run build_table.py` | 0 |
| **December 2 or later** (one day after the reset, to be safe with time zones) | `uv run numista.py --status`, then `uv run numista.py` | up to 1,950 |
| **January 2 or later** | `uv run numista.py` if `--status` still shows `pending_quota` | the rest (~200–400) |

Because coins run in priority order (silver/gold, then by LLM value guess), **the valuable
coins are done in November**. December and January only add cheap base-metal coins.
To stop earlier, skip the cheapest ones: `uv run numista.py --min-guess 1` marks
non-precious coins with an LLM guess under €1 (~600 coins) as `skipped_low_value`.

Never needed: running `numista.py` again in the same month after it stopped (it does nothing),
or for `build_table.py` changes (offline).
