# Architecture

## Web first, phone app later

- **PWA** (website installable to the home screen). Phone browsers have camera access
  (`getUserMedia` / `<input capture>`), so one codebase serves phone and desktop.
- Phone app later: wrap the PWA with **Capacitor**; switch to React Native only if the
  live camera guidance needs native performance.
- Frontend: React (Next.js or Vite) so the Capacitor path stays easy.

## Backend

- Keep Python. Turn the scripts into a package (`coinscan/`):
  `extract` → `pair` → `identify` → `match_catalogue` → `value` → `export`.
- **FastAPI** for the API, a **job-queue worker** (RQ / arq / Celery) for processing.
- **Postgres** + **object storage** for images.
  Option 1: Supabase (EU region; Postgres, auth and storage in one).
  Option 2: Hetzner VPS + Cloudflare R2.
- **Stripe** for payments.
- Host in the EU (GDPR).
- Clustering: drop the GPU step. It is only a consistency check; use a small CPU
  embedding model or skip it. Duplicates are found by identity key anyway.

## Processing tiers

| Tier | API | Time | Cost |
|---|---|---|---|
| Express | Messages API (synchronous) | minutes | ~2× |
| Economy | Message Batches API | usually < 1 hour | 50 % off |

Can be a pricing lever (Express included in paid tiers, Economy in free tier).

## Data model (sketch)

- `user`, `collection`, `sheet` (front image, back image, sheet type, quality flags)
- `coin` (collection, sheet, slot, front/back crop, diameter_mm, status: identified / confirmed / sold)
- `identification` (LLM JSON, model, confidence, cost)
- `catalogue_match` (Numista type/issue ID, score, fetched_at, respecting cache limits)
- `valuation` (grade, catalogue price, melt value, silver_g, gold_g, method, date)
- `correction` (user edits, for training data with consent)

## Review screen (the core of the product)

- Our crop next to the catalogue candidates' images (Numista thumbnails or own catalogue).
- One-click confirm, or pick an alternative, or edit fields.
- Queue order: high value first, then low confidence, then cluster disagreements.
- Confirmed coins count as "confirmed" in stats ([04](04-data-and-highlights.md)).
- Early on: a manual review queue for us (concierge), using the `identify-coins` skill.

## Exports

- **XLSX**: current `build_table.py` layout (Coins, Types, Top, Review sheets).
- **CSV**: one row per coin.
- **PDF report**: thumbnails, totals, metal content, top coins, disclaimer. Usable for
  insurance and estates.
- **JSON**: full data.
- **Share link**: read-only page for a dealer or family members.
- Later: Google Sheets; Numista collection import format if one exists.

## Values

- `est_value = max(melt, catalogue price at assumed grade)` as now.
- Live metal spot prices from an API instead of hand-edited `.env` values.
- Always show the melt floor and the catalogue estimate separately, with a disclaimer.
