---
name: identify-coins
description: Identify or double-check specific coins by looking at their crops (country, denomination, year, subject, Numista type, rough value) and record the result in overrides.csv. Use for the review queue, low-confidence or high-value coins, clusters that disagree, or when the user asks "what is coin X" or "is this worth anything".
---

# Identify coins (Claude Code as the reviewer)

The `identify.py` stage does the bulk identification over the API. This skill covers
the cases it gets wrong, plus spot checks. It runs on the user's Claude Code
subscription, so it adds no API cost. It writes the same fields that
`identify.py` produces (schema in `docs/PLAN.md`, Phase 1).

## 1. Decide which coins

- Explicit IDs: `b2_s1_coin3` = `f"{pair_key}_coin{coin_number}"`.
- A cluster: `--group <group_id>` (all members of a cluster).
- The review queue: the `Review` sheet of `results/<run>/coins.xlsx`. Before that
  exists, use `identifications.jsonl` rows with `confidence < 0.6` or null
  denomination/year.
- More than about 15 coins: work in chunks of 10 and write overrides after each chunk,
  so progress survives an interruption. Use subagents only if the user asks for them.

## 2. Render the crops

```bash
uv run python .claude/skills/identify-coins/crop_pair.py <coin_id> [...]   # or --group N
```

This writes one side-by-side JPG per coin (front `_v` left, back `_r` right) to
`results/<run>/review_crops/`. View each with the Read tool. Use `--size 768` if small text is unreadable.

## 3. Identify

Read from the coin, not from memory: legend, numeral and unit, year, mint mark,
portrait or subject. Then work out the following:

- **Country / issuer** as it was at minting time (ČSSR ≠ Czech Republic, West Germany/BRD, USSR).
- **Denomination** value and unit. The numeral matters: the same design exists in
  several denominations (e.g. 25 vs 50 Schilling).
- **Which side is the obverse.** Front/back labels are not reliable.
- **Commemorative subject**, if any.
- **Metal.** Many commemoratives are silver. Note the fineness if you know it.

If a field can't be read, leave it empty and say so. Never fill it with a plausible guess.

## 4. Verify and price

- Find the Numista type with WebSearch, e.g. `site:numista.com 100 korun 1978 Karel IV`.
  Open the type page and confirm with the pictures.
  Record `numista_type_id` (the number in `pieces<ID>.html`).
  This uses the website, not the API, so it does not touch the 2,000/month API quota.
- From the type page, note weight, composition, and the price table for the year if present.
- For coins that look valuable, also check sold listings:
  `https://www.ebay.de/sch/i.html?_nkw=<query>&LH_Sold=1&LH_Complete=1`.
  Prices from sold listings beat prices from asking listings.

## 5. Record

Append to `overrides.csv` in the repo root. Create it with this header if it is missing:

```csv
coin_id,field,value,source,note
b2_s1_coin3,denomination_value,100,claude-code,read from reverse
b2_s1_coin3,numista_type_id,1234,https://en.numista.com/catalogue/pieces1234.html,
```

- `field` is one of the identification fields (`country`, `denomination_value`,
  `denomination_text`, `currency`, `year`, `mint_mark`, `subject`, `composition_guess`,
  `km_number`, `front_is_obverse`), or `numista_type_id`, `numista_issue_id`,
  `est_value_eur`, `exclude` (`1` = not a coin / bad crop).
- Append only. Later rows win. Never edit or delete rows the user wrote.
- Put the proof URL in `source` whenever there is one.

Finally, give the user a short table: coin_id, what it is, confidence, value hint, and what you changed.
Remind them that `build_table.py` picks up overrides offline, and that `numista.py` only
needs re-running for new `numista_type_id`s or identities.
