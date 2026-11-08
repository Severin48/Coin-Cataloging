# Data collection and the highlights page

Goal: know how much value we identify for users, and turn it into a public
highlights page that doubles as marketing.

## What to track per coin

- Status: identified → confirmed (user or reviewer) → sold (optional, user reports)
- Country, year, denomination, catalogue ID
- Value: catalogue estimate, melt value, method used, grade, date of valuation
- Metal: pure silver grams, pure gold grams
- Identification confidence, model, cost
- Whether the user allows it to be featured publicly (opt-in)

## Public highlights page

Aggregates:

- Coins identified, coins confirmed
- **Confirmed value found** (total €)
- **kg of pure silver and gold**, also valued at today's spot price
- Countries covered (map), oldest coin, most common coin
- **Top 10 most valuable confirmed finds** (opt-in only, photo + name + value)
- "Most surprising find of the month"

First case study: our own collection, 1,665 coins, 449 highlights, about €13,100.

Rules for credible numbers:

- Only **confirmed** coins count toward headline values.
- Show the melt floor separately from catalogue estimates.
- Record value at identification time; optionally also show at today's spot price.

## Privacy

- **Public data is aggregate only.** Individual coins only with explicit opt-in, never
  with location or user name. A list of who owns valuable coins is a burglary risk.
- **Strip EXIF / GPS** from every upload.
- GDPR: EU hosting, data export and deletion, clear consent text.
- Terms of use: consent to use photos and corrections as training data
  (for our own classifier / catalogue later, see Phase D in [README.md](README.md)).

## Internal analytics

- Where users drop out (upload, quality check, review, export, payment).
- Quality-check failure reasons → improve instructions ([02](02-capture-and-pairing.md)).
- Correction rate per country / era → where identification is weak.
- Cost per coin (LLM + catalogue requests) vs revenue per coin.
