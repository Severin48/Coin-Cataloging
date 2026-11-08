# Product plan: from coin pipeline to web and phone app

Status (2026-11-08): idea / planning. The personal pipeline works end to end
(see [docs/PLAN.md](../docs/PLAN.md)): 1,665 coins identified for $13.22 (Batch API),
449 highlights worth about €9,450, collection about €13,100.

**One-line pitch:** photograph a whole box of coins on a sheet of paper and get a
priced inventory (Excel, PDF, CSV) plus a list of what to sell.

## Files

| File | Content |
|---|---|
| [01-market.md](01-market.md) | Competitors, the gap we fill, target users, the Numista licence risk |
| [02-capture-and-pairing.md](02-capture-and-pairing.md) | How users take photos; capture sheet with markers; front/back pairing; quality checks |
| [03-architecture.md](03-architecture.md) | Web app first (PWA), backend, processing tiers, review UI, exports |
| [04-data-and-highlights.md](04-data-and-highlights.md) | What we track; public stats/highlights page; privacy |
| [05-monetization.md](05-monetization.md) | Unit costs, pricing tiers, extra revenue |
| [06-nice-to-haves.md](06-nice-to-haves.md) | Feature backlog beyond the MVP |

## Roadmap

### Phase A: Validate (2–3 weeks)

- [ ] Email contact@numista.com about commercial terms (see [01-market.md](01-market.md)). **Blocker for everything else.**
- [ ] Refactor the scripts into an importable package (`coinscan/`): functions that take
      images and return data, no hardcoded paths, no GPU needed.
- [ ] Design the printable capture sheet with ArUco markers ([02](02-capture-and-pairing.md)).
- [ ] Add marker detection, slot-based pairing and diameter measurement (mm).
- [ ] Run 3–5 friends' or family members' collections by hand. Measure identification
      accuracy and note every way users get the photos wrong.
- [ ] Test the top 3–4 competitor apps on the same coins (accuracy, bulk support, export).

### Phase B: Web MVP (4–6 weeks)

- [ ] Upload (sheet pairs) → processing job → review screen → exports (XLSX, CSV, PDF).
- [ ] Accounts, Stripe, free tier (25–30 coins).
- [ ] Concierge mode: hard cases go to a manual review queue (us, with the `identify-coins` skill).

### Phase C: Growth

- [ ] Guided camera capture in the browser (overlay, live quality checks).
- [ ] PDF report (insurance / estate), shareable read-only links.
- [ ] Public highlights page.
- [ ] eBay Partner Network links on "sell".

### Phase D: App and scale

- [ ] App store release (Capacitor wrap of the PWA, or React Native if the camera needs it).
- [ ] Dealer tier, sell assistant.
- [ ] Own classifier and own catalogue from confirmed identifications, to cut LLM cost
      and reduce dependence on Numista.

## Recommended first build step

The capture sheet with markers and diameter measurement. It fixes front/back pairing
and also improves identification (it fixes the 25 vs 50 Schilling confusion).
