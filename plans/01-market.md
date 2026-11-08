# Market and competitors

Researched 2026-11-08 from app store listings and web search. Not verified by hands-on
testing yet; see the Phase A task in [README.md](README.md).

## Competitors

| Group | Examples | What they do | Price |
|---|---|---|---|
| Single-coin AI apps (crowded) | CoinSnap, CoinScan, Coin ID, Coin Identifier & Value, Coinoscope and many clones | One photo of one coin → ID and value. Some track total collection value; one or two export to PDF/Excel | Mostly subscriptions, ~$30/year or ~$50 lifetime; aggressive trials |
| Catalogues | Numista | The hobby's catalogue and collection manager. Image search only through the paid API | Free site; API image search paid |
| US / graded | PCGS CoinFacts | US coins, barcode lookup for slabbed coins | Free or cheap; earns from grading |

Common complaint in reviews: wrong identifications. Single-coin apps disclaim that they
should not be used for buying, selling or insurance.

## The gap

No app found that identifies **many coins from one photo** and turns a box of coins
into a **priced inventory with a sell list**. All listings describe one coin at a time.

Our differentiators:

- Bulk: 20–30 coins per sheet, thousands per collection.
- Proof: every value has a Numista link and an eBay sold-listings link.
- Metal floor: melt value from weight × fineness × spot price.
- Spreadsheet-first output (Excel, CSV, PDF), not locked into an app.
- Honest one-time pricing instead of subscription traps.

## Target users

1. **Inheritors**: "grandpa's coin box". One-off need, low knowledge, want to know what is valuable and how to sell. Largest group.
2. **Estate clearers / house clearance**: recurring, need fast valuation and lots.
3. **Small dealers**: intake of customer collections, offers based on melt value.
4. **Collectors**: catalogue a backlog, track value, find gaps.

## Risk: Numista licence (blocker)

From the [API licence](https://fr.numista.com/api/license.php) and [pricing page](https://en.numista.com/api/pricing.php):

- Commercial use is allowed, with attribution ("Source: Numista").
- **Caching is limited**: metadata at most 7 days; other catalogue data may not be stored.
  Our personal pipeline caches forever. Fine for personal use, not for the product.
- Redistributing the data as a database / API / bulk download is prohibited.
- Free plan: 2,000 requests/month. That is about one user's collection. Not enough.
- Paid image search: €100 activation, €100/month minimum, €0.03 per request.
- Pricing pages in different languages disagree. A custom plan exists on request.

**Action:** email contact@numista.com with the use case and expected volume before building.

**Plan B** if terms don't work:

- Own catalogue built from confirmed identifications (our own data: crops, LLM output, user confirmations).
- Values from metal spot price (melt floor) plus eBay sold-listing links.
- Numista only as a link-out ("look it up on Numista").

## Sources

- [Numista API pricing](https://en.numista.com/api/pricing.php)
- [Numista API licence](https://fr.numista.com/api/license.php)
- [Coin Identifier Value Finder](https://mwm.ai/apps/coin-identifier-value-finder/6757428708)
- [Coin Identifier & Value App](https://apps.appfollow.io/ios/coin-identifier-value-app/6462873087?country=pg)
- [Coin Value Identifier](https://apps.apple.com/app/id6746510305)
- [CoinScan](https://apps.apple.com/app/id6448908389)
- [Coin Scan Identifier](https://apps.apple.com/si/app/coin-scan-identifier-checker/id6463596132)
- [CoinKnow vs PCGS CoinFacts](https://www.tribuneindia.com/partner-exclusives/coinknow-vs-pcgs-coinfacts-a-comparison-of-coin-identifier-apps/)
- [Yulys: best coin apps](https://yulys.com/blog/free-coin-identifier-and-value-apps)
