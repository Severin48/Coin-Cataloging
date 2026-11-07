---
name: tune-clustering
description: Improve or re-run the duplicate-coin clustering (clustering.py / check_clusters.py) — choosing eps, switching embedding models, handling swapped sides and rotation, reviewing cluster folders. Use when the user mentions clusters, groups, duplicates, eps, over/under-merging, or same coin with different denominations.
---

# Tune the clustering

## What clustering is for (read this before tuning)

Per `docs/PLAN.md`, the clusters are **not** the source of truth for "same coin type".
The LLM identification is, because it reads the numeral. Clusters are used for:

1. a consistency check: flag clusters whose members got different identities,
2. reviewing similar coins together (`check_clusters.py` folders, `crop_pair.py --group N`).

So a slightly **over-merging** setting is fine and even useful. Missing a duplicate
costs more (it gets looked up twice) than merging 25 and 50 Schilling (the LLM
separates them). Don't spend hours on `eps`.

## Current state

- `clustering.py`: ResNet50 (ImageNet) embedding of the **front crop only**, then
  `DBSCAN(eps=0.08, metric="cosine", min_samples=1)`.
- On the August run: 1,687 coins → 1,364 clusters, 1,226 of them singletons.
- Paths are hardcoded at the top of both scripts. Point them at the canonical run
  (`results/2025_11_16_15_40_15`) before re-running.

## Why the tradeoff feels unavoidable

1. **`min_samples=1` makes DBSCAN single-linkage clustering.** A and C get merged when a
   chain A–B–C exists, even if A and C differ. Raising `eps` therefore merges whole chains at once.
   → Use `AgglomerativeClustering(metric="cosine", linkage="average", distance_threshold=t, n_clusters=None)`.
   It degrades more gracefully.
2. **Only the front is embedded, but front/back is not reliably obverse/reverse.**
   Two identical coins scanned with opposite sides up never match.
   That probably explains many of the singletons.
   → Embed both sides and use a swap-invariant pair distance:
   `d(A,B) = min( (d(Af,Bf)+d(Ab,Bb))/2 , (d(Af,Bb)+d(Ab,Bf))/2 )`.
   Pass it to the clusterer as a precomputed matrix (`metric="precomputed"`).
3. **Rotation.** Coins lie at random angles, and ImageNet CNNs aren't rotation-invariant.
   → Average the embedding over 4–8 rotations of a *circular-masked* crop
   (mask the corners so background and capsule edges don't dominate).
4. **Model choice.** ImageNet classification features capture "silver round thing" more than
   design details. Self-supervised features retrieve near-identical instances much better.
   → Try DINOv2 ViT-B/14 via `torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")`,
   using the CLS token, L2-normalised. It fits easily on the 5070 Ti.

## How to evaluate a setting (10 minutes, no guessing)

1. Hand-label about 40 coin pairs as same/different. Include known hard cases (same design,
   different denomination; same coin, swapped sides). Save them as
   `results/<run>/cluster_pairs_gt.csv` (`coin_a,coin_b,same`).
2. For each candidate (model × distance × threshold), print the precision and recall of
   "same cluster" on those pairs, the number of clusters, and the share of singletons.
3. Pick the threshold with recall ≥ 0.95 that keeps precision acceptable. Only identity mixes
   *within one denomination* count as real errors.

## Reviewing

- `uv run check_clusters.py` writes `cluster_review_output/group_<id>/` (clusters with
  more than one coin) and a histogram.
- `uv run python .claude/skills/identify-coins/crop_pair.py --group <id>` renders both sides of every member.
- Cache the embeddings (`embeddings.npy` + coin_id order) in the run folder, so that
  trying thresholds doesn't re-run the model.
