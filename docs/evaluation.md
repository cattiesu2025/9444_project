# Retrieval Evaluation (E)

Implementation: `src/evaluate.py` (numpy only, no torch/GPU needed). Inputs are the IR query and RGB candidate-pool `.npz` files exported by B with `src/export_features.py`. Training classification labels are never used.

## Protocol

| Item | Setting |
| --- | --- |
| Direction | IR query (cam3/6) → RGB gallery |
| Search mode | All-search: cam1/2/4/5 (primary); Indoor-search: cam1/2 (auxiliary, `--search indoor`) |
| Single-shot | One image per (identity, RGB camera); cameras with no images are skipped |
| Fixed galleries | Seeds 0–9 (`evaluation.gallery_seeds` in `configs/base.json`). Sampling follows PMT `process_gallery_sysu` exactly: `random.Random(seed)`, identities ascending × camera order, `choice` over candidates sorted by path |
| Distance | `-(query @ gallery.T)` on BN-neck, L2-normalised features; stable sort, so ties keep gallery order |
| Camera exclusion | cam3 queries drop all cam2 gallery images (same location), regardless of identity |
| CMC | SYSU identity-level: only the first occurrence of each identity is kept in the ranking; Rank-k = the first correct identity is within the top k distinct identities |
| mAP / mINP | Computed on the instance-level ranking after exclusion |
| Queries without a positive | Excluded from all metrics; counted in `num_invalid_queries` / `invalid_query_rows` |
| Aggregation | Per training seed, average over the 10 galleries first; then mean and sample SD (ddof=1) across training seeds. Gallery SD is reported separately and never treated as training replication |

Metrics are stored as fractions (0–1) and converted to percentages only for display.

## Usage

```bash
python -m src.evaluate run \
  --query artifacts/results/dev_M0_seed17_katana_20261025_retry1/query_val.npz \
  --gallery-pool artifacts/results/dev_M0_seed17_katana_20261025_retry1/gallery_pool_val.npz \
  --output-dir artifacts/results/dev_M0_seed17_katana_20261025_retry1/eval_all
```

- The first run writes the fixed gallery record `artifacts/manifests/gallery_val_all_single.json` (stored by image path, with a candidate-pool fingerprint). Every later model reuses it, and a different candidate pool is rejected, so all methods are compared on identical galleries.
- Writes `metrics.json`: run_id, split, training seed, completed epochs, SHA256 of checkpoint / inputs / gallery file, protocol, per-gallery CMC / mAP / mINP / Rank-1/5/10/20 and valid-query counts, the 10-gallery mean and gallery SD, and feature-norm range. Refuses to overwrite an existing file.
- Input checks: queries must all be IR from cameras 3/6 and the pool all RGB; matching feature dimensions, finite values, no duplicate paths; both exports must share checkpoint / split / run_id.

Aggregating across training seeds:

```bash
python -m src.evaluate summarize --metrics run17/eval_all/metrics.json run42/eval_all/metrics.json run2026/eval_all/metrics.json --output artifacts/results/summary_M0_all.json
```

Runs with different protocol, split or gallery fingerprint are rejected, as are duplicate training seeds.

Retrieval case figures (`scripts/retrieval_cases.py`) draw the real Top-k ranking for queries chosen by a recorded, seeded rule (`hit`: Rank-1 correct; `near`: Rank-1 wrong but correct at #2–3; `miss`: no correct match in Top-k), or for explicitly listed queries, which must be reported as purposive selection.

## Correctness checks

- `python -m pytest tests/test_evaluate.py -q`: hand-computed AP / mINP / CMC; identity-level vs instance-level CMC; cam3→cam2 exclusion (including wrong identities); exclusion and counting of queries without a positive; multi-query averaging; deterministic ties; gallery sampling identical to the PMT procedure and independent of row order; rejection of a changed candidate pool and of an empty gallery; rejection of mismatched inputs; sample SD; CLI creating and reusing the fixed gallery; no overwrite.
- `python scripts/check_eval_reference.py /path/to/PMT` (pinned commit `0b31da9`): compares CMC / mAP / mINP with upstream `eval_sysu` on 200 random SYSU-structured cases, and checks on a temporary SYSU directory that All/Indoor × 10 seeds produce exactly the same gallery images as `process_gallery_sysu`. Report written to `artifacts/results/eval_checks/reference.json`.

Known differences from upstream: upstream uses the default `np.argsort`, so tie order is undefined, while this implementation uses a stable sort; upstream fails on ragged CMC arrays when fewer distinct identities than max_rank remain, while this implementation pads to max_rank; upstream computes distances in float32, this implementation in float64.

**Cross-check results (2026-10-28, PASS):**

- Synthetic cases: over 200 cases, max difference CMC 3.0e-8, mAP 1.1e-16, mINP 0; all 20 galleries (All/Indoor × 10 seeds) identical to `process_gallery_sysu` image by image.
- Real features (M0/M1/F seed17 × 10 galleries), passed directly to upstream `eval_sysu`: M0 matches everywhere (max difference 2.7e-7). In the seed-8 gallery of M1/F, 4 queries have near-ties with distance gaps of about 1e-7; float32 vs float64 rounding swaps their order, moving one query across one CMC rank: that point differs by 1/1980 = 0.0005 and mAP by 5e-7. With float32 distances, as upstream uses, results match upstream (CMC 3e-8, mAP 0). The difference comes from floating-point rounding, not the protocol, and changes the reported 10-gallery means by at most 0.00005.

## Robustness under query perturbations (`src/robustness.py`)

Rules fixed in advance on the development split (design §7); only IR queries are perturbed and the RGB gallery features exported by B for the same checkpoint are reused unchanged.

| Item | Setting |
| --- | --- |
| Processing order | load → LANCZOS resize to model input (256×128) → perturbation → ToTensor → ImageNet Normalize (same as B's export) |
| Occlusion | Full-width horizontal band at top / middle / bottom covering 10% or 20% of the image (band height = round(ratio × height): 26 or 51 rows); filled with the ImageNet mean pixel (124, 116, 104), which normalises to about 0 and is the same fill used by RandomErasing during training |
| Blur | Gaussian blur, σ = 1 (kernel 7) and σ = 2 (kernel 13); kernel = 2⌈3σ⌉ + 1 |
| Conditions | clean + 6 occlusions + 2 blurs = 9, all deterministic (no random numbers) |
| Galleries | The same fixed gallery file as `src.evaluate`; metrics are means over the 10 galleries |
| Report | Per-condition mAP / mINP / Rank-k, and the absolute drop from clean in percentage points |

```bash
python -m src.robustness \
  --checkpoint checkpoints/dev_M0_seed17_katana_20261025_retry1/last.pt \
  --gallery-pool <run>/gallery_pool_val.npz --query <run>/query_val.npz \
  --manifest <run>/training_manifest.csv --root data/SYSU-MM01 \
  --output-dir artifacts/results/dev_M0_seed17_katana_20261025_retry1/robustness
```

Checks before running: checkpoint SHA256 must equal the gallery pool's `checkpoint_sha256`, and the manifest SHA256 must equal the one stored in the checkpoint. The clean condition is re-extracted and compared with B's exported queries (`export_consistency`, min cosine > 0.999), which confirms the model is loaded and preprocessed exactly as in the export. Output: `robustness.json` (refuses to overwrite) and `perturbation_examples.png` showing one query under every condition.

Self-tests (`tests/test_robustness.py`, 13 passing): band position and area for every region/ratio, fill value, pixels outside the band unchanged, blur deterministic and size-preserving with stronger blur lowering contrast, fixed condition list, end-to-end run on synthetic data with a tiny model (clean result identical to plain evaluation, gallery file untouched, no overwrite), perturbations change query features, rejection of a mismatched checkpoint or manifest, and strict loading of an M2/M3 checkpoint with a local head (the model is built with `src.train.build_model`, the same path as the exporter).

Runtime on an RTX 5070 Ti Laptop GPU: about 2 minutes per condition for the 1,980 validation queries (single loader worker), about 17 minutes for all 9.

**Results (2026-10-30, M2/M3 added 2026-11-04; internal val, All-search, seed 17, mean of 10 fixed galleries; mAP / Rank-1 in %, drop in mAP points vs each model's clean row in brackets):**

| Condition | M0 | M1 | F | M2 | M3 |
| --- | --- | --- | --- | --- | --- |
| clean | 34.4 / 32.0 | 34.1 / 31.4 | 34.4 / 31.7 | 33.8 / 31.5 | 33.4 / 30.6 |
| occlusion top 10% | 32.1 / 29.2 (−2.3) | 31.9 / 28.9 (−2.2) | 32.0 / 29.0 (−2.4) | 31.8 / 29.2 (−2.0) | 31.4 / 28.5 (−2.0) |
| occlusion top 20% | 28.8 / 25.2 (−5.6) | 28.4 / 24.8 (−5.7) | 28.7 / 25.0 (−5.7) | 28.9 / 25.7 (−4.9) | 28.5 / 25.4 (−4.8) |
| occlusion middle 10% | 33.5 / 30.6 (−0.9) | 33.2 / 29.8 (−1.0) | 33.5 / 30.2 (−0.9) | 33.3 / 30.6 (−0.5) | 32.4 / 29.3 (−1.0) |
| occlusion middle 20% | 32.5 / 29.6 (−1.9) | 32.1 / 28.6 (−2.0) | 32.5 / 29.0 (−1.9) | 32.0 / 28.9 (−1.8) | 31.3 / 28.1 (−2.0) |
| occlusion bottom 10% | 31.0 / 28.4 (−3.4) | 30.6 / 27.6 (−3.5) | 30.9 / 27.9 (−3.5) | 30.8 / 28.4 (−3.0) | 30.1 / 26.9 (−3.3) |
| occlusion bottom 20% | 26.2 / 22.1 (−8.2) | 25.8 / 21.5 (−8.3) | 26.1 / 21.8 (−8.3) | 25.9 / 22.0 (−7.8) | 25.4 / 21.4 (−7.9) |
| blur σ=1 | 34.1 / 31.7 (−0.3) | 33.8 / 30.8 (−0.4) | 34.0 / 30.9 (−0.4) | 33.5 / 30.9 (−0.2) | 33.1 / 30.2 (−0.3) |
| blur σ=2 | 33.1 / 30.4 (−1.2) | 32.8 / 29.4 (−1.3) | 32.9 / 29.5 (−1.5) | 32.5 / 29.6 (−1.3) | 32.1 / 29.1 (−1.2) |

Checkpoint SHA256 of all five `last.pt` files matched the exported features; clean re-extraction matched the exports (min cosine M0 0.9996, M1 0.9994, F 0.9998, M2 0.9998, M3 0.9993). The clean row comes from the re-extracted features, so it can differ from the first-results table by up to 0.1 point (GPU numerical noise).

Observations: bottom occlusion hurts most (20%: about −8 mAP points), then top (about −5.6); middle occlusion and blur cost at most about 2 points. M0, M1 and F degrade almost identically (differences ≤ 0.3 points, single seed), so S (M1) gives no robustness benefit over M0 or F under these perturbations.

**Local supervision (RQ2, M2/M3 seed17, 2026-11-04).** Paired over the same 10 galleries, M2's occlusion drop is smaller than M0's under every occlusion condition: top 20% by 0.7 mAP points (−4.9 vs −5.6; smaller on 10/10 galleries), bottom 10% by 0.4 (10/10), bottom 20% by 0.4 (9/10), top 10% by 0.3 (9/10), middle 10% by 0.4 (9/10), middle 20% by 0.2 (5/10). Under blur the drops are the same (±0.05), which fits L targeting missing body regions rather than image quality. However, M2 starts 0.6 mAP lower on clean queries (lower than M0 on 10/10 galleries), so under occlusion its absolute mAP only matches M0 (top 20%: +0.1; bottom 20%: −0.2) and never clearly exceeds it. M3 (S + L) shows the smaller top-occlusion drop (−0.75) but not the middle/bottom benefit, and is about 1 mAP point below M0 in absolute terms under most conditions. Conclusion for one seed: L trades a little clean accuracy for slightly more stable features under occlusion; the effect (≤ 0.7 points) is below the gallery SD and needs repeat seeds. Raw results: `artifacts/results/{M0,M1,F,M2,M3}_seed17/robustness/robustness.json` (local, not in Git); clean re-extraction matched the exports (min cosine 0.9998 M2, 0.9993 M3).

**Exploratory: local features at test time (`scripts/local_concat_eval.py`, 2026-11-04).** Not part of the main protocol: it changes the retrieval feature from 768 to 2,304 dimensions. Three variants were fixed before running: `global` (as exported); `global+parts` (global + two height-band means of the patch tokens, each L2-normalised, no BN; available for any PMT); `global+local` (M2 only: band features passed through the trained local-head BN necks). The three parts are equally weighted (concatenate, divide by √3); nothing was tuned.

Since 2026-11-07 the script reads the region count q from the checkpoint's local head (for M2-style variants with 3 regions, e.g. `L3_B025`); checkpoints without a local head use `--regions` (default 2). With q regions, the concatenation `[√w·g, √((1−w)/q)·l_1, …]` gives cosine similarity = w·(global) + (1−w)/q·Σ(region). `global+parts` and `global+local` use equal weights (w = 1/(q+1), identical to the formula above when q = 2); a fourth variant, `global+local_g50`, fixes w = 1/2 so runs with different q can be compared without the global share shrinking. Both weightings are fixed in advance and must not be chosen from the perturbed results. Self-tests: `tests/test_local_concat.py` (weighted-cosine identity, equivalence with the earlier 2-region formula, 3-region end-to-end run, `--regions` for models without a local head). The gallery is re-extracted with the same checkpoint (global part matches the export, min cosine ≥ 0.9995).

| Condition | M0 global | M0 global+parts | M2 global | M2 global+parts | M2 global+local |
| --- | --- | --- | --- | --- | --- |
| clean | 34.4 | 33.6 | 33.8 | 33.7 | 33.9 |
| occlusion top 20% | 28.8 | 28.8 | 28.9 | 29.4 | 29.1 |
| occlusion middle 20% | 32.4 | 31.5 | 32.0 | 31.5 | 32.4 |
| occlusion bottom 20% | 26.1 | 25.2 | 25.9 | 25.4 | 26.2 |
| blur σ=2 | 33.1 | 32.6 | 32.5 | 32.6 | 32.7 |

(mAP %, mean of the 10 fixed galleries; all 9 conditions in `artifacts/results/{M0,M2}_seed17/local_concat.json`.)

- Untrained band features hurt M0 under every condition except top 20% occlusion (clean −0.75 mAP, lower on 10/10 galleries), so simply adding local information at test time does not help.
- M2's trained local features add a small, consistent gain over M2 global under every condition (+0.1 to +0.4 mAP; middle 20% and bottom 20% occlusion better on 10/10 galleries), i.e. the local supervision does make the band features useful for retrieval.
- The gain is too small to close M2's clean gap: M2 global+local is still 0.5 mAP below M0 global on clean queries (lower on 9/10 galleries) and only matches M0 under occlusion (top 20%: +0.3; bottom 20%: +0.0). Single seed; not a reason to change the main protocol.

Artificial occlusion and blur are controlled stress tests, not the real occlusion distribution; they must not be used to tune methods. Natural occlusion examples are reported separately.

## Camera coverage of the internal validation split (important limitation)

In the internal validation split (99 identities), IR queries come only from cam3 (1,980 images) and RGB candidates only from cam4 (990) and cam5 (984); there are no cam1/2/6 images. Therefore:

- All-search on val is effectively cam3 → cam4/5, with 195 images per gallery; the cam3→cam2 exclusion has no effect on val (it is still covered by the tests).
- Indoor-search is impossible on val (no cam1/2 images); the evaluator raises an error.
- The camera mix differs from the test split, so validation metrics are for internal comparison only and cannot be compared directly with SYSU test results reported in the paper.

## First results (internal val, All-search, single training seed 17, mean of 10 fixed galleries)

| Experiment | run_id | mAP | mINP | Rank-1 | Rank-5 | Rank-10 | Rank-20 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M0 | dev_M0_seed17_katana_20261025_retry1 | 0.3438 | 0.2424 | 0.3207 | 0.6027 | 0.7264 | 0.8388 |
| M1 | dev_M1_seed17_katana_20261027_l40s | 0.3409 | 0.2401 | 0.3135 | 0.6056 | 0.7330 | 0.8401 |
| F | dev_F_seed17_katana_20261027_retry1 | 0.3439 | 0.2419 | 0.3167 | 0.6121 | 0.7396 | 0.8453 |
| M2 | dev_M2_seed17_katana_20261103 | 0.3378 | 0.2371 | 0.3152 | 0.5919 | 0.7171 | 0.8269 |
| M3 | dev_M3_seed17_katana_20261103 | 0.3335 | 0.2372 | 0.3063 | 0.5803 | 0.7060 | 0.8243 |

All three share the same gallery file (fingerprint `d799087d813f…`); all 1,980 queries are valid. Gallery SD: mAP about 0.015–0.017, Rank-1 about 0.019–0.024. Paired per-gallery differences vs M0: M1 mAP −0.003 (−0.010 to +0.005), Rank-1 −0.007; F mAP 0.000 (−0.009 to +0.009), Rank-1 −0.004. The differences are smaller than the gallery variation and come from a single training seed, so they do not show whether S helps or hurts. M2 and M3 (2026-11-04) are below M0 on every gallery: M2 mAP −0.006 (−0.012 to −0.001), Rank-1 −0.006 (10/10 galleries lower); M3 mAP −0.010 (−0.015 to −0.003), Rank-1 −0.014 (9/10). Consistent in sign but below 1 point and from one seed. Raw results are in each run's `eval_all/metrics.json` (local, not in Git).

## Three training seeds: M0 vs M1 (2026-11-08)

Same protocol and gallery file; all features exported by B on Katana (delivery validation PASS, feature `checkpoint_sha256` equals the checkpoint). As a cross-check, E also exported M0 seed42/2026 locally from the checkpoints: min cosine to B's features 0.9996, mAP within 0.01 points.

| Seed | M0 mAP | M1 mAP | M1 − M0 | M0 Rank-1 | M1 Rank-1 |
| --- | --- | --- | --- | --- | --- |
| 17 | 34.38 | 34.09 | −0.29 | 32.07 | 31.35 |
| 42 | 34.24 | 33.97 | −0.27 | 31.37 | 31.49 |
| 2026 | 35.10 | 34.99 | −0.11 | 33.18 | 32.74 |
| Mean ± SD | 34.57 ± 0.46 | 34.35 ± 0.56 | −0.22 | 32.21 ± 0.91 | 31.86 ± 0.76 |

M1 is below M0 in mAP for every training seed, but the mean difference (−0.22 mAP) is about half the seed SD. S does not improve retrieval; it is not shown to hurt either. The training-seed SD of M0 (0.46 mAP) is the yardstick for every single-seed comparison below.

## L exploration: regions q × loss weight β (seed 17, exploratory)

All four runs differ from M0 only in the local head (`local.regions`, `local.weight`); retrieval uses the 768-d global feature. Checkpoint SHA256, `.sha256` file and the features' `checkpoint_sha256` agree for every run; clean re-extraction consistency PASS (min cosine ≥ 0.9995).

Clean mAP (paired difference vs M0 seed17 over the 10 galleries, and galleries where the variant is higher):

| | β = 0.5 | β = 0.25 |
| --- | --- | --- |
| q = 2 | M2: 33.78 (−0.60, 0/10) | L2_B025: 34.67 (+0.29, 6/10) |
| q = 3 | **L3_B050: 34.92 (+0.54, 7/10)** | L3_B025: 34.42 (+0.04, 5/10) |

Rank-1: M2 31.52, L2_B025 32.82, L3_B050 32.18, L3_B025 31.94 (M0 32.07).

Absolute mAP under perturbation (drop vs clean in brackets):

| Condition | M0 | M2 | L2_B025 | L3_B050 | L3_B025 |
| --- | --- | --- | --- | --- | --- |
| clean | 34.4 | 33.8 | 34.7 | **34.9** | 34.4 |
| top 20% | 28.8 (−5.6) | 28.9 (−4.9) | 29.1 (−5.6) | 29.1 (−5.8) | 29.0 (−5.4) |
| middle 20% | 32.5 (−1.9) | 32.0 (−1.8) | 32.4 (−2.3) | **33.5 (−1.4)** | 32.6 (−1.8) |
| bottom 20% | 26.2 (−8.2) | 25.9 (−7.8) | 26.4 (−8.3) | 26.6 (−8.3) | 26.1 (−8.4) |
| blur σ=2 | 33.1 (−1.2) | 32.5 (−1.3) | 33.3 (−1.3) | 33.7 (−1.2) | 33.1 (−1.3) |

Paired difference in drop vs M0 (positive = drops less): M2 top 20% +0.69 (10/10), bottom 20% +0.39 (9/10); L3_B050 middle 20% +0.48 (8/10), top −0.23, bottom −0.09; L2_B025 and L3_B025 within ±0.35 everywhere.

Test-time concatenation (`scripts/local_concat_eval.py`, clean mAP, global → global+local): L3_B050 34.9 → 35.0, L2_B025 34.7 → 34.2, L3_B025 34.4 → 34.0, M2 33.8 → 33.9. Concatenation does not help consistently and stays outside the main protocol.

Reading:

- M2 (q=2, β=0.5) is the only configuration below M0 (seed 17 only; over three seeds M2 is level with M0, see "Baselines and early variants over three seeds"). Reducing β or using three regions removes the clean-accuracy cost; the two changes do not add up (L3_B025 is back at M0's level).
- L3_B050 is the highest on clean queries and in absolute mAP under every perturbation, but its +0.54 over M0 seed17 is about one M0 seed SD and inside M0's seed range (34.24–35.10).
- M2's smaller occlusion drop does not carry over to the other variants. Part of it is likely the lower starting point, so RQ2 has no consistent support from artificial occlusion.
- Basis for the freeze decision (validation only; the test split and natural-occlusion subset were not used): if the team adopts an L configuration for M2/M3 seeds 42/2026, L3_B050 is the candidate supported by these numbers. Any claim of improvement needs the three-seed comparison.

The team froze L3_B050. The three-seed comparison below supersedes the seed-17 reading above: the +0.54 does not hold.

## Frozen L3_B050 over three seeds (2026-11-14)

B trained L3_B050 seeds 42 and 2026 with the frozen configuration (archives match `SHA256SUMS`; checkpoint, `.sha256` file and the features' `checkpoint_sha256` agree; config identical to seed 17 apart from the seed). All comparisons are paired with M0 seeds 17/42/2026 on the same 10 galleries (30 seed–gallery pairs).

**Clean retrieval:** L3_B050 34.67 ± 0.34 mAP vs M0 34.57 ± 0.46 (per seed +0.54, +0.04, −0.29; mean +0.10; L higher on 15/30 pairs). Rank-1 31.96 ± 0.38 vs 32.21 ± 0.91 (−0.24). Training-time local supervision alone does not change clean accuracy.

**Robustness** (M0 seeds 42/2026 were run for this comparison; clean re-extraction consistency PASS for all six runs). Mean mAP over three seeds:

| Condition | M0 | L3_B050 | Difference | L higher (of 30) |
| --- | --- | --- | --- | --- |
| clean | 34.57 | 34.67 | +0.10 | 16 |
| top 20% | 28.92 | 28.79 | −0.13 | 14 |
| middle 20% | 32.55 | 32.88 | +0.33 | 18 |
| bottom 20% | 25.92 | 25.86 | −0.06 | 17 |
| middle 10% | 33.63 | 33.93 | +0.30 | 20 |
| blur σ=2 | 33.47 | 33.45 | −0.02 | 16 |

Only middle occlusion shows a small, mostly consistent advantage; top, bottom and blur are level. The seed-17 pattern (higher under every perturbation, +1.0 for middle) does not replicate.

**Test-time fusion of local features** (`scripts/local_concat_eval.py`, all 9 conditions, 3 regions for every model; M0 has no local head, so it uses untrained mean-pooled parts). Mean mAP over three seeds:

| Condition | M0 | M0 + untrained parts | L3_B050 | L3_B050 + local, equal | L3_B050 + local, global 50% |
| --- | --- | --- | --- | --- | --- |
| clean | 34.57 | 33.90 | 34.66 | 34.80 | **34.83** |
| top 10% | **32.37** | 32.21 | 32.14 | 32.12 | 32.19 |
| top 20% | 28.92 | **29.12** | 28.80 | 28.92 | 28.97 |
| middle 10% | 33.63 | 32.67 | 33.92 | 34.01 | **34.05** |
| middle 20% | 32.55 | 31.52 | 32.87 | 33.00 | **33.05** |
| bottom 10% | 31.07 | 30.38 | 30.91 | 31.09 | **31.10** |
| bottom 20% | 25.91 | 24.96 | 25.85 | 25.88 | **25.95** |
| blur σ=1 | 34.31 | 33.71 | 34.28 | **34.49** | **34.49** |
| blur σ=2 | 33.47 | 33.03 | 33.45 | **33.71** | **33.71** |

Paired differences (30 seed–gallery pairs; per-seed values in brackets):

| Condition | Fusion vs L3_B050 global | Fusion vs M0 global | M0 + parts vs M0 |
| --- | --- | --- | --- |
| clean | +0.17 (+0.12, +0.21, +0.17); 26/30 | +0.26 (+0.64, +0.26, −0.12); 19/30 | −0.67; 0/30 |
| top 10% | +0.06 (+0.13, +0.04, +0.01); 21/30 | −0.18 (+0.58, −0.32, −0.81); 13/30 | −0.17; 10/30 |
| top 20% | +0.17 (+0.20, +0.16, +0.16); 25/30 | +0.05 (+0.48, +0.37, −0.71); 16/30 | +0.20; 18/30 |
| middle 10% | +0.13 (+0.08, +0.22, +0.09); 23/30 | +0.42 (+1.00, +0.35, −0.09); 22/30 | −0.96; 0/30 |
| middle 20% | +0.18 (+0.19, +0.32, +0.02); 22/30 | +0.50 (+1.19, +0.17, +0.15); 21/30 | −1.03; 0/30 |
| bottom 10% | +0.19 (+0.22, +0.14, +0.20); 26/30 | +0.03 (+0.80, −0.15, −0.56); 16/30 | −0.68; 1/30 |
| bottom 20% | +0.10 (+0.26, −0.04, +0.08); 22/30 | +0.04 (+0.69, −0.09, −0.49); 17/30 | −0.95; 0/30 |
| blur σ=1 | +0.21 (+0.22, +0.22, +0.18); 28/30 | +0.18 (+0.59, +0.13, −0.18); 17/30 | −0.60; 2/30 |
| blur σ=2 | +0.26 (+0.31, +0.26, +0.20); 27/30 | +0.24 (+0.84, +0.09, −0.23); 20/30 | −0.44; 4/30 |

- Fusion improves L3_B050 under all 9 conditions, on 26 of 27 seed–condition pairs (the exception is −0.04), but by only 0.06–0.26 mAP.
- The same fusion on M0 hurts under 8 of 9 conditions (up to −1.0). Local features help only when trained with local identity losses.
- Against M0, L3_B050 + fusion gains on clean (+0.26), middle occlusion (+0.42/+0.50) and blur (+0.18/+0.24); top and bottom occlusion are level, and top 10% is slightly lower (−0.18). Only middle 20% is positive on all three seeds.
- Fusion makes Rank-1 no better on clean queries (31.93 vs 32.18 for M0).

Both fusion weightings were fixed before these runs; the 50% global weighting is reported as the main variant because it does not depend on q. Fusion changes the retrieval dimension (768 → 3,072), so it stays outside the main protocol and is reported as a separate variant.

Consistency: every run passes except the L3_B050 seed-42 gallery re-extraction, which reports CHECK (min cosine 0.9974). A re-run gave the identical value; a per-image check shows a single image (`cam5/0335/0005.jpg`) below 0.999, while all other 1,973 gallery images are at least 0.9997 (median 1.0). This is a numerical difference on one image and does not affect the conclusions.

Reading: training-time local supervision alone leaves clean accuracy unchanged. Combined with test-time fusion of its trained local features, it gives a small gain over L alone under every condition and a small gain over PMT on clean, middle-occluded and blurred queries, clearest for middle occlusion; top and bottom occlusion are level. All effects are below one mAP point.

## Band erasing: L3_B050_BE and M0_BE over three seeds (2026-11-15)

B trained both band-erasing experiments (PR #12) with seeds 17/42/2026. All eight archives match `SHA256SUMS`. For each of the six runs the checkpoint, its `.sha256` file and the `checkpoint_sha256` of both exported feature files agree, and the training manifest matches. Configs differ from L3_B050 / M0 only in `augmentation.band_erasing = 0.5` (M0_BE also sets `local.regions = 3`, which only places the bands; activation checkpointing differs from M0 seed 17 and affects memory only). The recorded `src/data.py` hash equals the file merged in PR #12, so the band-erasing code was used.

**Clean retrieval** (mean ± SD over three seeds):

| Model | mAP | Rank-1 |
| --- | --- | --- |
| M0 | 34.57 ± 0.46 | 32.21 ± 0.91 |
| M0_BE | 33.68 ± 0.28 | 32.09 ± 0.50 |
| L3_B050 | 34.67 ± 0.34 | 31.96 ± 0.38 |
| L3_B050_BE | 33.64 ± 0.59 | 31.92 ± 0.28 |

Band erasing lowers mAP on every seed: L3_B050_BE − L3_B050 = −1.03 (−1.82, −0.73, −0.54; higher on 6/30 pairs); M0_BE − M0 = −0.89 (−0.38, −0.71, −1.59; 5/30). Rank-1 is unchanged, so the loss is in lower-ranked matches.

**Robustness** (all 9 conditions, mean mAP over three seeds; paired difference and pairs where the band-erasing model is higher):

| Condition | M0 | M0_BE | L3_B050 | L3_B050_BE | L3_B050_BE − L3_B050 | M0_BE − M0 |
| --- | --- | --- | --- | --- | --- | --- |
| clean | 34.57 | 33.68 | 34.67 | 33.65 | −1.02; 6/30 | −0.89; 6/30 |
| top 10% | 32.37 | 31.82 | 32.14 | 31.51 | −0.63; 12/30 | −0.55; 7/30 |
| top 20% | 28.92 | 28.51 | 28.79 | 28.31 | −0.48; 11/30 | −0.41; 10/30 |
| middle 10% | 33.63 | 33.10 | 33.93 | 32.98 | −0.95; 5/30 | −0.53; 12/30 |
| middle 20% | 32.55 | 32.25 | 32.88 | 32.08 | −0.80; 7/30 | −0.29; 13/30 |
| bottom 10% | 31.07 | 30.16 | 30.91 | 30.12 | −0.79; 6/30 | −0.91; 2/30 |
| bottom 20% | 25.92 | 26.09 | 25.86 | 26.07 | +0.21; 20/30 | +0.17; 21/30 |
| blur σ=1 | 34.31 | 33.33 | 34.30 | 33.38 | −0.92; 5/30 | −0.99; 4/30 |
| blur σ=2 | 33.47 | 32.47 | 33.45 | 32.64 | −0.80; 7/30 | −1.00; 5/30 |

Band erasing is worse under 8 of 9 conditions for both models. Only bottom 20% improves slightly (+0.2), even though the artificial bands resemble the training augmentation and so favour it.

**Fusion** (3 regions, global 50%): fusion still helps L3_B050_BE slightly under every condition (+0.05 to +0.15; 15–23/30), but L3_B050_BE + fusion is below L3_B050 + fusion under 8 of 9 conditions (clean −1.13; 3/30), the exception again being bottom 20% (+0.26). Untrained parts hurt M0_BE under all 9 conditions (−0.49 to −1.13).

Consistency: robustness re-extraction reports CHECK for L3_B050_BE seed 17 (queries, min cosine 0.9979) and M0_BE seed 42 (min 0.9987 queries, 0.9980 gallery). A per-image check finds one image below 0.999 in each affected set (`cam3/0426/0020.jpg`; `cam3/0404/0014.jpg`, `cam5/0429/0008.jpg`); medians are ≥ 0.99995. All other runs PASS.

Reading: band erasing aligned with the local regions does not help. It costs about one mAP point on clean queries and under almost every perturbation, with or without local supervision, and with or without fusion. Erasing a third of the body in half of the training images is likely too strong on top of the authors' random erasing. It is reported as a negative result; L3_B050 (with optional fusion) remains the L configuration.

## Baselines and early variants over three seeds: V0, AGW, CAJ_fix, M2, M3 (2026-11-16)

B trained seeds 42 and 2026 for the five rows that had only seed 17 (archive `extra_seeds_20261116.zip`: 10 runs, features, logs and configs, **no checkpoints**). Checks: all ten runs report `status=completed` (CAJ_fix hit the 2 h walltime once and was resumed to 100 epochs); each feature file's `checkpoint_sha256` equals the run summary's checkpoint hash; every config equals the resolved repository config (`src.config.load_config`), and the two seeds differ only in `training.seed`; git commit, manifest and pretrained hashes match the earlier Katana runs. Against the seed-17 runs, M2/M3 differ only in `model.activation_checkpointing` (memory setting, same computation). E's evaluator reproduces B's `eval_all` and three-seed summaries to two decimals on all ten runs (gallery fingerprint `d799087d813f…`). Seed 17 of V0/AGW/CAJ_fix uses the team lead's earlier result (seed means only, so their paired counts cover 20 seed×gallery pairs).

| Configuration | seed17 | seed42 | seed2026 | mAP mean ± SD | Rank-1 | mINP |
| --- | --- | --- | --- | --- | --- | --- |
| V0 (ViT, no grey stage / MSEL / DCL) | 33.03 | 32.75 | 31.86 | 32.55 ± 0.61 | 28.09 | 24.24 |
| AGW (ResNet-50) | 30.24 | 30.57 | 31.21 | 30.67 ± 0.50 | 26.91 | 22.27 |
| CAJ_fix (ResNet-50) | 34.38 | 33.63 | 33.21 | 33.74 ± 0.59 | 32.23 | 23.03 |
| M0 (PMT) | 34.38 | 34.24 | 35.10 | 34.57 ± 0.46 | 32.21 | 24.36 |
| M1 (S) | 34.09 | 33.97 | 34.99 | 34.35 ± 0.56 | 31.86 | 24.08 |
| M2 (L, q=2, β=0.5) | 33.78 | 35.10 | 34.75 | 34.54 ± 0.68 | 32.08 | 24.40 |
| M3 (S+L) | 33.35 | 34.70 | 34.92 | 34.32 ± 0.85 | 32.03 | 23.91 |
| L3_B050 | 34.92 | 34.29 | 34.81 | 34.67 ± 0.34 | 31.96 | 24.71 |

| Paired comparison | per-seed mAP difference | mean | Rank-1 | higher seed×gallery |
| --- | --- | --- | --- | --- |
| M0 − V0 | +1.35 / +1.49 / +3.24 | +2.03 | +4.11 | 20/20 |
| CAJ_fix − AGW | +4.15 / +3.07 / +2.00 | +3.07 | +5.32 | 19/20 |
| M2 − M0 | −0.60 / +0.86 / −0.35 | −0.03 | −0.12 | 12/30 |
| M3 − M0 | −1.03 / +0.46 / −0.18 | −0.25 | −0.17 | 13/30 |
| M3 − M1 | −0.74 / +0.72 / −0.07 | −0.03 | +0.17 | 15/30 |
| M3 − M2 | −0.43 / −0.40 / +0.17 | −0.22 | −0.05 | 10/30 |
| L3_B050 − M2 | +1.14 / −0.81 / +0.06 | +0.13 | −0.12 | 18/30 |

Reading:

- PMT's grey stage and MSEL/DCL help: M0 is above V0 on every seed and gallery (+2.0 mAP, +4.1 Rank-1).
- CAJ's channel augmentation helps the ResNet-50 baseline on every seed (+3.1 mAP, +5.3 Rank-1). The seed-17 gap of 4.1 that motivated M0_CA was at the high end; the three-seed gap is smaller but still clear. This matches the +1.45 that the same augmentation gives PMT (previous section is the controlled test).
- M2's seed-17 deficit (−0.60, 0/10 galleries) does not replicate: over three seeds M2 is level with M0. The seed-to-seed spread (SD 0.5–0.9) is larger than every L/S effect, so M2, M3, M1 and L3_B050 are all level with M0. The earlier seed-17 ranking of L variants (q × β exploration) should be read as noise, not as evidence that β=0.25 or q=3 fixes a cost of q=2.
- S and L do not add up: M3 (S+L) is level with M1 and slightly below M2.
- Robustness (9 perturbations) and clean re-extraction were not run for these ten runs, because both need checkpoints. Their robustness rows stay seed 17 only.

## Channel augmentation: M0_CA and L3_B050_CA (validation, 2026-11-16)

CAJ's channel augmentation (PR #16; RGB channel exchange with p=0.5, IR adaptive gray) was trained by B with seeds 17/42/2026 for M0_CA and 42/2026 for L3_B050_CA. The L3_B050_CA seed-17 archive is listed in `SHA256SUMS` but was not uploaded. The five delivered archives match `SHA256SUMS`; checkpoint, `.sha256` file and both features' `checkpoint_sha256` agree; the training manifest matches; configs differ from M0 / L3_B050 only in `augmentation.channel_exchange = 0.5`; the recorded `src/data.py` equals the file merged in PR #16. E's evaluation reproduces B's numbers in [channel_augmentation.md](channel_augmentation.md).

| Model | Seeds | mAP | Rank-1 | mINP |
| --- | --- | --- | --- | --- |
| M0 | 3 | 34.57 ± 0.46 | 32.21 ± 0.91 | 24.36 ± 0.38 |
| **M0_CA** | 3 | **36.02 ± 0.37** | **34.24 ± 0.67** | **25.19 ± 0.20** |
| L3_B050 | 3 | 34.67 ± 0.34 | 31.96 ± 0.38 | 24.71 ± 0.85 |
| L3_B050_CA | 2 | 35.29 ± 0.91 | 33.23 ± 1.29 | 24.64 ± 0.31 |

- M0_CA − M0: mAP +1.45 (+1.47, +2.20, +0.67; higher on 28/30 pairs), Rank-1 +2.04 (26/30). This meets the rule in [channel_augmentation.md](channel_augmentation.md), written before the results.
- L3_B050_CA − M0_CA: mAP −0.81 (−0.50, −1.12; 2/20 pairs). Local supervision does not add to channel augmentation.

**Robustness** (9 conditions, mean mAP over three seeds; all five runs PASS clean consistency):

| Condition | M0 | M0_CA | M0_CA − M0 (pairs higher) | Change in drop vs M0 |
| --- | --- | --- | --- | --- |
| clean | 34.57 | 36.01 | +1.44 (27/30) | — |
| top 10% | 32.37 | 33.50 | +1.12 (27/30) | −0.32 |
| top 20% | 28.92 | 29.57 | +0.65 (19/30) | −0.79 |
| middle 10% | 33.63 | 35.08 | +1.45 (26/30) | +0.01 |
| middle 20% | 32.55 | 33.90 | +1.36 (27/30) | −0.08 |
| bottom 10% | 31.07 | 31.74 | +0.67 (21/30) | −0.77 |
| bottom 20% | 25.92 | 26.55 | +0.63 (23/30) | −0.81 |
| blur σ=1 | 34.31 | 35.62 | +1.30 (28/30) | −0.14 |
| blur σ=2 | 33.47 | 34.58 | +1.11 (28/30) | −0.33 |

M0_CA is higher under every condition, with all three seeds positive in each, but drops about 0.8 more than M0 under top and bottom occlusion, so its advantage there halves (+0.65).

**Fusion:** untrained parts hurt M0_CA (clean −1.00, −0.7 to −1.2 under perturbations); fusion leaves L3_B050_CA unchanged (+0.04 to +0.11). The L3_B050_CA seed-2026 fusion gallery re-extraction reports CHECK (min cosine 0.9977); this optional control does not affect the conclusions.

## Final test results: M0_final vs M0_CA_final (2026-11-16)

Frozen M0 and M0_CA retrained by B on train+val (395 identities), seeds 17/42/2026, evaluated once on the official test split ([final_stage.md](final_stage.md)). All six archives match `SHA256SUMS`. For each run: checkpoint, `.sha256` file and both test features' `checkpoint_sha256` agree; `stage = final`, `train_splits = train+val`, `evaluation_split = test`; the label map has 395 identities, including all 99 validation identities and no test identity; 24 epochs completed; training manifest matches. Test split: 3,803 IR queries, 6,775 RGB candidates, 96 identities; 10 fixed galleries (seeds 0–9) shared by both models (`gallery_test_all_single.json`, `gallery_test_indoor_single.json`). Retrieval uses the global feature only.

| Protocol | Metric | M0 | M0_CA | M0_CA − M0 (per seed; pairs higher) |
| --- | --- | --- | --- | --- |
| All-search | mAP | 64.59 ± 1.18 | **66.19 ± 0.55** | +1.60 (+0.31, +1.13, +3.35; 28/30) |
| | Rank-1 | 66.26 ± 1.64 | **68.18 ± 0.48** | +1.93 (+0.52, +1.64, +3.62; 28/30) |
| | mINP | 51.84 ± 1.00 | **53.25 ± 0.69** | +1.41 (24/30) |
| | Rank-10 | 95.30 ± 0.61 | **96.06 ± 0.13** | +0.76 (24/30) |
| Indoor-search | mAP | 76.85 ± 0.37 | **78.39 ± 0.40** | +1.55 (+1.20, +1.10, +2.33; 30/30) |
| | Rank-1 | 71.52 ± 0.56 | **73.28 ± 0.70** | +1.76 (+1.20, +0.89, +3.20; 27/30) |
| | mINP | 73.16 ± 0.38 | **74.78 ± 0.30** | +1.61 (30/30) |
| | Rank-10 | 97.23 ± 0.28 | 97.42 ± 0.19 | +0.19 (18/30) |

- M0 reproduces the PMT paper's SYSU-MM01 level (All-search mAP about 65, Rank-1 about 67.5); our M0 gives 64.59 / 66.26.
- Channel augmentation improves PMT on the test set under both protocols, on every seed, by an amount consistent with validation (+1.45). It also halves the seed SD on All-search (0.55 vs 1.18).

**Natural-occlusion subset** (`scripts/natural_occlusion_eval.py`, A's 224 occluded test IR queries, annotation file committed at `5c7625e`; descriptive only). Mean over three seeds, All-search:

| Group | Queries | M0 mAP | M0_CA mAP | Difference | Rank-1 difference |
| --- | --- | --- | --- | --- | --- |
| all | 3,803 | 64.59 | 66.19 | +1.60 | +1.93 |
| natural occlusion | 224 | 60.19 | 61.86 | +1.67 (+0.36, +2.65, +2.01) | +2.95 |
| not occluded | 3,579 | 64.87 | 66.46 | +1.59 | +1.86 |
| cam 6, occluded | 219 | 60.21 | 61.99 | +1.79 | +2.98 |
| cam 6, not occluded | 1,701 | 61.93 | 63.82 | +1.89 | +2.46 |
| occluded, both halves | 158 | 61.76 | 63.67 | +1.91 | +2.91 |
| occluded, lower half | 65 | 56.03 | 57.15 | +1.12 | +3.03 |
| occluded by an object | 208 | 61.12 | 62.75 | +1.63 | +2.98 |

Occluded queries are about 4.7 mAP harder. The gain from channel augmentation is similar on occluded and non-occluded queries, so it is a general improvement rather than an occlusion-specific one. Groups with very few queries (cam 3 occluded: 5; upper half: 1; occluded by a person: 16) are not interpreted.

**Robustness on the test split** (9 conditions, All-search, mean mAP over three seeds):

| Condition | M0 | M0_CA | M0_CA − M0 (per seed; pairs higher) |
| --- | --- | --- | --- |
| clean | 64.58 | 66.18 | +1.60 (+0.31, +1.12, +3.36; 27/30) |
| top 10% | 60.75 | 62.05 | +1.30 (−0.17, +1.17, +2.90; 24/30) |
| top 20% | 55.58 | 56.49 | +0.91 (−0.41, +0.89, +2.24; 21/30) |
| middle 10% | 63.43 | 65.10 | +1.67 (29/30) |
| middle 20% | 61.70 | 63.39 | +1.69 (28/30) |
| bottom 10% | 59.46 | 61.03 | +1.57 (25/30) |
| bottom 20% | 53.12 | 54.64 | +1.51 (27/30) |
| blur σ=1 | 64.53 | 66.20 | +1.67 (27/30) |
| blur σ=2 | 63.32 | 64.94 | +1.62 (25/30) |

M0_CA is higher under all 9 conditions. As on validation, top occlusion shows the smallest gain (seed 17 slightly negative); on the test split bottom occlusion keeps most of the gain (+1.5). Consistency: M0 PASS for all seeds; M0_CA seeds 17 and 42 report CHECK on query re-extraction (min cosine 0.9988, 0.9986). A per-image check finds one image below 0.999 in each (`cam3/0176/0019.jpg`, `cam3/0080/0017.jpg`; medians 0.99995), a numerical difference that does not affect the results.

## Status

The evaluator, robustness tool and concatenation script are complete and self-tested. Evaluated on val: V0, AGW, CAJ_fix, M0, M1, M2, M3, L3_B050, L3_B050_BE, M0_BE and M0_CA seeds 17/42/2026 (V0/AGW/CAJ_fix seed 17 by the team lead), L3_B050_CA seeds 42/2026, F seed17, the three L exploration runs, and robustness and fusion over three seeds for M0, M1, L3_B050, the BE and CA runs (V0/AGW/CAJ_fix/M2/M3 seeds 42/2026 were delivered without checkpoints, so no robustness). Evaluated on test (once, after the freeze): M0_final and M0_CA_final seeds 17/42/2026, All- and Indoor-search, natural-occlusion subset and robustness. Remaining: error analysis and retrieval-case figures for the report.
