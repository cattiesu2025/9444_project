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

- M2 (q=2, β=0.5) is the only configuration below M0. Reducing β or using three regions removes the clean-accuracy cost; the two changes do not add up (L3_B025 is back at M0's level).
- L3_B050 is the highest on clean queries and in absolute mAP under every perturbation, but its +0.54 over M0 seed17 is about one M0 seed SD and inside M0's seed range (34.24–35.10).
- M2's smaller occlusion drop does not carry over to the other variants. Part of it is likely the lower starting point, so RQ2 has no consistent support from artificial occlusion.
- Basis for the freeze decision (validation only; the test split and natural-occlusion subset were not used): if the team adopts an L configuration for M2/M3 seeds 42/2026, L3_B050 is the candidate supported by these numbers. Any claim of improvement needs the three-seed comparison.

## Status

The evaluator, robustness tool and concatenation script are complete and self-tested. Evaluated on val: V0/AGW/CAJ_fix (team lead), M0 and M1 seeds 17/42/2026, F/M2/M3 seed17, and the three L exploration runs, including robustness and concatenation for every L run. Remaining: three-seed evaluation of the frozen configuration, final test-split evaluation, natural-occlusion subset, error analysis.
