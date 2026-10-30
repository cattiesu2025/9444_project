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

Self-tests (`tests/test_robustness.py`, 12 passing): band position and area for every region/ratio, fill value, pixels outside the band unchanged, blur deterministic and size-preserving with stronger blur lowering contrast, fixed condition list, end-to-end run on synthetic data with a tiny model (clean result identical to plain evaluation, gallery file untouched, no overwrite), perturbations change query features, and rejection of a mismatched checkpoint or manifest.

Runtime on an RTX 5070 Ti Laptop GPU: about 2 minutes per condition for the 1,980 validation queries (single loader worker), about 17 minutes for all 9.

**Results (2026-10-30, internal val, All-search, seed 17, mean of 10 fixed galleries; mAP / Rank-1 in %, drop in mAP points vs each model's clean row in brackets):**

| Condition | M0 | M1 | F |
| --- | --- | --- | --- |
| clean | 34.4 / 32.0 | 34.1 / 31.4 | 34.4 / 31.7 |
| occlusion top 10% | 32.1 / 29.2 (−2.3) | 31.9 / 28.9 (−2.2) | 32.0 / 29.0 (−2.4) |
| occlusion top 20% | 28.8 / 25.2 (−5.6) | 28.4 / 24.8 (−5.7) | 28.7 / 25.0 (−5.7) |
| occlusion middle 10% | 33.5 / 30.6 (−0.9) | 33.2 / 29.8 (−1.0) | 33.5 / 30.2 (−0.9) |
| occlusion middle 20% | 32.5 / 29.6 (−1.9) | 32.1 / 28.6 (−2.0) | 32.5 / 29.0 (−1.9) |
| occlusion bottom 10% | 31.0 / 28.4 (−3.4) | 30.6 / 27.6 (−3.5) | 30.9 / 27.9 (−3.5) |
| occlusion bottom 20% | 26.2 / 22.1 (−8.2) | 25.8 / 21.5 (−8.3) | 26.1 / 21.8 (−8.3) |
| blur σ=1 | 34.1 / 31.7 (−0.3) | 33.8 / 30.8 (−0.4) | 34.0 / 30.9 (−0.4) |
| blur σ=2 | 33.1 / 30.4 (−1.2) | 32.8 / 29.4 (−1.3) | 32.9 / 29.5 (−1.5) |

Checkpoint SHA256 of all three `last.pt` files matched the exported features; clean re-extraction matched B's export (min cosine 0.9996 / 0.9994 / 0.9998). The clean row comes from the re-extracted features, so it can differ from the first-results table by up to 0.1 point (GPU numerical noise).

Observations: bottom occlusion hurts most (20%: about −8 mAP points), then top (about −5.6); middle occlusion and blur cost at most about 2 points. The three models degrade almost identically (differences ≤ 0.3 points, single seed), so S (M1) gives no robustness benefit over M0 or F under these perturbations. The M0 rows are the baseline for testing whether local supervision (M2/M3, D) reduces the occlusion drop (RQ2). Raw results: `artifacts/results/{M0,M1,F}_seed17/robustness/robustness.json` (local, not in Git).

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

All three share the same gallery file (fingerprint `d799087d813f…`); all 1,980 queries are valid. Gallery SD: mAP about 0.015–0.017, Rank-1 about 0.019–0.024. Paired per-gallery differences vs M0: M1 mAP −0.003 (−0.010 to +0.005), Rank-1 −0.007; F mAP 0.000 (−0.009 to +0.009), Rank-1 −0.004. The differences are smaller than the gallery variation and come from a single training seed, so they do not show whether S helps or hurts. Raw results are in each run's `eval_all/metrics.json` (local, not in Git).

## Status

The evaluator and its self-tests are complete and match the pinned PMT reference implementation; M0/M1/F seed17 have been re-evaluated on the internal validation split. The robustness tool is implemented, self-tested and has been run on the M0/M1/F seed17 checkpoints; M2/M3 will be run with the same conditions once D's checkpoints exist.
