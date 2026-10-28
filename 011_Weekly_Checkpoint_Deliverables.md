# Weekly Progress and Class Checkpoint (PMT Transformer)

**Last updated:** 2026-10-28, Australia/Sydney  
**Design document:** [Project Design](010_Project_Design.md)  
**Members:** A–E are member codes to be replaced with names. Unchecked items below are plans, not completed work.  
**Dates:** The official checkpoint time follows course announcements. The 10/25 data-interface and 10/28 results-summary dates are internal targets only; they neither replace nor postpone course deadlines.

**Class format:** The tutor asks about project progress orally; code is not inspected in class. The implementation and self-check lists below help the team move the full project forward and are not a code package that must be submitted in class. To prepare, each member explains what they have done, current results, problems met and next steps. Code always lives in the project directories by function.

## 1. Goals for This Week and Terminology

Course requirement: choose a baseline, implement it, train it on our data, evaluate it, and record and discuss initial results.

Our baseline is **PMT: a shared Transformer with the original paper's two-stage training and losses**. This week we first obtain initial results for one training seed; later improvements are not a prerequisite for completing the baseline.

- **Smooth grayscale transition (S, Smooth transition):** in stage 2, still use grayscale images with some probability and gradually reduce that probability to 0, instead of switching to RGB all at once. This is an improvement the group plans to test; owned by C.
- **Local feature supervision (L, Local supervision):** pool the Transformer's image patch features by horizontal region and add local identity supervision. This is an improvement the group plans to test; owned by D.
- S/L are factor codes in the design, not member names, and not the names of components in the original PMT paper.

**Division of work: each person owns one independent deliverable and implements, checks and documents it themselves.** B owns the full baseline; C and D each deliver an independent improvement module and do not verify B's baseline. E owns evaluation and the summary and does not write other members' technical write-ups. Each member maintains their own module and configs; nobody needs to edit the same training file at the same time.

## 2. What Each Member Delivers This Week

| Member | Owns | Deliverables this week | Order |
|---|---|---|---|
| A | Data preparation, splits, sampling and EDA | Data entry point, split/label notes, sampler, audit and EDA draft | Data interface first; EDA can follow |
| B | Implementation and training of the full PMT baseline | Environment, training/feature-export entry points, configs, training logs, model weights, method notes | Real training after A's data interface, then hand over to E |
| C | Smooth grayscale transition | Independent schedule module, probability examples, self-test results and method notes | In parallel with A/B; does not block the baseline |
| D | Local feature supervision | Independent local-head module, shape/gradient self-tests and method notes | In parallel with A/B; does not block the baseline |
| E | Retrieval evaluation and initial results summary | Evaluation code, metric self-tests, fixed galleries, baseline metrics, retrieval cases and summary page | Prepare the evaluator independently first; produce metrics once A's data and B's weights are available |

## 3. A: Data Preparation and EDA

**Location:** `src/data.py`, `data/`, `artifacts/manifests/`; EDA outputs go to `artifacts/results/`.

Required:

1. Data path and source notes; mapping rules for RGB/IR, identities and cameras.
2. `manifest.csv`, train/validation/test split files and check records: no identity overlap, all images readable, missing/abnormal files recorded.
3. Runnable data loading and an identity/modality-balanced sampling entry point; one command and a real batch example.
4. Each batch contains images, contiguous training labels, original identities, modality and camera; default `2PK=64` with P=8, K=4, with the configuration stated.
5. Initial EDA: camera/modality counts, per-identity sample distribution, RGB/IR examples of the same identity. Each figure states the observation and its implication for training.

The default development split is 296 training identities and 99 validation identities, with 96 test identities held out. Images must not be split at random in a way that mixes the same identity across splits.

**Done when:** the data interface can be read directly and check records are complete; the interface does not wait for all EDA plots. Sampling-control training is not required this week.

## 4. B: Full PMT Baseline

**Location:** `src/model.py`, `src/losses.py`, `src/train.py`, `configs/`; weights go to `checkpoints/`, logs to `artifacts/results/`.

Required:

1. Environment notes: author code URL/commit, dependencies, device, pretraining source and actual modifications. Record differences between the paper's MindSpore and the public PyTorch implementation.
2. Full PMT model and training entry point, including the shared ViT, two-stage input, PHT/MSEL/DCL, save and resume; B verifies the original method and stage boundary.
3. Config files: input, patch/stride, sampling, augmentation, losses, optimiser, seeds and training length. The paper's SYSU setting (24 epochs, first 6 epochs as stage 1) is the starting point for checking and must not be claimed identical without verification.
4. Training logs on our real data and checkpoint path/checksum. Logs include each loss, stage, epoch/global step, learning rate, time and peak GPU memory.
5. Model loading and feature-export entry point: features with original identities, cameras, modality and image paths for E's evaluation; state feature dimension, normalisation and distance.
6. Short method and usage notes: one training command, one feature-export command, completed training progress and known issues.

**Done when:** a model has actually been trained with loadable weights saved, and input/loss/resume self-tests pass. The baseline does not enable smooth transition or local supervision; B does not wait for the C/D modules.

Describe the full method and full training separately: results from partial epochs are "interim results"; the authors' pretrained model may only be used for verification and does not replace our own training. If only a plain ViT was actually completed, it must be labelled a plain ViT, not full PMT.

## 5. C: Smooth Grayscale Transition Module

**Location:** `src/schedules.py`, `configs/experiments/M1.json`, `configs/experiments/F.json`.

Required:

1. `src/schedules.py`: an independent probability schedule taking the stage-2 step, transition window and initial probability, and returning the current grayscale probability; provide an off switch and the state needed for resume.
2. Standalone example: print or plot the probability sequences for an immediate switch, a smooth transition and the fixed-probability control; explain how the fixed control matches the expected grayscale exposure.
3. Self-check records: probability range, window start/end, zero after the window, continuity across resume, and no change to the original path when switched off. Short examples may use simulated steps and must not be presented as training results.
4. Method notes: hypothesis, formula, default parameters, expected inputs/outputs, how it plugs into stage 2, and follow-up control experiments.

**Done when:** the module can run and be verified without depending on B's running training jobs. Grayscale conversion applies to visible-light sources only; IR is unchanged; PMT's loss-switching rules must not be changed at the same time.

**Not required this week:** training the improved model or showing an improvement. After B delivers the baseline, C builds their own improved config and training entry on B's fixed version; B is not expected to integrate it for C.

## 6. D: Local Feature Supervision Module

**Location:** `src/local_head.py`, `configs/experiments/M2.json`, `configs/experiments/M3.json`.

Required:

1. `local_head.py`: takes patch features with special tokens removed plus explicit H_p/W_p, and pools them by horizontal region; the first version uses q=2 with a local identity classifier per region.
2. Standalone example: feed simulated patch features and output local features, classification logits and local cross-entropy; record tensor shapes and gradient checks.
3. Self-check records: token count matches H_p×W_p, special-token exclusion convention, non-square grid, finite gradients, no effect on the global output when switched off.
4. Method notes: global/local structure, loss and beta, misalignment risk, integration point and follow-up experiments. The real token grid must be read from the patch-embedding output, not guessed by taking a square root.

**Done when:** the module runs independently without waiting for full baseline training. Simulated features only verify the module interface and backpropagation, not effectiveness on real images. The first version still tests with the original global feature only; the local head is auxiliary training supervision.

**Not required this week:** training the improved model or showing better occlusion performance. After B delivers the baseline, D independently connects real tokens and builds their own improved training config; B is not expected to integrate it for D.

## 7. E: Evaluation and Initial Results

**Location:** `src/evaluate.py`, `src/robustness.py`, `artifacts/manifests/`, `artifacts/results/`.

Required:

1. Evaluation code and protocol notes: internal validation identities, IR→RGB, All-search/Single-shot, camera exclusion, fixed galleries, distance and handling of valid queries.
2. Metric self-checks: hand-compute Rank-1/AP on a manual distance matrix and cross-check against a reference implementation; check SYSU identity-level CMC and the rule that cam3 queries exclude cam2 gallery images.
3. Fixed indices for 10 galleries, seeds 0–9; evaluate using A's identity/camera information and B's feature-export entry point.
4. `metrics.json`/`metrics.csv`: Rank-1/mAP per trial, mean, run_id, split, epoch, number of valid queries and checkpoint checksum.
5. Top-5 retrieval cases from real rankings, stating how samples were selected; initial analysis and limitations.
6. In the progress record at the end of this document, summarise the baseline, training progress, metrics and evidence paths, and record the team discussion, problems and next steps for the oral class report.

**Done when:** metrics come from our own trained weights and raw records are traceable. Report the internal validation split by default and never tune on the test split. 10 gallery repetitions are not 10 independent training runs; if only one ad-hoc evaluation exists, state the number of trials honestly.

E checks evaluation correctness themselves; there is no approval step by other members. Problems with upstream data or weights are reported to their owner to fix. Everyone is responsible for their own deliverable.

## 8. Delivery Order

**Baseline main line: A data interface → B trained weights and feature-export entry point → E metrics, cases and summary.**

**Improvement modules: C smooth transition and D local supervision are prepared independently in parallel → after B delivers the baseline, each integrates and trains their own improvement.** This week only the modules and self-checks are required; improvement training comes later.

| Step | Delivery | Internal target |
|---|---|---|
| 1 | A delivers the data interface/splits; B prepares the original model; C/D prepare independent modules; E prepares the evaluator | All start 10/24; data interface target 10/25 |
| 2 | B trains on A's data and delivers evaluable weights, configs and export command; A adds EDA; C/D finish self-checks | Starts as soon as the data interface is usable |
| 3 | E produces metrics and retrieval cases from the weights; all five deliver their notes and E compiles the discussion record | Internal target 10/28, and no later than the official checkpoint |

B's training does not depend on C/D verifying anything first, and nobody has to wait for all of A's EDA before starting. B may do a trial run on a verified small batch first and completes data and input checks before the full training run.

The later combined model is integrated by D on B's fixed baseline version using C's delivered schedule module; D owns the combined config and training, and C does not need to edit the same file as D. E evaluates every version consistently; B continues to own the original baseline.

## 9. Final Delivery Checklist

- [ ] A: data/sampling interface, split checks and EDA draft.
- [x] B: trained baseline, configs, logs, weights and feature-export entry point.
- [x] C: standalone smooth-transition module, self-checks and notes.
- [ ] D: standalone local-supervision module, self-checks and notes.
- [x] E: evaluation code, self-checks, real metrics, cases and summary.
- [ ] Whole team: discuss initial results; write down problems, limitations and follow-up tasks.
- [ ] Summary page distinguishes completed training, interim results, module prototypes and unfinished items.

No separate directory or code package is created for the class checkpoint. The final project is still integrated into one notebook: [submission/project.ipynb](submission/project.ipynb); `checkpoints/` only stores model weights and resume state and is not uploaded to Git.

## 10. Progress Record This Week (for the class report)

**Current status:** Data interface ready; M0 baseline trained for 24 epochs with validation features exported. The evaluator passes hand-computed self-tests and matches the PMT reference implementation; M0 (and the already-trained M1 and F) seed17 now have retrieval metrics on the internal validation split.

| Member | Done | To do | Next step |
| :---: | --- | --- | --- |
| A | Data interface, sampler and real-data checks | EDA | Add data analysis and examples |
| B | M0 seed17 training, resume and feature export | This week's tasks complete | Deliver baseline weights and features to E (done) |
| C | Smooth grayscale schedule `src/schedules.py` (linear / fixed expected exposure / off) wired into training; M1 and F configs; schedule self-tests (boundaries, equal expected exposure, rebuild from config and step); method write-up; M1 and F seed17 trained for 24 epochs on Katana and evaluated by E | End-to-end checkpoint-resume check; more seeds | Seeds 42/2026; training curves around the stage switch |
| D | No commits yet: `src/local_head.py` is still a placeholder; only the scaffold M2/M3 configs exist | Local head, non-square token grid and gradient self-tests | Per the design: shape/gradient/small-sample checks by 10/29; if still not working by 10/31, keep the failure evidence and downgrade to exploratory work |
| E | Evaluator (10 fixed galleries, cam3→cam2 exclusion, identity-level CMC, mAP/mINP, cross-seed summary); 14 hand-computed self-tests; cross-check vs PMT `eval_sysu`/`process_gallery_sysu` passed; M0/M1/F seed17 evaluated; M0 Top-5 retrieval cases | `metrics.csv` export; robustness evaluation (`src/robustness.py`); multi-seed summary | Summarise mean±SD once seeds 42/2026 arrive; failure-case analysis |

### Evidence and Notes

- **A:** [Data interface notes](docs/data_interface.md), [data preparation entry point](scripts/prepare_data.py). Real-data audit and batch checks pass; local records in `artifacts/manifests/data_audit.md` and `artifacts/results/data_check/check_report.md`.
- **B:** [Baseline usage notes](docs/baseline.md), [integrated notebook](submission/project.ipynb). M0 seed17 completed 24 epochs on Katana L40S; 50 A+B tests and delivery validation pass; validation features exported for 1,980 IR and 1,974 RGB images.
- **B's delivery:** logs, configs and features in `artifacts/results/dev_M0_seed17_katana_20261025_retry1/`; weights in the matching `checkpoints/` directory. The AMP issue is fixed and the official job finished successfully. Retrieval metrics are reported by E below.
- **C:** [Smooth transition method and checks](docs/smooth_transition.md), [schedule](src/schedules.py), [schedule tests](tests/test_schedule.py) (3 passing); merged into main (PR #1, #2). M1/F runs: `dev_M1_seed17_katana_20261027_l40s`, `dev_F_seed17_katana_20261027_retry1`; delivery validation PASS (24 epochs, export checksums and row alignment).
- **D:** As of 2026-10-28 there is no D branch or commit on GitHub.
- **E:** [Evaluation protocol and usage](docs/evaluation.md), [evaluator](src/evaluate.py), [self-tests](tests/test_evaluate.py), [reference cross-check](scripts/check_eval_reference.py), [retrieval-case script](scripts/retrieval_cases.py). SHA256 of all 6 feature `.npz` files delivered by B match their export records; all 3,954 validation images match the training manifest SHA256. Cross-check report `artifacts/results/eval_checks/reference.json`; fixed gallery `artifacts/manifests/gallery_val_all_single.json` (local, not in Git).

### Initial results (compiled by E, 2026-10-28)

Protocol: internal validation split (99 identities, not used for training), IR→RGB, All-search, single-shot; one training seed (17); mean over 10 fixed galleries (seeds 0–9); metrics in %. All three models use the same galleries; all 1,980 queries are valid.

| Experiment | run_id | Training | mAP | mINP | Rank-1 | Rank-5 | Rank-10 | Rank-20 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **M0 (PMT baseline)** | dev_M0_seed17_katana_20261025_retry1 | 24 epochs | **34.4** | 24.2 | **32.1** | 60.3 | 72.6 | 83.9 |
| M1 (PMT + smooth transition) | dev_M1_seed17_katana_20261027_l40s | 24 epochs | 34.1 | 24.0 | 31.3 | 60.6 | 73.3 | 84.0 |
| F (fixed-probability control) | dev_F_seed17_katana_20261027_retry1 | 24 epochs | 34.4 | 24.2 | 31.7 | 61.2 | 74.0 | 84.5 |

- Variation across galleries (SD): mAP about 1.5–1.7, Rank-1 about 1.9–2.4.
- Paired per-gallery differences vs M0: M1 mAP −0.3 (−1.0 to +0.5), Rank-1 −0.7; F mAP 0.0 (−0.9 to +0.9), Rank-1 −0.4.
- Raw records: each run's `eval_all/metrics.json` (E's local copy, extracted from B's result packages; includes checkpoint and input SHA256).

**Interpretation and limitations:**

1. Our baseline is established: PMT M0 reaches mAP 34.4% and Rank-1 32.1% on the internal validation split, the reference point for later comparisons.
2. M1 and F differ from M0 by less than the gallery variation, with a single training seed, so **we cannot yet tell whether the smooth transition helps or hurts**. Seeds 42/2026 are needed, aggregated by averaging galleries first and then taking mean±SD across seeds.
3. The internal validation split has IR only from cam3 and RGB only from cam4/5 (no cam1/2/6): All-search is effectively cam3→cam4/5 and Indoor-search is impossible. Validation metrics are for internal comparison only and **cannot be compared directly with the paper's SYSU test results** (395 training identities, test split, full camera set).
4. Top-5 retrieval cases (4 queries sampled at random from the 371 where Rank-1 is wrong but the correct identity is at #2–3; `artifacts/results/M0_seed17/retrieval_cases/near/`) show that dark clothing can appear bright in infrared, so colour and brightness cues fail, and the model confuses people with similar hair and build.

**For team discussion (record conclusions here afterwards):**

- Scheduling of seeds 42/2026 (M0, M1, and whether F needs more seeds).
- Whether to state the limited camera coverage of the validation split separately in the report, or adjust the validation protocol.
- M2/M3 (local supervision) should reuse the same gallery file and evaluation command once integrated.

**Team discussion conclusions:** to be filled in after the actual discussion.
