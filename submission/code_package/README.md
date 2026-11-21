# COMP9444 Group Project — Code

Visible–infrared person re-identification on SYSU-MM01, with the Progressive Modality-shared Transformer (PMT) as the baseline. This archive holds the complete code behind the submitted notebook (`project.ipynb`). The notebook itself runs without this code; it is provided for reference and reproduction.

Not included: the SYSU-MM01 dataset, pretrained weights, model checkpoints and exported features.

Some code comments and command-line help strings are in Chinese, written during development.

## Layout

| Path | Contents |
| --- | --- |
| `src/data.py` | Dataset manifest, identity splits, identity-balanced P/K sampler, augmentation |
| `src/model.py`, `src/losses.py` | PMT model (ViT-B/16, overlapping patches, BN neck) and its losses (ID, PHT, MSEL, DCL) |
| `src/schedules.py` | Smooth grey-to-colour transition (S) |
| `src/local_head.py` | Local supervision head (L) |
| `src/agw.py`, `src/train_agw.py` | AGW and CAJ baselines ported from the authors' code, incl. the CAJ masked-max correction (`CAJ_fix`) |
| `src/training.py`, `src/train.py` | Training loop, resumable checkpoints, AMP; entry point for all PMT-based runs |
| `src/export_features.py` | Feature export for query / gallery |
| `src/evaluate.py` | SYSU-MM01 all-search evaluation with fixed gallery draws (mAP, mINP, CMC) |
| `src/robustness.py` | Retrieval robustness under occlusion / blur applied to IR queries |
| `src/config.py`, `configs/` | `configs/base.json` merged with one `configs/experiments/<ID>.json` per run |
| `scripts/` | Data and pretrained-weight preparation, reference checks against the authors' code, Katana (PBS) job scripts, fusion and natural-occlusion evaluation |
| `tests/` | Unit tests on synthetic data (no dataset needed) |
| `ATTRIBUTION.md`, `third_party_licenses/` | Sources, pinned upstream commits and our changes |

## Experiment names

Each model in the notebook corresponds to one file in `configs/experiments/` (the `--experiment` argument):

| Model | `--experiment` |
| --- | --- |
| V0 (ViT without PMT's grey stage, MSEL and DCL) | `V0` |
| M0 (PMT baseline) | `M0` |
| AGW / CAJ / CAJfix | `AGW`, `CAJ`, `CAJ_fix` (via `src.train_agw`) |
| M1 (M0 + S) / F (fixed-probability control) | `M1`, `F` |
| M2 (L, q = 2) / M2\* (L, q = 3) / M3 (S + L) | `M2`, `L3_B050`, `M3` |
| Exploratory L variants | `L2_B025`, `L3_B025` |
| Sampler control (P = 16, K = 2) | `Q` |
| Band erasing / channel augmentation | `M0_BE`, `L3_B050_BE`, `M0_CA`, `L3_B050_CA` |
| Final stage (395 identities, test split) | `M0_final`, `M0_CA_final` |

Print the merged configuration of a run with `python scripts/inspect_config.py --experiment M0`.

## Setup

Python 3.11. Training and evaluation were run on Linux with one NVIDIA GPU (L40S, about 0.8 h per PMT run).

```bash
pip install -r requirements-katana-lock.txt              # CUDA environment used for all reported runs
python scripts/prepare_pretrained.py --model vit         # ViT-B/16 ImageNet weights linked by the PMT authors
python scripts/prepare_pretrained.py --model resnet50    # ResNet-50 weights for AGW / CAJ
python scripts/prepare_data.py                           # locate or download SYSU-MM01, build the manifest and run data checks
```

`requirements-lock.txt` is the local (macOS, CPU/MPS) environment used for development and tests.

## Train, export and evaluate

```bash
python -m src.train --experiment M0 --run-id dev_M0_seed17 --seed 17 --device cuda
python -m src.export_features --checkpoint checkpoints/dev_M0_seed17/last.pt --split val --modality ir  --output artifacts/results/dev_M0_seed17/query_val.npz
python -m src.export_features --checkpoint checkpoints/dev_M0_seed17/last.pt --split val --modality rgb --output artifacts/results/dev_M0_seed17/gallery_pool_val.npz
python -m src.evaluate run --query artifacts/results/dev_M0_seed17/query_val.npz --gallery-pool artifacts/results/dev_M0_seed17/gallery_pool_val.npz --output-dir artifacts/results/dev_M0_seed17/eval_all
```

All reported runs use seeds 17, 42 and 2026. AGW and CAJ use `python -m src.train_agw --experiment AGW|CAJ|CAJ_fix` with the same arguments. Final-stage runs (`M0_final`, `M0_CA_final`) train on train + val and export the test split instead of val. Training checkpoints are resumable with `--resume`.

## Tests

```bash
python -m pytest tests -q
```

The tests use a synthetic SYSU-style directory (`tests/synthetic_sysu.py`) and check the sampler and split rules, losses against hand-computed values, stage boundaries, bit-exact resume, the evaluator (hand-computed AP / CMC, camera exclusion), and the S, L, BE and CA modules. `scripts/check_pmt_reference.py` and `scripts/check_eval_reference.py` compare our implementation numerically with the authors' code at the pinned commit (they need a local checkout of that code).
