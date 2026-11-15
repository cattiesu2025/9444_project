"""Standard-library configuration loader for the project scaffold."""

from copy import deepcopy
import json
from pathlib import Path

EXPERIMENTS = ("V0", "M0", "M1", "M2", "M3", "L3_B025", "L3_B050", "L2_B025", "L3_B050_BE", "M0_BE", "M0_CA", "L3_B050_CA", "F", "Q", "AGW", "CAJ", "CAJ_fix",
               "M0_final", "M0_CA_final")
FINAL_TRAIN_SPLITS = ["train", "val"]
FINAL_TRAIN_IDENTITIES = 395


def train_splits(config: dict) -> list[str]:
    """Identity splits used for training: development ["train"], final ["train", "val"]."""
    data = config["data"]
    return list(data.get("train_splits", [data.get("train_split", "train")]))

def merge(base: dict, override: dict) -> dict:
    """Recursively override dictionaries; replace lists and scalar values."""
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(experiment: str, config_dir: Path | None = None) -> dict:
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment}")
    directory = config_dir or Path(__file__).resolve().parents[1] / "configs"
    base = json.loads((directory / "base.json").read_text(encoding="utf-8"))
    override = json.loads(
        (directory / "experiments" / f"{experiment}.json").read_text(encoding="utf-8")
    )
    config = merge(base, override)
    validate(config)
    return config


def validate(config: dict) -> None:
    """Check basic design invariants, not algorithm or training readiness."""
    train = config["training"]
    if train["identities_per_batch"] < 2 or train["images_per_identity_per_modality"] < 2:
        raise ValueError("P and K must both be >= 2 for metric learning")
    if not 0 <= train["stage1_epochs"] < train["epochs"]:
        raise ValueError("Invalid stage boundary")
    smooth = config["smooth"]
    if smooth["mode"] not in ("off", "linear", "fixed_expected_exposure"):
        raise ValueError("Unknown transition mode")
    if not 0 <= smooth["initial_probability"] <= 1 or smooth["window_epochs"] <= 0:
        raise ValueError("Invalid grayscale schedule")
    if config["local"]["regions"] < 1 or config["local"]["weight"] < 0:
        raise ValueError("Invalid local supervision configuration")
    band = config.get("augmentation", {}).get("band_erasing", 0.0)
    if not 0 <= band <= 1:
        raise ValueError("band_erasing must be a probability in [0, 1]")
    exchange = config.get("augmentation", {}).get("channel_exchange", 0.0)
    if not 0 <= exchange <= 1:
        raise ValueError("channel_exchange must be a probability in [0, 1]")
    if config["model"]["retrieval_feature"] != "global_only":
        raise ValueError("Core comparisons must use global-only retrieval")
    stage = config.get("stage", "development")
    splits, evaluation = train_splits(config), config["data"]["evaluation_split"]
    if stage == "development":
        if splits != ["train"] or evaluation != "val":
            raise ValueError("Development configurations must train on train and evaluate on validation")
    elif stage == "final":
        # Final stage: retrain on train+val (395 identities), evaluate once on the held-out test split.
        if sorted(splits) != FINAL_TRAIN_SPLITS or evaluation != "test":
            raise ValueError("Final configurations must train on train+val and evaluate on test")
        if config["data"]["train_identities"] != FINAL_TRAIN_IDENTITIES:
            raise ValueError("Final configurations must declare 395 training identities")
    else:
        raise ValueError(f"Unknown stage: {stage}")
