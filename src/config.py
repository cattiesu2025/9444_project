"""Standard-library configuration loader for the project scaffold."""

from copy import deepcopy
import json
from pathlib import Path

EXPERIMENTS = ("V0", "M0", "M1", "M2", "M3", "F", "Q", "AGW", "CAJ")


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
    if config["model"]["retrieval_feature"] != "global_only":
        raise ValueError("Core comparisons must use global-only retrieval")
    if config["data"]["evaluation_split"] != "val":
        raise ValueError("Development configurations must evaluate on validation")
