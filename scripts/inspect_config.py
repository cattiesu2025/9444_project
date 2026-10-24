"""Inspect a development proposal without installing ML dependencies."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import EXPERIMENTS, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=EXPERIMENTS, default="M0")
    args = parser.parse_args()
    config = load_config(args.experiment)
    train = config["training"]
    config["derived"] = {
        "batch_size": 2 * train["identities_per_batch"] * train["images_per_identity_per_modality"]
    }
    print(json.dumps(config, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
