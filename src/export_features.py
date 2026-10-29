"""Export BN-neck L2-normalized features and row-aligned metadata for E."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import get_eval_dataset, collate_eval
from .model import PMT
from .agw import AGW, BACKBONE as AGW_BACKBONE
from .training import load_checkpoint, sha256
from .train import choose_device, PROJECT


@torch.inference_mode()
def export_dataset(model, dataset, output, metadata, batch_size=128, num_workers=0, device="cpu"):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_eval)
    collected = {key: [] for key in ("features", "identities", "cameras", "modalities", "row_indices", "paths")}
    for batch in loader:
        images = batch["images"].to(device)
        result = model(images, batch["modalities"].to(device)) if getattr(model, "modality_specific", False) else model(images)
        collected["features"].append(result["retrieval_feature"].cpu().numpy())
        for key in ("identities", "cameras", "modalities", "row_indices"):
            collected[key].append(batch[key].numpy())
        collected["paths"].extend(batch["paths"])
    if not collected["features"]:
        raise ValueError("No evaluation images matched the requested split/modality")
    arrays = {k: np.asarray(v) if k == "paths" else np.concatenate(v) for k, v in collected.items()}
    if not np.isfinite(arrays["features"]).all():
        raise FloatingPointError("Nonfinite retrieval features")
    metadata = dict(metadata, count=len(arrays["features"]), feature_dim=arrays["features"].shape[1],
                    feature_location="after_bn", normalization="l2", distance="negative_cosine_similarity",
                    gallery_selection="candidate_pool_only; E selects fixed single-shot trials", horizontal_flip=False)
    output = Path(output)
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(f"Export already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays, metadata_json=np.asarray(json.dumps(metadata)))
    metadata["npz_sha256"] = sha256(output)
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, help="Output .npz file (must not already exist)")
    parser.add_argument("--root")
    parser.add_argument("--split", choices=("val",), default="val")
    parser.add_argument("--modality", choices=("rgb", "ir"), required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    if Path(args.output).suffix != ".npz":
        parser.error("output must end in .npz")
    checkpoint = load_checkpoint(args.checkpoint)
    cfg = checkpoint["config"]
    device = choose_device(args.device)
    network = AGW if cfg["model"]["backbone"] == AGW_BACKBONE else PMT
    model = network(checkpoint["num_classes"], cfg["model"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    dataset = get_eval_dataset(args.split, modality=args.modality, root=args.root, download=False,
                               image_size=tuple(cfg["model"]["image_size"]))
    manifest_hash = sha256(PROJECT / cfg["data"]["manifest"])
    if manifest_hash != checkpoint["manifest_sha256"]:
        raise ValueError("Evaluation manifest changed since training; verify data provenance before exporting")
    metadata = {"evaluation_manifest_sha256": manifest_hash, "run_id": checkpoint["run_id"], "split": args.split, "modality": args.modality,
                "checkpoint_sha256": sha256(args.checkpoint), "global_step": checkpoint["global_step"],
                "seed": checkpoint["seed"], "next_epoch_zero_based": checkpoint["sampler"]["epoch"],
                "next_batch": checkpoint["sampler"]["start_batch"],
                "training_manifest_sha256": checkpoint["manifest_sha256"]}
    print(json.dumps(export_dataset(model, dataset, args.output, metadata, args.batch_size, args.num_workers, device), indent=2))


if __name__ == "__main__":
    main()
