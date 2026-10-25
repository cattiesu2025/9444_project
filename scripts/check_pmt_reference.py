"""Numerically compare B's implementation to an external pinned author checkout.

Usage: python scripts/check_pmt_reference.py /path/to/PMT
No upstream files are copied into this repository.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.model import VisionTransformer
from src.losses import hard_triplet, msel, dcl
from src.config import load_config
from src.training import EpochSchedule

COMMIT = "0b31da97b06428473bc4a1134b322ea226340818"


def module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/results/baseline_checks/reference.json"))
    args = parser.parse_args()
    actual = subprocess.check_output(["git", "-C", str(args.checkout), "rev-parse", "HEAD"], text=True).strip()
    if actual != COMMIT:
        raise ValueError(f"Expected upstream commit {COMMIT}, got {actual}")
    root = args.checkout / "Pytorch-PMT-VI-ReID"
    torch.set_num_threads(1)
    torch.manual_seed(17)
    ref_vit = module(root / "model/vision_transformer.py").ViT(
        img_size=[32, 20], patch_size=8, stride_size=6, embed_dim=24, depth=2, num_heads=3)
    ours = VisionTransformer(image_size=(32, 20), patch_size=8, patch_stride=6,
                             dim=24, depth=2, heads=3, dropout=0., drop_path=0.)
    ours.load_state_dict(ref_vit.state_dict(), strict=True)
    images = torch.randn(4, 3, 32, 20)
    ours.eval(); ref_vit.eval()
    torch.testing.assert_close(ours(images)[0], ref_vit(images), atol=2e-6, rtol=2e-5)
    report = {"upstream_commit": actual, "backbone_forward": "PASS", "loss_cases": []}
    tri = module(root / "loss/Triplet.py").TripletLoss(.1, feat_norm="no")
    msel_class = module(root / "loss/MSEL.py").MSEL
    dcl_class = module(root / "loss/DCL.py").DCL
    for p, k in ((2, 2), (3, 4), (8, 4)):
        labels = torch.arange(p).repeat_interleave(k).repeat(2)
        for seed in (17, 42, 2026):
            torch.manual_seed(seed)
            x = torch.randn(2 * p * k, 24, requires_grad=True)
            comparisons = [("triplet", hard_triplet(x, labels), tri(x, x, labels)),
                           ("msel", msel(x, labels, k), msel_class(k)(x, labels)),
                           ("dcl", dcl(x, labels, k), dcl_class(k)(x, labels))]
            for name, value, reference in comparisons:
                torch.testing.assert_close(value, reference, rtol=2e-5, atol=2e-6)
                gradient = torch.autograd.grad(value, x, retain_graph=True)[0]
                ref_gradient = torch.autograd.grad(reference, x, retain_graph=True)[0]
                torch.testing.assert_close(gradient, ref_gradient, rtol=2e-4, atol=2e-5)
                report["loss_cases"].append({"P": p, "K": k, "seed": seed, "loss": name,
                    "absolute_error": abs(float(value.detach() - reference.detach())), "gradient": "PASS"})
    cfg = load_config("M0")["training"]
    def optimizer():
        return torch.optim.AdamW([{"params": [torch.nn.Parameter(torch.ones(1))], "lr": .0003},
                                 {"params": [torch.nn.Parameter(torch.ones(1))], "lr": .00015}])
    ours_opt, ref_opt = optimizer(), optimizer()
    ours_schedule = EpochSchedule(ours_opt, cfg)
    ref_schedule = module(root / "scheduler.py").create_scheduler(SimpleNamespace(MAX_EPOCH=24,
        LR_MIN=.01, BASE_LR=.0003, LR_INIT=.01, WARMUP_EPOCHS=3), ref_opt)
    for epoch in range(1, 25):
        ours_schedule.step(epoch); ref_schedule.step(epoch)
        assert [g["lr"] for g in ours_opt.param_groups] == [g["lr"] for g in ref_opt.param_groups]
    report["schedule_all_24_epochs"] = "PASS"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"PASS: backbone, 27 loss/gradient comparisons, all 24 epoch learning rates; {args.output}")


if __name__ == "__main__":
    main()
