"""B's development baseline. Run: python -m src.train --help."""
import argparse
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path
import torch
from .config import load_config
from .data import get_train_loader, check_pmt_layout
from .model import PMT
from .losses import PMTLoss
from .training import (seed_all, make_optimizer, EpochSchedule, isolated_batches,
                       save_checkpoint, load_checkpoint, restore_training, sha256, scaled_optimizer_step)


PROJECT = Path(__file__).resolve().parents[1]


def stage_at(cfg, epoch):
    if cfg["experiment"] == "V0":
        return "V0"
    return 1 if epoch < cfg["training"]["stage1_epochs"] else 2


def choose_device(value):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(value)


def provenance(cfg, device, dataset, pretrained):
    def git(*args):
        result = subprocess.run(["git", *args], cwd=PROJECT, text=True, capture_output=True)
        return result.stdout.strip()
    return {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
            "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.machine(),
            "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
            "cuda": torch.version.cuda, "git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--short"),
            "git_diff": git("diff"), "source_sha256": {str(p.relative_to(PROJECT)): sha256(p) for p in (PROJECT / "src").glob("*.py")},
            "data_root": str(dataset.root), "manifest_sha256": sha256(PROJECT / cfg["data"]["manifest"]),
            "pretrained": str(pretrained) if pretrained else None,
            "pretrained_sha256": sha256(pretrained) if pretrained else None}


def run(args):
    checkpoint = load_checkpoint(args.resume) if args.resume else None
    cfg = checkpoint["config"] if checkpoint else load_config(args.experiment)
    if cfg["experiment"] not in ("M0", "V0") or cfg["smooth"]["mode"] != "off" or cfg["local"]["enabled"]:
        raise ValueError("B's entry point supports M0/V0 only; C/D own extension integration")
    if checkpoint and checkpoint["run_id"] != args.run_id:
        raise ValueError("Resume requires the original --run-id")
    if not checkpoint and getattr(args, "activation_checkpointing", False):
        cfg["model"]["activation_checkpointing"] = True
    seed = checkpoint["seed"] if checkpoint else args.seed
    seed_all(seed)
    device = choose_device(args.device)
    if cfg["training"]["amp"] and device.type != "cuda":
        print("CUDA AMP unavailable; recording FP32 execution on", device)
    use_amp = bool(cfg["training"]["amp"] and device.type == "cuda")
    if checkpoint and checkpoint["amp_enabled"] != use_amp:
        raise ValueError("Resume requires the original AMP mode/device type")
    pretrained = Path(args.pretrained) if args.pretrained else PROJECT / "checkpoints/jx_vit_base_p16_224-80ecf9dd.pth"
    if not checkpoint and not pretrained.is_file():
        raise FileNotFoundError(f"ImageNet initialization required: {pretrained}. Run scripts/prepare_pretrained.py first.")
    loader, sampler, dataset, label_map = get_train_loader(cfg, root=args.root, download=False,
        num_workers=args.num_workers, seed=seed, pin_memory=device.type == "cuda")
    if len(label_map) != cfg["data"]["train_identities"]:
        raise ValueError("Unexpected development identity count; verify A's audit")
    if len(loader) == 0:
        raise ValueError("No training batches")
    evidence = provenance(cfg, device, dataset, pretrained if not checkpoint else None)
    if checkpoint:
        if checkpoint["label_map"] != label_map or checkpoint["manifest_sha256"] != evidence["manifest_sha256"]:
            raise ValueError("Data/label map changed since checkpoint; refusing incompatible resume")
    run_dir = PROJECT / "artifacts/results" / args.run_id
    ckpt_dir = PROJECT / "checkpoints" / args.run_id
    if not checkpoint and (run_dir.exists() or ckpt_dir.exists()):
        raise FileExistsError("run-id already exists; choose a new ID or --resume")
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model = PMT(len(label_map), cfg["model"], pretrained=None if checkpoint else pretrained).to(device)
    optimizer = make_optimizer(model, cfg["training"])
    scheduler = EpochSchedule(optimizer, cfg["training"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = PMTLoss(k=sampler.K, margin=cfg["loss"]["triplet_margin"],
                        msel_weight=cfg["loss"]["msel_weight"], dcl_weight=cfg["loss"]["dcl_weight"])
    global_step, start_epoch, start_batch = 0, 0, 0
    if checkpoint:
        restore_training(checkpoint, model, optimizer, scheduler, scaler, sampler)
        global_step, start_epoch, start_batch = checkpoint["global_step"], sampler.epoch, sampler.start_batch
        evidence["resumed_from"] = str(args.resume)
        evidence["checkpoint_sha256"] = sha256(args.resume)
    stamp = f"resume_{global_step}" if checkpoint else "environment"
    (run_dir / f"{stamp}.json").write_text(json.dumps(evidence, indent=2))
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    log_path = run_dir / "train.jsonl"
    if checkpoint and log_path.exists():
        records = [json.loads(line) for line in log_path.read_text().splitlines()]
        log_path.write_text("".join(json.dumps(r) + "\n" for r in records if r["global_step"] <= global_step))
    new_steps = 0
    consecutive_overflows = checkpoint.get("consecutive_overflows", 0) if checkpoint else 0
    successful_updates = 0
    started = time.perf_counter()

    def save(epoch, next_batch, stage):
        sampler.set_epoch(epoch, start_batch=next_batch)
        checksum = save_checkpoint(ckpt_dir / "last.pt", model, optimizer, scheduler, scaler, sampler,
            config=cfg, seed=seed, num_classes=len(label_map), label_map=label_map, global_step=global_step,
            stage=stage, run_id=args.run_id, amp_enabled=use_amp, consecutive_overflows=consecutive_overflows,
            manifest_sha256=evidence["manifest_sha256"],
            pretrained_sha256=checkpoint["pretrained_sha256"] if checkpoint else evidence["pretrained_sha256"])
        summary = {"run_id": args.run_id, "global_step": global_step, "next_epoch_zero_based": epoch,
                   "next_batch": next_batch, "status": "completed" if epoch >= cfg["training"]["epochs"] else "partial",
                   "checkpoint": str(ckpt_dir / "last.pt"), "sha256": checksum,
                   "elapsed_this_invocation_seconds": time.perf_counter() - started}
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    with log_path.open("a") as log:
        for epoch in range(start_epoch, cfg["training"]["epochs"]):
            offset = start_batch if epoch == start_epoch else 0
            sampler.set_epoch(epoch, start_batch=offset)
            stage = stage_at(cfg, epoch)
            probability = float(stage == 1)
            sampler.set_gray_probability(probability)
            scheduler.step(epoch + 1)
            model.train()
            for batch_index, batch in enumerate(isolated_batches(loader), start=offset):
                tick = time.perf_counter()
                check_pmt_layout(batch, sampler.P, sampler.K)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                images, labels = batch["images"].to(device), batch["labels"].to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_amp, dtype=torch.float16):
                    output = model(images)
                # Explicit FP32 losses, outside autocast.
                terms = criterion(output, labels, stage)
                if not torch.isfinite(terms["total"]):
                    raise FloatingPointError(f"Nonfinite loss at epoch {epoch}, batch {batch_index}")
                updated, scale_before, scale_after = scaled_optimizer_step(terms["total"], model, optimizer, scaler)
                consecutive_overflows = 0 if updated else consecutive_overflows + 1
                successful_updates += int(updated)
                if consecutive_overflows >= 16:
                    raise FloatingPointError("16 consecutive AMP overflows; investigate numerical instability")
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elif device.type == "mps":
                    torch.mps.synchronize()
                global_step += 1
                new_steps += 1
                record = {"epoch": epoch + 1, "batch": batch_index + 1, "global_step": global_step,
                          "stage": stage, "lr": sorted(set(g["lr"] for g in optimizer.param_groups)),
                          "p_gray": probability, "gray_count": int(batch["gray"].sum()),
                          "seconds": time.perf_counter() - tick,
                          "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                          "mps_allocated_bytes": torch.mps.current_allocated_memory() if device.type == "mps" else None,
                          "amp_enabled": use_amp, "optimizer_updated": updated,
                          "amp_scale_before": scale_before, "amp_scale_after": scale_after,
                          **{k: float(v.detach()) for k, v in terms.items()}}
                log.write(json.dumps(record) + "\n")
                log.flush()
                if new_steps == 1 or not updated or global_step % args.log_every == 0:
                    print(json.dumps(record), flush=True)
                end_epoch = batch_index + 1 == sampler.batches_per_epoch
                stop = args.max_steps is not None and new_steps >= args.max_steps
                if stop or end_epoch or global_step % args.save_every == 0:
                    save(epoch + 1 if end_epoch else epoch, 0 if end_epoch else batch_index + 1, stage)
                if stop:
                    if successful_updates == 0:
                        raise FloatingPointError("No successful optimizer updates in this invocation; preflight failed")
                    return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("M0", "V0"), default="M0")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--root")
    parser.add_argument("--pretrained")
    parser.add_argument("--resume")
    parser.add_argument("--activation-checkpointing", action="store_true", help="Recompute Transformer blocks to reduce training activation memory")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-steps", type=int, help="Stop after this many additional batches; checkpoint remains resumable")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in (".", ".."):
        parser.error("run-id must be a directory name")
    if args.save_every < 1 or args.log_every < 1 or (args.max_steps is not None and args.max_steps < 1):
        parser.error("step intervals must be positive")
    run(args)


if __name__ == "__main__":
    main()
