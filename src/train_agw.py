"""AGW / CAJ external baselines on the shared 296-identity split. Run: python -m src.train_agw --help."""
import argparse
import json
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .config import load_config
from .data import get_train_loader, collate_batch, check_pmt_layout
from .agw import AGW, AGWLoss, BACKBONE, make_sgd, StepWarmupSchedule, make_transforms, collate_caj, check_caj_layout
from .training import (seed_all, isolated_batches, save_checkpoint, load_checkpoint, restore_training,
                       sha256, scaled_optimizer_step)
from .train import choose_device, provenance, PROJECT

PRETRAINED = "checkpoints/resnet50-19c8e357.pth"


def build_loader(cfg, args, seed, pin_memory):
    """A's sampler and dataset (296 IDs, P=8/K=4, RGB then IR) with AGW/CAJ transforms and collate."""
    _, sampler, dataset, label_map = get_train_loader(cfg, root=args.root, download=False, num_workers=0,
                                                      seed=seed, pin_memory=False, preload=args.preload)
    sampler.set_gray_probability(0.0)
    dataset.transforms = make_transforms(cfg)
    caj = cfg["augmentation"]["channel_augmentation"]
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_caj if caj else collate_batch,
                        num_workers=args.num_workers, pin_memory=pin_memory)
    return loader, sampler, dataset, label_map, (check_caj_layout if caj else check_pmt_layout)


def run(args):
    checkpoint = load_checkpoint(args.resume) if args.resume else None
    cfg = checkpoint["config"] if checkpoint else load_config(args.experiment)
    if cfg["experiment"] not in ("AGW", "CAJ", "CAJ_fix") or cfg["model"]["backbone"] != BACKBONE:
        raise ValueError("This entry point supports AGW/CAJ/CAJ_fix only")
    if checkpoint and checkpoint["run_id"] != args.run_id:
        raise ValueError("Resume requires the original --run-id")
    seed = checkpoint["seed"] if checkpoint else args.seed
    seed_all(seed)
    device = choose_device(args.device)
    use_amp = bool(cfg["training"]["amp"] and device.type == "cuda")
    if checkpoint and checkpoint["amp_enabled"] != use_amp:
        raise ValueError("Resume requires the original AMP mode/device type")
    pretrained = Path(args.pretrained) if args.pretrained else PROJECT / PRETRAINED
    if not checkpoint and not pretrained.is_file():
        raise FileNotFoundError(f"ImageNet initialization required: {pretrained}. Run scripts/prepare_pretrained.py --model resnet50 first.")
    loader, sampler, dataset, label_map, check_layout = build_loader(cfg, args, seed, device.type == "cuda")
    if len(label_map) != cfg["data"]["train_identities"]:
        raise ValueError("Unexpected development identity count; verify A's audit")
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
    model = AGW(len(label_map), cfg["model"], pretrained=None if checkpoint else pretrained).to(device)
    optimizer = make_sgd(model, cfg["training"])
    scheduler = StepWarmupSchedule(optimizer, cfg["training"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = AGWLoss(cfg["loss"])
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
    new_steps, successful_updates = 0, 0
    consecutive_overflows = checkpoint.get("consecutive_overflows", 0) if checkpoint else 0
    started = time.perf_counter()

    def save(epoch, next_batch):
        sampler.set_epoch(epoch, start_batch=next_batch)
        checksum = save_checkpoint(ckpt_dir / "last.pt", model, optimizer, scheduler, scaler, sampler,
            config=cfg, seed=seed, num_classes=len(label_map), label_map=label_map, global_step=global_step,
            stage=cfg["experiment"], run_id=args.run_id, amp_enabled=use_amp,
            consecutive_overflows=consecutive_overflows, manifest_sha256=evidence["manifest_sha256"],
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
            scheduler.step(epoch)  # Upstream: zero-based epoch, before training it.
            model.train()
            for batch_index, batch in enumerate(isolated_batches(loader), start=offset):
                tick = time.perf_counter()
                check_layout(batch, sampler.P, sampler.K)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                images, labels = batch["images"].to(device), batch["labels"].to(device)
                modalities = batch["modalities"].to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_amp, dtype=torch.float16):
                    output = model(images, modalities)
                terms = criterion(output, labels)
                if not torch.isfinite(terms["total"]):
                    raise FloatingPointError(f"Nonfinite loss at epoch {epoch}, batch {batch_index}")
                updated, scale_before, scale_after = scaled_optimizer_step(terms["total"], model, optimizer, scaler)
                consecutive_overflows = 0 if updated else consecutive_overflows + 1
                successful_updates += int(updated)
                if consecutive_overflows >= 16:
                    raise FloatingPointError("16 consecutive AMP overflows; investigate numerical instability")
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                global_step += 1
                new_steps += 1
                accuracy = (output["logits"].argmax(1) == labels).float().mean()
                record = {"epoch": epoch + 1, "batch": batch_index + 1, "global_step": global_step,
                          "stage": cfg["experiment"], "lr": sorted(set(g["lr"] for g in optimizer.param_groups)),
                          "batch_rows": len(labels), "train_id_accuracy": float(accuracy),
                          "seconds": time.perf_counter() - tick,
                          "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
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
                    save(epoch + 1 if end_epoch else epoch, 0 if end_epoch else batch_index + 1)
                if stop:
                    if successful_updates == 0:
                        raise FloatingPointError("No successful optimizer updates in this invocation; preflight failed")
                    return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("AGW", "CAJ", "CAJ_fix"), default="AGW")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--root")
    parser.add_argument("--pretrained")
    parser.add_argument("--resume")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--preload", action="store_true", help="Decode and resize training images once into memory")
    parser.add_argument("--max-steps", type=int, help="Stop after this many additional batches; checkpoint remains resumable")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in (".", ".."):
        parser.error("run-id must be a directory name")
    if args.save_every < 1 or args.log_every < 1 or (args.max_steps is not None and args.max_steps < 1):
        parser.error("step intervals must be positive")
    run(args)


if __name__ == "__main__":
    main()
