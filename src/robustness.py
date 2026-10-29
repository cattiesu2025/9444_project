"""E：固定 IR query 扰动（遮挡/模糊）下的检索鲁棒性，RGB gallery 保持不变。

规则在开发验证集上预先固定（设计文档 §7）：
- 只扰动 IR query；RGB gallery 直接复用 B 导出的同一权重候选池特征，不重新提取、不做改动。
- 处理顺序：读取 → LANCZOS 缩放到模型输入尺寸 → 扰动 → ToTensor → ImageNet Normalize（与 B 的导出一致）。
- 遮挡：上/中/下水平带，占图像面积 10%/20%（带高 = round(比例 × 高)，整宽），填充 ImageNet 均值像素
  (124, 116, 104)，归一化后约为 0，即"中性"值。
- 模糊：两档 Gaussian blur，kernel 7/σ=1 与 kernel 13/σ=2（kernel = 2⌈3σ⌉+1）。
- 扰动是确定性的，不含随机数；所有模型使用相同被扰动 query 和同一份固定 gallery 文件。
报告各条件的 10 组 gallery 平均指标，以及相对干净条件的绝对下降（百分点）。人工扰动不等于真实遮挡分布。

python -m src.robustness --checkpoint checkpoints/<run_id>/last.pt \
    --gallery-pool <run>/gallery_pool_val.npz --query <run>/query_val.npz \
    --manifest <manifest.csv> --root data/SYSU-MM01 --output-dir artifacts/results/<run_id>/robustness
"""
import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .data import IMAGENET_MEAN, MODALITY_CODE, SYSUImageList, load_image, load_manifest, select_rows
from .evaluate import PROJECT, _write_new, check_inputs, evaluate, gallery_rows, load_features, sha256_file

FILL = tuple(round(255 * m) for m in IMAGENET_MEAN)
PROCESSING_ORDER = ["load", "LANCZOS resize to model input", "perturbation", "ToTensor", "ImageNet Normalize"]


@dataclass(frozen=True)
class Perturbation:
    """kind: clean | occlusion | blur。occlusion 用 region/ratio，blur 用 sigma（kernel 由 sigma 决定）。"""
    kind: str = "clean"
    region: str = ""
    ratio: float = 0.
    sigma: float = 0.

    @property
    def kernel(self):
        return 2 * math.ceil(3 * self.sigma) + 1

    def band(self, height):
        """遮挡带的行范围 [start, stop)。"""
        rows = round(self.ratio * height)
        start = {"top": 0, "middle": (height - rows) // 2, "bottom": height - rows}[self.region]
        return start, start + rows

    def __call__(self, image):
        if self.kind == "clean":
            return image
        if self.kind == "occlusion":
            array = np.array(image)
            start, stop = self.band(array.shape[0])
            array[start:stop] = FILL
            return Image.fromarray(array)
        if self.kind == "blur":
            from torchvision.transforms import functional as TF
            return TF.gaussian_blur(image, kernel_size=[self.kernel, self.kernel], sigma=[self.sigma, self.sigma])
        raise ValueError(f"unknown perturbation {self.kind}")


CONDITIONS = {"clean": Perturbation()}
CONDITIONS.update({f"occlusion_{region}_{int(ratio * 100)}": Perturbation("occlusion", region, ratio)
                   for region in ("top", "middle", "bottom") for ratio in (0.1, 0.2)})
CONDITIONS.update({f"blur_sigma{sigma:g}": Perturbation("blur", sigma=sigma) for sigma in (1., 2.)})


class PerturbedImageList(SYSUImageList):
    """与 B 的评价数据集相同，只在缩放后、ToTensor 前插入固定扰动。"""

    def __init__(self, *args, perturbation=Perturbation(), **kwargs):
        super().__init__(*args, **kwargs)
        self.perturbation = perturbation

    def __getitem__(self, i):
        row_index = self.indices[i]
        row = self.rows[row_index]
        image = self.perturbation(load_image(self.root / row["relative_path"], self.height, self.width))
        return {"image": self.transform(image), "identity": row["identity"], "camera": row["camera"],
                "modality": MODALITY_CODE[row["modality"]], "row_index": row_index, "path": row["relative_path"]}


def extract(model, dataset, batch_size=128, num_workers=0, device="cpu"):
    """与 export_features.export_dataset 相同的前向（BN 后 L2 特征），结果留在内存。"""
    import torch
    from torch.utils.data import DataLoader
    from .data import collate_eval
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_eval)
    out = {key: [] for key in ("features", "identities", "cameras", "modalities", "paths")}
    with torch.inference_mode():
        for batch in loader:
            out["features"].append(model(batch["images"].to(device))["retrieval_feature"].float().cpu().numpy())
            for key in ("identities", "cameras", "modalities"):
                out[key].append(batch[key].numpy())
            out["paths"].extend(batch["paths"])
    arrays = {k: np.asarray(v) if k == "paths" else np.concatenate(v) for k, v in out.items()}
    if not np.isfinite(arrays["features"]).all():
        raise FloatingPointError("nonfinite features")
    return arrays


def load_model(checkpoint_path, device):
    import torch
    from .model import PMT
    from .training import load_checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    model = PMT(checkpoint["num_classes"], checkpoint["config"]["model"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.eval(), checkpoint


def run_robustness(checkpoint_path, gallery_pool_path, manifest_path, root, output_dir, gallery_file=None,
                   query_path=None, conditions=None, device="cpu", batch_size=128, num_workers=0, example_path=None):
    conditions = conditions or CONDITIONS
    output = Path(output_dir) / "robustness.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    pool = load_features(gallery_pool_path)
    checkpoint_sha = sha256_file(checkpoint_path)
    if pool["metadata"].get("checkpoint_sha256", checkpoint_sha) != checkpoint_sha:
        raise ValueError("gallery pool was exported from a different checkpoint")
    model, checkpoint = load_model(checkpoint_path, device)
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != checkpoint["manifest_sha256"]:
        raise ValueError("manifest differs from the one used for training; verify data provenance")
    rows = load_manifest(manifest_path)
    split = pool["metadata"].get("split", "val")
    indices = select_rows(rows, [split], modality="ir")
    image_size = tuple(checkpoint["config"]["model"]["image_size"])
    gallery_file = Path(gallery_file or PROJECT / f"artifacts/manifests/gallery_{split}_all_single.json")
    if not gallery_file.exists():
        raise FileNotFoundError(f"{gallery_file} missing: run src.evaluate first so all evaluations share it")
    gallery = json.loads(gallery_file.read_text(encoding="utf-8"))
    trial_rows = gallery_rows(pool, gallery)

    report = {"run_id": checkpoint.get("run_id"), "split": split, "training_seed": checkpoint.get("seed"),
              "checkpoint_sha256": checkpoint_sha, "manifest_sha256": manifest_sha,
              "gallery_file": str(gallery_file), "gallery_file_sha256": sha256_file(gallery_file),
              "gallery_fingerprint": gallery["pool_fingerprint"], "gallery_pool_sha256": sha256_file(gallery_pool_path),
              "protocol": {"perturbed": "IR query only; RGB gallery features reused unchanged",
                           "processing_order": PROCESSING_ORDER, "occlusion_fill_rgb": list(FILL),
                           "image_size": list(image_size), "deterministic": True, "units": "fraction 0-1"},
              "conditions": {}}
    clean = None
    for name, perturbation in conditions.items():
        dataset = PerturbedImageList(rows, root, indices, image_size=image_size, perturbation=perturbation)
        query = extract(model, dataset, batch_size, num_workers, device)
        query["metadata"] = {"checkpoint_sha256": checkpoint_sha, "split": split}
        check_inputs(query, pool)
        result = evaluate(query, pool, trial_rows)
        mean = result["mean_over_gallery"]
        entry = {"perturbation": asdict(perturbation) | ({"kernel": perturbation.kernel} if perturbation.kind == "blur" else {}),
                 "num_queries": int(len(query["paths"])), "mean_over_gallery": mean,
                 "trials": [{k: t[k] for k in ("gallery_seed", "mAP", "rank1", "num_valid_queries")} for t in result["trials"]]}
        if name == "clean":
            clean = mean
            if query_path is not None:
                entry["export_consistency"] = compare_with_export(query, load_features(query_path))
        if clean is not None:
            entry["drop_vs_clean_points"] = {k: 100 * (clean[k] - mean[k]) for k in ("mAP", "rank1")}
        report["conditions"][name] = entry
        print(f"{name:22s} mAP {100 * mean['mAP']:5.1f}  R1 {100 * mean['rank1']:5.1f}  R5 {100 * mean['rank5']:5.1f}  "
              f"R10 {100 * mean['rank10']:5.1f}" + (f"  (mAP -{entry['drop_vs_clean_points']['mAP']:.1f} pts)"
                                                       if name != "clean" and clean is not None else ""), flush=True)
    _write_new(output, report)
    if example_path or indices:
        save_examples(rows, root, (example_path or rows[indices[0]]["relative_path"]), image_size, conditions,
                      Path(output_dir) / "perturbation_examples.png")
    return report


def compare_with_export(query, exported):
    """干净条件重新提取的特征应与 B 的导出一致（不同 GPU 只允许数值误差）。"""
    order = {p: i for i, p in enumerate(exported["paths"].tolist())}
    if set(order) != set(query["paths"].tolist()):
        raise ValueError("re-extracted clean queries differ from the exported query set")
    rows = [order[p] for p in query["paths"].tolist()]
    diff = float(np.abs(query["features"] - exported["features"][rows]).max())
    cosine = float((query["features"] * exported["features"][rows]).sum(1).min())
    return {"max_abs_diff": diff, "min_cosine": cosine, "status": "PASS" if cosine > 0.999 else "CHECK"}


def save_examples(rows, root, relative_path, image_size, conditions, output):
    """同一张 IR query 在各条件下的样子，供报告/展示说明扰动规则。"""
    from .data import find_dataset_root
    from PIL import ImageDraw
    height, width = image_size
    base = load_image(find_dataset_root(root) / relative_path, height, width)
    label_h, gap = 22, 6
    canvas = Image.new("RGB", (len(conditions) * (width + gap), height + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (name, perturbation) in enumerate(conditions.items()):
        canvas.paste(perturbation(base), (i * (width + gap), label_h))
        draw.text((i * (width + gap) + 2, 4), name.replace("occlusion_", "occ_"), fill=(27, 42, 58))
    canvas.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gallery-pool", required=True, help="B 用同一权重导出的 RGB 候选池 npz（不扰动）")
    parser.add_argument("--query", help="同一权重导出的干净 IR query npz，用于核对重新提取的一致性")
    parser.add_argument("--manifest", default=str(PROJECT / "artifacts/manifests/manifest.csv"))
    parser.add_argument("--root", required=True, help="含 cam1–cam6 的 SYSU-MM01 目录")
    parser.add_argument("--gallery-file", help="固定 gallery 记录（默认与 src.evaluate 相同）")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--conditions", nargs="+", choices=sorted(CONDITIONS), help="默认全部条件")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--example-path", help="示例图使用的 query 相对路径")
    args = parser.parse_args()
    import torch
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    conditions = {"clean": CONDITIONS["clean"]} | {c: CONDITIONS[c] for c in (args.conditions or CONDITIONS)}
    run_robustness(args.checkpoint, args.gallery_pool, args.manifest, args.root, args.output_dir, args.gallery_file,
                   args.query, conditions, torch.device(device), args.batch_size, args.num_workers, args.example_path)


if __name__ == "__main__":
    main()
