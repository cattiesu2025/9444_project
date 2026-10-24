"""A：SYSU-MM01 数据接口——manifest、官方身份划分、标签映射与 PMT 兼容的 P/K 采样。

给 B（训练）和 E（评价）的入口，详细说明见 docs/data_interface.md。最常用的两个：

    loader, sampler, dataset, label_map = get_train_loader("M0")   # 训练数据；final=True 用 395 身份
    query = get_eval_dataset("val", modality="ir")                  # 特征导出/评价，不含训练标签

两者会自动：查找已有数据（参数 root → 环境变量 SYSU_MM01_ROOT → data/SYSU-MM01），
找不到时从 Kaggle 下载到 data/SYSU-MM01；manifest 不存在时生成 manifest 与审计；审计未通过时拒绝提供训练数据。

其他公共函数：
    prepare_data(...)         -> 只做"确保数据 + manifest + 审计"，返回路径与审计结论
    build_train_loader(...)   -> 自己指定 manifest/root/P/K 等参数构建训练 DataLoader
    SYSUImageList(...)        -> 逐图数据集
    check_pmt_layout(...)     -> 校验 batch 是否满足 PMT 损失的排列要求
    as_pmt_tuple(batch)       -> 转成作者 train.py 使用的 (img_rgb, img_ir, label_rgb, label_ir)

batch 排列（PMT 的 MSEL/DCL 依赖该排列，故作为接口契约固定）：
    前 P*K 行为可见光来源（rgb，可能已转灰度），后 P*K 行为红外（ir）；
    两半的身份顺序完全相同；每个身份在每一半中连续占 K 行。
    同时显式返回 labels / identities / cameras / modalities，下游不应靠排列推断标签。

manifest/审计函数只依赖标准库、numpy 与 Pillow；训练/评价数据集还需要 torch 与 torchvision。
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence, Union

import numpy as np
from PIL import Image

try:  # 允许在没有 torch 的机器上只运行 manifest 与审计
    import torch
    from torch.utils.data import DataLoader, Dataset, Sampler
except ImportError:  # pragma: no cover - 由 _require_torch 给出明确报错
    torch = None
    DataLoader = None
    Dataset = object
    Sampler = object

# ---------------------------------------------------------------------------
# 数据集约定
# ---------------------------------------------------------------------------

# 默认位置均相对项目根目录，与从哪个目录运行无关
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "SYSU-MM01"
DEFAULT_MANIFEST_DIR = PROJECT_ROOT / "artifacts" / "manifests"
DATASET_POINTER = PROJECT_ROOT / "data" / "dataset_location.json"  # 自动下载的来源与日期（不入 Git）
DATA_ROOT_ENV = "SYSU_MM01_ROOT"                                     # 可用环境变量指定数据目录
KAGGLE_HANDLE = "coconutjean/sysumm01"                               # 课程提供的 Kaggle 镜像
KAGGLE_URL = "https://www.kaggle.com/datasets/coconutjean/sysumm01"

RGB_CAMERAS = (1, 2, 4, 5)
IR_CAMERAS = (3, 6)
ALL_CAMERAS = (1, 2, 3, 4, 5, 6)
CAMERA_MODALITY = {**{c: "rgb" for c in RGB_CAMERAS}, **{c: "ir" for c in IR_CAMERAS}}
MODALITY_CODE = {"rgb": 0, "ir": 1}  # batch["modalities"] 中的整数编码
SPLIT_FILES = {"train": "train_id.txt", "val": "val_id.txt", "test": "test_id.txt"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# 课程项目说明与作者仓库公布的规模；实际数量以审计为准，差异写入审计报告。
EXPECTED_COUNTS = {
    "identities": 491,
    "rgb_images": 30071,
    "ir_images": 15792,
    "total_images": 45863,
    "train_identities": 296,
    "val_identities": 99,
    "test_identities": 96,
}

MANIFEST_FIELDS = (
    # 设计文档 §3.2 要求的字段
    "relative_path", "identity", "camera", "modality", "split",
    "width", "height", "channels", "file_bytes", "sha256",
    # 额外审计字段：解码结果、原始色彩模式、EDA 用的简单统计
    "image_mode", "decode_ok", "decode_error",
    "mean_intensity", "std_intensity", "channel_spread",
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# 采样器内部的独立随机流编号：改变灰度概率不会改变身份/图片的抽取
_SAMPLER_STREAM = 101
_AUGMENT_STREAM = 202


def _require_torch() -> None:
    if torch is None:
        raise ImportError("该功能需要 torch 和 torchvision；manifest/审计部分不需要。")


# ---------------------------------------------------------------------------
# 目录扫描与官方划分
# ---------------------------------------------------------------------------

def find_dataset_root(path: str | os.PathLike, max_depth: int = 3) -> Path:
    """返回同时包含 cam1 与 exp 的目录；兼容 Kaggle 解压后多一层嵌套的情况。"""
    start = Path(path).expanduser().resolve()
    if not start.is_dir():
        raise FileNotFoundError(f"数据目录不存在：{start}")
    frontier = [start]
    for _ in range(max_depth + 1):
        next_frontier = []
        for directory in frontier:
            if (directory / "cam1").is_dir() and (directory / "exp").is_dir():
                return directory
            next_frontier.extend(sorted(p for p in directory.iterdir() if p.is_dir()))
        frontier = next_frontier
    raise FileNotFoundError(f"在 {start} 下 {max_depth} 层内未找到同时含 cam1/ 与 exp/ 的目录")


def read_id_file(path: str | os.PathLike) -> list[int]:
    """读取 SYSU exp/*_id.txt（逗号分隔，通常只有一行）。"""
    text = Path(path).read_text(encoding="utf-8")
    tokens = [t for t in re.split(r"[\s,]+", text.strip()) if t]
    ids = [int(t) for t in tokens]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path} 内有重复身份")
    return ids


def load_official_splits(root: str | os.PathLike) -> dict[str, list[int]]:
    """读取官方 train/val/test 身份列表；缺文件直接报错，不自行随机划分。"""
    root = Path(root)
    splits = {}
    for split, filename in SPLIT_FILES.items():
        file = root / "exp" / filename
        if not file.is_file():
            raise FileNotFoundError(f"缺少官方划分文件：{file}")
        splits[split] = read_id_file(file)
    return splits


def identity_to_split(splits: dict[str, Sequence[int]]) -> dict[int, str]:
    """身份 → split；同一身份出现在多个划分时报错（身份泄漏）。"""
    mapping: dict[int, str] = {}
    for split, ids in splits.items():
        for pid in ids:
            if pid in mapping:
                raise ValueError(f"身份 {pid} 同时出现在 {mapping[pid]} 和 {split}")
            mapping[pid] = split
    return mapping


def scan_dataset(root: str | os.PathLike) -> tuple[list[dict], dict]:
    """扫描 cam*/<身份>/<图片>，返回 (记录列表, 结构异常)。不读取图片内容。"""
    root = Path(root)
    records: list[dict] = []
    issues = {"missing_cameras": [], "non_numeric_dirs": [], "non_image_files": [], "empty_identity_dirs": []}
    for cam in ALL_CAMERAS:
        cam_dir = root / f"cam{cam}"
        if not cam_dir.is_dir():
            issues["missing_cameras"].append(cam)
            continue
        for pid_dir in sorted(p for p in cam_dir.iterdir() if p.is_dir()):
            if not pid_dir.name.isdigit():
                issues["non_numeric_dirs"].append(pid_dir.relative_to(root).as_posix())
                continue
            files = sorted(f for f in pid_dir.iterdir() if f.is_file())
            images = [f for f in files if f.suffix.lower() in IMAGE_EXTENSIONS]
            issues["non_image_files"].extend(
                f.relative_to(root).as_posix() for f in files if f.suffix.lower() not in IMAGE_EXTENSIONS
            )
            if not images:
                issues["empty_identity_dirs"].append(pid_dir.relative_to(root).as_posix())
            for f in images:
                records.append({
                    "relative_path": f.relative_to(root).as_posix(),
                    "identity": int(pid_dir.name),
                    "camera": cam,
                    "modality": CAMERA_MODALITY[cam],
                })
    return records, issues


# ---------------------------------------------------------------------------
# 逐图检查与 manifest
# ---------------------------------------------------------------------------

def inspect_image(path: str | os.PathLike) -> dict:
    """完整解码一张图，返回尺寸、通道、哈希与简单亮度/色彩统计。"""
    path = Path(path)
    raw = path.read_bytes()
    info = {
        "file_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "width": "", "height": "", "channels": "", "image_mode": "",
        "decode_ok": False, "decode_error": "",
        "mean_intensity": "", "std_intensity": "", "channel_spread": "",
    }
    try:
        with Image.open(path) as img:
            img.load()  # 强制完整解码，能发现截断文件
            info.update(width=img.width, height=img.height,
                        channels=len(img.getbands()), image_mode=img.mode)
            small = img.convert("RGB").resize((32, 64), Image.BILINEAR)
        arr = np.asarray(small, dtype=np.float32)
        gray = arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        spread = (np.abs(r - g) + np.abs(g - b) + np.abs(r - b)).mean() / 3.0
        info.update(decode_ok=True,
                    mean_intensity=round(float(gray.mean()), 3),
                    std_intensity=round(float(gray.std()), 3),
                    channel_spread=round(float(spread), 3))
    except Exception as exc:  # 记录而不中断，由审计决定是否阻塞
        info["decode_error"] = f"{type(exc).__name__}: {exc}"
    return info


def build_manifest(root: str | os.PathLike, workers: int = 8,
                   progress: Callable[[int, int], None] | None = None) -> tuple[list[dict], dict]:
    """扫描 + 解码 + 哈希，返回 (manifest 行, 结构异常)。行按 relative_path 排序。"""
    root = find_dataset_root(root)
    split_of = identity_to_split(load_official_splits(root))
    records, issues = scan_dataset(root)
    records.sort(key=lambda r: r["relative_path"])

    def work(rec: dict) -> dict:
        row = dict(rec)
        row["split"] = split_of.get(rec["identity"], "unassigned")
        row.update(inspect_image(root / rec["relative_path"]))
        return row

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for i, row in enumerate(pool.map(work, records), 1):
            rows.append(row)
            if progress and (i % 2000 == 0 or i == len(records)):
                progress(i, len(records))
    return rows, issues


def write_manifest(rows: Iterable[dict], path: str | os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in MANIFEST_FIELDS})


def load_manifest(path: str | os.PathLike) -> list[dict]:
    """读取 manifest.csv，并把数值字段转回 int/float/bool。"""
    int_fields = ("identity", "camera", "width", "height", "channels", "file_bytes")
    float_fields = ("mean_intensity", "std_intensity", "channel_spread")
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            for k in int_fields:
                row[k] = int(row[k]) if row.get(k, "") != "" else None
            for k in float_fields:
                row[k] = float(row[k]) if row.get(k, "") != "" else None
            row["decode_ok"] = str(row.get("decode_ok", "True")) == "True"
            if CAMERA_MODALITY[row["camera"]] != row["modality"]:
                raise ValueError(f"manifest 中摄像头与模态不一致：{row['relative_path']}")
            rows.append(row)
    return rows


def file_sha256(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 数据审计
# ---------------------------------------------------------------------------

def _percentile(values: Sequence[float], q: float) -> float | None:
    return round(float(np.percentile(values, q)), 3) if len(values) else None


def audit_manifest(rows: Sequence[dict], issues: dict, splits: dict[str, Sequence[int]],
                   K: int = 4, gray_spread_threshold: float = 2.0) -> dict:
    """根据 manifest 生成审计结果：数量、划分隔离、模态覆盖、解码、重复、尺寸与模态映射证据。

    gates 中的项目不通过时，不应开始正式长训练；warnings 需要在报告中披露。
    """
    split_of = {pid: s for s, ids in splits.items() for pid in ids}
    by_mod = Counter(r["modality"] for r in rows)
    by_cam = Counter(r["camera"] for r in rows)
    by_split_mod = Counter((r["split"], r["modality"]) for r in rows)
    image_ids = {r["identity"] for r in rows}

    # 每个身份在各划分、各模态、各摄像头的图片数
    per_id_mod = defaultdict(Counter)
    per_id_cam = defaultdict(set)
    for r in rows:
        if r["decode_ok"]:
            per_id_mod[r["identity"]][r["modality"]] += 1
            per_id_cam[r["identity"]].add(r["camera"])

    split_summary = {}
    for split, ids in splits.items():
        ids = set(ids)
        split_summary[split] = {
            "identities_in_file": len(ids),
            "identities_with_images": len(ids & image_ids),
            "identities_without_images": sorted(ids - image_ids),
            "missing_rgb": sorted(p for p in ids & image_ids if per_id_mod[p]["rgb"] == 0),
            "missing_ir": sorted(p for p in ids & image_ids if per_id_mod[p]["ir"] == 0),
            "rgb_images": by_split_mod[(split, "rgb")],
            "ir_images": by_split_mod[(split, "ir")],
            f"identities_below_K{K}_rgb": sorted(p for p in ids & image_ids if 0 < per_id_mod[p]["rgb"] < K),
            f"identities_below_K{K}_ir": sorted(p for p in ids & image_ids if 0 < per_id_mod[p]["ir"] < K),
        }
    names = list(splits)
    overlaps = {f"{a}&{b}": sorted(set(splits[a]) & set(splits[b]))
                for i, a in enumerate(names) for b in names[i + 1:]}
    unassigned = sorted(image_ids - set(split_of))

    # 重复文件：按内容哈希分组
    groups = defaultdict(list)
    for r in rows:
        groups[r["sha256"]].append(r)
    dup_groups = [g for g in groups.values() if len(g) > 1]
    cross_identity = [[r["relative_path"] for r in g] for g in dup_groups if len({r["identity"] for r in g}) > 1]
    cross_split = [[r["relative_path"] for r in g] for g in dup_groups if len({r["split"] for r in g}) > 1]

    decoded = [r for r in rows if r["decode_ok"]]
    failures = [{"path": r["relative_path"], "error": r["decode_error"]} for r in rows if not r["decode_ok"]]

    def size_stats(mod: str) -> dict:
        rs = [r for r in decoded if r["modality"] == mod]
        if not rs:
            return {}
        sizes = Counter((r["width"], r["height"]) for r in rs)
        ratios = [r["height"] / r["width"] for r in rs]
        return {
            "distinct_sizes": len(sizes),
            "top_sizes_wxh": [[f"{w}x{h}", n] for (w, h), n in sizes.most_common(8)],
            "width_min_median_max": [min(r["width"] for r in rs), int(np.median([r["width"] for r in rs])),
                                     max(r["width"] for r in rs)],
            "height_min_median_max": [min(r["height"] for r in rs), int(np.median([r["height"] for r in rs])),
                                      max(r["height"] for r in rs)],
            "aspect_h_over_w_p5_p50_p95": [_percentile(ratios, 5), _percentile(ratios, 50), _percentile(ratios, 95)],
            "image_modes": dict(Counter(r["image_mode"] for r in rs)),
            "channels": dict(Counter(r["channels"] for r in rs)),
        }

    # 模态映射证据：IR 图三个通道几乎相等（channel_spread≈0），RGB 图通常明显不同
    camera_evidence = {}
    for cam in ALL_CAMERAS:
        spreads = [r["channel_spread"] for r in decoded if r["camera"] == cam]
        if not spreads:
            continue
        gray_like = sum(s < gray_spread_threshold for s in spreads) / len(spreads)
        camera_evidence[f"cam{cam}"] = {
            "assigned_modality": CAMERA_MODALITY[cam],
            "images": len(spreads),
            "channel_spread_p50": _percentile(spreads, 50),
            "channel_spread_p95": _percentile(spreads, 95),
            "fraction_gray_like": round(gray_like, 4),
            "mean_intensity_p50": _percentile([r["mean_intensity"] for r in decoded if r["camera"] == cam], 50),
        }
    mapping_suspicious = [
        cam for cam, ev in camera_evidence.items()
        if (ev["assigned_modality"] == "ir" and ev["fraction_gray_like"] < 0.9)
        or (ev["assigned_modality"] == "rgb" and ev["fraction_gray_like"] > 0.5)
    ]

    cam_coverage = Counter(len(per_id_cam[p]) for p in image_ids)
    expected = EXPECTED_COUNTS
    actual = {
        "identities": len(image_ids),
        "rgb_images": by_mod["rgb"],
        "ir_images": by_mod["ir"],
        "total_images": len(rows),
        "train_identities": len(splits.get("train", [])),
        "val_identities": len(splits.get("val", [])),
        "test_identities": len(splits.get("test", [])),
    }

    gates = {
        "all_six_cameras_present": not issues.get("missing_cameras"),
        "official_split_files_disjoint": all(not v for v in overlaps.values()),
        "no_cross_split_duplicate_files": not cross_split,
        "every_split_identity_has_images": all(not s["identities_without_images"] for s in split_summary.values()),
        "train_identities_have_rgb_and_ir": not (split_summary.get("train", {}).get("missing_rgb")
                                                 or split_summary.get("train", {}).get("missing_ir")),
    }
    # 颜色统计只是映射的旁证（暗光 RGB 也可能接近灰度），因此作为需人工核对的警告，不阻塞训练
    warnings = {
        "camera_modality_suspicious": mapping_suspicious,
        "decode_failures": len(failures),
        "cross_identity_duplicate_groups": len(cross_identity),
        "unassigned_identities": len(unassigned),
        "non_image_files": len(issues.get("non_image_files", [])),
        "non_numeric_dirs": len(issues.get("non_numeric_dirs", [])),
        "count_mismatch_vs_published": {k: [expected[k], actual[k]] for k in expected if expected[k] != actual[k]},
        "val_or_test_missing_modality": {
            s: {"missing_rgb": split_summary[s]["missing_rgb"], "missing_ir": split_summary[s]["missing_ir"]}
            for s in ("val", "test") if s in split_summary
            and (split_summary[s]["missing_rgb"] or split_summary[s]["missing_ir"])
        },
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "warnings": warnings,
        "counts": {
            "published": expected,
            "actual": actual,
            "images_per_camera": {f"cam{c}": by_cam[c] for c in ALL_CAMERAS},
            "images_per_split_modality": {f"{s}/{m}": n for (s, m), n in sorted(by_split_mod.items())},
        },
        "splits": split_summary,
        "split_overlaps": overlaps,
        "unassigned_identities": unassigned,
        "identities_by_number_of_cameras": dict(sorted(cam_coverage.items())),
        "decode_failures": failures,
        "duplicates": {
            "groups": len(dup_groups),
            "files_in_groups": sum(len(g) for g in dup_groups),
            "cross_identity": cross_identity,
            "cross_split": cross_split,
        },
        "image_sizes": {m: size_stats(m) for m in ("rgb", "ir")},
        "camera_modality_evidence": camera_evidence,
        "camera_modality_suspicious": mapping_suspicious,
        "structure_issues": issues,
    }


# ---------------------------------------------------------------------------
# 训练标签
# ---------------------------------------------------------------------------

def select_rows(rows: Sequence[dict], splits: Sequence[str], modality: str | None = None,
                cameras: Sequence[int] | None = None, decodable_only: bool = True) -> list[int]:
    """返回满足条件的 manifest 行号。"""
    keep = []
    for i, r in enumerate(rows):
        if r["split"] not in splits:
            continue
        if modality is not None and r["modality"] != modality:
            continue
        if cameras is not None and r["camera"] not in cameras:
            continue
        if decodable_only and not r["decode_ok"]:
            continue
        keep.append(i)
    return keep


def build_label_map(rows: Sequence[dict], train_splits: Sequence[str] = ("train",)) -> dict[int, int]:
    """原始身份 → 连续训练标签 0..C-1（按身份升序）。

    只纳入在训练划分中同时拥有可解码 RGB 与 IR 图片的身份；其他身份由审计报告列出。
    开发阶段 train_splits=("train",)（296 身份）；最终阶段 ("train", "val")（395 身份）。
    """
    has = defaultdict(set)
    for i in select_rows(rows, train_splits):
        has[rows[i]["identity"]].add(rows[i]["modality"])
    eligible = sorted(pid for pid, mods in has.items() if mods == {"rgb", "ir"})
    return {pid: label for label, pid in enumerate(eligible)}


def write_label_map(label_map: dict[int, int], path: str | os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["identity", "label"])
        for pid, label in sorted(label_map.items(), key=lambda kv: kv[1]):
            w.writerow([pid, label])


# ---------------------------------------------------------------------------
# 图像读取与 PMT 增强
# ---------------------------------------------------------------------------

def load_image(path: str | os.PathLike, height: int, width: int) -> Image.Image:
    """读取为 3 通道 RGB 并缩放到 (width, height)。

    与作者 process_sysu.py 一致：预先用抗锯齿缩放（ANTIALIAS 即 LANCZOS）。
    单通道 IR 在 convert("RGB") 时复制为三通道。
    """
    with Image.open(path) as img:
        return img.convert("RGB").resize((width, height), Image.LANCZOS)


class PMTRandomErasing:
    """作者 transforms.py 中的 RandomErasing（p=0.5，用 ImageNet 均值填充），行为保持一致。

    来源：hulu88/PMT@0b31da9 Pytorch-PMT-VI-ReID/transforms.py（方法出自 Zhong et al.）。
    """

    def __init__(self, p=0.5, sl=0.02, sh=0.4, r1=0.3, mean=IMAGENET_MEAN):
        self.p, self.sl, self.sh, self.r1, self.mean = p, sl, sh, r1, mean

    def __call__(self, img):
        if random.uniform(0, 1) >= self.p:
            return img
        for _ in range(100):
            area = img.size()[1] * img.size()[2]
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                channels = img.size()[0]
                for c in range(channels):
                    img[c, x1:x1 + h, y1:y1 + w] = self.mean[c] if channels == 3 else self.mean[0]
                return img
        return img


class PMTTrainTransform:
    """作者训练增强的等价实现，灰度转换改为由采样器逐样本决定。

    rgb：flip → [Grayscale(3) 若 gray=True] → ToTensor → Normalize → RandomErasing
    ir ：flip → RandomChoice(ColorJitter(0.3, 0.3), GaussianBlur(21, (0.1, 3))) → ToTensor → Normalize → RandomErasing
    作者原实现中第一阶段对全部 RGB 转灰度，等价于 gray_probability=1.0。
    """

    def __init__(self, modality: str):
        _require_torch()
        from torchvision import transforms as T
        self.modality = modality
        self.flip = T.RandomHorizontalFlip()
        self.gray = T.Grayscale(num_output_channels=3)
        self.ir_aug = T.RandomChoice([T.ColorJitter(brightness=0.3, contrast=0.3),
                                      T.GaussianBlur(21, sigma=(0.1, 3))])
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self.erase = PMTRandomErasing(p=0.5)

    def __call__(self, img: Image.Image, gray: bool = False):
        img = self.flip(img)
        if self.modality == "rgb":
            if gray:
                img = self.gray(img)
        else:
            if gray:
                raise ValueError("灰度转换只作用于可见光来源图片，IR 不变")
            img = self.ir_aug(img)
        return self.erase(self.normalize(self.to_tensor(img)))


def make_test_transform():
    """评价/特征导出：缩放已在 load_image 完成，此处只做 ToTensor + Normalize。"""
    _require_torch()
    from torchvision import transforms as T
    return T.Compose([T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])


# ---------------------------------------------------------------------------
# 训练数据集与采样器
# ---------------------------------------------------------------------------

class SYSUTrainData(Dataset):
    """按采样器给出的 ticket=(manifest 行号, 是否灰度, 增强种子) 返回一张训练图。

    每张图的随机增强在 fork 出的局部 RNG 中按增强种子执行，因此结果与 num_workers 无关，
    也不会扰动主进程的全局随机数（模型初始化、dropout）。
    """

    def __init__(self, rows: Sequence[dict], root: str | os.PathLike, label_map: dict[int, int],
                 image_size: tuple[int, int] = (256, 128), transforms: dict | None = None,
                 preload: bool = False, preload_rows: Iterable[int] | None = None):
        _require_torch()
        self.rows = rows
        self.root = Path(root)
        self.label_map = label_map
        self.height, self.width = image_size
        self.transforms = transforms or {"rgb": PMTTrainTransform("rgb"), "ir": PMTTrainTransform("ir")}
        self._cache: dict[int, np.ndarray] = {}
        if preload:  # 类似作者的 .npy 预载；在主进程完成，Linux 下 worker 以写时复制共享
            for i in preload_rows if preload_rows is not None else range(len(rows)):
                self._cache[i] = np.asarray(self._load(i))

    def _load(self, row_index: int) -> Image.Image:
        if row_index in self._cache:
            return Image.fromarray(self._cache[row_index])
        return load_image(self.root / self.rows[row_index]["relative_path"], self.height, self.width)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, ticket: tuple[int, bool, int]) -> dict:
        row_index, gray, aug_seed = ticket
        row = self.rows[row_index]
        img = self._load(row_index)
        py_state = random.getstate()
        try:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(aug_seed)
                random.seed(aug_seed)
                image = self.transforms[row["modality"]](img, gray=bool(gray))
        finally:
            random.setstate(py_state)
        return {
            "image": image,
            "label": self.label_map[row["identity"]],
            "identity": row["identity"],
            "camera": row["camera"],
            "modality": MODALITY_CODE[row["modality"]],
            "gray": bool(gray),
            "row_index": row_index,
            "path": row["relative_path"],
        }


GrayProbability = Union[float, Callable[[int], float]]


class PMTBatchSampler(Sampler):
    """PMT 兼容的身份/模态平衡采样：每步 P 个身份，每身份 K 张 RGB + K 张 IR。

    - 每个 epoch 的内容只由 (seed, epoch) 决定，便于断点恢复与跨成员复现；
      身份/图片抽取与灰度/增强使用两条独立随机流，改变灰度概率不改变抽到的图片。
    - 每个身份在某模态不足 K 张时有放回抽取，并在 stats 中计数；身份本身每步不重复。
    - 默认每 epoch 步数 = max(#RGB, #IR) // (P*K)，与作者 IdentitySampler + drop_last 一致。
    - gray_probability 可为常数或 f(global_step)；global_step = epoch * batches_per_epoch + 步内序号。
      第一阶段用 1.0，M0 第二阶段用 0.0，C 的平滑过渡传入调度函数。
    """

    def __init__(self, rows: Sequence[dict], label_map: dict[int, int],
                 identities_per_batch: int = 8, images_per_identity_per_modality: int = 4,
                 seed: int = 17, batches_per_epoch: int | None = None,
                 gray_probability: GrayProbability = 0.0):
        if identities_per_batch < 2 or images_per_identity_per_modality < 2:
            raise ValueError("P 与 K 都必须 ≥ 2（度量学习需要正负样本，PMT 的 MSEL 需要 K>1）")
        self.P, self.K = identities_per_batch, images_per_identity_per_modality
        self.seed = int(seed)
        self.identities = sorted(label_map)
        if len(self.identities) < self.P:
            raise ValueError(f"可采样身份 {len(self.identities)} 少于 P={self.P}")
        self.pos = {"rgb": defaultdict(list), "ir": defaultdict(list)}
        # 每个身份只属于一个官方划分，所以按标签映射筛选即只取训练划分的图片
        for i, r in enumerate(rows):
            if r["identity"] in label_map and r["decode_ok"]:
                self.pos[r["modality"]][r["identity"]].append(i)
        for pid in self.identities:
            if not self.pos["rgb"][pid] or not self.pos["ir"][pid]:
                raise ValueError(f"身份 {pid} 缺少某一模态，不能进入标签映射")
        n_rgb = sum(len(v) for v in self.pos["rgb"].values())
        n_ir = sum(len(v) for v in self.pos["ir"].values())
        self.num_images = {"rgb": n_rgb, "ir": n_ir}
        self.batches_per_epoch = batches_per_epoch or max(n_rgb, n_ir) // (self.P * self.K)
        if self.batches_per_epoch < 1:
            raise ValueError("每 epoch 步数为 0，请检查数据量或 P/K")
        self.gray_probability = gray_probability
        self.epoch = 0
        self.start_batch = 0

    # -- 控制 --------------------------------------------------------------
    @property
    def batch_size(self) -> int:
        return 2 * self.P * self.K

    def set_epoch(self, epoch: int, start_batch: int = 0) -> None:
        """epoch 从 0 开始计数；断点恢复时用 start_batch 跳过已完成的步。"""
        if not 0 <= start_batch <= self.batches_per_epoch:
            raise ValueError("start_batch 超出范围")
        self.epoch, self.start_batch = int(epoch), int(start_batch)

    def set_gray_probability(self, gray_probability: GrayProbability) -> None:
        self.gray_probability = gray_probability

    def gray_probability_at(self, global_step: int) -> float:
        p = self.gray_probability(global_step) if callable(self.gray_probability) else self.gray_probability
        p = float(p)
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"灰度概率 {p} 不在 [0, 1]")
        return p

    def state_dict(self) -> dict:
        """恢复所需的全部状态；灰度调度函数本身由 C/B 的模块保存。"""
        return {"seed": self.seed, "epoch": self.epoch, "start_batch": self.start_batch,
                "P": self.P, "K": self.K, "batches_per_epoch": self.batches_per_epoch}

    def load_state_dict(self, state: dict) -> None:
        for key in ("seed", "P", "K", "batches_per_epoch"):
            if state[key] != getattr(self, key):
                raise ValueError(f"采样器配置不一致：{key} {state[key]} != {getattr(self, key)}")
        self.set_epoch(state["epoch"], state["start_batch"])

    # -- 生成 --------------------------------------------------------------
    def _pick(self, rng: np.random.Generator, pool: list[int]) -> tuple[list[int], bool]:
        replace = len(pool) < self.K
        return [pool[j] for j in rng.choice(len(pool), self.K, replace=replace)], replace

    def epoch_plan(self, epoch: int) -> list[dict]:
        """生成某 epoch 全部步的抽样计划（不读图片），供迭代、统计与复现检查使用。"""
        rng = np.random.default_rng([self.seed, _SAMPLER_STREAM, epoch])
        plan = []
        for b in range(self.batches_per_epoch):
            pids = [self.identities[j] for j in rng.choice(len(self.identities), self.P, replace=False)]
            rgb, ir, replaced = [], [], []
            for pid in pids:
                rows_rgb, rep_rgb = self._pick(rng, self.pos["rgb"][pid])
                rows_ir, rep_ir = self._pick(rng, self.pos["ir"][pid])
                rgb += rows_rgb
                ir += rows_ir
                if rep_rgb:
                    replaced.append((pid, "rgb"))
                if rep_ir:
                    replaced.append((pid, "ir"))
            plan.append({"batch": b, "identities": pids, "rows": rgb + ir, "replaced": replaced})
        return plan

    def _tickets(self, epoch: int, batch: int, rows: list[int]) -> list[tuple[int, bool, int]]:
        aug = np.random.default_rng([self.seed, _AUGMENT_STREAM, epoch, batch])
        n = self.P * self.K
        p = self.gray_probability_at(epoch * self.batches_per_epoch + batch)
        draws = aug.random(n)               # 次数固定，与 p 无关
        seeds = aug.integers(0, 2**31 - 1, size=2 * n)
        gray = [bool(d < p) for d in draws] + [False] * n
        return [(int(r), g, int(s)) for r, g, s in zip(rows, gray, seeds)]

    def __iter__(self) -> Iterator[list[tuple[int, bool, int]]]:
        epoch, start = self.epoch, self.start_batch
        self.start_batch = 0  # 跳步只作用于恢复后的第一个 epoch
        for step in self.epoch_plan(epoch)[start:]:
            yield self._tickets(epoch, step["batch"], step["rows"])

    def __len__(self) -> int:
        return self.batches_per_epoch - self.start_batch

    def epoch_stats(self, epoch: int = 0) -> dict:
        """单个 epoch 的覆盖与重复情况（Q 采样分析与审计用）。"""
        plan = self.epoch_plan(epoch)
        seen = Counter(pid for s in plan for pid in s["identities"])
        half = self.P * self.K
        rgb_rows = [r for s in plan for r in s["rows"][:half]]
        ir_rows = [r for s in plan for r in s["rows"][half:]]
        return {
            "P": self.P, "K": self.K, "batch_size": self.batch_size,
            "batches_per_epoch": self.batches_per_epoch,
            "identities_available": len(self.identities),
            "identities_covered": len(seen),
            "identity_coverage": round(len(seen) / len(self.identities), 4),
            "identity_visits_min_max": [min(seen.values()), max(seen.values())],
            "rows_drawn": len(rgb_rows) + len(ir_rows),
            "unique_rgb_images": len(set(rgb_rows)),
            "unique_ir_images": len(set(ir_rows)),
            "rgb_image_coverage": round(len(set(rgb_rows)) / self.num_images["rgb"], 4),
            "ir_image_coverage": round(len(set(ir_rows)) / self.num_images["ir"], 4),
            "with_replacement_draws": sum(len(s["replaced"]) for s in plan),
            "identities_below_K": {
                m: sorted(pid for pid in self.identities if len(self.pos[m][pid]) < self.K) for m in ("rgb", "ir")
            },
        }


def collate_batch(items: Sequence[dict]) -> dict:
    """把 2PK 个样本整理为 batch 字典；各字段逐行对齐。"""
    _require_torch()

    def long(key):
        return torch.tensor([it[key] for it in items], dtype=torch.long)

    return {
        "images": torch.stack([it["image"] for it in items]),
        "labels": long("label"),
        "identities": long("identity"),
        "cameras": long("camera"),
        "modalities": long("modality"),
        "gray": torch.tensor([it["gray"] for it in items], dtype=torch.bool),
        "row_indices": long("row_index"),
        "paths": [it["path"] for it in items],
    }


def check_pmt_layout(batch: dict, P: int, K: int) -> None:
    """检查 batch 是否满足 PMT MSEL/DCL/Triplet 依赖的排列；不满足时抛 AssertionError。"""
    n = P * K
    labels, ids = batch["labels"].tolist(), batch["identities"].tolist()
    mods, cams = batch["modalities"].tolist(), batch["cameras"].tolist()
    assert len(labels) == 2 * n, f"batch 行数 {len(labels)} != 2PK={2 * n}"
    assert mods[:n] == [MODALITY_CODE["rgb"]] * n and mods[n:] == [MODALITY_CODE["ir"]] * n, "前半须为 RGB、后半须为 IR"
    assert all(CAMERA_MODALITY[c] == ("rgb" if m == 0 else "ir") for c, m in zip(cams, mods)), "摄像头与模态不一致"
    assert labels[:n] == labels[n:], "RGB 与 IR 两半的身份顺序不一致"
    blocks = [labels[i * K:(i + 1) * K] for i in range(P)]
    assert all(len(set(b)) == 1 for b in blocks), "每个身份须连续占 K 行"
    assert len({b[0] for b in blocks}) == P, "同一步内身份重复"
    assert not any(batch["gray"].tolist()[n:]), "IR 行不应被转灰度"
    assert len(set(zip(labels, ids))) == P, "labels 与 identities 不是一一对应"


def as_pmt_tuple(batch: dict):
    """转为作者 train.py 的 (input1=RGB, input2=IR, label1, label2)。"""
    n = batch["images"].shape[0] // 2
    return batch["images"][:n], batch["images"][n:], batch["labels"][:n], batch["labels"][n:]


def build_train_loader(manifest: str | os.PathLike | Sequence[dict], root: str | os.PathLike,
                       train_splits: Sequence[str] = ("train",), identities_per_batch: int = 8,
                       images_per_identity_per_modality: int = 4, seed: int = 17,
                       gray_probability: GrayProbability = 1.0, image_size: tuple[int, int] = (256, 128),
                       num_workers: int = 4, pin_memory: bool = True, preload: bool = False,
                       batches_per_epoch: int | None = None):
    """一站式构建训练 DataLoader。返回 (loader, sampler, dataset, label_map)。

    每个 epoch 前调用 sampler.set_epoch(epoch)，切换阶段时调用 sampler.set_gray_probability(...)。
    """
    _require_torch()
    rows = load_manifest(manifest) if isinstance(manifest, (str, os.PathLike)) else list(manifest)
    root = find_dataset_root(root)
    label_map = build_label_map(rows, train_splits)
    sampler = PMTBatchSampler(rows, label_map, identities_per_batch, images_per_identity_per_modality,
                              seed=seed, batches_per_epoch=batches_per_epoch, gray_probability=gray_probability)
    train_rows = [i for i in select_rows(rows, train_splits) if rows[i]["identity"] in label_map]
    dataset = SYSUTrainData(rows, root, label_map, image_size=image_size,
                            preload=preload, preload_rows=train_rows)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_batch,
                        num_workers=num_workers, pin_memory=pin_memory)
    return loader, sampler, dataset, label_map


def build_train_loader_from_config(config: dict, **overrides):
    """从 src.config.load_config 的结果构建；最终 395 身份阶段传 train_splits=("train", "val")。"""
    data, train = config["data"], config["training"]
    kwargs = dict(
        manifest=_project_path(data.get("manifest", "artifacts/manifests/manifest.csv")),
        root=_project_path(data["root"]),
        train_splits=(data.get("train_split", "train"),),
        identities_per_batch=train["identities_per_batch"],
        images_per_identity_per_modality=train["images_per_identity_per_modality"],
        seed=train["seeds"][0],
        image_size=tuple(config["model"]["image_size"]),
    )
    kwargs.update(overrides)
    return build_train_loader(**kwargs)


# ---------------------------------------------------------------------------
# 评价 / 特征导出用数据集
# ---------------------------------------------------------------------------

class SYSUImageList(Dataset):
    """逐图返回测试变换后的图像与原始身份/摄像头/模态/路径，不含训练标签。

    用法：SYSUImageList(rows, root, select_rows(rows, ["val"], modality="ir"))
    """

    def __init__(self, rows: Sequence[dict], root: str | os.PathLike, indices: Sequence[int] | None = None,
                 image_size: tuple[int, int] = (256, 128)):
        _require_torch()
        self.rows, self.root = rows, find_dataset_root(root)
        self.indices = list(indices) if indices is not None else list(range(len(rows)))
        self.height, self.width = image_size
        self.transform = make_test_transform()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict:
        row_index = self.indices[i]
        row = self.rows[row_index]
        img = load_image(self.root / row["relative_path"], self.height, self.width)
        return {"image": self.transform(img), "identity": row["identity"], "camera": row["camera"],
                "modality": MODALITY_CODE[row["modality"]], "row_index": row_index, "path": row["relative_path"]}


def collate_eval(items: Sequence[dict]) -> dict:
    _require_torch()

    def long(key):
        return torch.tensor([it[key] for it in items], dtype=torch.long)

    return {"images": torch.stack([it["image"] for it in items]), "identities": long("identity"),
            "cameras": long("camera"), "modalities": long("modality"), "row_indices": long("row_index"),
            "paths": [it["path"] for it in items]}


# ---------------------------------------------------------------------------
# 审计报告与数据产物（scripts/build_manifest.py 与 prepare_data 共用）
# ---------------------------------------------------------------------------

def git_version() -> str:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
                                         stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True,
                                        stderr=subprocess.DEVNULL).strip()
        return commit + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def disk_usage(root: Path, rows) -> dict:
    allocated = 0
    for r in rows:
        try:
            allocated += os.stat(root / r["relative_path"]).st_blocks * 512
        except (OSError, AttributeError):
            pass
    dirs = sum(1 for _ in root.rglob("*") if _.is_dir())
    return {
        "image_file_bytes": sum(r["file_bytes"] for r in rows),
        "image_allocated_bytes": allocated,
        "image_files": len(rows),
        "directories": dirs,
        "approx_inodes": len(rows) + dirs,
    }


def fmt_ids(ids, limit=20) -> str:
    ids = list(ids)
    if not ids:
        return "无"
    shown = ", ".join(str(i) for i in ids[:limit])
    return shown + (f" …（共 {len(ids)} 个）" if len(ids) > limit else f"（{len(ids)} 个）")


def render_audit_markdown(audit: dict, info: dict, K: int) -> str:
    c = audit["counts"]
    lines = [
        "# SYSU-MM01 数据审计报告",
        "",
        f"- 生成时间：{info['generated_at']}",
        f"- 数据来源：{info['source'] or '未填写'}；下载日期：{info['downloaded_on'] or '未填写'}",
        f"- 数据根目录：`{info['dataset_root']}`",
        f"- manifest sha256：`{info['files']['manifest.csv']}`",
        f"- 代码版本：`{info['code_version']}`",
        "",
        f"## 结论：{audit['status']}",
        "",
        "| 门槛检查 | 结果 |",
        "| --- | --- |",
    ]
    lines += [f"| {k} | {'通过' if v else '**未通过**'} |" for k, v in audit["gates"].items()]
    w = audit["warnings"]
    lines += [
        "",
        "需要披露的事项：",
        "",
        f"- 无法解码的图片：{w['decode_failures']}（采样器已排除，列表见 data_audit.json）",
        f"- 跨身份的内容重复文件组：{w['cross_identity_duplicate_groups']}",
        f"- 不属于任何官方划分的身份：{w['unassigned_identities']}",
        f"- 非图片文件：{w['non_image_files']}；非数字身份目录：{w['non_numeric_dirs']}",
        f"- 验证/测试集缺某模态的身份：{w['val_or_test_missing_modality'] or '无'}",
        f"- 颜色统计与目录模态不符、需人工查看样例的摄像头：{w['camera_modality_suspicious'] or '无'}",
        "",
        "## 数量：公布值与实际值",
        "",
        "| 项目 | 公布值 | 实际值 | 差异 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for k, v in c["published"].items():
        a = c["actual"][k]
        lines.append(f"| {k} | {v} | {a} | {a - v:+d} |")
    lines += ["", "公布值来自课程项目说明与作者仓库；差异如实记录，未做静默过滤。", "",
              "## 摄像头与模态", "",
              "| 摄像头 | 指定模态 | 图片数 | 通道差异中位数 | 近灰度比例 | 亮度中位数 |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for cam, ev in audit["camera_modality_evidence"].items():
        lines.append(f"| {cam} | {ev['assigned_modality']} | {ev['images']} | {ev['channel_spread_p50']} | "
                     f"{ev['fraction_gray_like']:.1%} | {ev['mean_intensity_p50']} |")
    lines += ["",
              "模态由目录决定（RGB：cam1/2/4/5，IR：cam3/6，见 `src/data.py` 的 `CAMERA_MODALITY`）。"
              "“通道差异”为 R/G/B 两两差的均值；IR 图三通道几乎相同，可作为映射正确的独立证据。",
              f"可疑摄像头：{audit['camera_modality_suspicious'] or '无'}。", "",
              "## 官方划分", "",
              "| 划分 | 列表身份数 | 有图身份数 | RGB 图 | IR 图 | 缺 RGB | 缺 IR | "
              f"RGB<{K} 张 | IR<{K} 张 |",
              "| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |"]
    for s, v in audit["splits"].items():
        lines.append(f"| {s} | {v['identities_in_file']} | {v['identities_with_images']} | {v['rgb_images']} | "
                     f"{v['ir_images']} | {fmt_ids(v['missing_rgb'])} | {fmt_ids(v['missing_ir'])} | "
                     f"{fmt_ids(v[f'identities_below_K{K}_rgb'])} | {fmt_ids(v[f'identities_below_K{K}_ir'])} |")
    ov = audit["split_overlaps"]
    lines += ["", "划分两两交集：" + "；".join(f"{k} = {fmt_ids(v)}" for k, v in ov.items()),
              "", f"划分按身份进行（官方 exp/*_id.txt），不按图片随机拆分。某模态不足 {K} 张的训练身份由采样器有放回抽取并计数。",
              "", "## 身份覆盖的摄像头数", "",
              "| 覆盖摄像头数 | 身份数 |", "| ---: | ---: |"]
    lines += [f"| {k} | {v} |" for k, v in audit["identities_by_number_of_cameras"].items()]
    d = audit["duplicates"]
    lines += ["", "## 重复文件（按 sha256）", "",
              f"- 重复组：{d['groups']}，涉及文件 {d['files_in_groups']}",
              f"- 跨身份重复组：{len(d['cross_identity'])}（可能是标签问题）",
              f"- 跨划分重复组：{len(d['cross_split'])}（会造成泄漏，必须为 0）",
              "", "重复图像仅审计记录，未修改官方数据。", "", "## 图像尺寸", ""]
    for m, st in audit["image_sizes"].items():
        if not st:
            continue
        lines += [f"**{m.upper()}**：{st['distinct_sizes']} 种尺寸；宽 min/中位/max = {st['width_min_median_max']}，"
                  f"高 = {st['height_min_median_max']}；高宽比 p5/p50/p95 = {st['aspect_h_over_w_p5_p50_p95']}；"
                  f"色彩模式 {st['image_modes']}；常见尺寸 {st['top_sizes_wxh'][:5]}", ""]
    lines += ["所有图片在读取时统一缩放到 256×128（高×宽），与作者预处理一致。", "",
              "## 磁盘", "",
              f"- 图片文件总大小：{info['disk']['image_file_bytes'] / 2**30:.3f} GiB；"
              f"实际占用块：{info['disk']['image_allocated_bytes'] / 2**30:.3f} GiB",
              f"- 文件 {info['disk']['image_files']}，目录 {info['disk']['directories']}，"
              f"约 {info['disk']['approx_inodes']} 个 inode",
              "", "## 标签映射", "",
              f"- 开发阶段 `label_map_dev.csv`：{info['label_maps']['dev']} 个身份（train）",
              f"- 最终阶段 `label_map_final.csv`：{info['label_maps']['final']} 个身份（train+val）",
              "- 标签按原始身份升序编号为 0..C-1；只含同时有 RGB 与 IR 图片的身份。", ""]
    return "\n".join(lines)


def build_data_artifacts(root: str | os.PathLike, out_dir: str | os.PathLike | None = None,
                         source: str = "", downloaded_on: str = "", K: int = 4, workers: int = 8,
                         verbose: bool = True) -> dict:
    """生成 manifest、官方划分、标签映射、审计报告和 dataset_info，返回审计结果。"""
    say = print if verbose else (lambda *a, **k: None)
    root = find_dataset_root(root)
    out = _project_path(out_dir) if out_dir is not None else DEFAULT_MANIFEST_DIR
    (out / "splits").mkdir(parents=True, exist_ok=True)
    say(f"[data] dataset root: {root}")
    t0 = time.time()
    rows, issues = build_manifest(root, workers=workers,
                                  progress=lambda i, n: say(f"[data] inspected {i}/{n} images", flush=True))
    say(f"[data] {len(rows)} images inspected in {time.time() - t0:.1f}s")
    write_manifest(rows, out / "manifest.csv")
    rows = load_manifest(out / "manifest.csv")  # 以写出的文件为准，保证与下游读取一致

    splits = load_official_splits(root)
    for split, ids in splits.items():
        (out / "splits" / f"{split}_ids.txt").write_text("\n".join(str(i) for i in sorted(ids)) + "\n")
    dev_map = build_label_map(rows, ("train",))
    final_map = build_label_map(rows, ("train", "val"))
    write_label_map(dev_map, out / "label_map_dev.csv")
    write_label_map(final_map, out / "label_map_final.csv")

    audit = audit_manifest(rows, issues, splits, K=K)
    tracked = ["manifest.csv", "label_map_dev.csv", "label_map_final.csv"] + [f"splits/{s}_ids.txt" for s in splits]
    info = {
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": source,
        "downloaded_on": downloaded_on,
        "dataset_root": str(root),
        "code_version": git_version(),
        "official_split_files_sha256": {f"exp/{f}": file_sha256(root / "exp" / f) for f in SPLIT_FILES.values()},
        "files": {name: file_sha256(out / name) for name in tracked},
        "label_maps": {"dev": len(dev_map), "final": len(final_map)},
        "disk": disk_usage(root, rows),
    }
    (out / "dataset_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "data_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "data_audit.md").write_text(render_audit_markdown(audit, info, K), encoding="utf-8")

    say(f"[data] audit status: {audit['status']}")
    for k, v in audit["gates"].items():
        say(f"  {'PASS' if v else 'FAIL'}  {k}")
    say(f"[data] dev label map: {len(dev_map)} identities; final: {len(final_map)}")
    say(f"[data] outputs written to {out}")
    return audit


# ---------------------------------------------------------------------------
# 数据位置、自动下载与一站式入口（B / E 只需要这一节）
# ---------------------------------------------------------------------------

_DOWNLOAD_HELP = f"""无法自动下载 SYSU-MM01。任选一种办法：
  1) 配置 Kaggle API 后重新运行：登录 kaggle.com → 右上角头像 → Settings → API → Create New Token，
     把下载的 kaggle.json 放到 用户目录/.kaggle/kaggle.json（Windows 例：C:\\Users\\<你>\\.kaggle\\kaggle.json）；
  2) 手动下载：打开 {KAGGLE_URL} → Download → Download dataset as zip，
     解压到 {DEFAULT_DATA_ROOT}（多嵌套一层也可以），再运行一次。
也可以用环境变量 {DATA_ROOT_ENV} 或参数 root= 指向已有的数据目录。"""


def _project_path(path: str | os.PathLike) -> Path:
    """相对路径按项目根目录解释，绝对路径保持不变。"""
    p = Path(path).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p


def _try_root(path: str | os.PathLike | None) -> Path | None:
    if not path:
        return None
    try:
        return find_dataset_root(_project_path(path))
    except (FileNotFoundError, NotADirectoryError):
        return None


def _read_pointer() -> dict:
    try:
        return json.loads(DATASET_POINTER.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _is_within(path: Path, parent: Path) -> bool:
    path, parent = Path(path).resolve(), Path(parent).resolve()
    return path == parent or parent in path.parents


def _move_into_data_dir(src_root: Path, target: Path | None = None) -> Path:
    """把数据根目录下的全部内容移动到 data/SYSU-MM01（同一磁盘上是重命名，几乎瞬间完成，不占双份空间）。"""
    target = target or DEFAULT_DATA_ROOT
    target.mkdir(parents=True, exist_ok=True)
    children = list(Path(src_root).iterdir())
    conflicts = [c.name for c in children if (target / c.name).exists()]
    if conflicts:
        raise RuntimeError(f"无法移动数据：{target} 中已存在 {conflicts}，请先检查并清理该目录。")
    print(f"[data] moving dataset from {src_root} to {target} ...", flush=True)
    for child in children:
        shutil.move(str(child), str(target / child.name))
    return find_dataset_root(target)


def _record_download(root: Path, downloaded_on: str | None = None, previous: dict | None = None) -> None:
    """在 data/dataset_location.json 记录来源与下载日期，供审计报告使用。"""
    info = dict(previous or {})
    info.update(root=str(root), source=KAGGLE_URL, handle=KAGGLE_HANDLE)
    info.setdefault("downloaded_on", downloaded_on or dt.date.today().isoformat())
    DATASET_POINTER.parent.mkdir(parents=True, exist_ok=True)
    DATASET_POINTER.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")


def locate_dataset(root: str | os.PathLike | None = None) -> Path | None:
    """按顺序查找已有数据：参数 root → 环境变量 SYSU_MM01_ROOT → data/SYSU-MM01。"""
    if root is not None:
        return _try_root(root)
    for candidate in (os.environ.get(DATA_ROOT_ENV), DEFAULT_DATA_ROOT):
        found = _try_root(candidate)
        if found:
            return found
    return None


def _migrate_previous_download() -> Path | None:
    """旧版本会把数据下载到 kagglehub 缓存目录；若记录的位置在项目外，把它移进 data/SYSU-MM01。"""
    pointer = _read_pointer()
    old = _try_root(pointer.get("root"))
    if old is None or _is_within(old, DEFAULT_DATA_ROOT):
        return None
    root = _move_into_data_dir(old)
    _record_download(root, previous=pointer)
    print(f"[data] dataset now at {root}", flush=True)
    return root


def download_dataset(target: str | os.PathLike | None = None) -> Path:
    """用 kagglehub 把课程 Kaggle 镜像（约 2 GB 压缩包）直接下载到 data/SYSU-MM01，返回数据根目录。

    使用 kagglehub 的 output_dir，不经过系统缓存目录；旧版 kagglehub 不支持时先下载到缓存再移动过来。
    来源与日期记录在 data/dataset_location.json（不入 Git）。
    """
    target = _project_path(target) if target is not None else DEFAULT_DATA_ROOT
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError("自动下载需要 kagglehub：pip install kagglehub\n" + _DOWNLOAD_HELP) from exc
    target.mkdir(parents=True, exist_ok=True)
    print(f"[data] downloading {KAGGLE_HANDLE} from Kaggle into {target} (about 2 GB, first time only) ...",
          flush=True)
    try:
        try:
            path = kagglehub.dataset_download(KAGGLE_HANDLE, output_dir=str(target))
        except TypeError:  # kagglehub < 0.3 没有 output_dir
            path = kagglehub.dataset_download(KAGGLE_HANDLE)
    except Exception as exc:
        raise RuntimeError(f"下载失败：{type(exc).__name__}: {exc}\n" + _DOWNLOAD_HELP) from exc
    root = find_dataset_root(path)
    if not _is_within(root, target):
        root = _move_into_data_dir(root, target)
    _record_download(root)
    print(f"[data] dataset ready at {root}", flush=True)
    return root


def ensure_dataset(root: str | os.PathLike | None = None, download: bool = True) -> Path:
    """有数据就直接用，没有就（在 download=True 时）下载到 data/SYSU-MM01。返回含 cam1–cam6 与 exp 的目录。"""
    found = locate_dataset(root)
    if found:
        return found
    if root is not None:
        raise FileNotFoundError(f"指定的数据目录中找不到 cam1/ 与 exp/：{_project_path(root)}")
    migrated = _migrate_previous_download()
    if migrated:
        return migrated
    if not download:
        raise FileNotFoundError("没有找到 SYSU-MM01，且未允许下载。\n" + _DOWNLOAD_HELP)
    return download_dataset()


def prepare_data(root: str | os.PathLike | None = None, download: bool = True,
                 manifest_dir: str | os.PathLike | None = None, rebuild: bool = False,
                 source: str | None = None, downloaded_on: str | None = None,
                 workers: int = 8, verbose: bool = True) -> dict:
    """一站式准备：确保数据存在 → 如 manifest 缺失、过期或 rebuild=True 则重新生成 manifest 与审计。

    返回 {"root", "manifest", "manifest_dir", "audit_status", "rebuilt"}。
    已有 manifest 且数据位置未变时直接复用，不会重复扫描 4.6 万张图。
    """
    root = ensure_dataset(root, download)
    out = _project_path(manifest_dir) if manifest_dir is not None else DEFAULT_MANIFEST_DIR
    manifest, info_file, audit_file = out / "manifest.csv", out / "dataset_info.json", out / "data_audit.json"
    try:
        recorded_root = json.loads(info_file.read_text(encoding="utf-8")).get("dataset_root")
    except (OSError, ValueError):
        recorded_root = None
    stale = rebuild or not manifest.exists() or not audit_file.exists() or recorded_root != str(root)
    if stale:
        pointer = _read_pointer()
        pointer = pointer if pointer.get("root") == str(root) else {}
        audit = build_data_artifacts(
            root, out,
            source=source if source is not None else pointer.get("source", ""),
            downloaded_on=downloaded_on if downloaded_on is not None else pointer.get("downloaded_on", ""),
            workers=workers, verbose=verbose)
    else:
        audit = json.loads(audit_file.read_text(encoding="utf-8"))
        if verbose:
            print(f"[data] reusing manifest {manifest} (audit {audit['status']}); pass rebuild=True to regenerate")
    return {"root": root, "manifest": manifest, "manifest_dir": out,
            "audit_status": audit["status"], "rebuilt": stale}


def _require_passed_audit(prep: dict, allow_failed_audit: bool) -> None:
    if prep["audit_status"] != "PASS" and not allow_failed_audit:
        raise RuntimeError(f"数据审计未通过，见 {prep['manifest_dir'] / 'data_audit.md'}；"
                           "确认问题后可传 allow_failed_audit=True 继续（仅限调试）。")


def get_manifest(root: str | os.PathLike | None = None, download: bool = True) -> list[dict]:
    """返回 manifest 行（list[dict]），必要时自动下载并生成。"""
    return load_manifest(prepare_data(root, download, verbose=False)["manifest"])


def get_train_loader(experiment: str | dict = "M0", final: bool = False,
                     root: str | os.PathLike | None = None, download: bool = True,
                     allow_failed_audit: bool = False, **overrides):
    """B 的训练入口：一行拿到训练数据。返回 (loader, sampler, dataset, label_map)。

        loader, sampler, dataset, label_map = get_train_loader("M0")
        num_classes = len(label_map)

    experiment：实验 ID（读取 configs/）或已加载的配置字典。
    final=True：最终阶段，用 train+val（395 身份）训练；默认开发阶段 train（296 身份）。
    overrides：转给 build_train_loader，例如 num_workers=0、seed=42、gray_probability=1.0。
    """
    if isinstance(experiment, str):
        from .config import load_config
        config = load_config(experiment)
    else:
        config = experiment
    prep = prepare_data(root, download, verbose=False)
    _require_passed_audit(prep, allow_failed_audit)
    if final:
        overrides.setdefault("train_splits", ("train", "val"))
    return build_train_loader_from_config(config, manifest=prep["manifest"], root=prep["root"], **overrides)


def get_eval_dataset(split: str = "val", modality: str | None = None, cameras: Sequence[int] | None = None,
                     root: str | os.PathLike | None = None, download: bool = True,
                     image_size: tuple[int, int] = (256, 128)) -> "SYSUImageList":
    """B/E 的特征导出入口：某划分（可选模态、摄像头）的逐图数据集，不含训练标签。

        query = get_eval_dataset("val", modality="ir")
        gallery_pool = get_eval_dataset("val", modality="rgb")
    """
    prep = prepare_data(root, download, verbose=False)
    rows = load_manifest(prep["manifest"])
    return SYSUImageList(rows, prep["root"], select_rows(rows, [split], modality=modality, cameras=cameras),
                         image_size=image_size)
