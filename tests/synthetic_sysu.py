"""生成目录结构与 SYSU-MM01 相同的小型合成数据，仅用于测试数据代码，不代表真实数据。

结构：<root>/cam{1..6}/<4 位身份>/<4 位序号>.jpg 与 <root>/exp/{train,val,test}_id.txt。
RGB 摄像头 1/2/4/5 生成彩色图；IR 摄像头 3/6 生成三通道灰度图（与真实 IR 一致）。
可选注入异常：损坏文件、跨身份重复文件、IR 不足 K 张、非图片文件、缺少某摄像头目录。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

RGB = (1, 2, 4, 5)
IR = (3, 6)


def make_synthetic_sysu(root: str | Path, train=range(1, 13), val=range(13, 17), test=range(17, 21),
                        images_per_camera: int = 5, inject_anomalies: bool = True, seed: int = 0) -> Path:
    root = Path(root)
    rng = np.random.default_rng(seed)
    (root / "exp").mkdir(parents=True, exist_ok=True)
    for split, ids in (("train", train), ("val", val), ("test", test)):
        (root / "exp" / f"{split}_id.txt").write_text(",".join(str(i) for i in ids) + "\n")
    for cam in RGB + IR:
        (root / f"cam{cam}").mkdir(exist_ok=True)

    for pid in list(train) + list(val) + list(test):
        colour = rng.integers(30, 225, size=3)
        for cam in RGB + IR:
            # 真实数据中部分身份缺少某些摄像头；这里让身份在 cam4/cam5 上间隔缺失
            if cam in (4, 5) and (pid + cam) % 3 == 0:
                continue
            n = images_per_camera
            if inject_anomalies and pid == 2 and cam in IR:
                n = 1  # 身份 2：IR 共 2 张 < K=4，触发有放回采样
            d = root / f"cam{cam}" / f"{pid:04d}"
            d.mkdir(parents=True, exist_ok=True)
            for k in range(1, n + 1):
                h, w = int(rng.integers(220, 320)), int(rng.integers(90, 150))
                base = np.zeros((h, w, 3), dtype=np.float32)
                base[: h // 2] = colour
                base[h // 2:] = colour[::-1]
                base += rng.normal(0, 12, size=base.shape)
                if cam in IR:
                    base = np.repeat(base.mean(axis=2, keepdims=True), 3, axis=2)
                Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).save(d / f"{k:04d}.jpg", quality=90)

    if inject_anomalies:
        bad = root / "cam1" / f"{train[0]:04d}" / "0099.jpg"
        bad.write_bytes(b"\xff\xd8\xff\xe0 truncated jpeg")          # 损坏文件
        src = root / "cam2" / f"{train[1]:04d}" / "0001.jpg"
        dup = root / "cam2" / f"{train[2]:04d}" / "0098.jpg"
        dup.write_bytes(src.read_bytes())                               # 跨身份重复文件
        (root / "cam1" / f"{train[0]:04d}" / "Thumbs.db").write_bytes(b"x")  # 非图片文件
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--clean", action="store_true", help="不注入异常")
    args = parser.parse_args()
    make_synthetic_sysu(args.root, inject_anomalies=not args.clean)
    print(f"synthetic SYSU-like data written to {args.root}")


if __name__ == "__main__":
    main()
