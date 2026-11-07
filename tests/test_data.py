"""A：数据接口自检（使用合成的 SYSU 结构数据，不需要真实数据集）。

运行：python -m pytest tests/test_data.py -q
真实数据上的同类检查由 scripts/check_data.py 完成。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from src import data as D  # noqa: E402
from synthetic_sysu import make_synthetic_sysu  # noqa: E402

TRAIN, VAL, TEST = range(1, 21), range(21, 25), range(25, 29)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    # 多一层目录，模拟 Kaggle 解压后的嵌套结构
    root = make_synthetic_sysu(tmp_path_factory.mktemp("sysu") / "download" / "SYSU-MM01",
                               train=TRAIN, val=VAL, test=TEST)
    rows, issues = D.build_manifest(root.parent.parent, workers=4)
    manifest = root / "manifest.csv"
    D.write_manifest(rows, manifest)
    return root, D.load_manifest(manifest), issues


def make_loader(dataset, P=4, K=4, gray=1.0, workers=0, **kw):
    root, rows, _ = dataset
    return D.build_train_loader(rows, root, identities_per_batch=P, images_per_identity_per_modality=K,
                                gray_probability=gray, num_workers=workers, pin_memory=False, **kw)


# --- manifest 与划分 --------------------------------------------------------

def test_root_discovery_handles_nesting(dataset):
    root, _, _ = dataset
    assert D.find_dataset_root(root.parent.parent) == root


def test_camera_modality_and_identity_parsing(dataset):
    _, rows, _ = dataset
    for r in rows:
        cam_dir, pid_dir, _ = r["relative_path"].split("/")
        assert cam_dir == f"cam{r['camera']}"
        assert int(pid_dir) == r["identity"]
        assert r["modality"] == ("ir" if r["camera"] in (3, 6) else "rgb")


def test_manifest_has_required_fields(dataset):
    _, rows, _ = dataset
    required = {"relative_path", "identity", "camera", "modality", "split",
                "width", "height", "channels", "file_bytes", "sha256"}
    assert required <= set(rows[0])
    assert all(len(r["sha256"]) == 64 for r in rows)


def test_splits_follow_official_files(dataset):
    _, rows, _ = dataset
    for r in rows:
        expected = "train" if r["identity"] in TRAIN else "val" if r["identity"] in VAL else "test"
        assert r["split"] == expected


def test_overlapping_split_files_rejected():
    with pytest.raises(ValueError):
        D.identity_to_split({"train": [1, 2], "val": [2, 3]})


def test_audit_reports_anomalies(dataset):
    root, rows, issues = dataset
    audit = D.audit_manifest(rows, issues, D.load_official_splits(root))
    assert audit["status"] == "PASS"
    assert audit["warnings"]["decode_failures"] == 1
    assert audit["warnings"]["cross_identity_duplicate_groups"] == 1
    assert audit["warnings"]["non_image_files"] == 1
    assert audit["splits"]["train"]["identities_below_K4_ir"] == [2]
    assert not audit["warnings"]["camera_modality_suspicious"]


def test_audit_fails_on_cross_split_duplicate(dataset):
    root, rows, issues = dataset
    leaked = [dict(r) for r in rows]
    test_row = next(r for r in leaked if r["split"] == "test")
    next(r for r in leaked if r["split"] == "train")["sha256"] = test_row["sha256"]
    audit = D.audit_manifest(leaked, issues, D.load_official_splits(root))
    assert audit["status"] == "FAIL" and not audit["gates"]["no_cross_split_duplicate_files"]


# --- 标签映射 ----------------------------------------------------------------

def test_label_map_dev_and_final(dataset):
    _, rows, _ = dataset
    dev = D.build_label_map(rows, ("train",))
    final = D.build_label_map(rows, ("train", "val"))
    assert set(dev) == set(TRAIN)
    assert set(final) == set(TRAIN) | set(VAL)
    assert sorted(dev.values()) == list(range(len(dev)))
    assert list(dev) == sorted(dev)  # 按身份升序编号


# --- batch 契约 ---------------------------------------------------------------

@pytest.mark.parametrize("P,K", [(4, 4), (8, 2)])
def test_batch_layout_and_fields(dataset, P, K):
    loader, sampler, _, label_map = make_loader(dataset, P, K)
    sampler.set_epoch(0)
    for batch in list(loader)[:3]:
        D.check_pmt_layout(batch, P, K)
        assert batch["images"].shape == (2 * P * K, 3, 256, 128)
        assert batch["images"].dtype == torch.float32
        assert all(label_map[i] == l for i, l in zip(batch["identities"].tolist(), batch["labels"].tolist()))
        assert set(batch["identities"].tolist()) <= set(TRAIN)
        assert len(batch["paths"]) == 2 * P * K
        x1, x2, y1, y2 = D.as_pmt_tuple(batch)
        assert x1.shape == x2.shape and torch.equal(y1, y2)


def test_layout_check_catches_misalignment(dataset):
    loader, sampler, _, _ = make_loader(dataset, 4, 4)
    sampler.set_epoch(0)
    batch = next(iter(loader))
    n = 16
    perm = torch.cat([torch.arange(n), n + torch.roll(torch.arange(n), 4)])
    broken = {k: (v[perm] if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    with pytest.raises(AssertionError):
        D.check_pmt_layout(broken, 4, 4)


def test_corrupt_images_never_sampled(dataset):
    _, rows, _ = dataset
    bad = {i for i, r in enumerate(rows) if not r["decode_ok"]}
    _, sampler, _, _ = make_loader(dataset)
    assert bad and not bad & {r for s in sampler.epoch_plan(0) for r in s["rows"]}


def test_with_replacement_when_below_K(dataset):
    _, sampler, _, _ = make_loader(dataset, 4, 4)
    stats = sampler.epoch_stats(0)
    assert stats["identities_below_K"]["ir"] == [2]
    visits_of_2 = sum(2 in s["identities"] for s in sampler.epoch_plan(0))
    assert stats["with_replacement_draws"] == visits_of_2


def test_pk_must_exceed_one(dataset):
    _, rows, _ = dataset
    with pytest.raises(ValueError):
        D.PMTBatchSampler(rows, D.build_label_map(rows), 8, 1)


# --- 灰度、复现与恢复 ---------------------------------------------------------

def test_gray_only_on_rgb_and_follows_probability(dataset):
    _, rows, _ = dataset
    lm = D.build_label_map(rows)
    n = 16
    for p, expect in ((1.0, n), (0.0, 0)):
        s = D.PMTBatchSampler(rows, lm, 4, 4, gray_probability=p)
        t = next(iter(s))
        assert sum(g for _, g, _ in t[:n]) == expect and not any(g for _, g, _ in t[n:])


def test_gray_schedule_does_not_change_sampling(dataset):
    _, rows, _ = dataset
    lm = D.build_label_map(rows)
    a = D.PMTBatchSampler(rows, lm, 4, 4, gray_probability=0.0)
    b = D.PMTBatchSampler(rows, lm, 4, 4, gray_probability=lambda step: max(0.0, 1 - step / 5))
    ta, tb = list(a), list(b)
    assert [[(r, s) for r, _, s in x] for x in ta] == [[(r, s) for r, _, s in x] for x in tb]
    gray_counts = [sum(g for _, g, _ in x) for x in tb]
    assert gray_counts[0] == 16 and gray_counts[-1] == 0


def test_gray_probability_uses_global_step(dataset):
    _, rows, _ = dataset
    seen = []
    s = D.PMTBatchSampler(rows, D.build_label_map(rows), 4, 4,
                          gray_probability=lambda step: seen.append(step) or 0.0)
    s.set_epoch(2)
    list(s)
    assert seen == list(range(2 * s.batches_per_epoch, 3 * s.batches_per_epoch))


def test_deterministic_and_worker_independent(dataset):
    l0, s0, _, _ = make_loader(dataset, workers=0)
    l2, s2, _, _ = make_loader(dataset, workers=2)
    s0.set_epoch(1)
    s2.set_epoch(1)
    b0, b2 = next(iter(l0)), next(iter(l2))
    assert torch.equal(b0["images"], b2["images"]) and b0["paths"] == b2["paths"]


def test_global_rng_untouched(dataset):
    # 数据集的增强不消耗全局 RNG（DataLoader 创建迭代器时自身会取一次基础种子，这是 PyTorch 的标准行为）
    import random
    _, sampler, ds, _ = make_loader(dataset)
    tickets = next(iter(sampler))
    torch.manual_seed(123)
    random.seed(123)
    expected = (torch.rand(3), random.random())
    torch.manual_seed(123)
    random.seed(123)
    for t in tickets:
        ds[t]
    assert torch.equal(torch.rand(3), expected[0]) and random.random() == expected[1]


def test_seed_and_epoch_change_plan(dataset):
    _, rows, _ = dataset
    lm = D.build_label_map(rows)
    a, b = D.PMTBatchSampler(rows, lm, seed=17), D.PMTBatchSampler(rows, lm, seed=42)
    assert a.epoch_plan(0)[0]["rows"] != b.epoch_plan(0)[0]["rows"]
    assert a.epoch_plan(0)[0]["rows"] != a.epoch_plan(1)[0]["rows"]
    assert a.epoch_plan(3) == D.PMTBatchSampler(rows, lm, seed=17).epoch_plan(3)


def test_resume_mid_epoch(dataset):
    _, rows, _ = dataset
    lm = D.build_label_map(rows)
    s = D.PMTBatchSampler(rows, lm, 4, 4, gray_probability=0.5)
    s.set_epoch(1)
    full = list(s)
    s.set_epoch(1, start_batch=2)
    state = s.state_dict()
    restored = D.PMTBatchSampler(rows, lm, 4, 4, gray_probability=0.5)
    restored.load_state_dict(state)
    assert len(restored) == len(full) - 2
    assert list(restored) == full[2:]
    assert len(restored) == restored.batches_per_epoch  # 跳步只作用一次


def test_q_configuration_matches_steps(dataset):
    _, rows, _ = dataset
    lm = D.build_label_map(rows)
    base, q = D.PMTBatchSampler(rows, lm, 8, 4), D.PMTBatchSampler(rows, lm, 16, 2)
    assert base.batch_size == q.batch_size == 64
    assert base.batches_per_epoch == q.batches_per_epoch


# --- 评价数据集与配置 ----------------------------------------------------------

def test_eval_dataset_has_no_training_labels(dataset):
    root, rows, _ = dataset
    idx = D.select_rows(rows, ["val"], modality="ir")
    ds = D.SYSUImageList(rows, root, idx)
    batch = D.collate_eval([ds[i] for i in range(4)])
    assert "labels" not in batch
    assert batch["images"].shape == (4, 3, 256, 128)
    assert set(batch["identities"].tolist()) <= set(VAL)
    assert set(batch["cameras"].tolist()) <= {3, 6}


def test_build_from_project_config(dataset):
    from src.config import load_config
    root, rows, _ = dataset
    loader, sampler, _, _ = D.build_train_loader_from_config(
        load_config("M0"), manifest=rows, root=root, num_workers=0, pin_memory=False)
    assert (sampler.P, sampler.K, sampler.seed) == (8, 4, 17)


# --- 自动查找 / 下载 / 一站式入口 ------------------------------------------------

@pytest.fixture
def isolated_paths(tmp_path, monkeypatch):
    """把默认数据位置、下载记录和 manifest 目录指向临时目录，不触碰项目里的真实文件。"""
    monkeypatch.setattr(D, "DEFAULT_DATA_ROOT", tmp_path / "data" / "SYSU-MM01")
    monkeypatch.setattr(D, "DATASET_POINTER", tmp_path / "data" / "dataset_location.json")
    monkeypatch.setattr(D, "DEFAULT_MANIFEST_DIR", tmp_path / "manifests")
    monkeypatch.delenv(D.DATA_ROOT_ENV, raising=False)
    return tmp_path


def test_ensure_dataset_uses_existing_data_without_download(dataset, isolated_paths, monkeypatch):
    root, _, _ = dataset
    monkeypatch.setenv(D.DATA_ROOT_ENV, str(root))
    monkeypatch.setattr(D, "download_dataset", lambda: pytest.fail("不应下载"))
    assert D.ensure_dataset() == root


def _no_download(monkeypatch):
    import types
    monkeypatch.setitem(sys.modules, "kagglehub",
                        types.SimpleNamespace(dataset_download=lambda *a, **k: pytest.fail("不应重复下载")))


def test_download_goes_directly_into_data_dir(dataset, isolated_paths, monkeypatch):
    import shutil
    import types
    root, _, _ = dataset
    calls = []

    def fake_download(handle, output_dir=None):  # 模拟 kagglehub ≥0.3：直接写入 output_dir
        calls.append((handle, output_dir))
        shutil.copytree(root, output_dir, dirs_exist_ok=True)
        return output_dir

    monkeypatch.setitem(sys.modules, "kagglehub", types.SimpleNamespace(dataset_download=fake_download))
    got = D.ensure_dataset()
    assert got == D.DEFAULT_DATA_ROOT.resolve()
    assert calls == [(D.KAGGLE_HANDLE, str(D.DEFAULT_DATA_ROOT))]
    assert D._read_pointer()["source"] == D.KAGGLE_URL
    _no_download(monkeypatch)
    assert D.ensure_dataset() == got  # 第二次直接用 data/SYSU-MM01


def test_old_kagglehub_cache_download_is_moved_into_data_dir(dataset, isolated_paths, monkeypatch):
    import shutil
    import types
    root, _, _ = dataset
    cache = isolated_paths / "cache" / "versions" / "1"
    shutil.copytree(root, cache)

    def old_download(handle):  # 模拟旧版 kagglehub：没有 output_dir，只下载到缓存
        return str(cache)

    monkeypatch.setitem(sys.modules, "kagglehub", types.SimpleNamespace(dataset_download=old_download))
    got = D.ensure_dataset()
    assert got == D.DEFAULT_DATA_ROOT.resolve()
    assert (got / "cam1").is_dir() and not any(cache.iterdir())  # 移动而不是复制


def test_previous_cache_download_is_migrated(dataset, isolated_paths, monkeypatch):
    import json
    import shutil
    root, _, _ = dataset
    cache = isolated_paths / "home" / ".cache" / "kagglehub" / "versions" / "1"
    shutil.copytree(root, cache)
    D.DATASET_POINTER.parent.mkdir(parents=True, exist_ok=True)
    D.DATASET_POINTER.write_text(json.dumps({"root": str(cache), "downloaded_on": "2026-10-24"}))
    _no_download(monkeypatch)
    got = D.ensure_dataset(download=False)
    assert got == D.DEFAULT_DATA_ROOT.resolve() and (got / "exp" / "train_id.txt").exists()
    assert not any(cache.iterdir())
    pointer = D._read_pointer()
    assert pointer["root"] == str(got) and pointer["downloaded_on"] == "2026-10-24"


def test_ensure_dataset_without_download_gives_instructions(isolated_paths):
    with pytest.raises(FileNotFoundError, match="Download dataset as zip"):
        D.ensure_dataset(download=False)


def test_prepare_data_builds_then_reuses(dataset, isolated_paths, monkeypatch):
    root, _, _ = dataset
    monkeypatch.setenv(D.DATA_ROOT_ENV, str(root))
    first = D.prepare_data(verbose=False)
    assert first["rebuilt"] and first["audit_status"] == "PASS" and first["manifest"].exists()
    assert not D.prepare_data(verbose=False)["rebuilt"]
    assert D.prepare_data(rebuild=True, verbose=False)["rebuilt"]


def test_get_train_loader_one_call(dataset, isolated_paths, monkeypatch):
    root, _, _ = dataset
    monkeypatch.setenv(D.DATA_ROOT_ENV, str(root))
    loader, sampler, _, label_map = D.get_train_loader("M0", num_workers=0, pin_memory=False)
    sampler.set_epoch(0)
    D.check_pmt_layout(next(iter(loader)), sampler.P, sampler.K)
    assert set(label_map) == set(TRAIN)
    _, _, _, final_map = D.get_train_loader("M0", final=True, num_workers=0, pin_memory=False)
    assert set(final_map) == set(TRAIN) | set(VAL)


def test_get_eval_dataset_one_call(dataset, isolated_paths, monkeypatch):
    root, _, _ = dataset
    monkeypatch.setenv(D.DATA_ROOT_ENV, str(root))
    query = D.get_eval_dataset("val", modality="ir")
    assert len(query) > 0 and {query[i]["camera"] for i in range(len(query))} <= {3, 6}


# --- 自然遮挡子集（服务 L / E） ------------------------------------------------

def _write_subset(path, records):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=D.OCCLUSION_FIELDS)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in D.OCCLUSION_FIELDS})


def _val_ir(rows, n):
    return [rows[i] for i in D.select_rows(rows, ["val"], modality="ir")][:n]


def _record(row, **kw):
    base = {"relative_path": row["relative_path"], "identity": row["identity"], "camera": row["camera"],
            "occluder": "object", "region": "bottom", "half": "lower", "severity": "mild", "note": ""}
    base.update(kw)
    return base


def test_natural_occlusion_loads_and_filters(dataset, tmp_path):
    _, rows, _ = dataset
    a, b, c = _val_ir(rows, 3)
    path = tmp_path / "occ.csv"
    _write_subset(path, [_record(a), _record(b, occluder="person", region="top", half="upper"),
                         _record(c, occluder="frame_edge", region="side", half="both", severity="heavy")])
    recs = D.load_natural_occlusion(path, rows, "val")
    assert [r["relative_path"] for r in recs] == [a["relative_path"], b["relative_path"], c["relative_path"]]
    assert D.natural_occlusion_paths(path, rows, "val", half="upper") == [b["relative_path"]]
    assert D.natural_occlusion_paths(path, rows, "val", occluder="frame_edge", severity="heavy") == [c["relative_path"]]
    with pytest.raises(ValueError):
        D.natural_occlusion_paths(path, rows, "val", identity="1")
    with pytest.raises(ValueError, match="test 划分"):      # 默认划分是 test，验证集样例会被拒绝
        D.load_natural_occlusion(path, rows)


@pytest.mark.parametrize("bad", [{"occluder": "shadow"}, {"half": "middle"}, {"region": "left"}, {"severity": "x"}])
def test_natural_occlusion_rejects_unknown_labels(dataset, tmp_path, bad):
    _, rows, _ = dataset
    path = tmp_path / "occ.csv"
    _write_subset(path, [_record(_val_ir(rows, 1)[0], **bad)])
    with pytest.raises(ValueError):
        D.load_natural_occlusion(path, rows, "val")


def test_natural_occlusion_rejects_non_val_ir_and_duplicates(dataset, tmp_path):
    _, rows, _ = dataset
    train_ir = rows[D.select_rows(rows, ["train"], modality="ir")[0]]
    val_rgb = rows[D.select_rows(rows, ["val"], modality="rgb")[0]]
    for wrong in (train_ir, val_rgb):
        path = tmp_path / "wrong.csv"
        _write_subset(path, [_record(wrong)])
        with pytest.raises(ValueError, match="val 划分的 IR"):
            D.load_natural_occlusion(path, rows, "val")
    a = _val_ir(rows, 1)[0]
    _write_subset(tmp_path / "dup.csv", [_record(a), _record(a)])
    with pytest.raises(ValueError, match="重复"):
        D.load_natural_occlusion(tmp_path / "dup.csv", rows, "val")
    _write_subset(tmp_path / "id.csv", [_record(a, identity=a["identity"] + 1)])
    with pytest.raises(ValueError, match="不一致"):
        D.load_natural_occlusion(tmp_path / "id.csv", rows, "val")


def test_subset_by_paths_keeps_order_and_is_strict():
    import numpy as np
    arrays = {"features": np.arange(8.).reshape(4, 2), "identities": np.array([1, 2, 3, 4]),
              "paths": np.array(["a", "b", "c", "d"]), "metadata": {"split": "val"}}
    part = D.subset_by_paths(arrays, ["d", "b"])
    assert part["paths"].tolist() == ["b", "d"] and part["identities"].tolist() == [2, 4]
    assert part["features"].tolist() == [[2., 3.], [6., 7.]] and part["metadata"] == {"split": "val"}
    with pytest.raises(ValueError):
        D.subset_by_paths(arrays, ["b", "zzz"])
    assert D.subset_by_paths(arrays, ["b", "zzz"], strict=False)["paths"].tolist() == ["b"]


def test_get_eval_dataset_natural_occlusion_subset(dataset, isolated_paths, monkeypatch, tmp_path):
    root, rows, _ = dataset
    monkeypatch.setenv(D.DATA_ROOT_ENV, str(root))
    picked = _val_ir(D.get_manifest(), 2)
    path = tmp_path / "occ.csv"
    _write_subset(path, [_record(r) for r in picked])
    monkeypatch.setattr(D, "NATURAL_OCCLUSION_FILE", path)
    ds = D.get_eval_dataset("val", modality="ir", subset="natural_occlusion")
    assert sorted(ds[i]["path"] for i in range(len(ds))) == sorted(r["relative_path"] for r in picked)
    with pytest.raises(ValueError):
        D.get_eval_dataset("val", modality="ir", subset="nope")


def test_natural_occlusion_eval_script(dataset, tmp_path):
    """用合成特征端到端跑评价脚本：分组数量正确、差值正确、与 E 的评价函数结果一致。"""
    import json
    import subprocess
    import numpy as np
    from src.evaluate import build_gallery_trials, evaluate, gallery_rows, load_features
    _, rows, _ = dataset
    rng = np.random.default_rng(0)

    def export(split_rows, modality_code, path, shift, run_id):
        ids = np.array([r["identity"] for r in split_rows])
        feats = rng.normal(size=(len(ids), 16)) + np.eye(16)[ids % 16] * shift
        feats /= np.linalg.norm(feats, axis=1, keepdims=True)
        np.savez_compressed(path, features=feats.astype(np.float32), identities=ids,
                            cameras=np.array([r["camera"] for r in split_rows]),
                            modalities=np.full(len(ids), modality_code), paths=np.array([r["relative_path"] for r in split_rows]),
                            metadata_json=np.asarray(json.dumps({"split": "val", "run_id": run_id})))

    q_rows = [rows[i] for i in D.select_rows(rows, ["val"], modality="ir")]
    g_rows = [rows[i] for i in D.select_rows(rows, ["val"], modality="rgb")]
    for name, shift in (("m0", 2.0), ("m2", 4.0)):
        export(q_rows, 1, tmp_path / f"q_{name}.npz", shift, name)
        export(g_rows, 0, tmp_path / f"g_{name}.npz", shift, name)
    pool = load_features(tmp_path / "g_m0.npz")
    gallery = build_gallery_trials(pool, [0, 1, 2], "all", "val")
    (tmp_path / "gallery.json").write_text(json.dumps(gallery))
    picked = q_rows[:3]
    _write_subset(tmp_path / "occ.csv", [_record(picked[0]), _record(picked[1], half="upper", region="top"),
                                         _record(picked[2], occluder="person")])
    manifest = tmp_path / "manifest.csv"
    D.write_manifest(rows, manifest)
    out = tmp_path / "res.json"
    cmd = [sys.executable, str(ROOT / "scripts" / "natural_occlusion_eval.py"),
           "--run", "M0", str(tmp_path / "q_m0.npz"), str(tmp_path / "g_m0.npz"),
           "--run", "M2", str(tmp_path / "q_m2.npz"), str(tmp_path / "g_m2.npz"),
           "--subset", str(tmp_path / "occ.csv"), "--manifest", str(manifest),
           "--gallery-file", str(tmp_path / "gallery.json"), "--split", "val", "--output", str(out)]
    # 标注文件不在 Git 中：默认拒绝，保证标注先于结果固定
    refused = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert refused.returncode != 0 and "未提交" in refused.stderr and not out.exists()
    cmd.append("--allow-uncommitted-subset")
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=ROOT)
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["split"] == "val" and res["subset_committed"] is False
    g = res["runs"]["M0"]["groups"]
    assert g["all"]["num_queries"] == len(q_rows)
    assert g["natural_occlusion"]["num_queries"] == 3
    assert g["not_occluded"]["num_queries"] == len(q_rows) - 3
    assert g["half_lower"]["num_queries"] == 2 and g["half_upper"]["num_queries"] == 1
    assert g["occluder_person"]["num_queries"] == 1 and g["occluder_frame_edge"]["num_queries"] == 0
    cams = {r["camera"] for r in q_rows}
    for cam in cams:   # 每个摄像头：遮挡 + 未遮挡 = 全部
        assert (g[f"cam{cam}_natural_occlusion"]["num_queries"] + g[f"cam{cam}_not_occluded"]["num_queries"]
                == g[f"cam{cam}_all"]["num_queries"] == sum(r["camera"] == cam for r in q_rows))
    # 全集结果与直接调用 E 的 evaluate 完全一致
    direct = evaluate(load_features(tmp_path / "q_m0.npz"), pool, gallery_rows(pool, gallery))["mean_over_gallery"]
    assert g["all"]["mAP"] == pytest.approx(direct["mAP"])
    diff = res["difference_vs_M0"]["M2"]["natural_occlusion"]["mAP"]
    assert diff == pytest.approx(res["runs"]["M2"]["groups"]["natural_occlusion"]["mAP"] - g["natural_occlusion"]["mAP"])
    with pytest.raises(subprocess.CalledProcessError):   # 拒绝覆盖
        subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=ROOT)
