"""E：评价器自检。期望值均为手算（注释给出算式），不依赖真实数据或模型。"""
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.evaluate import (build_gallery_trials, check_inputs, evaluate, gallery_rows, sample_gallery,
                          summarize, sysu_metrics)

PROJECT = Path(__file__).resolve().parents[1]


def ranked(order):
    """距离矩阵的一行：order[k] 是排在第 k 位的 gallery 列号。"""
    row = np.empty(len(order))
    row[np.asarray(order)] = np.arange(len(order))
    return row


def one_query(gallery_ids, gallery_cams, query_id, query_cam, order=None, max_rank=5):
    order = range(len(gallery_ids)) if order is None else order
    return sysu_metrics(ranked(order)[None], [query_id], gallery_ids, [query_cam], gallery_cams, max_rank)


def test_ap_inp_and_identity_cmc_by_hand():
    # 排序身份 [2, 1, 1, 3]，query 身份 1（cam6，不排除）：正样本在第 2、3 位
    m = one_query([2, 1, 1, 3], [1, 4, 5, 1], 1, 6)
    assert m["mAP"] == pytest.approx((1 / 2 + 2 / 3) / 2)   # 0.5833
    assert m["mINP"] == pytest.approx(2 / 3)                  # 正样本数 / 最后一个正样本位置
    assert m["cmc"] == [0, 1, 1, 1, 1]                        # 首个正确前只有身份 2


def test_identity_level_cmc_collapses_repeated_wrong_identity():
    # 实例级首中在第 4 位，但前 3 张同为身份 2，身份级名次为第 2
    m = one_query([2, 2, 2, 1], [1, 2, 4, 5], 1, 6)
    assert m["cmc"] == [0, 1, 1, 1, 1]
    assert m["mAP"] == pytest.approx(1 / 4)


def test_cam3_query_drops_cam2_gallery():
    ids, cams = [1, 2, 1], [2, 1, 4]                          # 排序：身份1@cam2, 身份2@cam1, 身份1@cam4
    cam3 = one_query(ids, cams, 1, 3)                         # 丢弃 cam2 后为 [2, 1]
    assert cam3["cmc"][:2] == [0, 1] and cam3["mAP"] == pytest.approx(1 / 2)
    cam6 = one_query(ids, cams, 1, 6)                         # 不丢弃：[1, 2, 1]
    assert cam6["cmc"][0] == 1 and cam6["mAP"] == pytest.approx((1 + 2 / 3) / 2)
    # cam2 中的错误身份也要丢弃：身份 2@cam2 排第一，丢弃后 cam3 query 首位命中
    assert one_query([2, 1], [2, 4], 1, 3)["rank1"] == 1


def test_queries_without_positive_are_excluded_and_counted():
    g_ids, g_cams = [1, 2, 1, 3], [2, 1, 4, 2]                # 排序固定为列顺序
    q_ids, q_cams = [1, 9, 2, 3], [6, 6, 3, 3]
    m = sysu_metrics(np.tile(ranked([0, 1, 2, 3]), (4, 1)), q_ids, g_ids, q_cams, g_cams, max_rank=3)
    # q0：[1,2,1,3] AP=(1+2/3)/2；q1 身份 9 不在 gallery；q2 cam3：丢弃后 [2,1]，AP=1；
    # q3 cam3：身份 3 唯一正样本在 cam2，被丢弃后无正样本
    assert m["num_valid_queries"] == 2 and m["invalid_query_rows"] == [1, 3]
    assert m["mAP"] == pytest.approx(((1 + 2 / 3) / 2 + 1) / 2)
    assert m["rank1"] == pytest.approx(1.0)


def test_no_valid_query_raises():
    with pytest.raises(ValueError, match="no query"):
        sysu_metrics(ranked([0, 1])[None], [1], [1, 2], [3], [2, 1], max_rank=2)


def test_multi_query_average_by_hand():
    g_ids, g_cams = [1, 2, 3], [1, 1, 1]
    dist = np.stack([ranked([0, 1, 2]), ranked([0, 1, 2]), ranked([2, 1, 0])])
    m = sysu_metrics(dist, [1, 2, 1], g_ids, [6, 6, 6], g_cams, max_rank=3)
    # 身份级名次：q0 第1、q1 第2、q2 第3 → CMC = [1/3, 2/3, 1]；AP = [1, 1/2, 1/3]
    assert m["cmc"] == pytest.approx([1 / 3, 2 / 3, 1])
    assert m["mAP"] == pytest.approx((1 + 1 / 2 + 1 / 3) / 3)


def test_stable_tie_breaking_is_deterministic():
    m = sysu_metrics(np.zeros((1, 3)), [2], [1, 2, 3], [6], [1, 1, 1], max_rank=3)
    assert m["cmc"] == [0, 1, 1]                              # 平局保持 gallery 原顺序


def pool_rows(ids=range(1, 6), cams=(1, 2, 4, 5), per_cam=3, skip=lambda i, c: (i + c) % 4 == 0):
    rows = [(f"cam{c}/{i:04d}/{k:04d}.jpg", i, c) for i in ids for c in cams if not skip(i, c) for k in range(1, per_cam + 1)]
    paths, identities, cameras = map(np.asarray, zip(*rows))
    return {"paths": paths.astype(str), "identities": identities, "cameras": cameras}


def pmt_reference_sampling(pool, seed, cameras):
    """按 PMT process_gallery_sysu 的步骤独立重写：random.seed、身份升序、摄像头顺序、排序后 choice。"""
    random.seed(seed)
    out = []
    for pid in sorted(set(pool["identities"].tolist())):
        for cam in cameras:
            files = sorted(p for p, i, c in zip(pool["paths"], pool["identities"], pool["cameras"]) if i == pid and c == cam)
            if files:
                out.append(random.choice(files))
    return out


@pytest.mark.parametrize("search,cameras", [("all", (1, 2, 4, 5)), ("indoor", (1, 2))])
def test_gallery_sampling_matches_pmt_procedure(search, cameras):
    pool = pool_rows()
    for seed in range(10):
        rows = sample_gallery(pool["identities"], pool["cameras"], pool["paths"], seed, search)
        assert pool["paths"][rows].tolist() == pmt_reference_sampling(pool, seed, cameras)
        pairs = list(zip(pool["identities"][rows], pool["cameras"][rows]))
        assert len(pairs) == len(set(pairs))                  # 每个 (身份, 摄像头) 恰好一张
        expected = {(i, c) for i, c in zip(pool["identities"], pool["cameras"]) if c in cameras}
        assert set(pairs) == expected                         # 缺失摄像头被跳过


def test_gallery_fixed_across_row_order_and_seeds_differ():
    pool = pool_rows()
    trials = build_gallery_trials(pool, range(10))
    perm = np.random.default_rng(0).permutation(len(pool["paths"]))
    shuffled = {k: v[perm] for k, v in pool.items()}
    rows = gallery_rows(shuffled, trials)
    for seed in range(10):
        assert shuffled["paths"][rows[seed]].tolist() == trials["trials"][str(seed)]
    assert len({tuple(v) for v in trials["trials"].values()}) > 1
    with pytest.raises(ValueError, match="indoor-search impossible"):
        build_gallery_trials(pool_rows(cams=(4, 5)), [0], "indoor")   # 内部 val 仅有 cam4/5
    changed = {k: v[1:] for k, v in pool.items()}
    with pytest.raises(ValueError, match="differs"):
        gallery_rows(changed, trials)


def features(part, modality, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(len(part["paths"]), dim)) + np.eye(dim)[part["identities"] % dim] * 3
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return dict(part, features=x.astype(np.float32), modalities=np.full(len(x), modality), metadata={})


def test_check_inputs_rejects_bad_exports():
    pool = features(pool_rows(), 0)
    query = features(pool_rows(cams=(3, 6), skip=lambda i, c: False), 1, seed=1)
    check_inputs(query, pool)
    with pytest.raises(ValueError, match="modality"):
        check_inputs(dict(query, modalities=np.zeros(len(query["paths"]))), pool)
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        check_inputs(dict(query, metadata={"checkpoint_sha256": "a"}), dict(pool, metadata={"checkpoint_sha256": "b"}))
    bad = query["features"].copy(); bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        check_inputs(dict(query, features=bad), pool)


def test_evaluate_matches_per_trial_direct_computation():
    pool = features(pool_rows(), 0)
    query = features(pool_rows(cams=(3, 6), skip=lambda i, c: False), 1, seed=1)
    trials = build_gallery_trials(pool, [0, 1, 2])
    result = evaluate(query, pool, gallery_rows(pool, trials), max_rank=5)
    for t in result["trials"]:
        rows = gallery_rows(pool, trials)[t["gallery_seed"]]
        direct = sysu_metrics(-(query["features"] @ pool["features"][rows].T), query["identities"],
                              pool["identities"][rows], query["cameras"], pool["cameras"][rows], 5)
        assert t["mAP"] == pytest.approx(direct["mAP"])
    assert result["mean_over_gallery"]["mAP"] == pytest.approx(np.mean([t["mAP"] for t in result["trials"]]))


def test_summarize_uses_sample_sd_over_training_seeds():
    def run(seed, value):
        return {"run_id": f"r{seed}", "training_seed": seed, "split": "val", "protocol": {"search": "all"},
                "gallery_fingerprint": "f", "mean_over_gallery": {k: value for k in
                                                                  ("mAP", "mINP", "rank1", "rank5", "rank10", "rank20")}}
    s = summarize([run(17, 0.5), run(42, 0.7)])
    assert s["mAP"]["mean"] == pytest.approx(0.6)
    assert s["mAP"]["sample_sd"] == pytest.approx(np.sqrt(0.02))   # ((0.1²+0.1²)/(2-1))^0.5
    assert summarize([run(17, 0.5)])["mAP"]["sample_sd"] is None
    with pytest.raises(ValueError, match="duplicate"):
        summarize([run(17, 0.5), run(17, 0.6)])
    with pytest.raises(ValueError, match="gallery_fingerprint"):
        summarize([run(17, 0.5), dict(run(42, 0.6), gallery_fingerprint="g")])


def save_export(path, part, metadata):
    np.savez_compressed(path, features=part["features"], identities=part["identities"], cameras=part["cameras"],
                        modalities=part["modalities"], paths=part["paths"], row_indices=np.arange(len(part["paths"])),
                        metadata_json=np.asarray(json.dumps(metadata)))


def test_cli_writes_metrics_and_reuses_fixed_gallery(tmp_path):
    meta = {"run_id": "synthetic", "split": "val", "seed": 17, "checkpoint_sha256": "abc"}
    save_export(tmp_path / "q.npz", features(pool_rows(cams=(3, 6), skip=lambda i, c: False), 1, seed=1), meta)
    save_export(tmp_path / "g.npz", features(pool_rows(), 0), meta)
    gallery = tmp_path / "gallery.json"

    def cli(out):
        return subprocess.run([sys.executable, "-m", "src.evaluate", "run", "--query", str(tmp_path / "q.npz"),
                               "--gallery-pool", str(tmp_path / "g.npz"), "--output-dir", str(tmp_path / out),
                               "--gallery-file", str(gallery), "--seeds", "0", "1", "2"],
                              cwd=PROJECT, capture_output=True, text=True)
    first = cli("a")
    assert first.returncode == 0, first.stderr
    before = gallery.read_text()
    assert cli("b").returncode == 0 and gallery.read_text() == before
    a, b = (json.loads((tmp_path / d / "metrics.json").read_text()) for d in "ab")
    assert a["trials"] == b["trials"] and len(a["trials"]) == 3 and a["training_seed"] == 17
    assert cli("a").returncode != 0                           # 拒绝覆盖已有 metrics
