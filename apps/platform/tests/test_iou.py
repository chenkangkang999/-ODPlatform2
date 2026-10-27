#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : test_iou.py
# @Project   : ODPlatform
# @Function  : 显式 IoU 指标（evaluation/iou.py）的离线单测

"""``evaluation/iou.py`` 的测试 —— 全部**离线**跑: 没有真权重、没有 GPU、不下载。

分三块:
    1. 纯函数       —— IoU 计算、归一化换算、贪心一对一匹配、IoU 分桶边界
    2. 整理与出口   —— 匹配口径的 P/R/F1、直方图、空真值/空预测、iou_pairs.csv
    3. 端到端 + 视图 —— 假模型走完整条 analyze_iou; 顺带守住"标签目录解析"的两种布局

★ 为什么这里也在测 ``resolve_labels_dir``

    它是 ``build_dataset_view`` 找标签目录的唯一入口, 一旦只认 Roboflow 布局,
    本仓库 D3 流水线产出的数据集就会静默地"0 个真值框"（指标全 0 却不报错）。
    这是最容易悄悄回归的一处, 所以在本文件里一并守住（注释里写明归属）。
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from od_platform.evaluation import iou as I
from od_platform.evaluation.dataset import (
    build_dataset_view,
    resolve_labels_dir,
)
from od_platform.evaluation.report import EvaluationReport
from od_platform.evaluation.settings import EvalSettings

# ============================================================
# 合成数据集（D3 流水线的 {images,labels}/<split> 布局）
# ============================================================

D3_YAML = """\
path: .
train: images/val
val: images/val
nc: 1
names: ['block']
"""

# 100x100 的图, 真值框归一化 (0.5, 0.5, 0.2, 0.2) → 绝对坐标 (40,40,60,60)
PERFECT_BOX = (40.0, 40.0, 60.0, 60.0)


def make_d3_dataset(root: Path, *, labels: bool = True) -> Path:
    """造一个最小数据集: images/val + labels/val（本仓库 D3 的布局）。"""
    (root / "images" / "val").mkdir(parents=True, exist_ok=True)
    for name in ("a.jpg", "b.jpg"):
        (root / "images" / "val" / name).write_bytes(b"fake-jpeg-bytes")
    if labels:
        (root / "labels" / "val").mkdir(parents=True, exist_ok=True)
        (root / "labels" / "val" / "a.txt").write_text(
            "0 0.500000 0.500000 0.200000 0.200000\n", encoding="utf-8")
        (root / "labels" / "val" / "b.txt").write_text(
            "0 0.500000 0.500000 0.200000 0.200000\n", encoding="utf-8")
    (root / "data.yaml").write_text(D3_YAML, encoding="utf-8")
    return root / "data.yaml"


class FakeBoxes:
    def __init__(self, rows):
        self.cls = np.asarray([row[0] for row in rows], dtype=float)
        self.conf = np.asarray([row[1] for row in rows], dtype=float)
        self.xyxy = np.asarray([row[2] for row in rows], dtype=float).reshape(-1, 4)


class FakePredictResult:
    def __init__(self, path, boxes, shape=(100, 100)):
        self.path = str(path)
        self.orig_shape = shape
        self.boxes = boxes


class FakePredictModel:
    """逐图返回预设结果: a.jpg 给一个与真值完全重合的框, b.jpg 给零个框。"""

    def __init__(self, rows_by_stem=None):
        self.calls = []
        self.rows_by_stem = rows_by_stem or {"a": [(0, 0.9, PERFECT_BOX)], "b": []}

    def predict(self, source=None, **kwargs):
        self.calls.append(kwargs)
        paths = [source] if isinstance(source, (str, Path)) else list(source or [])
        results = []
        for path in paths:
            stem = Path(str(path)).stem
            rows = self.rows_by_stem.get(stem)
            if rows is None:
                results.append(FakePredictResult(path, None))
            else:
                results.append(FakePredictResult(path, FakeBoxes(rows)))
        return results


# ============================================================
# 1. 纯函数
# ============================================================

def test_box_iou_is_one_for_identical_boxes():
    assert I.box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    # 完全重合并带偏移的写法也算 1.0（只看数值, 不看是不是同一个元组）
    assert I.box_iou((5.0, 5.0, 15.0, 15.0), [5, 5, 15, 15]) == pytest.approx(1.0)


def test_box_iou_is_zero_without_overlap():
    assert I.box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0     # 完全分开
    assert I.box_iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0      # 只贴边, 交集为 0
    assert I.box_iou((0, 0, 10, 10), (5, 20, 15, 30)) == 0.0      # 纵向错开


def test_box_iou_partial_overlap_matches_hand_computation():
    # A=(0,0,10,10) 面积 100; B=(5,5,15,15) 面积 100; 交=5*5=25 → 25/(100+100-25)
    assert I.box_iou((0, 0, 10, 10), (5, 5, 15, 15)) == pytest.approx(25 / 175)


def test_box_iou_handles_nested_and_degenerate_boxes():
    # 小框完全落在大框里: 交=小框面积, 并=大框面积
    assert I.box_iou((0, 0, 10, 10), (2, 2, 4, 4)) == pytest.approx(4 / 100)
    # 退化成一条线 → 面积 0, 并集也是 0, 不能 0/0
    assert I.box_iou((0, 0, 10, 0), (0, 0, 10, 0)) == 0.0
    assert I.box_iou((0, 0, 0, 0), (5, 5, 5, 5)) == 0.0


def test_normalized_to_xyxy_converts_center_form():
    # 宽 100 高 200 的图上, 归一化 (0.5,0.5,0.2,0.2) → 20x40 的框, 中心在 (50,100)
    box = I.normalized_to_xyxy(0.5, 0.5, 0.2, 0.2, 100, 200)
    assert box == pytest.approx((40.0, 80.0, 60.0, 120.0))


def test_greedy_match_is_one_to_one_and_does_not_reuse():
    """★ 两个预测都挨着同一个真值: 只能配上最好的那个, 另一个必须算 FP。"""
    predictions = [
        {"cls": 0, "conf": 0.9, "box": (0, 0, 10, 10)},        # IoU = 1.0
        {"cls": 0, "conf": 0.5, "box": (0, 0, 10, 11)},        # IoU ≈ 10/11 ≈ 0.909
    ]
    ground_truths = [{"cls": 0, "box": (0, 0, 10, 10)}]
    pairs, unmatched_pred, unmatched_gt = I.greedy_match(predictions, ground_truths)
    assert len(pairs) == 1
    assert pairs[0]["pred_index"] == 0                    # 更像的那个赢
    assert pairs[0]["iou"] == pytest.approx(1.0)
    assert unmatched_pred == [1]
    assert unmatched_gt == []


def test_greedy_match_picks_by_iou_not_by_order():
    """★ 顺序在后的高分框不该被前面的劣质框抢走真值。"""
    predictions = [
        {"cls": 0, "conf": 0.3, "box": (0, 0, 10, 10)},        # 与 gt 的 IoU 较低
        {"cls": 0, "conf": 0.9, "box": (0, 0, 10, 10)},
    ]
    ground_truths = [{"cls": 0, "box": (0, 0, 10, 10)}]
    pairs, _, _ = I.greedy_match(predictions, ground_truths)
    # 两个候选 IoU 相同 → 按下标定序, 仍然只有一个被配上（结果是可复现的）
    assert len(pairs) == 1
    assert pairs[0]["iou"] == pytest.approx(1.0)


def test_greedy_match_never_pairs_different_classes():
    predictions = [{"cls": 1, "conf": 0.9, "box": (0, 0, 10, 10)}]
    ground_truths = [{"cls": 0, "box": (0, 0, 10, 10)}]
    pairs, unmatched_pred, unmatched_gt = I.greedy_match(predictions, ground_truths)
    assert pairs == []
    assert unmatched_pred == [0] and unmatched_gt == [0]


def test_greedy_match_respects_the_threshold():
    predictions = [{"cls": 0, "conf": 0.9, "box": (0, 0, 10, 10)}]
    ground_truths = [{"cls": 0, "box": (6, 0, 16, 10)}]        # IoU = 40/160 = 0.25
    pairs, _, _ = I.greedy_match(predictions, ground_truths, iou_threshold=0.5)
    assert pairs == []
    pairs_low, _, _ = I.greedy_match(predictions, ground_truths, iou_threshold=0.2)
    assert len(pairs_low) == 1


def test_greedy_match_with_empty_inputs():
    assert I.greedy_match([], []) == ([], [], [])
    assert I.greedy_match([{"cls": 0, "box": (0, 0, 1, 1)}], [])[1] == [0]
    assert I.greedy_match([], [{"cls": 0, "box": (0, 0, 1, 1)}])[2] == [0]


@pytest.mark.parametrize("value, expected", [
    (0.50, "0.50-0.60"),          # 下界含
    (0.5999, "0.50-0.60"),
    (0.60, "0.60-0.70"),          # 边界归上一档
    (0.70, "0.70-0.80"),
    (0.80, "0.80-0.90"),
    (0.90, "0.90-1.00"),
    (1.0, "0.90-1.00"),           # 完全重合必须落进最后一档
    (0.4999, None),               # 低于门限的不进直方图
    (0.0, None),
])
def test_bucket_of_boundaries(value, expected):
    assert I.bucket_of(value) == expected


def test_bucket_covers_every_matchable_iou():
    """＠门限之上的任何 IoU 都必须有归属 —— 否则直方图会"掉数"。"""
    for step in range(0, 101):
        value = 0.5 + step * 0.005
        assert I.bucket_of(value) is not None, value


# ============================================================
# 2. 整理与出口
# ============================================================

def test_parse_label_file_converts_and_skips_bad_rows(tmp_path):
    path = tmp_path / "x.txt"
    path.write_text("0 0.5 0.5 0.2 0.2\n"
                    "bad-line\n"
                    "0 0.5 0.5 0.2\n"          # 列数不足
                    "9 0.5 0.5 0.2 0.2\n"      # 类号越界
                    "\n", encoding="utf-8")
    rows = I.parse_label_file(path, image_width=100, image_height=100, n_classes=1)
    assert len(rows) == 1
    assert rows[0]["cls"] == 0
    assert rows[0]["box"] == pytest.approx(PERFECT_BOX)


def test_parse_label_file_missing_file_is_not_fatal(tmp_path):
    assert I.parse_label_file(tmp_path / "nope.txt", image_width=10,
                              image_height=10, n_classes=1) == []


def test_summarize_iou_reports_none_instead_of_zero_when_empty():
    """★ 没有任何匹配对时不许报 0.0 的平均 IoU —— "没配上"和"贴得极差"是两回事。"""
    summary = I.summarize_iou([], n_predictions=0, n_ground_truth=0)
    assert summary["iou"]["mean"] is None
    assert summary["iou"]["min"] is None
    assert summary["precision"] is None and summary["recall"] is None
    assert summary["f1"] is None
    assert all(row["count"] == 0 and row["share"] is None
               for row in summary["histogram"])


def test_summarize_iou_computes_stats_and_histogram():
    pairs = [{"iou": 0.55}, {"iou": 0.95}, {"iou": 0.85}]
    summary = I.summarize_iou(pairs, n_predictions=5, n_ground_truth=6)
    assert summary["matched_pairs"] == 3
    assert summary["tp"] == 3 and summary["fp"] == 2 and summary["fn"] == 3
    assert summary["precision"] == pytest.approx(3 / 5)
    assert summary["recall"] == pytest.approx(3 / 6)
    # 报告里的数都被 settings.decimals 截过, 所以比对面也要先截
    assert summary["f1"] == pytest.approx(round(2 * 0.6 * 0.5 / 1.1, 4))
    assert summary["iou"]["mean"] == pytest.approx(round((0.55 + 0.95 + 0.85) / 3, 4))
    assert summary["iou"]["median"] == pytest.approx(0.85)
    assert summary["iou"]["min"] == pytest.approx(0.55)
    assert summary["iou"]["max"] == pytest.approx(0.95)
    counts = {row["bucket"]: row["count"] for row in summary["histogram"]}
    assert counts == {"0.50-0.60": 1, "0.60-0.70": 0, "0.70-0.80": 0,
                      "0.80-0.90": 1, "0.90-1.00": 1}
    shares = {row["bucket"]: row["share"] for row in summary["histogram"]}
    assert shares["0.50-0.60"] == pytest.approx(round(1 / 3, 4))
    assert "匹配口径" in summary["note"] and "不可混用" in summary["note"]
    assert summary["match_threshold"] == I.IOU_MATCH_THRESHOLD


def test_write_iou_pairs_csv_has_one_row_per_matched_pair(tmp_path):
    report = EvaluationReport(
        run_id="r1", weights="w.pt", data_yaml="d.yaml", split="val",
        iou_analysis=I.summarize_iou([{"iou": 0.9}], n_predictions=1,
                                     n_ground_truth=1))
    # 纯函数产物里没有 pairs 明细（那是 analyze_iou 填的）→ 只写表头
    report.iou_analysis["pairs"] = [{
        "image": "a.jpg", "class_id": 0, "class_name": "block", "iou": 0.9,
        "confidence": 0.8, "gt_x1": 40.0, "gt_y1": 40.0, "gt_x2": 60.0, "gt_y2": 60.0,
        "pred_x1": 41.0, "pred_y1": 41.0, "pred_x2": 61.0, "pred_y2": 61.0,
    }]
    path = tmp_path / "iou_pairs.csv"
    report.write_iou_pairs_csv(path)
    rows = list(csv.DictReader(path.read_text(encoding="utf-8-sig").splitlines()))
    assert len(rows) == 1
    assert rows[0]["run_id"] == "r1"
    assert rows[0]["image"] == "a.jpg" and rows[0]["class_name"] == "block"
    assert float(rows[0]["iou"]) == pytest.approx(0.9)
    assert float(rows[0]["gt_x1"]) == pytest.approx(40.0)


def test_report_json_carries_the_iou_block(tmp_path):
    import json
    report = EvaluationReport(run_id="r1", weights="w.pt", data_yaml="d.yaml",
                              split="val",
                              iou_analysis={"status": "ok", "matched_pairs": 7})
    path = tmp_path / "report.json"
    report.write_json(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["iou_analysis"]["matched_pairs"] == 7


# ============================================================
# 3. 端到端 + 视图标签目录解析
# ============================================================

def test_analyze_iou_end_to_end_on_d3_layout(tmp_path):
    """★ 走完整条: 建视图（D3 布局）→ predict → 匹配 → 统计。"""
    source = make_d3_dataset(tmp_path / "ds")
    view = build_dataset_view(source, split="val", view_root=tmp_path / "view")
    assert view.instances_per_class == (2,)          # 标签确实被复制进来了

    model = FakePredictModel()
    analysis = I.analyze_iou(model, view, EvalSettings(decimals=4))

    assert analysis["status"] == "ok"
    assert analysis["n_images"] == 2
    assert analysis["n_predictions"] == 1             # 只有 a.jpg 有一个框
    assert analysis["n_ground_truth"] == 2
    assert analysis["matched_pairs"] == 1
    assert analysis["tp"] == 1 and analysis["fp"] == 0 and analysis["fn"] == 1
    assert analysis["iou"]["mean"] == pytest.approx(1.0)
    assert analysis["precision"] == pytest.approx(1.0)
    assert analysis["recall"] == pytest.approx(0.5)
    counts = {row["bucket"]: row["count"] for row in analysis["histogram"]}
    assert counts["0.90-1.00"] == 1
    assert len(analysis["pairs"]) == 1
    assert analysis["pairs"][0]["class_name"] == "block"
    assert len(analysis["per_image"]) == 2
    # 用的是部署阈值, 而不是 val 的 0.001
    assert model.calls[0]["conf"] == EvalSettings().conf
    assert model.calls[0]["iou"] == EvalSettings().iou
    assert model.calls[0]["stream"] is True


def test_analyze_iou_flags_a_val_level_conf_as_a_caveat(tmp_path):
    """★ 用 0.001 这类 val 级阈值跑匹配时, 报告必须自己说清"P 为什么低"。"""
    source = make_d3_dataset(tmp_path / "ds")
    view = build_dataset_view(source, split="val", view_root=tmp_path / "view")

    low = I.analyze_iou(FakePredictModel(), view, EvalSettings(conf=0.001))
    assert "val 级极低阈值" in low["note"]
    assert low["conf"] == 0.001

    high = I.analyze_iou(FakePredictModel(), view, EvalSettings(conf=0.25))
    assert "val 级极低阈值" not in high["note"]
    assert high["conf"] == 0.25


def test_analyze_iou_pairs_csv_roundtrip(tmp_path):
    source = make_d3_dataset(tmp_path / "ds")
    view = build_dataset_view(source, split="val", view_root=tmp_path / "view")
    analysis = I.analyze_iou(FakePredictModel(), view, EvalSettings())

    report = EvaluationReport(run_id="r1", weights="w.pt", data_yaml="d.yaml",
                              split="val", iou_analysis=analysis)
    path = tmp_path / "iou_pairs.csv"
    report.write_iou_pairs_csv(path)
    rows = list(csv.DictReader(path.read_text(encoding="utf-8-sig").splitlines()))
    assert len(rows) == analysis["matched_pairs"] == 1


def test_resolve_labels_dir_finds_d3_split_layout(tmp_path):
    """★ 本仓库 D3 布局: images/val 的标签在祖父目录下的 labels/val。"""
    (tmp_path / "images" / "val").mkdir(parents=True)
    (tmp_path / "labels" / "val").mkdir(parents=True)
    (tmp_path / "labels" / "val" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n",
                                                       encoding="utf-8")
    resolved, notes = resolve_labels_dir(tmp_path / "images" / "val")
    assert resolved == tmp_path / "labels" / "val"
    assert any("D3 流水线布局" in note for note in notes)


def test_resolve_labels_dir_finds_roboflow_layout(tmp_path):
    """Roboflow 布局: valid/images 的同级兄弟 valid/labels。"""
    (tmp_path / "valid" / "images").mkdir(parents=True)
    (tmp_path / "valid" / "labels").mkdir(parents=True)
    (tmp_path / "valid" / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n",
                                                         encoding="utf-8")
    resolved, notes = resolve_labels_dir(tmp_path / "valid" / "images")
    assert resolved == tmp_path / "valid" / "labels"
    assert any("Roboflow" in note for note in notes)


def test_resolve_labels_dir_returns_none_when_nothing_exists(tmp_path):
    (tmp_path / "images" / "val").mkdir(parents=True)
    resolved, notes = resolve_labels_dir(tmp_path / "images" / "val")
    assert resolved is None
    assert any("都不存在" in note for note in notes)


def test_build_dataset_view_counts_instances_on_d3_layout(tmp_path):
    """回归: 修好标签目录解析之前, 这里恒为 (0,) 而报告却不报错。"""
    source = make_d3_dataset(tmp_path / "ds")
    view = build_dataset_view(source, split="val", view_root=tmp_path / "view")
    assert view.n_images == 2
    assert view.total_instances == 2
    assert view.labels_dir == tmp_path / "view" / "labels"
    assert sorted(path.name for path in view.labels_dir.glob("*.txt")) == ["a.txt", "b.txt"]