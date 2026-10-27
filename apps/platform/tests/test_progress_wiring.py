#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""进度反馈接线测试（D4 1.1 与 1.2 第二条）。

守两件事:
    · orchestrator 的两段长循环（清点图像 / 扫描图像目录 / 读取标签）真的接了
      ProgressReporter, 且 `update` 是"逐项推进"而不是只在末尾拨一下;
    · manifest_lineage 的逐样本 SHA256 循环真的按比例打点 —— 用注入时钟让
      大数据集才有的中间行在测试里稳定出现（真实集合太小会因 MIN_INTERVAL 闭嘴）。
"""
from __future__ import annotations

import logging
import re
import shutil

import pytest

from od_platform.common.progress import ProgressReporter
from od_platform.data_pipeline import orchestrator
from od_platform.data_validation.checks import manifest_lineage
from od_platform.data_validation.registry import CheckContext, get_check
from od_platform.data_validation.snapshot import build_snapshot

_REPORT_LINE = r"\d+/\d+ 个样本 \(\d+%\) \d+\.\d+/s 剩余约"


class RecordingReporter:
    """把"接了没有 / 怎么接的"记下来的假进度器（不产日志）。"""

    def __init__(self, name, total, *, logger=None, **kwargs):
        self.name = name
        self.total = total
        self.kwargs = kwargs
        self.updates = []
        self.finished = False
        self.done = None
        RecordingReporter.instances.append(self)

    def update(self, done):
        self.updates.append(done)
        return None

    def finish(self, *, done=None):
        self.finished = True
        self.done = done
        return ""


@pytest.fixture()
def recorder(monkeypatch):
    RecordingReporter.instances = []
    monkeypatch.setattr(orchestrator, "ProgressReporter", RecordingReporter)
    return RecordingReporter.instances


# ---------------------------------------------------------------- orchestrator

def test_check_coverage_reports_unknown_total_scan(recorder):
    """清点图像走"总量未知"模式: 逐张推进, 不编百分比。"""
    orchestrator.DatasetPipeline("demo", "pascal_voc")._check_coverage()
    (progress,) = recorder
    assert progress.name == "清点图像"
    assert progress.total == 0                       # 总量未知
    assert progress.kwargs.get("unit") == "张图"
    assert progress.updates == list(range(1, len(progress.updates) + 1))
    assert progress.finished and progress.done == len(progress.updates) > 0


def test_scan_reports_both_image_and_label_loops(recorder, demo_yaml, tmp_path):
    """_scan 的两段循环各接一条进度, 且 update 覆盖每一个条目（含非图条目）。"""
    staging = tmp_path / "staging"
    staging.mkdir()
    for lbl in orchestrator.paths.dataset_processed_dir("demo").glob("labels/*/*.txt"):
        shutil.copy2(lbl, staging / lbl.name)

    pipeline = orchestrator.DatasetPipeline("demo", "pascal_voc")
    image_index, labels_per_image, label_bytes = pipeline._scan(staging, ["c0", "c1"])

    assert image_index and labels_per_image
    names = [p.name for p in recorder]
    assert names == ["扫描图像目录", "读取标签"]
    images_progress, labels_progress = recorder
    assert images_progress.total > 0 and labels_progress.total == len(label_bytes)
    assert images_progress.updates == list(range(1, images_progress.total + 1))
    assert labels_progress.updates == list(range(1, labels_progress.total + 1))
    assert images_progress.finished and images_progress.done == images_progress.total
    assert labels_progress.finished and labels_progress.done == labels_progress.total


# ---------------------------------------------------------------- manifest_lineage

def test_manifest_lineage_emits_intermediate_progress_for_large_dataset(
        demo_yaml, caplog, monkeypatch):
    """63 个真实样本 + 注入时钟: 中间进度行必须真的出现（含百分比与速率）。"""
    ticker = {"t": 0.0}

    def clock():                    # 每次读数走 1s → 触发 MIN_INTERVAL_SEC
        ticker["t"] += 1.0
        return ticker["t"]

    def factory(name, total, *, logger=None, **kwargs):
        return ProgressReporter(name, total, logger=logger, clock=clock, **kwargs)

    monkeypatch.setattr(manifest_lineage, "ProgressReporter", factory)
    ctx = CheckContext(yaml_path=demo_yaml, snapshot=build_snapshot(demo_yaml))
    with caplog.at_level(logging.INFO):
        result = get_check("manifest_lineage").func(ctx)

    lines = [r.getMessage() for r in caplog.records]
    intermediate = [l for l in lines if re.search(_REPORT_LINE, l)]
    assert len(intermediate) >= 2, lines
    assert any("完成 " in l and "个样本" in l for l in lines)     # 收尾行
    assert result.name == "manifest_lineage"


def test_manifest_lineage_small_dataset_only_emits_finish(demo_yaml, caplog, monkeypatch):
    """样本数低于阈值时中间一行都不打, 但收尾行永远有 —— 与 progress.py 的口径一致。"""
    ticker = {"t": 0.0}

    def clock():
        ticker["t"] += 1.0
        return ticker["t"]

    def factory(name, total, *, logger=None, **kwargs):
        return ProgressReporter(name, 10, logger=logger, clock=clock, **kwargs)  # 装成只有 10 个

    monkeypatch.setattr(manifest_lineage, "ProgressReporter", factory)
    ctx = CheckContext(yaml_path=demo_yaml, snapshot=build_snapshot(demo_yaml))
    with caplog.at_level(logging.INFO):
        get_check("manifest_lineage").func(ctx)

    lines = [r.getMessage() for r in caplog.records]
    assert not [l for l in lines if re.search(_REPORT_LINE, l)], lines
    assert any("完成 " in l for l in lines)