#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""``odp-clean`` CLI 的测试（清单 2.1）。

守四件事:
    · 默认只预演, 一个字节都不动;
    · 只有显式 ``--yes``（或旧名 ``--apply``）才真删;
    · ``--protect`` 的额外豁免组与 ``runs/lineage/`` 一样不可删;
    · 策略参数非法时以退出码 1 收场, 而不是把 ValueError 抛到用户脸上。
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pytest

from od_platform.cli import clean_runs
from od_platform.cli.clean_runs import EXIT_OK, EXIT_USAGE, main

DAY = 86400.0


@pytest.fixture(autouse=True)
def _no_repo_logging(monkeypatch):
    """``--yes`` 分支会调 get_logger 往仓库 logging/ 写文件; 测试换成哑 logger, 保持无副作用。"""
    monkeypatch.setattr(clean_runs, "get_logger",
                        lambda **kwargs: logging.getLogger("odp-clean-test"))


def make_run(runs_dir: Path, group: str, name: str, *, age_days: float,
             size_bytes: int = 1024, now: float) -> Path:
    path = runs_dir / group / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "artifact.bin").write_bytes(b"x" * size_bytes)
    stamp = now - age_days * DAY
    os.utime(path, (stamp, stamp))
    os.utime(path / "artifact.bin", (stamp, stamp))
    return path


@pytest.fixture()
def now():
    return time.time()


@pytest.fixture()
def scenario(tmp_path: Path, now: float):
    """3 个很旧的 detect_infer 现场 + 1 个很旧的 lineage 现场。"""
    runs = tmp_path / "runs"
    for index in range(3):
        make_run(runs, "detect_infer", f"old{index}", age_days=100.0 + index, now=now)
    make_run(runs, "lineage", "stamp", age_days=500.0, now=now)
    return runs


ARGS = ["--keep-latest", "1", "--max-age-days", "0"]


def test_dry_run_is_the_default_and_deletes_nothing(scenario, capsys):
    code = main(["--runs-dir", str(scenario), *ARGS])
    out = capsys.readouterr().out

    assert code == EXIT_OK
    assert "[预演]" in out and "计划清理 2 个现场" in out
    assert sorted(p.name for p in (scenario / "detect_infer").iterdir()) == \
        ["old0", "old1", "old2"]


def test_yes_really_deletes_and_prints_release_size(scenario, capsys):
    code = main(["--runs-dir", str(scenario), "--yes", *ARGS])
    out = capsys.readouterr().out

    assert code == EXIT_OK
    assert "已清理 2 个现场" in out
    remaining = sorted(p.name for p in (scenario / "detect_infer").iterdir())
    assert remaining == ["old0"]            # keep-latest=1: 最新那个必须留下


def test_apply_is_accepted_as_the_reference_alias(scenario, capsys):
    code = main(["--runs-dir", str(scenario), "--apply", *ARGS])
    out = capsys.readouterr().out

    assert code == EXIT_OK
    assert "已清理 2 个现场" in out


def test_lineage_survives_even_a_forced_clean(scenario, capsys):
    main(["--runs-dir", str(scenario), "--yes", *ARGS])
    capsys.readouterr()

    lineage = scenario / "lineage" / "stamp"
    assert lineage.is_dir() and (lineage / "artifact.bin").is_file()


def test_protect_flag_whitelists_a_group(tmp_path, now, capsys):
    runs = tmp_path / "runs"
    for index in range(3):
        make_run(runs, "golden_set", f"g{index}", age_days=500.0, now=now)
    code = main(["--runs-dir", str(runs), "--yes", *ARGS, "--protect", "golden_set"])
    out = capsys.readouterr().out

    assert code == EXIT_OK
    assert "没有需要清理" in out
    assert len(list((runs / "golden_set").iterdir())) == 3


def test_keep_latest_controls_how_many_survive(tmp_path, now, capsys):
    runs = tmp_path / "runs"
    for index in range(6):
        make_run(runs, "detect_infer", f"run{index}", age_days=100.0 + index, now=now)
    main(["--runs-dir", str(runs), "--yes", "--keep-latest", "4", "--max-age-days", "0"])
    capsys.readouterr()

    remaining = sorted(p.name for p in (runs / "detect_infer").iterdir())
    assert remaining == ["run0", "run1", "run2", "run3"]


def test_max_age_days_keeps_fresh_runs(tmp_path, now, capsys):
    runs = tmp_path / "runs"
    for index in range(3):
        make_run(runs, "detect_infer", f"run{index}", age_days=1.0, now=now)
    main(["--runs-dir", str(runs), "--yes", "--keep-latest", "1", "--max-age-days", "30"])
    out = capsys.readouterr().out

    assert "没有需要清理" in out
    assert len(list((runs / "detect_infer").iterdir())) == 3


def test_summary_only_prints_overview_and_touches_nothing(scenario, capsys):
    code = main(["--runs-dir", str(scenario), "--summary", *ARGS])
    out = capsys.readouterr().out

    assert code == EXIT_OK
    assert "运行现场根目录" in out and "当前总占用" in out
    assert "[豁免]" in out                       # lineage 要被标出来
    assert "计划清理" not in out
    assert len(list((scenario / "detect_infer").iterdir())) == 3


def test_invalid_policy_exits_with_usage_code(scenario, capsys):
    code = main(["--runs-dir", str(scenario), "--keep-latest", "0"])
    captured = capsys.readouterr()

    assert code == EXIT_USAGE
    assert "策略参数无效" in captured.err


def test_missing_runs_dir_is_not_an_error(tmp_path, capsys):
    code = main(["--runs-dir", str(tmp_path / "nope")])
    out = capsys.readouterr().out

    assert code == EXIT_OK
    assert "没有需要清理" in out
