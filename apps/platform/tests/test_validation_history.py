#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""质检报告历史对比（D4 1.2 第三条）的测试。

守这些事:
    · "新出现的问题"与"已修复的问题"分得清（这是这一项存在的全部意义）;
    · 条目级差集: 整项还没修好、但其中若干条已经消失 / 又冒出新条, 也要看得见;
    · INFO 不算问题（与 registry.passed 同口径, 否则每份不均衡数据集都常年飘红）;
    · 历史定位不能认错数据集、也不能把自己当成上一次;
    · 首次质检优雅降级（无历史可比, 不报错）, 且照样落一份 compare.json。
"""
from __future__ import annotations

import json
from pathlib import Path

from od_platform.data_validation.compare import (
    COMPARE_FILENAME,
    REPORT_FILENAME,
    check_issue_items,
    compare_reports,
    compare_with_history,
    find_previous_report,
    is_problem,
    severity_counts,
    severity_rank,
    write_compare_json,
)
from od_platform.data_validation.registry import CheckSeverity


def make_report(run_id: str, yaml_path: Path, overall: str, results) -> dict:
    """results: [(name, severity)] 或 [(name, severity, details)]。"""
    entries = []
    for item in results:
        name, sev = item[0], item[1]
        details = item[2] if len(item) > 2 else {}
        entries.append({"name": name, "severity": sev,
                        "summary": f"{name} 结论", "details": details})
    return {
        "run_id": run_id,
        "yaml_path": str(yaml_path),
        "overall_severity": overall,
        "results": entries,
    }


def write_report(runs_dir: Path, run_id: str, payload: dict) -> Path:
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / REPORT_FILENAME
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------- 口径

def test_info_is_not_treated_as_a_problem():
    assert is_problem(CheckSeverity.WARNING) and is_problem(CheckSeverity.ERROR)
    assert not is_problem(CheckSeverity.INFO)
    assert not is_problem(CheckSeverity.PASS)


def test_unknown_severity_ranks_lowest_without_exploding():
    assert severity_rank(None) == severity_rank(CheckSeverity.PASS)
    assert severity_rank("NOT_A_SEVERITY") == severity_rank(CheckSeverity.PASS)
    assert severity_rank(CheckSeverity.ERROR) > severity_rank(CheckSeverity.WARNING)


# ---------------------------------------------------------------- 对比（check 级）

def test_compare_classifies_fixed_newly_failing_and_still_failing(tmp_path):
    yaml_path = tmp_path / "data.yaml"
    previous = make_report("20260101-000001", yaml_path, CheckSeverity.WARNING, [
        ("bbox_within_image", CheckSeverity.WARNING),     # → 后来修好了
        ("pair_existence", CheckSeverity.ERROR),          # → 一直没修
        ("class_balance", CheckSeverity.PASS),            # → 后来变差了
    ])
    current = make_report("20260102-000002", yaml_path, CheckSeverity.ERROR, [
        ("bbox_within_image", CheckSeverity.PASS),
        ("pair_existence", CheckSeverity.ERROR),
        ("class_balance", CheckSeverity.WARNING),
    ])
    result = compare_reports(previous, current)

    assert [c.name for c in result.fixed] == ["bbox_within_image"]
    assert [c.name for c in result.newly_failing] == ["class_balance"]
    assert [c.name for c in result.still_failing] == ["pair_existence"]
    assert result.previous_overall == CheckSeverity.WARNING
    assert result.current_overall == CheckSeverity.ERROR
    assert result.improved is False


def test_compare_reports_improvement_when_severity_drops(tmp_path):
    yaml_path = tmp_path / "data.yaml"
    previous = make_report("20260101-000001", yaml_path, CheckSeverity.WARNING,
                           [("bbox_within_image", CheckSeverity.WARNING)])
    current = make_report("20260102-000002", yaml_path, CheckSeverity.INFO,
                          [("bbox_within_image", CheckSeverity.PASS)])
    result = compare_reports(previous, current)
    assert result.improved is True
    assert [c.name for c in result.fixed] == ["bbox_within_image"]


def test_added_check_that_warns_counts_as_a_newly_found_problem(tmp_path):
    """新加进来的检查项一上来就报警 —— 那正是"新发现的问题"。"""
    yaml_path = tmp_path / "data.yaml"
    previous = make_report("20260101-000001", yaml_path, CheckSeverity.PASS, [])
    current = make_report("20260102-000002", yaml_path, CheckSeverity.WARNING,
                          [("image_decodable", CheckSeverity.WARNING),
                           ("yaml_schema", CheckSeverity.PASS)])
    result = compare_reports(previous, current)
    assert result.added_checks == ["image_decodable", "yaml_schema"]
    assert [c.name for c in result.newly_failing] == ["image_decodable"]


def test_removed_checks_are_reported(tmp_path):
    yaml_path = tmp_path / "data.yaml"
    previous = make_report("20260101-000001", yaml_path, CheckSeverity.PASS,
                           [("legacy_check", CheckSeverity.PASS)])
    current = make_report("20260102-000002", yaml_path, CheckSeverity.PASS, [])
    result = compare_reports(previous, current)
    assert result.removed_checks == ["legacy_check"]
    assert result.fixed == [] and result.newly_failing == []


def test_render_lines_mention_the_important_parts(tmp_path):
    yaml_path = tmp_path / "data.yaml"
    previous = make_report("20260101-000001", yaml_path, CheckSeverity.WARNING,
                           [("bbox_within_image", CheckSeverity.WARNING)])
    current = make_report("20260102-000002", yaml_path, CheckSeverity.INFO,
                          [("bbox_within_image", CheckSeverity.PASS)])
    text = "\n".join(compare_reports(previous, current).render_lines())
    assert "20260101-000001" in text and "20260102-000002" in text
    assert "已修复" in text and "bbox_within_image" in text


def test_trend_line_reports_severity_count_movement(tmp_path):
    """趋势行要能回答"较上次：ERROR 2 → 0"这种问题。"""
    yaml_path = tmp_path / "data.yaml"
    previous = make_report("20260101-000001", yaml_path, CheckSeverity.ERROR, [
        ("a", CheckSeverity.ERROR), ("b", CheckSeverity.ERROR),
        ("c", CheckSeverity.WARNING)])
    current = make_report("20260102-000002", yaml_path, CheckSeverity.WARNING, [
        ("a", CheckSeverity.PASS), ("b", CheckSeverity.PASS),
        ("c", CheckSeverity.WARNING)])
    result = compare_reports(previous, current)
    assert result.previous_counts[CheckSeverity.ERROR] == 2
    assert result.current_counts[CheckSeverity.ERROR] == 0
    assert "ERROR 2→0" in result.trend_line()
    assert "ERROR 2→0" in "\n".join(result.render_lines())


# ---------------------------------------------------------------- 对比（条目级）

def test_check_issue_items_extracts_previews_and_kind_maps():
    result = {"name": "manifest_lineage", "severity": "ERROR", "details": {
        "n_changed": 2,
        "changed_preview": ["c_001", "c_002"],
        "missing_preview": [{"stem": "c_003"}],
        "moved_preview": [],
        "thresholds": {"error": 0.5},                    # 不是条目
    }}
    items = check_issue_items(result)
    assert "c_001" in items and "c_002" in items
    assert "stem=c_003" in items
    assert items == sorted(items)                        # 稳定排序
    kinds = check_issue_items({"details": {"error_kinds": {"coord_out_of_range": 3}}})
    assert kinds == ["kind=coord_out_of_range"]


def test_item_level_diff_sees_resolved_and_new_items(tmp_path):
    """整项仍 ERROR（check 级看是"仍存在"）, 但其中 2 条已消失、1 条新增。"""
    yaml_path = tmp_path / "data.yaml"
    previous = make_report("20260101-000001", yaml_path, CheckSeverity.ERROR, [
        ("label_format", CheckSeverity.ERROR,
         {"bad_lines_preview": [{"stem": "a"}, {"stem": "b"}, {"stem": "c"}]})])
    current = make_report("20260102-000002", yaml_path, CheckSeverity.ERROR, [
        ("label_format", CheckSeverity.ERROR,
         {"bad_lines_preview": [{"stem": "a"}, {"stem": "d"}]})])
    result = compare_reports(previous, current)

    assert [c.name for c in result.still_failing] == ["label_format"]     # check 级没变
    assert [i.key for i in result.disappeared_items] == ["stem=b", "stem=c"]
    assert [i.key for i in result.new_items] == ["stem=d"]
    assert [i.key for i in result.persisting_items] == ["stem=a"]
    text = "\n".join(result.render_lines())
    assert "已消失条目" in text and "stem=b" in text
    assert "新增条目" in text and "stem=d" in text


def test_item_diff_ignores_items_of_non_problem_checks(tmp_path):
    """PASS/INFO 的 details 里就算带着历史 preview, 也不算"问题条目"。"""
    yaml_path = tmp_path / "data.yaml"
    previous = make_report("20260101-000001", yaml_path, CheckSeverity.ERROR,
                           [("dup", CheckSeverity.ERROR, {"previews": [{"label": "x"}]})])
    current = make_report("20260102-000002", yaml_path, CheckSeverity.PASS,
                          [("dup", CheckSeverity.PASS, {"previews": []})])
    result = compare_reports(previous, current)
    assert result.disappeared_items == []                # 整项已修复, 不再逐条重复报
    assert [c.name for c in result.fixed] == ["dup"]


def test_comparison_serialises_for_the_report_json(tmp_path):
    yaml_path = tmp_path / "data.yaml"
    previous = make_report("20260101-000001", yaml_path, "WARNING",
                           [("bbox_within_image", "WARNING")])
    current = make_report("20260102-000002", yaml_path, "PASS",
                          [("bbox_within_image", "PASS")])
    payload = compare_reports(previous, current).to_dict()
    assert payload["improved"] is True
    assert payload["fixed"][0]["name"] == "bbox_within_image"
    assert payload["has_history"] is True
    json.dumps(payload)                       # 必须可序列化（要写进 report.json）


# ---------------------------------------------------------------- 历史定位

def test_find_previous_report_picks_the_newest_for_the_same_yaml(tmp_path):
    runs_dir = tmp_path / "runs"
    yaml_path = tmp_path / "data.yaml"
    write_report(runs_dir, "20260101-000001", make_report("20260101-000001", yaml_path, "PASS", []))
    newest = write_report(runs_dir, "20260103-000003",
                          make_report("20260103-000003", yaml_path, "PASS", []))

    found = find_previous_report(yaml_path, runs_dir=runs_dir)
    assert found == newest


def test_find_previous_report_never_returns_the_current_run(tmp_path):
    runs_dir = tmp_path / "runs"
    yaml_path = tmp_path / "data.yaml"
    write_report(runs_dir, "20260101-000001", make_report("20260101-000001", yaml_path, "PASS", []))
    write_report(runs_dir, "20260102-000002", make_report("20260102-000002", yaml_path, "PASS", []))

    found = find_previous_report(yaml_path, exclude_run_id="20260102-000002", runs_dir=runs_dir)
    assert found is not None and found.parent.name == "20260101-000001"


def test_find_previous_report_ignores_other_datasets(tmp_path):
    runs_dir = tmp_path / "runs"
    write_report(runs_dir, "20260101-000001",
                 make_report("20260101-000001", tmp_path / "other.yaml", "PASS", []))
    assert find_previous_report(tmp_path / "data.yaml", runs_dir=runs_dir) is None


def test_find_previous_report_skips_broken_files(tmp_path):
    runs_dir = tmp_path / "runs"
    yaml_path = tmp_path / "data.yaml"
    broken = runs_dir / "20260102-000002"
    broken.mkdir(parents=True)
    (broken / REPORT_FILENAME).write_text("{ 这不是 json", encoding="utf-8")
    good = write_report(runs_dir, "20260101-000001",
                        make_report("20260101-000001", yaml_path, "PASS", []))

    assert find_previous_report(yaml_path, runs_dir=runs_dir) == good


def test_no_history_directory_returns_none(tmp_path):
    assert find_previous_report(tmp_path / "data.yaml", runs_dir=tmp_path / "missing") is None


# ---------------------------------------------------------------- 端到端

def test_compare_with_history_end_to_end(tmp_path):
    runs_dir = tmp_path / "runs"
    yaml_path = tmp_path / "data.yaml"
    write_report(runs_dir, "20260101-000001", make_report(
        "20260101-000001", yaml_path, CheckSeverity.WARNING,
        [("bbox_within_image", CheckSeverity.WARNING)]))
    current = make_report("20260102-000002", yaml_path, CheckSeverity.INFO,
                          [("bbox_within_image", CheckSeverity.PASS)])

    comparison = compare_with_history(yaml_path, current, runs_dir=runs_dir)
    assert comparison is not None
    assert comparison.previous_run_id == "20260101-000001"
    assert [c.name for c in comparison.fixed] == ["bbox_within_image"]


def test_compare_with_history_returns_none_on_first_run(tmp_path):
    runs_dir = tmp_path / "runs"
    yaml_path = tmp_path / "data.yaml"
    current = make_report("20260102-000002", yaml_path, CheckSeverity.PASS, [])
    assert compare_with_history(yaml_path, current, runs_dir=runs_dir) is None


def test_write_compare_json_degrades_gracefully_without_history(tmp_path):
    """首次运行也要落 compare.json, 但如实写 has_history=false, 且不报错。"""
    path = tmp_path / COMPARE_FILENAME
    payload = write_compare_json(path, None, "20260102-000002")
    assert payload["has_history"] is False
    assert payload["previous_run_id"] is None
    assert json.loads(path.read_text(encoding="utf-8"))["has_history"] is False


def test_write_compare_json_writes_the_real_diff(tmp_path):
    yaml_path = tmp_path / "data.yaml"
    previous = make_report("20260101-000001", yaml_path, "ERROR",
                           [("bbox_within_image", "ERROR")])
    current = make_report("20260102-000002", yaml_path, "PASS",
                          [("bbox_within_image", "PASS")])
    comparison = compare_reports(previous, current)
    path = tmp_path / COMPARE_FILENAME
    payload = write_compare_json(path, comparison, "20260102-000002")
    assert payload["has_history"] is True
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["previous_run_id"] == "20260101-000001"
    assert on_disk["fixed"][0]["name"] == "bbox_within_image"


def test_severity_counts_covers_all_four_levels(tmp_path):
    report = make_report("r", tmp_path / "d.yaml", "ERROR", [
        ("a", "ERROR"), ("b", "ERROR"), ("c", "WARNING"),
        ("d", "INFO"), ("e", "PASS")])
    counts = severity_counts(report)
    assert counts == {"ERROR": 2, "WARNING": 1, "INFO": 1, "PASS": 1}


# ---------------------------------------------------------------- 真跑两次 validate

def test_validate_twice_emits_comparison_and_compare_json(tmp_path, monkeypatch, demo_yaml):
    """端到端: 同一份 yaml 连跑两次 validate, 第二次必须带上对比结论与 compare.json。"""
    import shutil

    from od_platform.common import paths
    from od_platform.data_validation import service as svc

    runs_dir = tmp_path / "runs"
    monkeypatch.setattr(paths, "VALIDATION_RUNS_DIR", runs_dir)

    # 用 conftest 的真实 demo 数据集做一份拷贝, 再人为改一个标签制造 manifest_lineage ERROR
    src_yaml = demo_yaml
    import yaml as _yaml

    dst_root = tmp_path / "demo_copy"
    shutil.copytree(paths.dataset_processed_dir("demo"), dst_root)
    doc = _yaml.safe_load(src_yaml.read_text(encoding="utf-8"))
    doc["path"] = dst_root.as_posix()
    yp = tmp_path / "demo_copy.yaml"
    yp.write_text(_yaml.safe_dump(doc, allow_unicode=True, sort_keys=False),
                  encoding="utf-8")

    first = svc.validate_dataset(yp, run_id="20260101-000001")
    assert first.comparison is None
    assert (runs_dir / "20260101-000001" / COMPARE_FILENAME).is_file()
    assert "comparison" not in json.loads(
        (runs_dir / "20260101-000001" / REPORT_FILENAME).read_text(encoding="utf-8"))

    # 人为制造一个问题: 往一个标签里加一行非法内容 → manifest_lineage ERROR
    lbl = sorted((dst_root / "labels" / "train").glob("*.txt"))[0]
    lbl.write_text(lbl.read_text(encoding="utf-8") + "0 0.5 0.5 0.2\n", encoding="utf-8")

    second = svc.validate_dataset(yp, run_id="20260102-000002")
    assert second.comparison is not None
    assert second.comparison.previous_run_id == "20260101-000001"
    assert any(c.name == "manifest_lineage" for c in second.comparison.newly_failing)

    payload = json.loads(
        (runs_dir / "20260102-000002" / REPORT_FILENAME).read_text(encoding="utf-8"))
    assert payload["comparison"]["previous_run_id"] == "20260101-000001"
    compare_payload = json.loads(
        (runs_dir / "20260102-000002" / COMPARE_FILENAME).read_text(encoding="utf-8"))
    assert compare_payload["has_history"] is True
    newly = {c["name"] for c in compare_payload["newly_failing"]}
    assert "manifest_lineage" in newly and "label_format" in newly