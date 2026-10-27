#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : compare.py
# @Project   : ODPlatform
# @Function  : 质检报告历史对比（本次 vs 上次: 新增了什么、修好了什么、还剩下什么）

"""质检报告历史对比。

★ 解决什么问题

    单次报告只回答"现在有没有问题", 回答不了**"这一次比上一次好了还是坏了"**。
    修数据的人最想知道的就是这两件事: 我刚修的那个问题真的消失了吗? 有没有
    顺手引入新的? 此前每次质检都是孤立的一跑, 这些信息只能靠人肉 diff 两份 json。

★ 设计取舍

    · **两层口径: check 级 + 条目级**。
      check 级用 severity 的变化判定"整项修好了/整项新坏了"——severity 是所有
      check 的共同语言, 不依赖任何 check 的 details 结构;
      条目级把 details 里逐条列出的证据（``changed_preview`` / ``bad_lines_preview``
      / ``findings`` / ``collisions`` / ``*_kinds`` ...）抽成稳定 key 做集合差集,
      回答"整项还没修好、但其中 3 条已经消失了"这种 check 级看不见的问题。
    · **条目的身份是 (check 名, 条目 key)**, severity 只作为展示/筛选用途, 不参与
      条目匹配 —— 否则同一批条目在整项降级（ERROR→WARNING）时会被误报成"全消失
      又全新出现"。
    · **历史报告靠 ``yaml_path`` 精确匹配**（报告里没记数据集名）。同一份数据换个
      路径会被当成不同数据集 —— 这是有意的: 路径变了, 数据就不是同一批。
    · INFO 不算"问题": 与 registry 的定义保持一致（``passed = severity in
      (PASS, INFO)``）。把 INFO 当问题会让每份不均衡数据集的报告都常年飘红。
    · 条目 key 来自各 check 的 preview —— preview 是**截断**的（各 check 有各自的
      上限）。所以条目级差集是"已列出的那些条目"的差集, 不是全量; 报告里如实
      标注 ``items_truncated`` 的可能。check 级的判定不受影响。
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from od_platform.common import paths
from od_platform.data_validation.registry import CheckSeverity

logger = logging.getLogger(__name__)

REPORT_FILENAME = "report.json"
# 对比结论的落盘文件名（放在本次 run 目录, 与 report.json 并列）
COMPARE_FILENAME = "compare.json"
# "算作问题"的严重级: INFO 是提示不是问题（与 registry.passed 同一口径）
PROBLEM_SEVERITIES = (CheckSeverity.WARNING, CheckSeverity.ERROR)

# 条目 key 的优先标识字段: 命中就用它, 让条目身份不随计数/描述文字漂移
_ITEM_ID_FIELDS = ("stem", "file", "label", "expect", "kind", "split", "line_no")
# 计数型的 kinds 映射也逐条算问题条目（label_format 的 error_kinds 等）
_KIND_MAP_SUFFIXES = ("_kinds", "kinds")


def severity_rank(severity: Optional[str]) -> int:
    """严重级排序; 未知/缺失按最低处理（不炸）。"""
    try:
        return CheckSeverity.rank(severity)
    except KeyError:
        return CheckSeverity.rank(CheckSeverity.PASS)


def is_problem(severity: Optional[str]) -> bool:
    return severity in PROBLEM_SEVERITIES


def severity_counts(report: Dict[str, Any]) -> Dict[str, int]:
    """统计一份报告里各严重级的 check 数（趋势行用: "ERROR 2 → 0"）。"""
    counts = {CheckSeverity.ERROR: 0, CheckSeverity.WARNING: 0,
              CheckSeverity.INFO: 0, CheckSeverity.PASS: 0}
    for result in _by_name(report).values():
        severity = result.get("severity")
        if severity in counts:
            counts[severity] += 1
    return counts


@dataclass(frozen=True)
class CheckChange:
    """单个检查项在两份报告之间的变化（check 级）。"""
    name: str
    previous: Optional[str]
    current: Optional[str]
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "previous": self.previous,
                "current": self.current, "summary": self.summary}


@dataclass(frozen=True)
class ItemChange:
    """单个具体条目在两份报告之间的变化（条目级）。"""
    check: str
    severity: str
    key: str

    def to_dict(self) -> Dict[str, Any]:
        return {"check": self.check, "severity": self.severity, "key": self.key}


@dataclass(frozen=True)
class Comparison:
    """两份质检报告的对比结论。"""
    previous_run_id: str
    current_run_id: str
    previous_overall: str
    current_overall: str
    newly_failing: List[CheckChange] = field(default_factory=list)   # 新增问题
    fixed: List[CheckChange] = field(default_factory=list)           # 已修复问题
    still_failing: List[CheckChange] = field(default_factory=list)   # 仍存在问题
    added_checks: List[str] = field(default_factory=list)
    removed_checks: List[str] = field(default_factory=list)
    new_items: List[ItemChange] = field(default_factory=list)            # 新增条目
    disappeared_items: List[ItemChange] = field(default_factory=list)    # 已消失条目
    persisting_items: List[ItemChange] = field(default_factory=list)     # 仍存在条目
    previous_counts: Dict[str, int] = field(default_factory=dict)
    current_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def improved(self) -> bool:
        """整体变好: 分数下降, 或者修好的比新出的多。"""
        before = severity_rank(self.previous_overall)
        after = severity_rank(self.current_overall)
        if after != before:
            return after < before
        return len(self.fixed) > len(self.newly_failing)

    def trend_line(self) -> str:
        """趋势行: "较上次问题计数: ERROR 2→0, WARNING 1→1"。"""
        parts = [f"{name} {self.previous_counts.get(name, 0)}"
                 f"→{self.current_counts.get(name, 0)}"
                 for name in PROBLEM_SEVERITIES]
        return "较上次问题计数: " + ", ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_history": True,
            "previous_run_id": self.previous_run_id,
            "current_run_id": self.current_run_id,
            "previous_overall": self.previous_overall,
            "current_overall": self.current_overall,
            "improved": self.improved,
            "previous_counts": dict(self.previous_counts),
            "current_counts": dict(self.current_counts),
            "trend": self.trend_line(),
            "newly_failing": [c.to_dict() for c in self.newly_failing],
            "fixed": [c.to_dict() for c in self.fixed],
            "still_failing": [c.to_dict() for c in self.still_failing],
            "added_checks": self.added_checks,
            "removed_checks": self.removed_checks,
            "new_items": [i.to_dict() for i in self.new_items],
            "disappeared_items": [i.to_dict() for i in self.disappeared_items],
            "persisting_items": [i.to_dict() for i in self.persisting_items],
        }

    def render_lines(self) -> List[str]:
        """给人看的对照表（CLI/日志直接打这些行）。"""
        lines = [
            f"与上次对比(run {self.previous_run_id} → {self.current_run_id}): "
            f"总评 {self.previous_overall} → {self.current_overall} "
            f"({'改善' if self.improved else '未改善'})",
            "  " + self.trend_line(),
        ]
        if self.fixed:
            lines.append("  [已修复] " + ", ".join(
                f"{c.name}({c.previous}→{c.current})" for c in self.fixed))
        if self.newly_failing:
            lines.append("  [新增问题] " + ", ".join(
                f"{c.name}({c.previous}→{c.current})" for c in self.newly_failing))
        if self.still_failing:
            lines.append("  [仍存在] " + ", ".join(
                f"{c.name}({c.previous}→{c.current})" for c in self.still_failing))
        if self.disappeared_items:
            lines.append(f"  [已消失条目] {len(self.disappeared_items)} 条: "
                         + ", ".join(f"{i.check}/{i.key}" for i in self.disappeared_items))
        if self.new_items:
            lines.append(f"  [新增条目] {len(self.new_items)} 条: "
                         + ", ".join(f"{i.check}/{i.key}" for i in self.new_items))
        if self.persisting_items:
            lines.append(f"  [仍存在条目] {len(self.persisting_items)} 条: "
                         + ", ".join(f"{i.check}/{i.key}" for i in self.persisting_items))
        if self.added_checks:
            lines.append(f"  新增检查项: {', '.join(self.added_checks)}")
        if self.removed_checks:
            lines.append(f"  移除检查项: {', '.join(self.removed_checks)}")
        if not (self.fixed or self.newly_failing or self.still_failing
                or self.new_items or self.disappeared_items):
            lines.append("  两次报告的问题集合没有变化")
        return lines


# ---------------------------------------------------------------- 条目抽取

def _item_key(item: Any) -> Optional[str]:
    """把一条证据压成一个稳定 key。

    优先用标识字段（stem/file/label/expect/kind/split/line_no）——它们才是"这条
    问题是谁"; 没有标识字段才退回 message / 整个 dict 的规范化 json。
    """
    if item is None:
        return None
    if isinstance(item, str):
        return item or None
    if isinstance(item, dict):
        if not item:
            return None
        picked = {k: item[k] for k in _ITEM_ID_FIELDS if k in item}
        if not picked:
            if "message" in item:
                picked = {"message": item["message"]}
            else:
                picked = item
        parts = []
        for key in sorted(picked):
            value = picked[key]
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False, sort_keys=True)
            parts.append(f"{key}={value}")
        return "; ".join(parts)
    return str(item)


def check_issue_items(result: Dict[str, Any]) -> List[str]:
    """从一条 check 结果里抽出"具体条目"key（去重 + 排序）。

    规则（对所有 check 一视同仁, 不写脆弱特例）:
        · details 里任何 **list** 值都当成逐条证据（``*_preview`` / ``findings``
          / ``collisions`` ...皆然）;
        · details 里以 ``kinds`` 结尾的 **dict** 当成"类别 → 计数", 每个类别算一条。
    """
    details = result.get("details")
    if not isinstance(details, dict):
        return []
    keys: Set[str] = set()
    for name, value in details.items():
        if isinstance(value, list):
            for item in value:
                key = _item_key(item)
                if key:
                    keys.add(key)
        elif isinstance(value, dict) and str(name).endswith(_KIND_MAP_SUFFIXES):
            for kind in value:
                keys.add(f"kind={kind}")
    return sorted(keys)


def report_issue_items(report: Dict[str, Any]) -> Dict[str, List[str]]:
    """一份报告 → {check 名: [条目 key]}, 只保留报了问题的 check。"""
    items: Dict[str, List[str]] = {}
    for name, result in _by_name(report).items():
        if not is_problem(result.get("severity")):
            continue
        found = check_issue_items(result)
        if found:
            items[name] = found
    return items


# ---------------------------------------------------------------- 历史定位

def load_report(path: Path) -> Optional[Dict[str, Any]]:
    """读一份 report.json; 读不了或结构不对返回 None（历史对比失败不该影响本次质检）。"""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("历史报告读取失败, 跳过对比: %s (%s)", path, exc)
        return None
    return data if isinstance(data, dict) else None


def find_previous_report(yaml_path: Path, *, exclude_run_id: Optional[str] = None,
                         runs_dir: Optional[Path] = None) -> Optional[Path]:
    """找同一份数据集（同一 yaml）上一次的质检报告。

    Args:
        yaml_path: 本次质检的数据集 yaml
        exclude_run_id: 本次的 run_id（绝不能把自己当成上一次）
        runs_dir: 报告根目录; None 时用 paths.VALIDATION_RUNS_DIR

    Returns:
        上一份 report.json 的路径; 没有历史返回 None
    """
    runs_dir = Path(runs_dir) if runs_dir else paths.VALIDATION_RUNS_DIR
    if not runs_dir.is_dir():
        return None
    target = Path(yaml_path).resolve()
    # run_id 形如 20261026-195456, 字典序即时间序
    for run_dir in sorted((d for d in runs_dir.iterdir() if d.is_dir()), reverse=True):
        if exclude_run_id and run_dir.name == exclude_run_id:
            continue
        candidate = run_dir / REPORT_FILENAME
        if not candidate.is_file():
            continue
        data = load_report(candidate)
        if not data:
            continue
        recorded = data.get("yaml_path")
        if not recorded:
            continue
        try:
            if Path(recorded).resolve() == target:
                return candidate
        except OSError:                      # 路径在当前机器上无法解析
            continue
    return None


# ---------------------------------------------------------------- 对比

def _by_name(report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    results = report.get("results")
    if not isinstance(results, list):
        return {}
    return {str(r.get("name")): r for r in results if isinstance(r, dict) and r.get("name")}


def _item_changes(check: str, severity: str, keys: Iterable[str]) -> List[ItemChange]:
    return [ItemChange(check=check, severity=severity, key=key) for key in sorted(keys)]


def compare_reports(previous: Dict[str, Any], current: Dict[str, Any]) -> Comparison:
    """对比两份报告, 给出新增/已修复/仍存在的问题（check 级 + 条目级）。"""
    before, after = _by_name(previous), _by_name(current)
    newly_failing: List[CheckChange] = []
    fixed: List[CheckChange] = []
    still_failing: List[CheckChange] = []

    for name, current_result in sorted(after.items()):
        current_sev = current_result.get("severity")
        summary = str(current_result.get("summary", ""))
        previous_result = before.get(name)
        if previous_result is None:
            # 新加进来的检查项: 如果它一上来就报警, 那是"新发现的问题"
            if is_problem(current_sev):
                newly_failing.append(CheckChange(name, None, current_sev, summary))
            continue
        previous_sev = previous_result.get("severity")
        if severity_rank(current_sev) > severity_rank(previous_sev):
            newly_failing.append(CheckChange(name, previous_sev, current_sev, summary))
        elif severity_rank(current_sev) < severity_rank(previous_sev) and is_problem(previous_sev):
            fixed.append(CheckChange(name, previous_sev, current_sev, summary))
        elif is_problem(current_sev):
            still_failing.append(CheckChange(name, previous_sev, current_sev, summary))

    # ---- 条目级差集 ----
    # 只对"两次都还是问题"的 check 做逐条差集: 整项已修复/整项新坏的, check 级
    # 那两行已经说清楚了, 再逐条列一遍只是噪音。
    before_items = report_issue_items(previous)
    after_items = report_issue_items(current)
    new_items: List[ItemChange] = []
    disappeared_items: List[ItemChange] = []
    persisting_items: List[ItemChange] = []
    still_failing_names = {c.name for c in still_failing}
    for name in sorted((set(before_items) | set(after_items)) & still_failing_names):
        prev_keys = set(before_items.get(name, ()))
        curr_keys = set(after_items.get(name, ()))
        current_sev = str(after.get(name, {}).get("severity")
                          or before.get(name, {}).get("severity") or CheckSeverity.PASS)
        new_items += _item_changes(name, current_sev, curr_keys - prev_keys)
        disappeared_items += _item_changes(name, current_sev, prev_keys - curr_keys)
        persisting_items += _item_changes(name, current_sev, curr_keys & prev_keys)

    return Comparison(
        previous_run_id=str(previous.get("run_id", "?")),
        current_run_id=str(current.get("run_id", "?")),
        previous_overall=str(previous.get("overall_severity", CheckSeverity.PASS)),
        current_overall=str(current.get("overall_severity", CheckSeverity.PASS)),
        newly_failing=newly_failing, fixed=fixed, still_failing=still_failing,
        added_checks=sorted(set(after) - set(before)),
        removed_checks=sorted(set(before) - set(after)),
        new_items=new_items, disappeared_items=disappeared_items,
        persisting_items=persisting_items,
        previous_counts=severity_counts(previous),
        current_counts=severity_counts(current),
    )


def compare_with_history(yaml_path: Path, current_report: Dict[str, Any], *,
                         exclude_run_id: Optional[str] = None,
                         runs_dir: Optional[Path] = None) -> Optional[Comparison]:
    """端到端: 找到同一数据集的上一次报告并对比; 没有历史返回 None。"""
    previous_path = find_previous_report(
        yaml_path, exclude_run_id=exclude_run_id or current_report.get("run_id"),
        runs_dir=runs_dir)
    if previous_path is None:
        return None
    previous = load_report(previous_path)
    if previous is None:
        return None
    return compare_reports(previous, current_report)


def write_compare_json(path: Path, comparison: Optional[Comparison],
                       current_run_id: str) -> Dict[str, Any]:
    """把对比结论落进本次 run 目录。

    没有历史（首次质检）也落一份, 但如实写 ``has_history: false`` —— 让"本次没有
    历史可比"本身成为可追溯的产物, 而不是一个"文件不存在"的沉默。
    """
    if comparison is not None:
        payload: Dict[str, Any] = comparison.to_dict()
    else:
        payload = {
            "has_history": False,
            "previous_run_id": None,
            "current_run_id": current_run_id,
            "note": "没有同数据集的历史质检报告, 本次为首次质检(无历史可比)",
        }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload