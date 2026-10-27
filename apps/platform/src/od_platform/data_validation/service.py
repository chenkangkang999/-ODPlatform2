# apps/platform/src/od_platform/data_validation/service.py
"""质检服务层:跑全部 check 并聚合。
唯一的 except Exception 护栏在这里:单 check 崩溃降级为 ERROR 结果,不拖垮整场质检。"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from od_platform.common import paths
from od_platform.common import stamp
from od_platform.common.lineage import compute_data_fingerprint
from od_platform.data_validation import VALIDATOR_VERSION
from od_platform.data_validation.checks._lineage import load_manifest_from_yaml
from od_platform.data_validation.compare import (
    COMPARE_FILENAME, compare_with_history, write_compare_json,
)
from od_platform.data_validation.registry import (
    CheckContext, CheckEntry, CheckResult, CheckSeverity, get_all_checks,
)
from od_platform.data_validation.report import ValidationReport
from od_platform.data_validation.snapshot import build_snapshot

logger = logging.getLogger(__name__)


def _safe_run_one(entry: CheckEntry, ctx: CheckContext) -> CheckResult:
    try:
        return entry.func(ctx)
    except Exception as e:                          # noqa: BLE001 —— 护栏,必须宽
        return CheckResult(entry.name, CheckSeverity.ERROR,
                           f"check 自身崩溃: {type(e).__name__}: {e}",
                           {"exception": repr(e)})


def run_all_checks(ctx: CheckContext) -> List[CheckResult]:
    return [_safe_run_one(e, ctx) for e in get_all_checks()]


def validate_dataset(yaml_path, task_type: Optional[str] = None,
                     run_id: Optional[str] = None,
                     write_report: bool = True) -> ValidationReport:
    """端到端:快照 → 全量 check → 报告(logger + json/csv)。"""
    t0 = time.perf_counter()
    run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    yaml_path = Path(yaml_path)
    snapshot = build_snapshot(yaml_path, task_type=task_type)
    results = run_all_checks(CheckContext(yaml_path=yaml_path, snapshot=snapshot))
    report = ValidationReport(run_id=run_id, snapshot=snapshot, results=results,
                              duration_seconds=time.perf_counter() - t0)
    # ★ 历史对比: 找同一份 yaml 上一次的 report.json 做差集。
    #   必须在写本次 report.json **之前**算 —— 否则会把刚写下的自己当成"上一次"。
    #   没有历史(首次质检)时优雅降级: 不是错误, 只如实说一句"无历史可比"。
    comparison = None
    if write_report:
        try:
            comparison = compare_with_history(yaml_path, report.to_dict(),
                                              exclude_run_id=run_id)
        except Exception as e:                      # noqa: BLE001 —— 对比失败不该影响质检
            logger.warning("历史对比失败, 跳过: %s", e)
            comparison = None
        if comparison is not None:
            report.comparison = comparison
            logger.info("找到同数据集的历史报告 run_id=%s, 已生成对比",
                        comparison.previous_run_id)
        else:
            logger.info("没有同数据集的历史质检报告, 本次为首次质检(无历史可比)")
    report.render_to_logger()
    if write_report:
        out_dir = paths.validation_run_dir(run_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        report.write_json(out_dir / "report.json")
        report.write_results_csv(out_dir / "results.csv")
        # 对比结论单独落一份 compare.json: 便于只看 diff、不必翻整份 report。
        write_compare_json(out_dir / COMPARE_FILENAME, comparison, run_id)

    # ★ 盖 validate 章: 我(D4)验过这批数据。仅在【能定位到 manifest】时盖 ——
    #   没有 manifest 的数据集(非 odp-transform 产出)谈不上流转卡, 跳过。
    manifest, _err, _odp = load_manifest_from_yaml(snapshot.yaml_data)
    if manifest is not None:
        # data_fingerprint 从【manifest.samples 自己算】—— 不读 D3 的章、不抄它的指纹
        data_fp = compute_data_fingerprint(manifest.samples)
        stamp.write_stamp(
            stage_file=stamp.STAMP_VALIDATE,
            data_fingerprint=data_fp,
            run_id=run_id,
            tool_version=VALIDATOR_VERSION,
            payload={
                "contract_fingerprint": manifest.contract_fingerprint,
                "overall_severity": report.overall_severity,
                "n_checks": len(report.results),
                "report_ref": f"runs/data_validation/{run_id}/report.json",
            },
        )
    return report