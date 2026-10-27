# apps/platform/src/od_platform/data_validation/checks/pair_existence.py
"""check: pair_existence —— 每张图都有对应标签吗(按缺失比例分级:≥50% ERROR / ≥5% WARNING)。"""
from __future__ import annotations

from typing import List

from od_platform.common.constants import (
    PAIR_MAX_DETAIL, PAIR_MISSING_ERROR_RATIO, PAIR_MISSING_WARN_RATIO,
)
from od_platform.data_validation.registry import (
    CheckContext, CheckResult, CheckSeverity, check,
)

_NAME = "pair_existence"


@check(_NAME)
def validate_pair_existence(ctx: CheckContext) -> CheckResult:
    snap = ctx.snapshot
    # ★ 阈值来自 D4 配置(ctx.settings); 读不到回落 constants(唯一默认值出处)
    max_detail = ctx.setting("pair_max_detail", PAIR_MAX_DETAIL)
    error_ratio = ctx.setting("pair_missing_error_ratio", PAIR_MISSING_ERROR_RATIO)
    warn_ratio = ctx.setting("pair_missing_warn_ratio", PAIR_MISSING_WARN_RATIO)
    missing: List[dict] = []
    total = 0
    for split in snap.splits:
        for lbl in snap.labels_per_split[split]:
            total += 1
            if not lbl.exists():
                missing.append({"split": split, "expect": str(lbl)})
    if total == 0:
        return CheckResult(_NAME, CheckSeverity.WARNING, "数据集没有任何图像", {})
    ratio = len(missing) / total
    details = {"total_images": total, "missing": len(missing),
               "missing_ratio": round(ratio, 4),
               "missing_preview": missing[:max_detail],
               "thresholds": {"error": error_ratio,
                              "warn": warn_ratio}}
    if ratio >= error_ratio:
        return CheckResult(_NAME, CheckSeverity.ERROR,
                           f"{len(missing)}/{total} 张图缺标签({ratio:.1%})— 疑似导出中断", details)
    if ratio >= warn_ratio:
        return CheckResult(_NAME, CheckSeverity.WARNING,
                           f"{len(missing)}/{total} 张图缺标签({ratio:.1%})", details)
    if missing:
        return CheckResult(_NAME, CheckSeverity.INFO,
                           f"{len(missing)}/{total} 张图缺标签({ratio:.1%})— 多为无目标图,确认即可",
                           details)
    return CheckResult(_NAME, CheckSeverity.PASS, f"全部 {total} 张图均有标签", details)