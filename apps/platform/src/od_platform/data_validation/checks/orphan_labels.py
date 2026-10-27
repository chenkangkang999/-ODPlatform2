# apps/platform/src/od_platform/data_validation/checks/orphan_labels.py
"""check: orphan_labels —— 有标签没图的孤儿 .txt。占比 ≥10% WARNING,有但 <10% INFO。"""
from __future__ import annotations

from typing import List

from od_platform.common.constants import DETAILS_PREVIEW_LIMIT, ORPHAN_LABELS_WARN_RATIO
from od_platform.data_validation.registry import (
    CheckContext, CheckResult, CheckSeverity, check,
)

_NAME = "orphan_labels"


@check(_NAME)
def validate_orphan_labels(ctx: CheckContext) -> CheckResult:
    snap = ctx.snapshot
    # ★ 阈值来自 D4 配置(ctx.settings); 读不到回落 constants(唯一默认值出处)
    warn_ratio = ctx.setting("orphan_labels_warn_ratio", ORPHAN_LABELS_WARN_RATIO)
    preview_limit = ctx.setting("details_preview_limit", DETAILS_PREVIEW_LIMIT)
    orphans: List[dict] = []
    total_labels = 0
    for split in snap.splits:
        image_stems = {img.stem for img in snap.images_per_split.get(split, ())}
        files = snap.label_files_per_split.get(split, ())
        total_labels += len(files)
        for lbl in files:
            if lbl.stem not in image_stems:
                orphans.append({"split": split, "label": str(lbl)})
    if not orphans:
        return CheckResult(_NAME, CheckSeverity.PASS, "无孤儿标签", {})
    ratio = len(orphans) / max(total_labels, 1)
    details = {"n_orphans": len(orphans), "total_label_files": total_labels,
               "orphan_ratio": round(ratio, 4),
               "orphans_preview": orphans[:preview_limit],
               "thresholds": {"warn": warn_ratio}}
    if ratio >= warn_ratio:
        return CheckResult(_NAME, CheckSeverity.WARNING,
                           f"{len(orphans)} 个孤儿标签({ratio:.1%})— 疑似图被删了", details)
    return CheckResult(_NAME, CheckSeverity.INFO,
                       f"{len(orphans)} 个孤儿标签({ratio:.1%})", details)