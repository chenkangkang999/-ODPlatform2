# apps/platform/src/od_platform/cli/validate_data.py
"""odp-validate:数据质检 CLI(只读,不改任何文件)。

★ 阈值来源(D4 配置化):configs/runtime/validate.yaml(odp-gen-config validate 生成),
  三源合并 CLI > YAML > DEFAULT。16 个 check 统一从 ctx.setting() 读生效值 ——
  settings 优先, 读不到回落 common/constants.py 的常量(默认值的唯一出处)。
  --yaml 不传时:若 configs/runtime/validate.yaml 存在就用它,不存在就全用默认值。

退出码:ERROR→2 / WARNING→1 / PASS·INFO→0。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from od_platform.common import paths
from od_platform.common.constants import Task
from od_platform.common.logging_utils import get_logger
from od_platform.data_validation.registry import CheckSeverity
from od_platform.data_validation.service import validate_dataset
from od_platform.runtime_config.registry import resolve_config_ref
from od_platform.runtime_config.report import render_effective
from od_platform.runtime_config.validate import build_validate_config

_EXIT_CODE = {CheckSeverity.ERROR: 2, CheckSeverity.WARNING: 1,
              CheckSeverity.INFO: 0, CheckSeverity.PASS: 0}

logger = get_logger(base_path=paths.LOGGING_DIR, log_type="validate")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="odp-validate",
                                description="数据集质检(只读):16 项 check,按最严聚合成退出码")
    p.add_argument("yaml_path", help="数据集 yaml(如 configs/datasets/demo.yaml)")
    p.add_argument("--task", default=None, choices=Task.all(),
                   help="不传则取 yaml 里 odp_meta.task")
    p.add_argument("--run-id", default=None)
    p.add_argument("--no-report", action="store_true", help="不写 json/csv 报告")
    # ★ 注意 dest: 位置参数已经叫 yaml_path(那是【数据集】yaml), 配置 yaml 另起一名
    p.add_argument("--yaml", dest="config_ref", default=None,
                   help="质检阈值配置(名字 / 路径; 默认 configs/runtime/validate.yaml)")
    args = p.parse_args(argv)

    # 阈值配置: 三源合并(YAML 目前是唯一可覆盖源; CLI 暂无阈值开关)
    settings, merger = build_validate_config(
        yaml_path=resolve_config_ref(args.config_ref, "validate"))
    for line in render_effective(settings, merger):
        logger.info(line)

    report = validate_dataset(Path(args.yaml_path), task_type=args.task,
                              run_id=args.run_id, write_report=not args.no_report,
                              settings=settings)
    return _EXIT_CODE.get(report.overall_severity, 2)


if __name__ == "__main__":
    sys.exit(main())