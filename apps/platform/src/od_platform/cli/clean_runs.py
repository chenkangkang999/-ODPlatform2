#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : clean_runs.py
# @Author    : ODPlatform Team
# @Project   : ODPlatform
# @Function  : odp-clean —— 按保留策略清理 runs/ 下的历史现场

"""odp-clean: 清理 ``runs/`` 下的历史运行现场。

★ 默认是**预演**: 只打印"打算删哪些、能释放多少", 不动盘; 真的要删必须显式
  ``--yes``。这是本仓库对破坏性操作的统一态度（``odp-reset`` 同理）。
  ``--apply`` 是参考实现里的旧名, 保留为 ``--yes`` 的同义开关。

★ ``runs/lineage/`` 永远豁免: 那是 D3/D4/D6 盖的章, 删了数据身份核验就对不上账,
  而且无法重建。它不靠"默认参数恰好没选到", 而是硬编码在保留策略里。

★ 策略来源(配置化):configs/runtime/clean.yaml(odp-gen-config clean 生成),
  三源合并 CLI > YAML > DEFAULT —— 显式给出的 --keep-latest / --max-age-days /
  --protect 永远赢 yaml。命令行参数一个都没给、也没有 clean.yaml 时, 默认值来自
  common/retention.py 的 DEFAULT_*(唯一出处, 由 CleanSettings 引用)。

退出码: 0=成功; 1=用法/策略参数错误。
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from od_platform.common import paths
from od_platform.common.logging_utils import get_logger
from od_platform.common.retention import (
    PROTECTED_SUBDIRS,
    RetentionPolicy,
    apply_cleanup,
    plan_cleanup,
    summarize_runs,
)
from od_platform.runtime_config.clean import build_clean_config
from od_platform.runtime_config.registry import resolve_config_ref
from od_platform.runtime_config.report import render_effective

EXIT_OK = 0
EXIT_USAGE = 1

_MB = 1048576.0

#: 配置来源的溯源日志走模块 logger —— 预演阶段不该往 logging/ 落文件
#  (真删除时才 get_logger 挂 handler, 与改动前的副作用面保持一致)
_log = logging.getLogger(__name__)

#: 不进配置字典的命令行开关(它们是"怎么跑这条命令", 不是保留策略)
_CLI_CONTROL = {"yes", "summary", "runs_dir", "config_ref"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="odp-clean",
        description="按保留策略清理 runs/ 下的历史运行现场（默认只预演, 不删除）")
    # --yes 是本仓库破坏性操作的统一确认开关（见 odp-reset）;
    # --apply 保留为参考实现的旧名, 两者等价, 只留一个即可。
    parser.add_argument("--yes", "--apply", dest="yes", action="store_true",
                        help="真的执行删除（不加则只预演）")
    # 策略字段 default 一律 None: 默认值唯一真相是 CleanSettings 的 Field(default=...)
    # (它又引用 retention 的 DEFAULT_*);此时 yaml 才有机会生效。
    parser.add_argument("--keep-latest", type=int, default=None,
                        help="每个子目录至少保留最新的 N 个（覆盖 yaml; 默认取配置）")
    parser.add_argument("--max-age-days", type=float, default=None,
                        help="这个天数内的一律不动（覆盖 yaml; 默认取配置）")
    parser.add_argument("--protect", nargs="*", default=None,
                        help="额外豁免的子目录名（空格分隔; 覆盖 yaml）")
    parser.add_argument("--yaml", dest="config_ref", default=None,
                        help="保留策略配置(名字 / 路径; 默认 configs/runtime/clean.yaml)")
    parser.add_argument("--summary", action="store_true",
                        help="只打印现状概览, 不计算清理计划")
    parser.add_argument("--runs-dir", default=None,
                        help="运行现场根目录（默认为仓库 runs/; 便于测试/离线巡检）")
    return parser


def _collect_cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """用户真正显式给出的策略字段。没给 -> None -> 不覆盖 yaml(0 是显式值, 保留)。"""
    return {k: v for k, v in vars(args).items()
            if v is not None and k not in _CLI_CONTROL}


def _print_summary(summary: dict) -> None:
    print(f"运行现场根目录: {summary['runs_dir']}")
    print(f"当前总占用: {summary['total_bytes'] / _MB:.1f} MB")
    for name, info in summary["groups"].items():
        mark = "   [豁免]" if info["protected"] else ""
        print(f"  {name:<18} {info['subdirs']:>4} 个现场  "
              f"{info['size_bytes'] / _MB:>9.1f} MB{mark}")


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cli_overrides = _collect_cli_overrides(args)

    # 策略: --yaml(名字/路径) → 三源合并(CLI > YAML > DEFAULT) → CleanSettings → 策略对象
    try:
        settings, merger = build_clean_config(
            yaml_path=resolve_config_ref(args.config_ref, "clean"),
            cli_args=cli_overrides or None,
        )
        policy = RetentionPolicy.from_settings(settings)
    except FileNotFoundError as exc:
        print(f"配置文件不存在: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except ValueError as exc:                      # 含 pydantic 的 ValidationError
        print(f"策略参数无效: {exc}", file=sys.stderr)
        return EXIT_USAGE

    _log.info("启动 odp-clean | CLI 显式给出: %s", sorted(cli_overrides))
    for line in render_effective(settings, merger):
        _log.info(line)

    runs_dir = paths.RUNS_DIR if not args.runs_dir else Path(args.runs_dir)
    _print_summary(summarize_runs(runs_dir))
    if args.summary:
        return EXIT_OK

    print(f"\n豁免（永不清理）: {', '.join(PROTECTED_SUBDIRS)}")
    print(f"策略: 每组保留最新 {policy.keep_latest_per_group} 个, "
          f"且 {policy.max_age_days:g} 天内的不动\n")
    if policy.extra_protected:
        print(f"额外豁免: {', '.join(policy.extra_protected)}\n")

    plan = plan_cleanup(runs_dir, policy=policy)
    for line in plan.render_lines():
        print(line)

    if not args.yes:
        print("\n[预演] 没有删除任何东西。确认无误后加 --yes 执行。")
        return EXIT_OK

    # 真删除才挂仓库日志 handler(与改动前一致: 预演不落日志文件)
    logger = get_logger(base_path=paths.LOGGING_DIR, log_type="clean")
    logger.info("odp-clean 执行: policy=%s plan=%d 个",
                policy, len(plan.candidates))
    result = apply_cleanup(plan, runs_dir=runs_dir)
    logger.info("清理完成: %s", result)
    print(f"\n已清理 {result['removed']} 个现场, "
          f"释放 {result['freed_bytes'] / _MB:.1f} MB"
          + (f", 跳过 {result['skipped']} 个（见日志）" if result["skipped"] else ""))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())