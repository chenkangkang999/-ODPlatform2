# apps/platform/src/od_platform/cli/transform_data.py
"""odp-transform:数据流水线 CLI。

★ 阈值来源(D3 配置化):configs/runtime/transform.yaml(odp-gen-config transform 生成),
  三源合并 CLI > YAML > DEFAULT —— 显式给出的命令行参数永远赢 yaml。
  --yaml 不传时:若 configs/runtime/transform.yaml 存在就用它,不存在就全用默认值
  (默认值的唯一出处是 common/constants.py,由 TransformSettings 引用)。

退出码:0=成功;1=数据/参数错误(脏数据、覆盖率不足等);2=用法错误(argparse)。
"""
from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List, Optional

from od_platform.common import paths
from od_platform.common.constants import AnnotationFormat, SplitStrategy, Task
from od_platform.common.logging_utils import get_logger
from od_platform.common.run_context import RunContext
from od_platform.data_pipeline.orchestrator import DatasetPipeline
from od_platform.runtime_config.registry import resolve_config_ref
from od_platform.runtime_config.report import render_effective
from od_platform.runtime_config.transform import build_transform_config

EXIT_OK = 0
EXIT_DATA_ERR = 1
EXIT_USAGE = 2

#: 不进配置字典的命令行开关(它们是"怎么跑这条命令", 不是"数据怎么处理")
_CLI_CONTROL = {"dataset", "format", "task", "yaml_path"}

logger = get_logger(base_path=paths.LOGGING_DIR, log_type="transform")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="odp-transform",
        description="把原始数据集转换为可训练的 YOLO 成品(转换+划分+物化+契约)")
    p.add_argument("--dataset", required=True, help="数据集名(data/raw/<name>)")
    p.add_argument("--format", required=True, choices=AnnotationFormat.all())
    p.add_argument("--task", default=Task.DETECT, choices=Task.all())
    # 配置字段 default 一律不给(全 None): 默认值唯一真相是 TransformSettings 的
    # Field(default=...)(它又引用 constants);argparse 再给一份会永远盖掉 yaml。
    p.add_argument("--yaml", dest="yaml_path", default=None,
                   help="运行配置(名字 / 路径; 默认 configs/runtime/transform.yaml)")
    p.add_argument("--train-rate", type=float, default=None,
                   help="训练集比例(覆盖 yaml; 默认取配置)")
    p.add_argument("--val-rate", type=float, default=None,
                   help="验证集比例(覆盖 yaml; 默认取配置)")
    p.add_argument("--seed", type=int, default=None,
                   help="划分随机种子(覆盖 yaml 的 random_state)")
    p.add_argument("--split-strategy", default=None, choices=SplitStrategy.all(),
                   help="划分策略(覆盖 yaml; 默认取配置)")
    p.add_argument("--classes", nargs="*", default=None,
                   help="类别白名单(顺序即 class_id);不传=按数据自带;传空=全排除")
    return p


def _collect_cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """用户真正显式给出的字段。没传 -> None -> 不进 dict -> 不覆盖 yaml。"""
    return {k: v for k, v in vars(args).items()
            if v is not None and k not in _CLI_CONTROL}


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    # 1) 配置: --yaml(名字/路径) → 三源合并(CLI > YAML > DEFAULT) → TransformSettings
    cli_overrides = _collect_cli_overrides(args)
    settings, merger = build_transform_config(
        yaml_path=resolve_config_ref(args.yaml_path, "transform"),
        cli_args=cli_overrides or None,
        # --seed 是 CLI 名, random_state 是字段名(改名映射只在这一次声明)
        cli_mapping={"seed": "random_state"},
    )
    logger.info("启动 odp-transform | CLI 显式给出: %s", sorted(cli_overrides))
    for line in render_effective(settings, merger):
        logger.info(line)

    try:
        with RunContext("data_pipeline") as run:
            result = DatasetPipeline(
                args.dataset, args.format, task=args.task,
                settings=settings, run=run,
            ).run()
    except (FileNotFoundError, ValueError) as e:
        logger.error("%s", e)
        return EXIT_DATA_ERR
    logger.info("yaml: %s", result["yaml"])
    logger.info("manifest: %s", result["manifest_path"])
    logger.info("fingerprint: %s", result["contract_fingerprint"])
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())