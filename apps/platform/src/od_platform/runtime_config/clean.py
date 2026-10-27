#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : clean.py
# @Author    : 雨霓同学 (ODPlatform team)
# @Project   : ODPlatform
# @Function  : runtime_config 子系统——runs/ 保留策略配置类
"""``runs/`` 保留策略配置: CleanSettings.

对应 ``configs/runtime/clean.yaml`` (odp-gen-config clean 生成)。

★ 覆盖哪些可调项(逐个核对过 common/retention.py + cli/clean_runs.py):
    keep_latest  → RetentionPolicy.keep_latest_per_group(每组至少留最新的几个)
    max_age_days → RetentionPolicy.max_age_days(这个天数内的一律不动)
    protect      → RetentionPolicy.extra_protected(额外豁免的子目录名)

★ 默认值只此一处(代码上可验证):
    默认值【直接引用】common/retention.py 的 DEFAULT_KEEP_LATEST / DEFAULT_MAX_AGE_DAYS
    —— 保留策略的两个数字此前就只在那里声明过一次, 本类只引用、不抄写。
    (为什么不让 retention.py 反过来 import 本类: common/ 是整个仓库的依赖底层,
      "底层依赖子系统"会把分层倒过来。SSoT 放在 retention.py, 配置模型引用它,
      仍然是"代码上可验证的唯一出处"。)

★ protect 的语义(与 retention.PROTECTED_SUBDIRS 的分工):
    PROTECTED_SUBDIRS("lineage")是【硬豁免】, 写死在保留策略里, 任何人都改不了
    —— 删了它 D3/D4/D6 的数据身份核验就再也对不上账。
    protect 只承载【额外】豁免(如 golden_set), 是可配置的那一部分。
    None / [] 都表示"不额外豁免"。

★ 三源合并优先级由 ConfigMerger 保证: CLI > YAML > DEFAULT。
    (--keep-latest / --max-age-days / --protect 依旧是 CLI 侧的显式覆盖, 优先级不变。)
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import Field

from od_platform.common import retention as R
from od_platform.runtime_config.base import SettingsBase


class CleanSettings(SettingsBase):
    """``odp-clean``(runs/ 保留策略)的全部可调项。"""

    keep_latest: int = Field(
        default=R.DEFAULT_KEEP_LATEST,
        ge=1,
        description="每个子目录至少保留最新的 N 个现场",
        json_schema_extra={
            "group":    "保留策略",
            "examples": [1, 5, 10],
            "tips": [
                "按修改时间排序, 最新的 N 个无条件保留(哪怕很旧)",
                "与 max_age_days 是「或」关系: 满足任一条就保留",
                "必须 >= 1(0 没有意义: 等于不留)",
                "命令行等价参数是 --keep-latest",
            ],
            "yaml_comment": "每组至少保留最新的 N 个",
        },
    )

    max_age_days: float = Field(
        default=R.DEFAULT_MAX_AGE_DAYS,
        ge=0.0,
        description="这个天数之内的现场一律不动",
        json_schema_extra={
            "group":    "保留策略",
            "examples": [7.0, 30.0, 90.0],
            "tips": [
                "按修改时间算: 太新的一律不删(跑得多也不会误删最近的产物)",
                "0 表示只按 keep_latest 判定(测试/离线巡检常用)",
                "与 keep_latest 是「或」关系: 满足任一条就保留",
                "命令行等价参数是 --max-age-days",
            ],
            "yaml_comment": "多少天内的现场一律不删",
        },
    )

    protect: Optional[List[str]] = Field(
        default=None,
        description="额外豁免的子目录名(在 lineage 之外)",
        json_schema_extra={
            "group":    "豁免名单",
            "examples": [None, ["golden_set"], ["golden_set", "handcheck"]],
            "tips": [
                "None / []: 只豁免 lineage(硬编码的出身链目录)",
                "list: 这些组名下的现场一律不进清理计划",
                "runs/lineage 无论如何都删不掉 —— 它不靠这里配",
                "命令行等价参数是 --protect",
            ],
            "yaml_comment": "额外豁免的子目录名(None = 不额外豁免)",
        },
    )

    # ============================================================
    # 派生量
    # ============================================================

    @property
    def extra_protected(self) -> tuple:
        """额外豁免名单(元组形态, 直接喂 RetentionPolicy.extra_protected)."""
        return tuple(self.protect or ())


# ============================================================
# 一键 build (与 build_train_config 同构, 走同一套 _build_config 流水线)
# ============================================================

def build_clean_config(
    yaml_path=None,
    cli_args=None,
    *,
    yaml_dir=None,
    cli_exclude=None,
    cli_mapping=None,
    extra_sources=None,
    track_sources: bool = True,
    dry_run: bool = False,
):
    """一键完成 clean 配置流水线: YAML + CLI (+ 自定义源) → 合并 → 验证.

    Args:
        yaml_path:     YAML 文件名 / 路径. None = 不读 yaml, 全用默认值
                       (yaml 引用由 odp-clean --yaml 显式给出.)
        cli_args:      argparse.Namespace 或 dict(None = 没有 CLI 覆盖).
        yaml_dir:      YAML 目录, 默认 paths.RUNTIME_CONFIGS_DIR.
        cli_exclude:   CLI 额外排除字段 (并入 CLILoader.DEFAULT_EXCLUDE).
        cli_mapping:   CLI 参数名映射 (CLI 名 → Pydantic 字段名).
        extra_sources: 额外配置源 [(source_id, dict), ...], 插在 YAML 与 CLI 之间.
        track_sources: 是否追踪溯源 (默认 True).
        dry_run:       True = 只合并不验证, 返回 (None, merger).

    Returns:
        正常模式: (CleanSettings, ConfigMerger)
        dry_run:  (None, ConfigMerger)
    """
    from od_platform.runtime_config import _build_config

    return _build_config(
        CleanSettings,
        yaml_path=yaml_path,
        cli_args=cli_args,
        yaml_dir=yaml_dir,
        cli_exclude=cli_exclude,
        cli_mapping=cli_mapping,
        extra_sources=extra_sources,
        track_sources=track_sources,
        dry_run=dry_run,
    )