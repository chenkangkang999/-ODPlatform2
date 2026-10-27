#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : transform.py
# @Author    : 雨霓同学 (ODPlatform team)
# @Project   : ODPlatform
# @Function  : runtime_config 子系统——D3 数据流水线配置类
"""D3 数据流水线配置: TransformSettings.

对应 ``configs/runtime/transform.yaml`` (odp-gen-config transform 生成)。

★ 覆盖哪些可调项(逐个核对过真实代码, 不多不少):
    覆盖率闸门   → orchestrator.DatasetPipeline._check_coverage
    稀有类预警   → orchestrator 传给 data_pipeline.report.analyze_balance
    划分比例/种子 → orchestrator 传给 data_pipeline.split.split_service
    类别白名单   → orchestrator 传给 convert.registry.ConvertOptions

★ 默认值只此一处(代码上可验证):
    每个字段的默认值【直接引用】common/constants.py 的常量, 不抄字面量。
    于是"默认值"这件事永远只有 constants.py 一个出处 —— 改常量, 配置默认值一起动。

★ 与 D5 的 train/val/infer 配置的分工:
    那些继承 BaseConfig(带 model/batch/imgsz 等 16 个字段);
    本类继承 SettingsBase(0 个业务字段的"能力基类"), 模板里只有 D3 真正用得到的键。

★ 三源合并优先级由 ConfigMerger 保证: CLI > YAML > DEFAULT。
    (--train-rate / --val-rate / --seed / --split-strategy / --classes 依旧是 CLI 侧
     的显式覆盖, 优先级不变。)
"""
from __future__ import annotations

import math
import warnings
from typing import List, Optional

from pydantic import Field, field_validator, model_validator

from od_platform.common import constants as C
from od_platform.runtime_config.base import SettingsBase


def _epsilon_decimals(epsilon: float) -> int:
    """由容差自身决定"取几位小数": RATE_EPSILON=1e-6 → 6.

    ★ 不写死 6: 这个位数本来就是"容差的量级", 写死就是第二个真相来源。
    """
    if epsilon <= 0:
        return 0
    return max(0, -math.floor(math.log10(epsilon)))


#: test 比例取整位数 —— 由 RATE_EPSILON 推导, 保证 0.8+0.1+0.1 精确闭合
_TEST_RATE_DECIMALS: int = _epsilon_decimals(C.RATE_EPSILON)


class TransformSettings(SettingsBase):
    """odp-transform(D3)的全部可调项。"""

    # ============================================================
    # 覆盖率闸门
    # ============================================================

    coverage_hard_threshold: float = Field(
        default=C.COVERAGE_HARD_THRESHOLD,
        ge=0.0, le=1.0,
        description="图像-标注覆盖率的硬阈值(低于此值直接拒跑)",
        json_schema_extra={
            "group":    "覆盖率闸门",
            "examples": [0.5, 0.8, 0.99],
            "tips": [
                "覆盖率的定义: 标注文件数 / 图像数(stem 级)",
                "低于硬阈值 → raise, 不写任何产物(疑似导出中断 / 路径配错)",
                "调高 = 更严: 例如 0.99 要求几乎每张图都有标注",
                "COCO 是单 json 格式, 跳过 stem 级覆盖率检查",
            ],
            "yaml_comment": "覆盖率硬阈值(低于此值直接拒跑)",
        },
    )

    coverage_soft_threshold: float = Field(
        default=C.COVERAGE_SOFT_THRESHOLD,
        ge=0.0, le=1.0,
        description="图像-标注覆盖率的软阈值(低于此值只警告)",
        json_schema_extra={
            "group":    "覆盖率闸门",
            "examples": [0.9, 0.95, 1.0],
            "tips": [
                "低于软阈值但高于硬阈值 → 打 WARNING 后继续",
                "建议 >= 硬阈值(反了只 warn, 软警告永不触发)",
                "软阈值 = 1.0 表示「只要有缺标注就提醒」",
            ],
            "yaml_comment": "覆盖率软阈值(低于此值只警告)",
        },
    )

    # ============================================================
    # 类别平衡报告
    # ============================================================

    rare_warn_min_expected: float = Field(
        default=C.RARE_WARN_MIN_EXPECTED,
        gt=0.0,
        description="稀有类预警线(最小 split 里的期望样本量)",
        json_schema_extra={
            "group":    "类别平衡报告",
            "examples": [1.0, 3.0, 5.0],
            "tips": [
                "期望样本量 = 该类出现的图数 × 最小 split 比例",
                "低于此值 → 报告里标「稀有类!」, 只预警不拦截",
                "调高 = 更早提醒补数据",
            ],
            "yaml_comment": "稀有类预警线(最小 split 的期望样本量)",
        },
    )

    # ============================================================
    # 划分策略
    # ============================================================

    train_rate: float = Field(
        default=C.DEFAULT_TRAIN_RATE,
        ge=0.0, le=1.0,
        description="训练集比例",
        json_schema_extra={
            "group":    "划分策略",
            "examples": [0.7, 0.8, 0.9],
            "tips": [
                "默认 8:1:1(train 0.8 / val 0.1 / test 0.1)",
                "test 比例 = 1 - train_rate - val_rate(本模型自动算, 不用配)",
                "train_rate + val_rate 不能超过 1",
                "小数据集可提高 val_rate 让验证更稳",
            ],
            "yaml_comment": "训练集比例",
        },
    )

    val_rate: float = Field(
        default=C.DEFAULT_VAL_RATE,
        ge=0.0, le=1.0,
        description="验证集比例",
        json_schema_extra={
            "group":    "划分策略",
            "examples": [0.1, 0.15, 0.2],
            "tips": [
                "验证集用于选模型 / 早停, 不能省",
                "剩余全部进 test(最终诚实评估)",
                "与 train_rate 之和超过 1 会被直接拒绝",
            ],
            "yaml_comment": "验证集比例",
        },
    )

    random_state: int = Field(
        default=C.DEFAULT_RANDOM_STATE,
        ge=0,
        description="划分随机种子",
        json_schema_extra={
            "group":    "划分策略",
            "examples": [0, 42, 2026],
            "tips": [
                "同种子 = 同划分(契约指纹可复现的前提)",
                "换种子 = 换划分, 会换 manifest 指纹",
                "命令行等价参数是 --seed",
            ],
            "yaml_comment": "划分随机种子(可复现性)",
        },
    )

    split_strategy: str = Field(
        default=C.DEFAULT_SPLIT_STRATEGY,
        description="划分策略名(SSoT, 封闭取值)",
        json_schema_extra={
            "group":    "划分策略",
            "examples": list(C.SplitStrategy.all()),
            "tips": [
                "限定取值: " + " / ".join(C.SplitStrategy.all()),
                "random: 纯随机洗牌, 不需要标签",
                "stratified_multilabel: 多标签分层, 稀有类也能进每个 split",
                "命令行等价参数是 --split-strategy",
            ],
            "yaml_comment": "划分策略(random / stratified_multilabel)",
        },
    )

    # ============================================================
    # 转换选项
    # ============================================================

    classes: Optional[List[str]] = Field(
        default=None,
        description="类别白名单(顺序即 class_id)",
        json_schema_extra={
            "group":    "转换选项",
            "examples": [None, ["head", "反光衣"], ["defect"]],
            "tips": [
                "None: 按数据自带顺序(原样保留类别)",
                "list: 只保留这些类, 顺序即 class_id(会重排!)",
                "[] 空列表: 全排除(常用于先看清有什么类再决定)",
                "命令行等价参数是 --classes",
            ],
            "yaml_comment": "类别白名单(None=按数据自带; [] = 全排除)",
        },
    )

    # ============================================================
    # 验证器 / 跨字段校验
    # ============================================================

    @field_validator("split_strategy")
    @classmethod
    def _validate_split_strategy(cls, v: str) -> str:
        """split_strategy 必须在 SplitStrategy SSoT 范围内(拒绝拼错的策略名)"""
        if v not in C.SplitStrategy.all():
            raise ValueError(
                f"split_strategy 必须是 {sorted(C.SplitStrategy.all())} 之一, "
                f"当前值: {v!r}"
            )
        return v

    @model_validator(mode="after")
    def _cross_field_validation(self) -> "TransformSettings":
        """跨字段验证: 划分比例必须留得下 test 划分; 阈值次序反了只提醒不拦截.

        ★ 为什么软/硬阈值次序反了只 warn 不 raise(与 train.py 的
          "mosaic=0 却设了 close_mosaic" 同一处理):
          把硬阈值调到软阈值之上是合法意图(等于"不要软警告了, 一缺就拒"),
          只 warn 才能让 coverage_hard_threshold: 0.99 单独生效。
        """
        if self.coverage_soft_threshold < self.coverage_hard_threshold:
            warnings.warn(
                f"coverage_soft_threshold({self.coverage_soft_threshold}) 小于 "
                f"coverage_hard_threshold({self.coverage_hard_threshold}): "
                f"软阈值将永远不会触发(低于硬阈值时已经拒跑了).",
                UserWarning,
            )
        total = self.train_rate + self.val_rate
        if total > 1.0 + C.RATE_EPSILON:
            raise ValueError(
                f"train_rate({self.train_rate}) + val_rate({self.val_rate}) = {total} > 1, "
                f"没有位置留给 test 划分(容差 {C.RATE_EPSILON})."
            )
        return self

    # ============================================================
    # 派生量
    # ============================================================

    @property
    def test_rate(self) -> float:
        """test 比例 = 1 - train_rate - val_rate(按 RATE_EPSILON 量级取整).

        ★ 取整位数由容差推导(_TEST_RATE_DECIMALS), orchestrator 不再自己写
          ``round(1 - train - val, 6)`` —— 那个 6 曾经是第 4 处魔法数字。
        """
        return max(0.0, round(1.0 - self.train_rate - self.val_rate,
                              _TEST_RATE_DECIMALS))


# ============================================================
# 一键 build (与 build_train_config 同构, 走同一套 _build_config 流水线)
# ============================================================

def build_transform_config(
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
    """一键完成 transform 配置流水线: YAML + CLI (+ 自定义源) → 合并 → 验证.

    Args:
        yaml_path:     YAML 文件名 / 路径. None = 不读 yaml, 全用默认值
                       (★ 与 build_train_config 默认 "train.yaml" 不同:
                        D3 的 yaml 引用由 odp-transform --yaml 显式给出,
                        编排层 DatasetPipeline 直接吃默认值, 不吃"模板必须存在"的亏.)
        cli_args:      argparse.Namespace 或 dict. None = 没有 CLI 覆盖.
        yaml_dir:      YAML 目录, 默认 paths.RUNTIME_CONFIGS_DIR.
        cli_exclude:   CLI 额外排除字段 (并入 CLILoader.DEFAULT_EXCLUDE).
        cli_mapping:   CLI 参数名映射 (CLI 名 → Pydantic 字段名, 如 seed → random_state).
        extra_sources: 额外配置源 [(source_id, dict), ...], 插在 YAML 与 CLI 之间.
        track_sources: 是否追踪溯源 (默认 True).
        dry_run:       True = 只合并不验证, 返回 (None, merger).

    Returns:
        正常模式: (TransformSettings, ConfigMerger)
        dry_run:  (None, ConfigMerger)
    """
    from od_platform.runtime_config import _build_config

    return _build_config(
        TransformSettings,
        yaml_path=yaml_path,
        cli_args=cli_args,
        yaml_dir=yaml_dir,
        cli_exclude=cli_exclude,
        cli_mapping=cli_mapping,
        extra_sources=extra_sources,
        track_sources=track_sources,
        dry_run=dry_run,
    )