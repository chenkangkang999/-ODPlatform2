#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : validate.py
# @Author    : 雨霓同学 (ODPlatform team)
# @Project   : ODPlatform
# @Function  : runtime_config 子系统——D4 数据质检配置类
"""D4 数据质检配置: ValidateSettings.

对应 ``configs/runtime/validate.yaml`` (odp-gen-config validate 生成)。

★ 覆盖哪些阈值(逐个核对过 15 个 check 的真实 import, 不多不少):
    pair_existence       → pair_missing_error_ratio / pair_missing_warn_ratio / pair_max_detail
    bbox_within_image    → bbox_out_error_ratio / bbox_edge_tolerance / details_preview_limit
    box_geometry         → degenerate_box_min_size / stats_max_aspect_ratio / details_preview_limit
    annotation_coverage  → unlabeled_warn_ratio / unlabeled_info_ratio
    orphan_labels        → orphan_labels_warn_ratio / details_preview_limit
    split_presence       → split_min_images
    class_balance        → stats_max_imbalance_ratio / class_imbalance_info_ratio
    manifest_lineage     → lineage_max_detail
    label_format /
    duplicate_annotations /
    split_uniqueness /
    stem_collision       → details_preview_limit
    (yaml_schema / fingerprint_match / class_presence 不读任何阈值 —— 它们的判据是
     "结构对不对 / 指纹一致不一致 / 类在不在", 没有可调数字。)

★ 默认值只此一处(代码上可验证):
    每个字段的默认值【直接引用】common/constants.py 的常量, 不抄字面量。
    这正是本课要收敛的 13 + 2 个散落阈值 —— 它们此前只躺在 constants.py 里,
    check 直接 import 常量, 用户既不能配 yaml 也不能用 CLI 覆盖。

★ check 侧读取姿势(见 data_validation/registry.py 的 setting()):
    ctx.settings.<字段> 优先, 缺失/None 时回落 constants.<常量> ——
    于是"旧构造点(没传 settings 的 CheckContext)"照样能跑。
"""
from __future__ import annotations

import warnings

from pydantic import Field, model_validator

from od_platform.common import constants as C
from od_platform.runtime_config.base import SettingsBase


class ValidateSettings(SettingsBase):
    """odp-validate(D4)的全部可调阈值。"""

    # ============================================================
    # 配对与覆盖率
    # ============================================================

    pair_missing_error_ratio: float = Field(
        default=C.PAIR_MISSING_ERROR_RATIO,
        ge=0.0, le=1.0,
        description="缺标签图占比的 ERROR 线",
        json_schema_extra={
            "group":    "配对与覆盖率",
            "examples": [0.3, 0.5, 0.8],
            "tips": [
                "缺标签图占比 >= 此值 → ERROR(疑似导出中断, 必须修)",
                "建议 >= pair_missing_warn_ratio(反了只 warn: WARNING 档永远轮不到)",
                "调低 = 更严: 少量缺失就升级成 ERROR",
            ],
            "yaml_comment": "缺标签占比 ERROR 线",
        },
    )

    pair_missing_warn_ratio: float = Field(
        default=C.PAIR_MISSING_WARN_RATIO,
        ge=0.0, le=1.0,
        description="缺标签图占比的 WARNING 线",
        json_schema_extra={
            "group":    "配对与覆盖率",
            "examples": [0.01, 0.05, 0.1],
            "tips": [
                "占比 >= 此值(且 < ERROR 线) → WARNING",
                "低于此值但有缺失 → INFO(多为无目标图, 确认即可)",
                "0 表示「一张都不许缺」",
            ],
            "yaml_comment": "缺标签占比 WARNING 线",
        },
    )

    unlabeled_warn_ratio: float = Field(
        default=C.UNLABELED_WARN_RATIO,
        ge=0.0, le=1.0,
        description="无有效标注图占比的 WARNING 线",
        json_schema_extra={
            "group":    "配对与覆盖率",
            "examples": [0.2, 0.3, 0.5],
            "tips": [
                "无有效标注(标签文件为空/无有效行)的图占比 >= 此值 → WARNING",
                "与 pair_existence 的分工: 那边看「文件在不在」, 这边看「里面有没有东西」",
                "建议 >= unlabeled_info_ratio(反了只 warn: INFO 档永远轮不到)",
            ],
            "yaml_comment": "无标注图占比 WARNING 线",
        },
    )

    unlabeled_info_ratio: float = Field(
        default=C.UNLABELED_INFO_RATIO,
        ge=0.0, le=1.0,
        description="无有效标注图占比的 INFO 线",
        json_schema_extra={
            "group":    "配对与覆盖率",
            "examples": [0.0, 0.05, 0.1],
            "tips": [
                "占比 >= 此值(且 < WARNING 线) → INFO",
                "低于此值 → PASS",
                "空图(背景图)本身是合法的, 所以这里不设 ERROR",
            ],
            "yaml_comment": "无标注图占比 INFO 线",
        },
    )

    orphan_labels_warn_ratio: float = Field(
        default=C.ORPHAN_LABELS_WARN_RATIO,
        ge=0.0, le=1.0,
        description="孤儿标签占比的 WARNING 线",
        json_schema_extra={
            "group":    "配对与覆盖率",
            "examples": [0.05, 0.1, 0.3],
            "tips": [
                "孤儿标签 = 有 .txt 没有对应图像",
                "占比 >= 此值 → WARNING(疑似图被删了)",
                "低于此值但有孤儿 → INFO",
            ],
            "yaml_comment": "孤儿标签占比 WARNING 线",
        },
    )

    # ============================================================
    # 边界框几何
    # ============================================================

    bbox_out_error_ratio: float = Field(
        default=C.BBOX_OUT_ERROR_RATIO,
        ge=0.0, le=1.0,
        description="出界框占比的 ERROR 线",
        json_schema_extra={
            "group":    "边界框几何",
            "examples": [0.01, 0.1, 0.5],
            "tips": [
                "出界框(cx±w/2 超出 [0,1])占比 >= 此值 → ERROR, 否则 WARNING",
                "调低 = 更严: 极少数越界框也会被升级成 ERROR",
                "只适用 detect; segment 的多边形顶点允许在外",
            ],
            "yaml_comment": "出界框占比 ERROR 线",
        },
    )

    bbox_edge_tolerance: float = Field(
        default=C.BBOX_EDGE_TOLERANCE,
        ge=0.0,
        description="出界判定的浮点容差",
        json_schema_extra={
            "group":    "边界框几何",
            "examples": [0.0, 1e-6, 1e-3],
            "tips": [
                "归一化坐标的小数误差不该被当成「越界」",
                "出界量 <= 此容差 → 视为没出界",
                "一般不要改(1e-6 足够吃掉浮点噪声)",
            ],
            "yaml_comment": "出界判定浮点容差",
        },
    )

    degenerate_box_min_size: float = Field(
        default=C.DEGENERATE_BOX_MIN_SIZE,
        ge=0.0,
        description="退化框的最小宽/高",
        json_schema_extra={
            "group":    "边界框几何",
            "examples": [1e-4, 1e-3, 1e-2],
            "tips": [
                "归一化 w 或 h 低于此值 → 记一个 tiny_area 退化框",
                "w<=0 / h<=0 会先被 zero_width / zero_height 抓走",
                "该 check 只出 WARNING(可训但建议清理)",
            ],
            "yaml_comment": "退化框最小宽/高",
        },
    )

    stats_max_aspect_ratio: float = Field(
        default=C.STATS_MAX_ASPECT_RATIO,
        gt=0.0,
        description="极端宽高比的判定线",
        json_schema_extra={
            "group":    "边界框几何",
            "examples": [10.0, 20.0, 50.0],
            "tips": [
                "宽高比 max(w/h, h/w) >= 此值 → 极端形状(细长条)",
                "细长缺陷(裂缝/划痕)是合法数据, 所以只出 WARNING",
                "D3 的只读报告与 D4 共用这一个常量口径",
            ],
            "yaml_comment": "极端宽高比判定线",
        },
    )

    # ============================================================
    # 类别分布
    # ============================================================

    stats_max_imbalance_ratio: float = Field(
        default=C.STATS_MAX_IMBALANCE_RATIO,
        gt=0.0,
        description="类别失衡比的 WARNING 线",
        json_schema_extra={
            "group":    "类别分布",
            "examples": [20.0, 50.0, 100.0],
            "tips": [
                "失衡比 = 最多类实例数 / 最少类实例数(只看非零类)",
                ">= 此值 → WARNING(长尾提醒, 建议关注 class_weights / 重采样)",
                "建议 >= class_imbalance_info_ratio(反了只 warn: INFO 档永远轮不到)",
            ],
            "yaml_comment": "类别失衡比 WARNING 线",
        },
    )

    class_imbalance_info_ratio: float = Field(
        default=C.CLASS_IMBALANCE_INFO_RATIO,
        gt=0.0,
        description="类别失衡比的 INFO 线",
        json_schema_extra={
            "group":    "类别分布",
            "examples": [5.0, 10.0, 20.0],
            "tips": [
                "失衡比 >= 此值(且 < WARNING 线) → INFO",
                "低于此值 → PASS(未见显著失衡)",
                "建议 > 1(比值小于 1 没有意义)",
            ],
            "yaml_comment": "类别失衡比 INFO 线",
        },
    )

    # ============================================================
    # 划分规模
    # ============================================================

    split_min_images: int = Field(
        default=C.SPLIT_MIN_IMAGES,
        ge=1,
        description="split 规模的提示线",
        json_schema_extra={
            "group":    "划分规模",
            "examples": [10, 30, 100],
            "tips": [
                "某个 split 的图数低于此值 → INFO(评估方差偏大)",
                "只提示不拦截: 小数据集本来就凑不够",
                "train 缺失才是 ERROR, 与这个数字无关",
            ],
            "yaml_comment": "split 规模提示线(图数)",
        },
    )

    # ============================================================
    # 明细展示上限
    # ============================================================

    pair_max_detail: int = Field(
        default=C.PAIR_MAX_DETAIL,
        ge=0,
        description="pair_existence 明细清单的展示上限",
        json_schema_extra={
            "group":    "明细展示上限",
            "examples": [5, 20, 100],
            "tips": [
                "details.missing_preview 最多列这么多条, 避免报告爆掉",
                "0 表示只报汇总数字, 不列清单",
                "只影响报告体积, 不影响判定",
            ],
            "yaml_comment": "缺失清单展示上限(条)",
        },
    )

    lineage_max_detail: int = Field(
        default=C.LINEAGE_MAX_DETAIL,
        ge=0,
        description="manifest_lineage 明细清单的展示上限",
        json_schema_extra={
            "group":    "明细展示上限",
            "examples": [5, 20, 100],
            "tips": [
                "changed / moved / missing 三个清单各自的上限",
                "计数是全量的, 只有清单被截断",
                "0 表示只报计数, 不列样本名",
            ],
            "yaml_comment": "契约核验清单展示上限(条)",
        },
    )

    details_preview_limit: int = Field(
        default=C.DETAILS_PREVIEW_LIMIT,
        ge=0,
        description="各 check details 预览列表的统一上限",
        json_schema_extra={
            "group":    "明细展示上限",
            "examples": [5, 20, 50],
            "tips": [
                "label_format / bbox_within_image / box_geometry / duplicate_annotations /"
                " orphan_labels / split_uniqueness / stem_collision 共用",
                "统一一个数字, 免得每份 report 预览长度都不一样",
                "只影响报告体积, 不影响判定",
            ],
            "yaml_comment": "details 预览统一上限(条)",
        },
    )

    # ============================================================
    # 跨字段校验
    # ============================================================

    @model_validator(mode="after")
    def _cross_field_validation(self) -> "ValidateSettings":
        """跨字段验证: 三组"严重度阶梯"的次序反了 → 只提醒, 不拦截.

        ★ 为什么只 warn: 阶梯反了的后果是"较轻那一档永远轮不到"(更强的档先命中),
          判定结果依然是安全的一侧(更严), 没有任何数据会被误放行。
          于是跟 train.py 的跨字段风格保持一致(UserWarning), 保住"一切皆配置"的
          可配置性 —— 用户把 threshold 调乱, 不该连跑都跑不起来。
        """
        pairs = [
            ("pair_missing_error_ratio", self.pair_missing_error_ratio,
             "pair_missing_warn_ratio", self.pair_missing_warn_ratio),
            ("unlabeled_warn_ratio", self.unlabeled_warn_ratio,
             "unlabeled_info_ratio", self.unlabeled_info_ratio),
            ("stats_max_imbalance_ratio", self.stats_max_imbalance_ratio,
             "class_imbalance_info_ratio", self.class_imbalance_info_ratio),
        ]
        for high_name, high, low_name, low in pairs:
            if high < low:
                warnings.warn(
                    f"{high_name}({high}) 小于 {low_name}({low}): "
                    f"严重度阶梯反了, {low_name} 这一档将永远轮不到.",
                    UserWarning,
                )
        return self


# ============================================================
# 一键 build (与 build_train_config 同构, 走同一套 _build_config 流水线)
# ============================================================

def build_validate_config(
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
    """一键完成 validate 配置流水线: YAML + CLI (+ 自定义源) → 合并 → 验证.

    Args:
        yaml_path:     YAML 文件名 / 路径. None = 不读 yaml, 全用默认值
                       (D4 的 yaml 引用由 odp-validate --yaml 显式给出;
                        服务层不带 settings 时直接吃默认值, 向后兼容.)
        cli_args:      argparse.Namespace 或 dict(None = 没有 CLI 覆盖).
        yaml_dir:      YAML 目录, 默认 paths.RUNTIME_CONFIGS_DIR.
        cli_exclude:   CLI 额外排除字段 (并入 CLILoader.DEFAULT_EXCLUDE).
        cli_mapping:   CLI 参数名映射 (CLI 名 → Pydantic 字段名).
        extra_sources: 额外配置源 [(source_id, dict), ...], 插在 YAML 与 CLI 之间.
        track_sources: 是否追踪溯源 (默认 True).
        dry_run:       True = 只合并不验证, 返回 (None, merger).

    Returns:
        正常模式: (ValidateSettings, ConfigMerger)
        dry_run:  (None, ConfigMerger)
    """
    from od_platform.runtime_config import _build_config

    return _build_config(
        ValidateSettings,
        yaml_path=yaml_path,
        cli_args=cli_args,
        yaml_dir=yaml_dir,
        cli_exclude=cli_exclude,
        cli_mapping=cli_mapping,
        extra_sources=extra_sources,
        track_sources=track_sources,
        dry_run=dry_run,
    )