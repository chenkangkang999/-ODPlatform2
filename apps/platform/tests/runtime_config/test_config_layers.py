#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : test_config_layers.py
# @Project   : ODPlatform
# @Function  : 「一切皆配置」(ADR-006) 的守卫测试 —— 覆盖 D3/D4/runs 保留策略三套新配置层
"""把 ADR-006 的五条纪律变成机器断言。

本文件补的是 2026-10-27 那轮"一切皆配置"改造里**三个新配置层**
（`TransformSettings` / `ValidateSettings` / `CleanSettings`）的守卫，
它们此前只有实现、没有测试：

1. **默认值只此一处** —— 每个字段的默认值必须等于 `common/constants.py` 里的同名常量
   （不是"抄了一份字面量"，而是**直接引用**；常量一改这里就会红）；
2. **三源合并优先级** —— `CLI > YAML > DEFAULT`，三者逐级可覆盖；
3. **拼写防线** —— `extra="forbid"`：字段名拼错在**构造时**就报错；
4. **阈值真生效** —— 改配置必须真的改变行为（这里用两个 hermetic 实验证明：
   D3 的覆盖率硬闸门、D4 的出界框严重度）；
5. **接线** —— 配置值必须真的传到消费者（`DatasetPipeline` / `RetentionPolicy` /
   `CheckContext`），而不是"配了但没人读"。

全部用例都不依赖仓库现有数据集、不写仓库目录（用 `tmp_path` 与 monkeypatch）。
"""
from __future__ import annotations

import warnings

import pytest
import yaml
from pydantic import ValidationError

from od_platform.common import constants as C
from od_platform.common import paths
from od_platform.common.retention import RetentionPolicy
from od_platform.data_pipeline.orchestrator import DatasetPipeline
from od_platform.data_validation.checks.bbox_within_image import validate_bbox_within_image
from od_platform.data_validation.registry import CheckContext, CheckSeverity
from od_platform.data_validation.snapshot import build_snapshot
from od_platform.runtime_config.clean import CleanSettings, build_clean_config
from od_platform.runtime_config.transform import TransformSettings, build_transform_config
from od_platform.runtime_config.validate import ValidateSettings, build_validate_config

pytestmark = [pytest.mark.unit]


# ============================================================
# 纪律 1: 默认值只此一处(逐个字段对齐 common/constants.py)
# ============================================================

#: (配置类, 字段名, constants 里的常量名) —— 只挑有代表性的, 不追求穷举
_DEFAULT_ALIGNMENT = [
    # D3 数据流水线
    (TransformSettings, "coverage_hard_threshold", "COVERAGE_HARD_THRESHOLD"),
    (TransformSettings, "coverage_soft_threshold", "COVERAGE_SOFT_THRESHOLD"),
    (TransformSettings, "rare_warn_min_expected", "RARE_WARN_MIN_EXPECTED"),
    (TransformSettings, "train_rate", "DEFAULT_TRAIN_RATE"),
    (TransformSettings, "val_rate", "DEFAULT_VAL_RATE"),
    (TransformSettings, "random_state", "DEFAULT_RANDOM_STATE"),
    (TransformSettings, "split_strategy", "DEFAULT_SPLIT_STRATEGY"),
    # D4 数据质检
    (ValidateSettings, "stats_max_aspect_ratio", "STATS_MAX_ASPECT_RATIO"),
    (ValidateSettings, "stats_max_imbalance_ratio", "STATS_MAX_IMBALANCE_RATIO"),
    (ValidateSettings, "pair_missing_error_ratio", "PAIR_MISSING_ERROR_RATIO"),
    (ValidateSettings, "pair_missing_warn_ratio", "PAIR_MISSING_WARN_RATIO"),
    (ValidateSettings, "pair_max_detail", "PAIR_MAX_DETAIL"),
    (ValidateSettings, "lineage_max_detail", "LINEAGE_MAX_DETAIL"),
    (ValidateSettings, "details_preview_limit", "DETAILS_PREVIEW_LIMIT"),
    (ValidateSettings, "bbox_edge_tolerance", "BBOX_EDGE_TOLERANCE"),
    (ValidateSettings, "bbox_out_error_ratio", "BBOX_OUT_ERROR_RATIO"),
    (ValidateSettings, "degenerate_box_min_size", "DEGENERATE_BOX_MIN_SIZE"),
    (ValidateSettings, "unlabeled_warn_ratio", "UNLABELED_WARN_RATIO"),
    (ValidateSettings, "unlabeled_info_ratio", "UNLABELED_INFO_RATIO"),
    (ValidateSettings, "split_min_images", "SPLIT_MIN_IMAGES"),
    (ValidateSettings, "class_imbalance_info_ratio", "CLASS_IMBALANCE_INFO_RATIO"),
    (ValidateSettings, "orphan_labels_warn_ratio", "ORPHAN_LABELS_WARN_RATIO"),
]


@pytest.mark.parametrize(
    ("settings_cls", "field", "constant_name"),
    _DEFAULT_ALIGNMENT,
    ids=[f"{cls.__name__}.{field}" for cls, field, _ in _DEFAULT_ALIGNMENT],
)
def test_field_default_equals_constant(settings_cls, field, constant_name):
    """默认值必须"直接引用"常量 —— 常量改了而配置没改, 这里立刻红。"""
    expected = getattr(C, constant_name)
    actual = getattr(settings_cls(), field)
    assert actual == expected, (
        f"{settings_cls.__name__}.{field} 默认值 {actual!r} != "
        f"constants.{constant_name}={expected!r}（默认值出现了第二处出处?）")


def test_clean_settings_defaults_live_in_retention_module():
    """保留策略的默认值出处是 `common/retention.py` 的 DEFAULT_* 常量。"""
    from od_platform.common import retention

    s = CleanSettings()
    assert s.keep_latest == retention.DEFAULT_KEEP_LATEST
    assert s.max_age_days == retention.DEFAULT_MAX_AGE_DAYS


# ============================================================
# 纪律 2: 三源合并优先级 CLI > YAML > DEFAULT
# ============================================================

_BUILDERS = [
    ("transform", build_transform_config, TransformSettings, "train_rate", 0.7, 0.55),
    ("validate", build_validate_config, ValidateSettings, "bbox_out_error_ratio", 0.5, 0.05),
    ("clean", build_clean_config, CleanSettings, "keep_latest", 9, 3),
]


@pytest.mark.parametrize(
    ("name", "builder", "settings_cls", "field", "yaml_value", "cli_value"),
    _BUILDERS,
    ids=[b[0] for b in _BUILDERS],
)
def test_three_source_precedence(name, builder, settings_cls, field, yaml_value, cli_value, tmp_path):
    """DEFAULT → YAML → CLI 三级覆盖, 且 CLI 永远赢。"""
    default_value = getattr(settings_cls(), field)

    # ① 只有 DEFAULT
    s_default, _ = builder()
    assert getattr(s_default, field) == default_value

    # ② YAML 覆盖 DEFAULT
    yaml_file = tmp_path / f"{name}.yaml"
    yaml_file.write_text(yaml.safe_dump({field: yaml_value}, allow_unicode=True), encoding="utf-8")
    s_yaml, merger = builder(yaml_path=str(yaml_file))
    assert getattr(s_yaml, field) == yaml_value
    assert getattr(s_yaml, field) != default_value

    # ③ CLI 覆盖 YAML(且 YAML 的值仍记录在溯源里)
    s_cli, _ = builder(yaml_path=str(yaml_file), cli_args={field: cli_value})
    assert getattr(s_cli, field) == cli_value
    assert getattr(s_cli, field) != yaml_value

    # ④ 溯源: 必须能回答"这个值从哪来"(来源报告 + 覆盖链)
    metadata = merger.get_metadata(field)
    assert metadata is not None, f"{field} 没有溯源记录 —— 配了不知道谁生效?"
    assert metadata.chain_str(), "覆盖链为空: 无法回答值从哪来"
    report_text = merger.get_source_report()
    assert isinstance(report_text, str)
    assert merger.to_audit_log(), "审计日志为空"


def test_yaml_not_found_is_loud(tmp_path):
    """给了 yaml 路径却不存在 → fail-fast, 不许静默用默认值跑。"""
    missing = tmp_path / "definitely_missing.yaml"
    with pytest.raises(FileNotFoundError):
        build_transform_config(yaml_path=str(missing))


# ============================================================
# 纪律 3: 拼写防线 (extra="forbid")
# ============================================================

_TYPO_CASES = [
    (TransformSettings, {"coverage_hard_thresho": 0.5}),
    (TransformSettings, {"train_rat": 0.9}),
    (ValidateSettings, {"bbox_out_error_ratios": 0.1}),
    (ValidateSettings, {"split_min_image": 30}),
    (CleanSettings, {"keep_lated": 3}),
    (CleanSettings, {"max_age_day": 7}),
]


@pytest.mark.parametrize(
    ("settings_cls", "bad_kwargs"),
    _TYPO_CASES,
    ids=[f"{cls.__name__}:{next(iter(kw))}" for cls, kw in _TYPO_CASES],
)
def test_typo_is_rejected_at_construction(settings_cls, bad_kwargs):
    """拼错的字段名必须在构造时就炸, 而不是"静默用默认值跑 5 小时"。"""
    with pytest.raises(ValidationError):
        settings_cls(**bad_kwargs)


def test_out_of_range_is_rejected():
    """取值域也走配置模型的校验(不是只在文档里写"建议 0~1")。"""
    with pytest.raises(ValidationError):
        ValidateSettings(bbox_out_error_ratio=1.5)
    with pytest.raises(ValidationError):
        CleanSettings(keep_latest=0)          # >= 1
    with pytest.raises(ValidationError):
        TransformSettings(train_rate=1.5)     # 比例不能 > 1


def test_severity_ladder_inverted_only_warns():
    """严重度阶梯反了只 warn 不拦 —— 用户把阈值调乱, 不该连跑都跑不起来。"""
    with pytest.warns(UserWarning):
        s = ValidateSettings(bbox_out_error_ratio=0.1, pair_missing_error_ratio=0.01,
                             pair_missing_warn_ratio=0.5)
    assert s.pair_missing_error_ratio == 0.01


# ============================================================
# 纪律 4a: D3 覆盖率硬闸门真的读配置
# ============================================================

def _make_raw_dataset(root, n_images: int, n_labels: int):
    """造一个最小的 raw 数据集(不写仓库目录)。"""
    img_dir = root / "images"
    lbl_dir = root / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_images):
        (img_dir / f"i_{i:03d}.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 16)
    for i in range(n_labels):
        (lbl_dir / f"i_{i:03d}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")


def test_coverage_gate_reads_settings(monkeypatch, tmp_path):
    """同一份数据(覆盖率 10%): 默认配置拒跑, 配置放宽后放行。"""
    monkeypatch.setattr(paths, "RAW_DATA_DIR", tmp_path)
    _make_raw_dataset(tmp_path / "gated", n_images=10, n_labels=1)

    # ① 默认配置(硬阈值 0.5) → 直接拒
    strict = DatasetPipeline("gated", "yolo", classes=["a"])
    with pytest.raises(ValueError) as excinfo:
        strict._check_coverage()
    assert "低于硬阈值" in str(excinfo.value)

    # ② 配置把硬阈值关掉 → 放行(证明"配置真的生效", 而不是写死的 0.5)
    relaxed = DatasetPipeline(
        "gated", "yolo", classes=["a"],
        settings=TransformSettings(coverage_hard_threshold=0.0, coverage_soft_threshold=0.0),
    )
    relaxed._check_coverage()          # 不抛异常即通过


def test_pipeline_takes_rates_from_settings(monkeypatch, tmp_path):
    """划分比例/种子/策略必须来自 settings, 而不是构造器里的字面量。"""
    monkeypatch.setattr(paths, "RAW_DATA_DIR", tmp_path)
    _make_raw_dataset(tmp_path / "rates", n_images=3, n_labels=3)

    s = TransformSettings(train_rate=0.5, val_rate=0.25, random_state=7,
                          split_strategy="random")
    p = DatasetPipeline("rates", "yolo", classes=["a"], settings=s)
    assert (p.train_rate, p.val_rate, p.random_state, p.split_strategy) == (0.5, 0.25, 7, "random")

    # 显式覆盖(给"不走 yaml 的调用点"留的后门)优先级高于 settings
    # 注意: 覆盖值要过完整校验 —— train+val 之和不能 > 1(这里 0.7+0.25=0.95 合法)
    p2 = DatasetPipeline("rates", "yolo", classes=["a"], settings=s, train_rate=0.7)
    assert p2.train_rate == 0.7

    # 非法覆盖值必须当场被拒(而不是静默生效)
    with pytest.raises(ValidationError):
        DatasetPipeline("rates", "yolo", classes=["a"], settings=s, train_rate=0.95)


# ============================================================
# 纪律 4b: D4 出界框严重度真的读配置
# ============================================================

def _make_processed_dataset(root, n_in_bounds: int, n_out_of_bounds: int):
    """造一个最小的 processed 数据集: 1 张图 + 若干行标签(含越界框)。"""
    (root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "val").mkdir(parents=True, exist_ok=True)
    (root / "images" / "val" / "a.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 16)
    lines = ["0 0.5 0.5 0.2 0.2"] * n_in_bounds
    lines += ["0 0.95 0.5 0.3 0.3"] * n_out_of_bounds     # cx+w/2 = 1.1 > 1 → 越界
    (root / "labels" / "val" / "a.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        yaml.safe_dump({"path": str(root).replace("\\", "/"), "val": "images/val",
                        "nc": 1, "names": {0: "a"}}, allow_unicode=True, sort_keys=False),
        encoding="utf-8")
    return yaml_path


def test_bbox_severity_reads_settings(tmp_path):
    """同一份数据(越界框占比 5%): 默认判 WARNING, 阈值调到 0 后升级为 ERROR。"""
    yaml_path = _make_processed_dataset(tmp_path / "ds", n_in_bounds=19, n_out_of_bounds=1)
    snapshot = build_snapshot(yaml_path)

    # ① 默认(bbox_out_error_ratio=0.1): 5% < 10% → WARNING
    default_ctx = CheckContext(yaml_path=yaml_path, snapshot=snapshot)
    result_default = validate_bbox_within_image(default_ctx)
    assert result_default.severity == CheckSeverity.WARNING, result_default.summary
    assert result_default.details["thresholds"]["error_ratio"] == C.BBOX_OUT_ERROR_RATIO

    # ② 配置调到 0.0 → 同一个框升级为 ERROR
    strict_ctx = CheckContext(yaml_path=yaml_path, snapshot=snapshot,
                              settings=ValidateSettings(bbox_out_error_ratio=0.0))
    result_strict = validate_bbox_within_image(strict_ctx)
    assert result_strict.severity == CheckSeverity.ERROR, result_strict.summary
    assert result_strict.details["thresholds"]["error_ratio"] == 0.0

    # ③ 兜底: 没传 settings 的旧构造点走"回落 constants", 不许炸
    fallback_ctx = CheckContext(yaml_path=yaml_path, snapshot=snapshot)
    assert fallback_ctx.setting("bbox_out_error_ratio", C.BBOX_OUT_ERROR_RATIO) == C.BBOX_OUT_ERROR_RATIO
    assert fallback_ctx.setting("不存在的字段", 123) == 123          # 读不到 → 回落


# ============================================================
# 纪律 5: 接线(配置值必须真的到达消费者)
# ============================================================

def test_clean_settings_reach_retention_policy():
    """CleanSettings → RetentionPolicy 必须真的传值(不是各存一份)。"""
    policy = RetentionPolicy.from_settings(CleanSettings(keep_latest=7, max_age_days=1.5))
    assert policy.keep_latest_per_group == 7
    assert policy.max_age_days == 1.5


def test_generated_templates_match_code_defaults(tmp_path):
    """三套新模板与代码默认值必须一致(模板漂移 → 这里红)。

    与 `test_generator.py` 里按 registry 参数化的契约测试互补:
    这里显式钉住"新加的三套", 免得 registry 哪天漏注册一个就再也测不到。
    """
    from od_platform.runtime_config.generator import ConfigGenerator
    from od_platform.runtime_config.registry import CONFIG_REGISTRY

    for name, settings_cls in (("transform", TransformSettings),
                               ("validate", ValidateSettings),
                               ("clean", CleanSettings)):
        assert name in CONFIG_REGISTRY, f"registry 里没有注册 {name} 模板"
        config_class, title = CONFIG_REGISTRY[name]
        assert config_class is settings_cls, (
            f"registry 里 {name} 指向的是 {config_class.__name__}, 期望 {settings_cls.__name__}")

        out = tmp_path / f"{name}.yaml"
        ConfigGenerator().generate(config_class, out, overwrite=True, title=title)
        parsed = yaml.safe_load(out.read_text(encoding="utf-8"))

        defaults = settings_cls().model_dump()
        for field, expected in defaults.items():
            if field not in parsed:          # None 值可能被渲染成 null 或省略, 以"在就用"为准
                continue
            actual = parsed[field]
            assert actual == expected, (
                f"{name}.yaml 的 {field} = {actual!r} 与代码默认值 {expected!r} 不一致"
                f"（模板漂移? 还是默认值出现了第二处出处?）")