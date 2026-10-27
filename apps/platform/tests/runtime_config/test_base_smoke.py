"""runtime_config 端到端冒烟测试 —— 三种模式(train / val / infer)走一遍全链路.

★ 与 test_base.py 的分工(刻意不重复):
    - test_base.py        字段级单测: 单个字段的默认值 / 验证器 / audit 快照 /
                          sensitive 机制. 大多用最小内部子类
                          (_ConcreteTestConfig / _SensitiveTestConfig),
                          不碰真实配置类, 也不跑完整流水线.
    - test_train_val.py / test_infer.py
                          真实配置类的字段级行为契约(逐字段参数化).
    - 本文件              子系统级冒烟: 用真实 YOLOTrain/Val/Infer 配置类,
                          把"构造 → 加载 yaml → 三源合并 → 溯源 → 验证/报错"
                          整条链路各走一遍, 回答一个问题:
                          **配置系统现在还活着吗?**
                          这里不做逐字段穷举(那是上面两个文件的活), 只钉住
                          跨模块的、一坏就全坏的接缝.

覆盖的接缝:
    1. 三种模式默认值构造(不依赖 GPU / 网络)
    2. 缺 yaml → fail-fast + 修复指引
    3. 三源合并优先级(CLI > YAML > DEFAULT)
    4. extra="forbid" 拼写防线(且错误信息带来源链)
    5. batch 的 bool / float / 负值 / 自动(-1) 语义
    6. imgsz 非 32 倍数只 warning(不 raise)
    7. 敏感字段在报告 / dump 中被 mask
    8. 溯源链完整: 每个生效字段都有溯源节点, 来源只能是内置三档(无"未知")

约束: 不依赖 GPU / 不联网 / 不写仓库文件(只用 tmp_path) / 1 秒内跑完.
"""
from __future__ import annotations

import warnings
from argparse import Namespace
from typing import Optional

import pytest
from pydantic import Field, ValidationError

from od_platform.common.constants import Task

from od_platform.runtime_config import (
    build_infer_config,
    build_train_config,
    build_val_config,
)
from od_platform.runtime_config.base  import BaseConfig
from od_platform.runtime_config.infer import YOLOInferConfig
from od_platform.runtime_config.merger import ConfigMerger, ConfigSource
from od_platform.runtime_config.train  import YOLOTrainConfig
from od_platform.runtime_config.val    import YOLOValConfig


#: 出处报告里允许出现的内置来源(出现别的 = 溯源"未知")
BUILTIN_SOURCES = {source.value for source in ConfigSource}

ALL_MODES = [YOLOTrainConfig, YOLOValConfig, YOLOInferConfig]

# 三个 build_* 入口 + 各自默认 yaml 名
BUILDERS = [
    (build_train_config, YOLOTrainConfig, "train.yaml"),
    (build_val_config,   YOLOValConfig,   "val.yaml"),
    (build_infer_config, YOLOInferConfig, "infer.yaml"),
]


def _non_none_default_fields(config_class) -> set[str]:
    """反射出"默认值不是 None"的字段名(这些字段必然会进 DEFAULT 源)."""
    return {
        name for name, field in config_class.model_fields.items()
        if field.default is not None
    }


# ============================================================
# 1. 三种模式默认值构造 —— 不需要 GPU / 网络
# ============================================================

class TestThreeModesConstruct:
    @pytest.mark.parametrize("config_class", ALL_MODES,
                             ids=lambda c: c.__name__)
    def test_defaults_construct(self, config_class):
        """三种模式都能只靠默认值构造出来(纯 CPU, 不碰 ultralytics)."""
        config = config_class()
        assert config.batch == 16
        assert config.imgsz == 640
        assert config.task  == Task.DETECT
        assert config.amp   is True

    def test_mode_specific_defaults_still_differ(self):
        """三模式的默认值差异必须还在(infer/val 的 conf 差两个数量级)."""
        assert YOLOTrainConfig().epochs == 100
        assert YOLOValConfig().conf     == 0.001
        assert YOLOInferConfig().conf   == 0.25

    @pytest.mark.parametrize("config_class,frame_field", [
        (YOLOTrainConfig, "experiment_name"),
        (YOLOValConfig,   "task"),
        (YOLOInferConfig, "task"),
    ], ids=["train", "val", "infer"])
    def test_framework_only_fields_not_leaked(self, config_class, frame_field):
        """框架级字段(track/val/infer 各自不同)不能泄给 ultralytics."""
        config = config_class()
        kwargs = config.to_ultralytics_kwargs()
        for name in config_class.FRAMEWORK_ONLY_FIELDS:
            assert name not in kwargs, f"{name} 不该传给 ultralytics"
        assert frame_field in config_class.FRAMEWORK_ONLY_FIELDS


# ============================================================
# 2. 缺 yaml → fail-fast + 修复指引(三入口都要有)
# ============================================================

class TestMissingYamlFailFast:
    @pytest.mark.parametrize("builder,config_class,default_name",
                             BUILDERS, ids=["train", "val", "infer"])
    def test_missing_file_raises_with_repair_hint(
        self, builder, config_class, default_name, tmp_path
    ):
        with pytest.raises(FileNotFoundError) as exc_info:
            builder(yaml_path="definitely_missing.yaml", yaml_dir=tmp_path)
        msg = str(exc_info.value)
        assert "YAML 配置文件不存在" in msg
        assert "odp-gen-config" in msg, "错误信息必须给修复命令"
        assert "definitely_missing" in msg

    def test_yaml_path_none_skips_yaml_entirely(self):
        """yaml_path=None: 不加载 yaml, 也不报错."""
        config, merger = build_train_config(
            yaml_path=None, cli_args=Namespace(epochs=50)
        )
        assert config.epochs == 50
        yaml_keys = [
            key for key, meta in merger._metadata.items()
            if meta.source == ConfigSource.YAML
        ]
        assert yaml_keys == []


# ============================================================
# 3. 三源合并 + 8. 溯源链完整
# ============================================================

class TestThreeSourceMergeAndProvenance:
    def test_priority_and_chain(self, write_yaml, tmp_path):
        yaml_file = write_yaml(
            "model: yolo11n.pt\ndata: rsod.yaml\nepochs: 200\n"
            "batch: 32\nlr0: 0.005\n",
            name="train.yaml",
        )
        config, merger = build_train_config(
            yaml_path=yaml_file.name,
            yaml_dir=tmp_path,
            cli_args=Namespace(epochs=300, lr0=0.001),
        )
        # CLI > YAML > DEFAULT
        assert config.epochs  == 300
        assert config.batch   == 32
        assert config.lr0     == 0.001
        assert config.workers == 8          # 没被覆盖 → DEFAULT

        assert merger.get_metadata("epochs").chain_str() == (
            "300(CLI) ← 200(YAML) ← 100(DEFAULT)"
        )

    def test_source_report_groups_by_source(self, write_yaml, tmp_path):
        yaml_file = write_yaml(
            "model: yolo11n.pt\ndata: rsod.yaml\nepochs: 200\n"
            "batch: 32\nlr0: 0.005\n",
            name="train.yaml",
        )
        _, merger = build_train_config(
            yaml_path=yaml_file.name,
            yaml_dir=tmp_path,
            cli_args=Namespace(epochs=300, lr0=0.001),
        )
        report = merger.get_source_report()
        assert "CLI (2 项)"     in report
        assert "YAML (3 项)"    in report
        assert "DEFAULT (57 项)" in report

    def test_provenance_is_complete_and_uses_builtin_sources(
        self, write_yaml, tmp_path
    ):
        """★ 溯源链完整: 每个生效字段恰好一条溯源, 来源只能是内置三档."""
        yaml_file = write_yaml(
            "model: yolo11n.pt\ndata: rsod.yaml\nepochs: 200\n",
            name="train.yaml",
        )
        config, merger = build_train_config(
            yaml_path=yaml_file.name,
            yaml_dir=tmp_path,
            cli_args=Namespace(epochs=300),
        )
        assert isinstance(config, YOLOTrainConfig)

        # eff = 本次合并真正生效的字段集合(每个都该有一个溯源节点)
        eff = set(merger._metadata)

        # 所有"有非 None 默认值"的字段都必须有溯源
        non_none = _non_none_default_fields(YOLOTrainConfig)
        assert non_none <= eff, (
            f"这些默认值字段没有溯源: {sorted(non_none - eff)}"
        )
        # 溯源键不能是模型里不存在的字段
        assert eff <= set(YOLOTrainConfig.model_fields)

        # 无"未知"来源: 只允许内置三档
        labels = {meta.source_label for meta in merger._metadata.values()}
        assert labels <= BUILTIN_SOURCES, f"出现未知来源: {labels - BUILTIN_SOURCES}"

        # "字段数 + 3": 完整溯源域 = 生效字段 + 三个内置来源档位
        assert len(eff | BUILTIN_SOURCES) == len(eff) + 3

        audit = merger.to_audit_log()
        assert audit["fields_count_total"] == len(eff)
        assert set(audit["fields_by_source"]) <= BUILTIN_SOURCES
        # 每个生效字段恰好落进一个来源桶(不多不少)
        bucketed = [
            key for keys in audit["fields_by_source"].values() for key in keys
        ]
        assert sorted(bucketed) == sorted(eff)

    def test_extra_sources_inserted_between_yaml_and_cli(self, write_yaml, tmp_path):
        yaml_file = write_yaml("epochs: 200\n", name="train.yaml")
        config, merger = build_train_config(
            yaml_path=yaml_file.name,
            yaml_dir=tmp_path,
            extra_sources=[("ENV", {"epochs": 250})],
            cli_args=Namespace(epochs=300),
        )
        assert config.epochs == 300
        assert merger.get_metadata("epochs").chain_str() == (
            "300(CLI) ← 250(ENV) ← 200(YAML) ← 100(DEFAULT)"
        )

    def test_val_and_infer_modes_end_to_end(self, write_yaml, tmp_path):
        val_file = write_yaml("split: test\nconf: 0.5\n", name="val.yaml")
        val_config, val_merger = build_val_config(
            yaml_path=val_file.name, yaml_dir=tmp_path,
            cli_args=Namespace(iou=0.7),
        )
        assert isinstance(val_config, YOLOValConfig)
        assert (val_config.split, val_config.conf, val_config.iou) == ("test", 0.5, 0.7)
        assert val_config.to_ultralytics_kwargs().get("task") is None

        infer_config, infer_merger = build_infer_config(
            yaml_path=None, cli_args={"source": "video.mp4", "conf": 0.6},
        )
        assert isinstance(infer_config, YOLOInferConfig)
        assert infer_config.source == "video.mp4"
        assert infer_config.to_ultralytics_kwargs()["source"] == "video.mp4"
        assert infer_merger.get_metadata("source").chain_str() == "video.mp4(CLI)"

    def test_dry_run_returns_no_config_but_keeps_trace(self, write_yaml, tmp_path):
        yaml_file = write_yaml("epochs: 200\n", name="train.yaml")
        config, merger = build_train_config(
            yaml_path=yaml_file.name, yaml_dir=tmp_path, dry_run=True
        )
        assert config is None
        assert isinstance(merger, ConfigMerger)
        assert merger.get_metadata("epochs").chain_str() == "200(YAML) ← 100(DEFAULT)"


# ============================================================
# 4. extra="forbid" 拼写防线
# ============================================================

class TestExtraForbidSpellingGuard:
    def test_typo_in_cli_is_rejected_with_source_chain(self):
        with pytest.raises(ValidationError) as exc_info:
            build_train_config(yaml_path=None, cli_args={"epchs": 300})
        msg = str(exc_info.value)
        assert "epchs" in msg
        assert "Extra inputs are not permitted" in msg
        assert "[来源:" in msg, "合并器应把来源链注入错误信息"

    def test_typo_in_yaml_is_rejected(self, write_yaml, tmp_path):
        yaml_file = write_yaml("epochs: 200\nepchs: 300\n", name="train.yaml")
        with pytest.raises(ValidationError) as exc_info:
            build_train_config(yaml_path=yaml_file.name, yaml_dir=tmp_path)
        assert "epchs" in str(exc_info.value)


# ============================================================
# 5. batch 的 bool / float / 负值语义
# ============================================================

class TestBatchSemantics:
    @pytest.mark.parametrize("value,expected", [
        (8,    8),      # 固定值
        (-1,  -1),      # 自动批次
        (0.7,  0.7),    # 显存利用率
        (1.0,  1.0),    # 上界
    ])
    def test_valid_values_kept(self, value, expected):
        assert YOLOTrainConfig(batch=value).batch == expected

    def test_bool_is_rejected_with_type_error(self):
        """★ bool 是 int 子类, 必须在类型转换前拦掉."""
        with pytest.raises(TypeError, match="batch 不能是 bool 类型"):
            YOLOTrainConfig(batch=True)

    @pytest.mark.parametrize("bad", [0, -5, -100])
    def test_non_minus_one_int_rejected(self, bad):
        with pytest.raises(ValidationError, match="-1"):
            YOLOTrainConfig(batch=bad)

    @pytest.mark.parametrize("bad", [1.5, 0.0, -0.3])
    def test_float_out_of_range_rejected(self, bad):
        with pytest.raises(ValidationError):
            YOLOTrainConfig(batch=bad)

    def test_batch_semantics_survive_the_build_pipeline(self):
        config, _ = build_train_config(yaml_path=None, cli_args={"batch": -1})
        assert config.batch == -1


# ============================================================
# 6. imgsz 非 32 倍数: 只 warning, 不 raise
# ============================================================

class TestImgszWarningOnly:
    def test_non_multiple_of_32_warns_but_passes(self):
        with pytest.warns(UserWarning, match="不是 32 的倍数"):
            config = YOLOTrainConfig(imgsz=641)
        assert config.imgsz == 641       # ★ 只是提醒, 值照收

    def test_multiple_of_32_is_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")          # 任何 warning 都算失败
            assert YOLOValConfig(imgsz=672).imgsz == 672

    def test_below_minimum_raises(self):
        """非 32 倍数只 warn; 但小于下界(32)仍是硬错."""
        with pytest.raises(ValidationError):
            YOLOTrainConfig(imgsz=16)


# ============================================================
# 7. 敏感字段 mask
# ============================================================

class _SmokeSensitiveTrain(YOLOTrainConfig):
    """给真实训练配置挂一个 sensitive 字段, 验证 mask 链路端到端可通."""
    api_key: Optional[str] = Field(
        default=None,
        description="冒烟用的敏感字段",
        json_schema_extra={"group": "核心参数", "sensitive": True},
    )


class TestSensitiveMask:
    def test_masked_in_dump_and_report(self):
        config = _SmokeSensitiveTrain(api_key="super_secret_123")
        assert config.mask_sensitive_dump()["api_key"] == "***"

        merger = ConfigMerger()
        merger.merge(_SmokeSensitiveTrain, sources=[
            (ConfigSource.CLI, {"api_key": "super_secret_123", "epochs": 7}),
        ])
        report = merger.get_source_report()
        assert "super_secret_123" not in report, "敏感值绝不能出现在报告里"
        assert "api_key = ***" in report

    def test_none_sensitive_field_not_masked(self):
        assert _SmokeSensitiveTrain().mask_sensitive_dump()["api_key"] is None

    def test_base_config_exposes_sensitive_field_names(self):
        assert BaseConfig().sensitive_field_names() == set()
        assert _SmokeSensitiveTrain.sensitive_field_names() == {"api_key"}