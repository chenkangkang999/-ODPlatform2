# test_generator.py
"""ConfigGenerator: YAML 模板生成.

覆盖生成器五类行为:
    1. 正常生成          —— 三份 train / val / infer, 字段与注释齐全
    2. 文件已存在跳过     —— 不覆盖、不重复追加键
    3. --overwrite 备份   —— 备份文件名可断言, 原内容进备份
    4. 空字段(None/"")   —— 渲染成 null / "" 而非被跳过
    5. 特殊字符转义       —— 写出的 yaml 能被 safe_load 原样读回(round-trip)
  + CLI(argparse) 非法 choice 的退出码语义
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

import pytest
import yaml
from pydantic import Field

from od_platform.runtime_config.base      import BaseConfig
from od_platform.runtime_config.generator import ConfigGenerator, main
from od_platform.runtime_config.registry  import CONFIG_REGISTRY
from od_platform.runtime_config.train     import YOLOTrainConfig
from od_platform.runtime_config.val       import YOLOValConfig
from od_platform.runtime_config.infer     import YOLOInferConfig
from od_platform.runtime_config.loaders   import YAMLLoader


class TestFormatValue:
    def setup_method(self):
        self.g = ConfigGenerator()

    def test_none(self):
        assert self.g._format_value(None) == "null"

    @pytest.mark.parametrize("val,expected", [
        (True,  "true"),
        (False, "false"),    # ★ bool 必须在 int 之前判
    ])
    def test_bool(self, val, expected):
        assert self.g._format_value(val) == expected

    @pytest.mark.parametrize("val,expected", [
        (16,    "16"),
        (-1,    "-1"),
        (0.001, "0.001"),
    ])
    def test_numbers(self, val, expected):
        assert self.g._format_value(val) == expected

    @pytest.mark.parametrize("val,expected", [
        ("yolo.pt", "yolo.pt"),       # 普通字符串无引号
        ("a:b",    '"a:b"'),          # 含 ':' 加引号
        ("#tag",   '"#tag"'),         # 含 '#'
        ("[x]",    '"[x]"'),          # 含 '['
    ])
    def test_strings(self, val, expected):
        assert self.g._format_value(val) == expected

    def test_lists(self):
        assert self.g._format_value([1, 2, 3]) == "[1, 2, 3]"
        assert self.g._format_value([])        == "[]"
        assert self.g._format_value([True, False]) == "[true, false]"


class TestGenerate:
    def test_basic_generation(self, tmp_path):
        g = ConfigGenerator()
        out = tmp_path / "train.yaml"
        assert g.generate(YOLOTrainConfig, out, overwrite=True) is True
        assert out.exists()

        content = out.read_text(encoding="utf-8")
        # 头部
        assert "自动生成时间" in content
        assert "odp-gen-config" in content        # ★ FAQ 含"如何重新生成"指引
        # 分组
        for group in ["核心参数", "训练控制", "优化器配置", "高级设置"]:
            assert f"# {group}" in content
        # 关键字段
        for field in ["task", "experiment_name", "epochs", "lr0"]:
            assert f"{field}:" in content
        # 注释格式
        assert "# 示例: " in content
        assert "# 提示:"  in content
        # FAQ
        assert "CLI > YAML > DEFAULT" in content

    def test_generated_yaml_roundtrips(self, tmp_path):
        """★ 关键: 生成的 YAML 必须能被 YAMLLoader 完整解析回来."""
        out = tmp_path / "train.yaml"
        ConfigGenerator().generate(YOLOTrainConfig, out, overwrite=True)

        parsed = YAMLLoader().load(out)
        # 默认值都对应得上
        assert parsed["epochs"]    == 100
        assert parsed["batch"]     == 16
        assert parsed["amp"]       is True       # bool 保持 bool
        assert parsed["task"]      == "detect"

    def test_default_no_overwrite(self, tmp_path):
        out = tmp_path / "train.yaml"
        g = ConfigGenerator()
        g.generate(YOLOTrainConfig, out, overwrite=True)

        # 用户改 yaml
        original = out.read_text(encoding="utf-8")
        modified = original.replace("epochs: 100", "epochs: 999")
        out.write_text(modified, encoding="utf-8")

        # 再生成 — 不该覆盖
        assert g.generate(YOLOTrainConfig, out) is False
        assert "epochs: 999" in out.read_text(encoding="utf-8")

    def test_overwrite_true_overrides(self, tmp_path):
        out = tmp_path / "train.yaml"
        out.write_text("epochs: 999\n", encoding="utf-8")
        ConfigGenerator().generate(YOLOTrainConfig, out, overwrite=True)
        assert "epochs: 100" in out.read_text(encoding="utf-8")

    def test_val_generation(self, tmp_path):
        out = tmp_path / "val.yaml"
        ConfigGenerator().generate(YOLOValConfig, out, overwrite=True, title="YOLO 验证配置")
        content = out.read_text(encoding="utf-8")
        assert "split: val"      in content
        assert "conf: 0.001"     in content
        assert "iou: 0.6"        in content
        assert "save_json: true" in content

    def test_creates_parent_dir(self, tmp_path):
        nested = tmp_path / "deep" / "subdir" / "train.yaml"
        assert ConfigGenerator().generate(YOLOTrainConfig, nested, overwrite=True) is True
        assert nested.exists()


# ============================================================
# 1. 正常生成 —— 三份模板(train / val / infer)
# ============================================================

class TestAllModesGeneration:
    @pytest.mark.parametrize("name", sorted(CONFIG_REGISTRY))
    def test_template_is_complete(self, name, tmp_path):
        """三份模板都要: 标题 + 分组 + 每个字段 + 示例/提示注释 + FAQ."""
        config_class, title = CONFIG_REGISTRY[name]
        out = tmp_path / f"{name}.yaml"
        assert ConfigGenerator().generate(
            config_class, out, overwrite=True, title=title
        ) is True

        content = out.read_text(encoding="utf-8")
        assert title in content
        assert "自动生成时间" in content
        assert "CLI > YAML > DEFAULT" in content

        for group in config_class().get_field_groups():
            assert f"# {group}" in content, f"{name} 缺分组注释: {group}"
        for field in config_class.model_fields:
            assert f"\n{field}: " in content, f"{name} 缺字段: {field}"

        assert "# 示例: " in content
        assert "# 提示:"  in content

    def test_parsed_template_matches_defaults(self, tmp_path):
        """模板解析回来后, 非 None 默认值必须逐字段对上."""
        for name, (config_class, title) in CONFIG_REGISTRY.items():
            out = tmp_path / f"{name}.yaml"
            ConfigGenerator().generate(config_class, out, overwrite=True, title=title)
            parsed = YAMLLoader().load(out)
            defaults = config_class()
            for field in config_class.model_fields:
                value = getattr(defaults, field)
                if value is None:
                    assert field not in parsed, f"{name}.{field} None 应被 loader 过滤"
                else:
                    assert parsed[field] == value, f"{name}.{field} 值不一致"

    def test_infer_specific_fields_present(self, tmp_path):
        out = tmp_path / "infer.yaml"
        ConfigGenerator().generate(YOLOInferConfig, out, overwrite=True)
        content = out.read_text(encoding="utf-8")
        for group in ["推理控制", "视频流", "输出控制"]:
            assert f"# {group}" in content
        assert "conf: 0.25" in content


# ============================================================
# 2. 文件已存在 → 跳过且绝不覆盖
# ============================================================

class TestSkipExisting:
    def test_skip_leaves_file_byte_identical(self, tmp_path):
        out = tmp_path / "train.yaml"
        ConfigGenerator().generate(YOLOTrainConfig, out, overwrite=True)

        modified = out.read_text(encoding="utf-8").replace(
            "epochs: 100", "epochs: 999"
        )
        out.write_text(modified, encoding="utf-8")
        snapshot = out.read_bytes()

        assert ConfigGenerator().generate(YOLOTrainConfig, out) is False
        assert out.read_bytes() == snapshot, "跳过时文件必须一个字节都不变"

        text = out.read_text(encoding="utf-8")
        assert "epochs: 999" in text
        # warmup_epochs 等键不含 "\nepochs: "; 这里钉住顶级 epochs 键只出现一次
        assert text.count("\nepochs: ") == 1, "跳过时不应重复追加键"

    def test_skip_on_each_of_three_modes(self, tmp_path):
        for name, (config_class, title) in CONFIG_REGISTRY.items():
            out = tmp_path / f"{name}.yaml"
            gen = ConfigGenerator()
            assert gen.generate(config_class, out, overwrite=True) is True
            before = out.read_bytes()
            assert gen.generate(config_class, out) is False
            assert out.read_bytes() == before


# ============================================================
# 3. --overwrite → 覆盖前备份
# ============================================================

class TestOverwriteBackup:
    def test_backup_created_and_holds_original(self, tmp_path):
        out = tmp_path / "train.yaml"
        ConfigGenerator().generate(YOLOTrainConfig, out, overwrite=True)

        original = out.read_text(encoding="utf-8").replace(
            "epochs: 100", "epochs: 999"
        )
        out.write_text(original, encoding="utf-8")

        assert ConfigGenerator().generate(YOLOTrainConfig, out, overwrite=True) is True

        backups = sorted(tmp_path.glob("train.yaml.bak.*"))
        assert len(backups) == 1
        backup = backups[0]
        # 命名契约: <name>.bak.<YYYYMMDD_HHMMSS>
        stamp = backup.name.rsplit(".bak.", 1)[1]
        assert re.fullmatch(r"\d{8}_\d{6}", stamp), f"时间戳格式不对: {stamp}"
        assert backup.read_text(encoding="utf-8") == original, "原内容必须进备份"
        assert "epochs: 100" in out.read_text(encoding="utf-8")

    def test_backup_false_skips_backup(self, tmp_path):
        out = tmp_path / "train.yaml"
        out.write_text("epochs: 999\n", encoding="utf-8")
        assert ConfigGenerator().generate(
            YOLOTrainConfig, out, overwrite=True, backup=False
        ) is True
        assert list(tmp_path.glob("*.bak.*")) == []

    def test_skip_path_creates_no_backup(self, tmp_path):
        out = tmp_path / "train.yaml"
        gen = ConfigGenerator()
        gen.generate(YOLOTrainConfig, out, overwrite=True)   # 首建, 无原文件可备份
        gen.generate(YOLOTrainConfig, out)                   # 跳过
        assert list(tmp_path.glob("*.bak.*")) == []


# ============================================================
# 4. 空字段 —— None / "" 的渲染契约
# ============================================================

class TestEmptyFieldRendering:
    def test_none_defaults_render_as_null_not_skipped(self, tmp_path):
        out = tmp_path / "train.yaml"
        ConfigGenerator().generate(YOLOTrainConfig, out, overwrite=True)
        content = out.read_text(encoding="utf-8")

        none_fields = [
            name for name, f in YOLOTrainConfig.model_fields.items()
            if f.default is None
        ]
        assert none_fields, "前提: train 存在默认 None 的字段"
        for field in none_fields:
            assert f"\n{field}: null" in content, f"{field} 应渲染成 null"

    def test_null_fields_dropped_by_loader(self, tmp_path):
        """写出的 null 在加载时等同"没写"(交给默认值)."""
        out = tmp_path / "train.yaml"
        ConfigGenerator().generate(YOLOTrainConfig, out, overwrite=True)
        parsed = YAMLLoader().load(out)
        for name, f in YOLOTrainConfig.model_fields.items():
            if f.default is None:
                assert name not in parsed, f"{name} 的 null 应被过滤"

    def test_empty_string_is_quoted_to_stay_string(self, tmp_path):
        assert ConfigGenerator()._format_value("") == '""'

        out = tmp_path / "special.yaml"
        ConfigGenerator().generate(_SpecialCharConfig, out, overwrite=True)
        parsed = yaml.safe_load(out.read_text(encoding="utf-8"))
        assert parsed["empty"] == ""          # 不是 None


# ============================================================
# 5. 特殊字符转义 —— round-trip 必须等价
# ============================================================

class _SpecialCharConfig(BaseConfig):
    """带 YAML 特殊字符默认值的配置(模拟用户把路径/实验名写成默认值)."""
    win_path:  Optional[str] = Field(default="C:\\data\\images",          description="Windows 路径",
                                     json_schema_extra={"group": "核心参数"})
    hash_name: Optional[str] = Field(default="D:\\my data\\a #1.yaml",    description="路径含空格 + #",
                                     json_schema_extra={"group": "核心参数"})
    quoted:    Optional[str] = Field(default='he said "hi"',              description="含双引号",
                                     json_schema_extra={"group": "核心参数"})
    chinese:   Optional[str] = Field(default="中文实验: 测试#1",           description="中文 + 冒号 + #",
                                     json_schema_extra={"group": "核心参数"})
    multiline: Optional[str] = Field(default="line1\nline2",              description="含换行",
                                     json_schema_extra={"group": "核心参数"})
    reserved:  Optional[str] = Field(default="null",                      description="YAML 保留字",
                                     json_schema_extra={"group": "核心参数"})
    leading:   Optional[str] = Field(default="  padded  ",                description="首尾空白",
                                     json_schema_extra={"group": "核心参数"})
    empty:     Optional[str] = Field(default="",                          description="空串",
                                     json_schema_extra={"group": "核心参数"})
    colon_only: Optional[str] = Field(default="a:b",                      description="只含冒号",
                                     json_schema_extra={"group": "核心参数"})


class TestSpecialCharEscaping:
    @pytest.mark.parametrize("raw", [
        "C:\\data\\images",          # ★ 缺陷现场: 反斜杠 + 冒号
        'he said "hi"',
        "中文实验: 测试#1",
        "line1\nline2",
        "tab\there",
        "null",
        "123",
        "  padded  ",
        "",
        "a:b",
        "#tag",
        "[x]",
    ])
    def test_format_value_roundtrips(self, raw):
        """单值层面: _format_value 的输出必须能被 safe_load 原样读回."""
        encoded = ConfigGenerator()._format_value(raw)
        assert yaml.safe_load(encoded) == raw

    @pytest.mark.parametrize("raw", ["null", "line1\nline2", "a:b", "C:\\d\\x"])
    def test_nested_list_strings_roundtrip(self, raw):
        encoded = ConfigGenerator()._format_value([raw, "plain"])
        assert yaml.safe_load(encoded) == [raw, "plain"]

    def test_generated_template_roundtrips_special_chars(self, tmp_path):
        """★ 整份模板层面: 特殊字符默认值写出的 yaml 必须能原样读回."""
        out = tmp_path / "special.yaml"
        ConfigGenerator().generate(_SpecialCharConfig, out, overwrite=True)

        text = out.read_text(encoding="utf-8")
        parsed = yaml.safe_load(text)          # 关键: 不能抛 ScannerError
        defaults = _SpecialCharConfig().model_dump()
        for field in [
            "win_path", "hash_name", "quoted", "chinese", "multiline",
            "reserved", "leading", "empty", "colon_only",
        ]:
            assert parsed[field] == defaults[field], (
                f"{field}: 写出的 YAML 读回来不一致 "
                f"({parsed[field]!r} != {defaults[field]!r})"
            )

    def test_win_path_is_escaped_not_raw(self, tmp_path):
        """回归钉子: Windows 路径会被双引号包裹并转义反斜杠."""
        encoded = ConfigGenerator()._format_value("C:\\data\\images")
        assert encoded == '"C:\\\\data\\\\images"'
        assert yaml.safe_load(f"p: {encoded}")["p"] == "C:\\data\\images"


# ============================================================
# 6. CLI 入口(argparse)退出码语义
# ============================================================

class TestCliEntryPoint:
    def test_invalid_choice_exits_2(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["odp-gen-config", "not_a_mode"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 2
        assert "invalid choice" in capsys.readouterr().err

    def test_missing_name_exits_2(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["odp-gen-config"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 2

    def test_valid_choice_writes_template(self, monkeypatch, tmp_path, capsys):
        out = tmp_path / "infer.yaml"
        monkeypatch.setattr(
            sys, "argv", ["odp-gen-config", "infer", "-o", str(out)]
        )
        main()
        assert "已生成" in capsys.readouterr().out
        assert "YOLO 推理配置" in out.read_text(encoding="utf-8")

    def test_cli_skip_existing_prints_hint(self, monkeypatch, tmp_path, capsys):
        out = tmp_path / "train.yaml"
        monkeypatch.setattr(
            sys, "argv", ["odp-gen-config", "train", "-o", str(out)]
        )
        main()
        capsys.readouterr()
        main()                                   # 第二次: 已存在 → 跳过
        captured = capsys.readouterr().out
        assert "文件已存在" in captured
        assert "--overwrite" in captured

    def test_cli_overwrite_creates_backup(self, monkeypatch, tmp_path):
        out = tmp_path / "val.yaml"
        monkeypatch.setattr(sys, "argv", ["odp-gen-config", "val", "-o", str(out)])
        main()
        out.write_text("split: test\n", encoding="utf-8")

        monkeypatch.setattr(
            sys, "argv",
            ["odp-gen-config", "val", "-o", str(out), "--overwrite"],
        )
        main()
        backups = list(tmp_path.glob("val.yaml.bak.*"))
        assert len(backups) == 1
        assert backups[0].read_text(encoding="utf-8") == "split: test\n"

    def test_cli_no_backup_flag(self, monkeypatch, tmp_path):
        out = tmp_path / "val.yaml"
        monkeypatch.setattr(sys, "argv", ["odp-gen-config", "val", "-o", str(out)])
        main()
        out.write_text("split: test\n", encoding="utf-8")
        monkeypatch.setattr(
            sys, "argv",
            ["odp-gen-config", "val", "-o", str(out), "--overwrite", "--no-backup"],
        )
        main()
        assert list(tmp_path.glob("*.bak.*")) == []