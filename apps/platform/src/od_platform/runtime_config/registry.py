#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : registry.py
# @Author    : 雨霓同学 (ODPlatform team)
# @Project   : ODPlatform
# @Function  : runtime_config 子系统——"名字 → 配置类"的唯一真相
"""配置注册表: 一张共享表, 把 CLI 里的名字映射到配置类.

谁读它:
    - generator.main()  —— odp-gen-config <name> 靠它拿配置类
                           (argparse choices 也直接取自它的 keys)
    - __init__.build_*  —— 便捷构建函数保持同一份映射
    - cli/{transform_data,validate_data,clean_runs} —— resolve_config_ref() 解析 --yaml

为什么是一张显式 dict, 而不是 registry_utils 的自动扫描:
    自动扫描是给"实现集会无限增长、第三方会加"的场景的(D3 的 converters/).
    这里只有几个固定核心模式, 一张表就够 —— 加新模式 = 加一行.
    哪天真要支持第三方注册, 再换成 import_submodules, 接口不变.

★ 追加式扩表(D3 数据流水线 / D4 质检 / runs 保留策略):
    transform / validate / clean 是【新加的行】, 与既有条目各自独立
    —— 加行不影响既有模板的字节内容(test_generator.py 的契约测试守着这点)。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from od_platform.common.paths import RUNTIME_CONFIGS_DIR, runtime_config_path
from od_platform.runtime_config.clean  import CleanSettings
from od_platform.runtime_config.demo  import YOLODemoConfig
from od_platform.runtime_config.infer import YOLOInferConfig
from od_platform.runtime_config.train import YOLOTrainConfig
from od_platform.runtime_config.transform import TransformSettings
from od_platform.runtime_config.val   import YOLOValConfig
from od_platform.runtime_config.validate import ValidateSettings

# 名字 → (配置类, 模板标题). ★ 加新模式 = 加一行, 分发逻辑一行不改.
CONFIG_REGISTRY: dict[str, tuple[type, str]] = {
    "train": (YOLOTrainConfig, "YOLO 训练配置"),
    "val":   (YOLOValConfig,   "YOLO 验证配置"),
    "infer": (YOLOInferConfig, "YOLO 推理配置"),
    # odp-demo: Gradio 演示界面的配置(界面旋钮 + 默认权重名)。
    # ★ 推理参数(conf / iou / imgsz / device ...)【不】在 demo.yaml 里 ——
    #   它们的默认值只在 configs/runtime/infer.yaml, 由 build_infer_config 合并。
    "demo":  (YOLODemoConfig,  "ODPlatform 演示界面配置"),
    # ---- D3 数据流水线(odp-transform)----
    "transform": (TransformSettings, "数据流水线配置(odp-transform)"),
    # ---- D4 数据质检(odp-validate)----
    "validate":  (ValidateSettings,  "数据质检配置(odp-validate)"),
    # ---- runs/ 保留策略(odp-clean)----
    "clean":     (CleanSettings,     "运行现场保留策略配置(odp-clean)"),
}


# ============================================================
# --yaml 引用解析 (odp-transform / odp-validate / odp-clean 共用)
# ============================================================

def resolve_config_ref(
    ref: Optional[Union[str, Path]],
    name: str,
    *,
    require_exists: bool = True,
) -> Optional[Path]:
    """把 CLI 的 ``--yaml`` 引用解析成 yaml 路径.

    三种写法都接受(与 odp-train 的 --config 口径一致):
        不传(None)        → <CONFIGS_DIR>/runtime/<name>.yaml
        裸名字("my_cfg")   → <CONFIGS_DIR>/runtime/my_cfg.yaml
        路径("a/b.yaml")   → 原样使用(含绝对路径)

    Args:
        ref:            CLI 给的引用; None = 没表态
        name:           该命令的默认配置名(transform / validate / clean)
        require_exists: True(默认)= 没传且默认模板不存在时返回 None —— 让
                        "还没生成过模板"退化成"全用默认值", 而不是 fail-fast
                        报错(这三个都是既有命令, 不能因为多了一个【可选】的
                        yaml 就变得更容易失败); False = 无条件返回路径。

    Returns:
        要加载的 yaml 路径, 或 None(= 不读 yaml, 全用默认值)。
    """
    if ref is None:
        default_path = runtime_config_path(name)
        if require_exists and not default_path.is_file():
            return None
        return default_path

    ref_str = str(ref)
    if "/" in ref_str or "\\" in ref_str:
        return Path(ref_str)                    # 路径: 原样(配错了就该报错)
    return RUNTIME_CONFIGS_DIR / (
        ref_str if ref_str.endswith(".yaml") else f"{ref_str}.yaml")


__all__ = ["CONFIG_REGISTRY", "resolve_config_ref"]