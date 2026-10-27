#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : demo.py
# @Author    : 雨霓同学 (ODPlatform team)
# @Project   : ODPlatform
# @Function  : runtime_config 子系统——odp-demo(Gradio 演示界面)配置类
"""odp-demo 的配置: YOLODemoConfig.

★ 为什么单独一个配置类, 而不是复用 YOLOInferConfig:
  演示界面有**自己的旋钮**(标题 / 绑定地址 / 端口 / 是否分享 / 单次最多几张图 / 默认权重),
  这些跟"怎么推理"无关; 而推理参数(conf / iou / imgsz / device / max_det / batch ...)的
  默认值**只认** ``configs/runtime/infer.yaml`` —— 那是 D5 的推理配置, 由
  :func:`od_platform.runtime_config.build_infer_config` 做三源合并(CLI > YAML > DEFAULT)。
  两套配置各管一半、互不重复, 于是不存在"同一个默认值写两处"的漂移。

★ 默认值在这里**只出现一次**: 字段的 ``Field(default=...)`` 是唯一真相,
  ``configs/runtime/demo.yaml`` 由 ``odp-gen-config demo`` 从本类反射生成
  (生成器读的正是 :meth:`get_field_groups` / :meth:`get_field_metadata`)。
  运行时的生效值 = DEFAULT(本类) < YAML(demo.yaml) < CLI(odp-demo 的参数), 全程带溯源
  (``runtime_config.merger.ConfigMerger``)。

★ 继承 :class:`SettingsBase`, 不继承 ``BaseConfig``:
  SettingsBase 只带**能力**(``extra="forbid"`` + 字段元数据反射 + 审计快照), 一个业务字段都没有;
  继承 BaseConfig 则会把 17 个 YOLO 训练/推理字段(batch / imgsz / workers / device / amp / ...)
  塞进 demo.yaml —— 那些项在演示界面里要么不生效、要么已经由 ``infer.yaml`` 负责,
  写了也不生效, 还让 ``extra="forbid"`` 的拼写防线形同虚设(这正是 SettingsBase 存在的理由)。
"""
from __future__ import annotations

from typing import Optional

from pydantic import Field

from od_platform.runtime_config.base import SettingsBase


class YOLODemoConfig(SettingsBase):
    """odp-demo 演示界面的配置(界面旋钮 + 默认权重名)。"""

    # ============================================================
    # 界面设置
    # ============================================================

    title: str = Field(
        default="ODPlatform · odp-demo 目标检测演示",
        description="演示界面标题",
        json_schema_extra={
            "group": "界面设置",
            "examples": ["ODPlatform · odp-demo 目标检测演示", "鱼类细分类检测演示"],
            "tips": [
                "显示在界面顶部的标题(也作为浏览器标签标题)",
                "改这里不影响任何推理行为",
            ],
            "yaml_comment": "演示界面标题",
        },
    )

    server_name: str = Field(
        default="127.0.0.1",
        description="界面绑定地址",
        json_schema_extra={
            "group": "界面设置",
            "examples": ["127.0.0.1", "0.0.0.0"],
            "tips": [
                "127.0.0.1: 只有本机能访问(默认, 演示/开发用)",
                "0.0.0.0: 局域网内其它机器也能访问(对外演示时才用)",
            ],
            "yaml_comment": "界面绑定地址(127.0.0.1=仅本机)",
        },
    )

    server_port: int = Field(
        default=7860,
        ge=1, le=65535,
        description="界面监听端口",
        json_schema_extra={
            "group": "界面设置",
            "examples": [7860, 7861, 8000],
            "tips": [
                "被占用时换一个端口, 或用 --server-port 传入",
                "端口 < 1024 在 Linux 上需要 root, 不建议",
            ],
            "yaml_comment": "界面监听端口",
        },
    )

    share: bool = Field(
        default=False,
        description="是否生成公网分享链接",
        json_schema_extra={
            "group": "界面设置",
            "examples": [False, True],
            "tips": [
                "True: gradio 生成一个临时公网链接(会经过第三方服务, 慎用)",
                "False: 只有本机/局域网访问",
            ],
            "yaml_comment": "是否生成公网分享链接(会经过第三方服务)",
        },
    )

    inbrowser: bool = Field(
        default=False,
        description="启动后是否自动打开浏览器",
        json_schema_extra={
            "group": "界面设置",
            "examples": [False, True],
            "tips": [
                "False: 手动打开终端打印出来的 URL(服务器/CI 环境必须 False)",
                "True: 本机演示图省事",
            ],
            "yaml_comment": "启动后是否自动打开浏览器",
        },
    )

    max_images: int = Field(
        default=8,
        ge=1,
        description="单次最多处理几张图片",
        json_schema_extra={
            "group": "界面设置",
            "examples": [1, 4, 8, 16],
            "tips": [
                "一次点'开始检测'最多跑几张图(防止误传几百张把显存/内存顶爆)",
                "超出的部分会被截断, 并在结果区给出提示",
            ],
            "yaml_comment": "单次最多处理几张图片",
        },
    )

    # ============================================================
    # 权重与绘制
    # ============================================================

    weight: Optional[str] = Field(
        default="train-20261027-124755-fish-yolo11n-best",
        description="默认权重(按名字解析, 不写路径)",
        json_schema_extra={
            "group": "权重与绘制",
            "examples": [
                "train-20261027-124755-fish-yolo11n-best",
                "yolo11n",
                "train-20261027-121740-synth-yolo11n-best",
            ],
            "tips": [
                "只写【名字】(可带或不带 .pt 后缀): 由 common/refs.py 依次在"
                " models/trained → models/pretrained → models/checkpoints 里找",
                "默认值 = 本仓库已训好的 fish(BengalDeltaFish 33 类)权重",
                "界面上可以随时换成下拉里的其它权重, 也可以直接输入名字",
                "留空(null)时退化成'第一个可用的已训练权重'",
            ],
            "yaml_comment": "默认权重名(按 common/refs.py 的搜索顺序解析, 不写盘符/绝对路径)",
        },
    )

    pipeline_yaml: Optional[str] = Field(
        default=None,
        description="美化/帧源配置(颜色·标签映射·字体·美化开关)",
        json_schema_extra={
            "group": "权重与绘制",
            "examples": [None, "infer_pipeline", "infer_pipeline.yaml"],
            "tips": [
                "None: 用 defaults 里与 odp-infer 完全相同的落点"
                " (paths.runtime_config_path('infer_pipeline'))",
                "配色/标签映射/字体/美化开关都读这份 yaml 的 visualization 段,"
                " 界面代码里不另写一套颜色",
                "只写名字会补 .yaml 并在 configs/runtime/ 下找; 也可给路径",
            ],
            "yaml_comment": "美化/帧源配置(None=用与 odp-infer 同一个默认落点)",
        },
    )


# ============================================================
# 一键 build (与 build_transform_config / build_validate_config 同构,
#            走同一套 runtime_config._build_config 流水线)
# ============================================================

def build_demo_config(
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
    """一键完成 demo 配置流水线: YAML + CLI (+ 自定义源) → 合并 → 验证.

    Args:
        yaml_path:     YAML 文件名 / 路径. None = 不读 yaml, 全用默认值.
                       界面的调用方(visualization/gradio_app.py)显式传
                       ``paths.runtime_config_path("demo")``; 传 None 则表示
                       "不读模板, 全部走模型默认值"。
        cli_args:      argparse.Namespace 或 dict. None = 没有 CLI 覆盖.
        yaml_dir:      YAML 目录, 默认 paths.RUNTIME_CONFIGS_DIR.
        cli_exclude:   CLI 额外排除字段 (并入 CLILoader.DEFAULT_EXCLUDE).
        cli_mapping:   CLI 参数名映射 (CLI 名 → Pydantic 字段名).
        extra_sources: 额外配置源 [(source_id, dict), ...], 插在 YAML 与 CLI 之间.
        track_sources: 是否追踪溯源 (默认 True).
        dry_run:       True = 只合并不验证, 返回 (None, merger).

    Returns:
        正常模式: ``(YOLODemoConfig, ConfigMerger)``
        dry_run:  ``(None, ConfigMerger)``
    """
    # 延迟 import: _build_config 住在 runtime_config/__init__.py, 顶层 import 会成环
    from od_platform.runtime_config import _build_config

    return _build_config(
        YOLODemoConfig,
        yaml_path=yaml_path,
        cli_args=cli_args,
        yaml_dir=yaml_dir,
        cli_exclude=cli_exclude,
        cli_mapping=cli_mapping,
        extra_sources=extra_sources,
        track_sources=track_sources,
        dry_run=dry_run,
    )


__all__ = ["YOLODemoConfig", "build_demo_config"]