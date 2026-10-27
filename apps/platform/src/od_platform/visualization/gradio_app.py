#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : gradio_app.py
# @Author    : 雨霓同学 (ODPlatform team)
# @Project   : ODPlatform
# @Function  : odp-demo —— Gradio 演示界面（D7 visualization 的"让人看得见"那一环）
"""Gradio 演示界面: 上传图片 → 推理 → 标注图 + 检出表 + 汇总.

★ 本模块只做**外壳**: 配置、读图、绘制全部复用既有子系统, 一行推理逻辑都不重写。

============================================================================
本模块的默认可调项与其【唯一来源】（界面代码里没有任何隐藏魔法数）
============================================================================
===================  ==========================================================
界面项               默认值的唯一来源
===================  ==========================================================
conf / iou / imgsz   ``configs/runtime/infer.yaml`` —— 经 D5 的
/ device             :func:`runtime_config.build_infer_config` 做三源合并
                     (CLI > YAML > DEFAULT) + Pydantic 校验。
                     ★ 界面把滑块的值当成"CLI 源"传进去, 与 odp-infer 完全同路;
                     控件上显示的"默认值 + 来源(YAML/DEFAULT)"取自合并器的溯源链,
                     不是猜的。
max_det / batch /    同上(同一个 infer.yaml 的生效值)。界面上不暴露, 由配置生效,
classes / augment /  与 odp-infer 一致 —— 想调就改 infer.yaml。
agnostic_nms / ...
标题 / 绑定地址 /    ``configs/runtime/demo.yaml``(= ``YOLODemoConfig`` 的 Field 默认值,
端口 / 是否分享 /    见 ``runtime_config/demo.py``; demo.yaml 由
自动开浏览器 /       ``odp-gen-config demo`` 从该模型反射生成)
单次最多几张图 /
默认权重名
配色 / 标签映射 /    ``configs/runtime/infer_pipeline.yaml`` 的 ``visualization`` 段,
字体 / 美化开关      经 :func:`inference.pipeline_config.load_pipeline_config` 读入
                     (与 odp-infer 读的是同一份文件、同一批字段)。
                     ``visualization.enabled=false`` → 退回 YOLO 原生 ``plot()``,
                     等价于 odp-infer 的 ``--no-viz``。本界面**不**画 HUD:
                     HUD 是实时视频的帧率面板, 静态图没有它的位置。
                     ★ 同一份 yaml 的 ``output`` 段(落盘文件名/编码器/审计快照)
                     对本界面**不生效** —— 它不落盘, 结果只回到浏览器。
权重文件路径         名字 → 路径由 ``common/refs.py`` 解析
                     (``models/trained`` → ``pretrained`` → ``checkpoints``);
                     候选清单由 ``common/paths.py`` 的三个目录动态扫出, 不写死。
日志                  ``common/logging_utils.get_logger(base_path=paths.LOGGING_DIR,
                     log_type="demo")`` —— 端私有日志区, 绝对路径, 不随工作目录漂移。
滑块的 min/max        ``YOLOInferConfig`` 字段的 Pydantic 约束(ge/le)反射而来;
                     步长 = 区间 / ``SLIDER_STEPS``(纯界面刻度, 不是业务参数)。
===================  ==========================================================

============================================================================
为什么没有直接调 ``inference.InferService.predict()``（唯一一处"绕开"引擎）
============================================================================
InferService 的两个业务接缝都**拿不到"每个框的类别/置信度/坐标"**:

  · ``sinks.OutputSink.write(frame, annotated)`` 只给"画好的图", 没有检测明细;
  · ``hooks.InferHooks.on_frame`` 的 ``FrameEvent`` 虽然声明了 ``detections`` 字段,
    但 ``inference/pipeline.py`` 派发时写死了 ``detections=None``。

于是走 service 只能拿到"标注图 + 每类计数", 拿不到本界面要求的**检出表**
(品种 / 置信度 / 框坐标)。另外 ``InferService.predict()`` 每次调用都会在
``runs/<task>_infer/<name>/`` 下新建一个输出目录(名字默认还自增: predict / predict2 / …),
对一个"点一下就推一次"的演示界面来说, 那是每点一次留一份运行现场 —— 即使靠
``infer_pipeline.yaml`` 的 ``output`` 段关掉审计快照, 目录本身照样要建。

所以这里保留**同一套配置与绘制**, 只把最外层驱动换成"单张图直调":

    YOLO(权重)([BGR 图], **predict_kwargs)  →  BeautifyVisualizer.draw()

  · ``predict_kwargs`` 的取法(白名单 :data:`PREDICT_KEYS` + 过滤 None + ``verbose=False``)
    与 ``InferService.predict()`` 逐行同构;
  · 绘制调用 ``DrawStyle.from_image_size(h, w, **style_overrides)`` 与
    ``inference/service._FrameProcessor.draw()`` 完全一致;
  · 读图走 ``frame_source.create_frame_source``(与 odp-infer 同一个帧源)。
  因此结果与 ``odp-infer --source <图> --model <同名权重>`` 等价。

想彻底走 service? 只差引擎侧两处**最小改动**(本模块没有改引擎代码, 建议见交付报告):
  1. ``inference/pipeline.py`` 派发 ``FrameEvent`` 时把 ``detections`` 真正填上
     (``result.boxes`` → ``[{label, conf, xyxy}, ...]``);
  2. ``inference/service.py`` 把 ``_PREDICT_KEYS`` 提为公开名, 让业务侧共用同一份白名单。
有了这两条, 本模块的 :meth:`DemoRunner.run` 可以整体换成
``InferService.predict(output_sink=..., hooks=...)``。

============================================================================
gradio 是**可选依赖**（pyproject 的 ``[project.optional-dependencies].ui``）
============================================================================
顶层 import 不碰 gradio(也不碰 torch/cv2), :func:`main` 里才延迟 import ——
所以 CI 不装 gradio 也能 import 本模块、跑纯函数与 ``pytest``
(见 ``apps/platform/tests/test_gradio_app.py``)。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from od_platform.common.logging_utils import get_logger
from od_platform.common.paths import (
    CHECKPOINTS_DIR,
    LOGGING_DIR,
    PRETRAINED_MODELS_DIR,
    ROOT_DIR,
    RUNTIME_CONFIGS_DIR,
    TRAINED_MODELS_DIR,
    runtime_config_path,
)
from od_platform.common.refs import resolve_ref, resolve_trained_model
from od_platform.inference.pipeline_config import PipelineConfig, load_pipeline_config
from od_platform.runtime_config import (
    ConfigMerger,
    ConfigSource,
    YOLOInferConfig,
    build_infer_config,
)
from od_platform.runtime_config.demo import YOLODemoConfig, build_demo_config
from od_platform.visualization import BeautifyVisualizer, DrawStyle

LOGGER = logging.getLogger(__name__)


# ============================================================================
# 常量 —— 只放"约定名 / 组件标识", 不放任何可调参数的默认值
# ============================================================================
# D5 / D8 里这两份配置的约定名(与 load_pipeline_config 的默认落点、
# build_infer_config 的默认 yaml 同名, 是框架约定而不是可调项)
DEMO_CONFIG_NAME = "demo"
INFER_CONFIG_NAME = "infer"
PIPELINE_CONFIG_NAME = "infer_pipeline"

# 界面暴露的推理旋钮 —— 都是 YOLOInferConfig 的字段, 默认值一律由 D5 从 infer.yaml 合并出来
INFER_UI_KNOBS: tuple[str, ...] = ("conf", "iou", "imgsz", "device")

# "配置来源"面板里一并展示的推理参数: 前 4 个可调, 后 2 个由 infer.yaml 直接生效(界面不暴露)
INFER_SOURCE_FIELDS: tuple[str, ...] = ("conf", "iou", "imgsz", "device", "max_det", "batch")

# CLI 自己的行为开关: 不进任何配置合并(与 odp-infer 的 --log-level / --no-viz 同一纪律)
CLI_ONLY_SWITCHES: tuple[str, ...] = ("log_level", "no_viz")

# 逐帧 predict 的计算参数白名单 —— 与 inference/service.py 的 _PREDICT_KEYS 同一份口径。
# ★ 为什么要有白名单: 盲传整个 config 会带进 batch / workers / cache / rect 等训练向字段,
#   它们传给逐帧 model() 要么报错要么被忽略。
PREDICT_KEYS: tuple[str, ...] = (
    "conf", "iou", "imgsz", "max_det", "classes",
    "agnostic_nms", "augment", "device", "retina_masks",
)

# 权重按名字解析的搜索顺序(与 common/refs.py 的语义一致: 自训练 → 预训练 → 推理查找目录)
WEIGHT_SEARCH_DIRS: tuple[Path, ...] = (
    TRAINED_MODELS_DIR, PRETRAINED_MODELS_DIR, CHECKPOINTS_DIR,
)

# 界面上的"自动"选项 —— 对应 YOLOInferConfig.device 的 None(即"由 ultralytics 自己选")
DEVICE_AUTO = "auto"

# 滑块刻度: 把字段约束区间(ge~le)等分成多少格。★ 这是界面细节, 不是可调业务参数
SLIDER_STEPS = 100

# 检出表列结构(与 detection_rows 一一对应; 多图时"图片"列区分来源)
TABLE_HEADERS: list[str] = ["图片", "#", "品种", "置信度", "x1", "y1", "x2", "y2"]
TABLE_DATATYPES: list[str] = ["str", "number", "str", "number", "number", "number", "number", "number"]

# 退出码(与 README / manual 的说明一致)
EXIT_OK = 0
EXIT_LAUNCH_FAILED = 1
EXIT_USAGE = 2
EXIT_NO_GRADIO = 2

GRADIO_MISSING_MESSAGE = (
    "odp-demo 需要可选依赖 gradio, 但当前环境里没有找到它。\n"
    "请先安装(可选依赖组 ui), 例如在仓库根执行:\n"
    '    .\\.venv\\Scripts\\python.exe -m pip install -e "./apps/platform[ui]"\n'
    "或者只装 gradio 本身:\n"
    "    .\\.venv\\Scripts\\python.exe -m pip install gradio\n"
    "(引擎本体 odp-init / odp-infer / odp-train / odp-eval / odp-clean 都不需要 gradio。)"
)


class GradioMissingError(ImportError):
    """环境里没有 gradio(可选依赖组 ui 未安装)。"""


# gradio "真的可用吗"的探针: 下面这些组件 build_app 全都要用。
# ★ 为什么不能只捕 ImportError: ``pip uninstall gradio`` 之后 site-packages 里的
#   ``gradio/`` 目录往往还有残留(gradio 运行时写下的 media_assets / hash_seed.txt),
#   于是 ``import gradio`` 会成功拿到一个**空的命名空间包**(没有 __file__、也没有任何组件),
#   一路走到 ``gr.Blocks`` 才炸 AttributeError。这里把这种"半卸载"状态也识别成"没装"。
GRADIO_REQUIRED_ATTRS: tuple[str, ...] = (
    "Blocks", "Markdown", "Row", "Column", "Files", "Dropdown", "Slider",
    "Number", "Checkbox", "Button", "Gallery", "Dataframe",
)


def _import_gradio():
    """延迟 import gradio —— 本模块顶层不碰它, 所以没装 gradio 也能 import / 跑纯函数。"""
    try:
        import gradio as gr
    except ImportError as exc:
        raise GradioMissingError(GRADIO_MISSING_MESSAGE) from exc

    missing = [name for name in GRADIO_REQUIRED_ATTRS if not hasattr(gr, name)]
    if missing:
        raise GradioMissingError(
            f"{GRADIO_MISSING_MESSAGE}\n"
            f"(另外: 这个 gradio 是残缺的 —— 缺少 {missing}。通常是 pip 卸载后 "
            f"site-packages 里还残留着 gradio/ 的运行时文件, 重装一次即可。)"
        )
    return gr


# ============================================================================
# 纯函数 1: 检测明细 → 表格行 / 汇总 / Markdown（不依赖 gradio, 可直接单元测试）
# ============================================================================
def _relpath(path: Any) -> str:
    """显示用路径: 能相对仓库根就相对(README/日志里不出现盘符), 否则原样。"""
    try:
        return str(Path(path).resolve().relative_to(ROOT_DIR))
    except (ValueError, OSError):
        return str(path)


def _format_value(value: Any) -> str:
    """配置值 → 给人看的字符串(None / bool 按 YAML 的写法展示)。"""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _to_numpy(value: Any):
    """torch 张量 / numpy 数组 / 列表 → numpy 数组(鸭子类型, 不 import torch)。"""
    if value is None:
        return None
    import numpy as np            # numpy 是 torch/ultralytics 的硬依赖; 延迟 import 保持模块轻量
    for attr in ("detach", "cpu"):
        if hasattr(value, attr):
            value = getattr(value, attr)()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _class_name(names: Any, class_id: int) -> str:
    """类别 id → 类别名。ultralytics 的 ``model.names`` 是 ``{0: 'name'}``; 也容忍 list。"""
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    try:
        return str(names[class_id])
    except (IndexError, KeyError, TypeError):
        return str(class_id)


def _display_name(detection: dict, display_name: Any = None) -> str:
    """检测明细里的类名 → 显示名(display_name 通常是 BeautifyVisualizer.display_name)。"""
    class_name = detection.get("cls")
    if display_name is None:
        return "" if class_name is None else str(class_name)
    return str(display_name(class_name))


def detections_from_yolo_result(result: Any, names: Any) -> list[dict]:
    """ultralytics ``Results[0]`` → 绘制/表格用的检测明细。

    返回格式就是 D7 的 :meth:`BeautifyVisualizer.from_yolo_results` 的输出::

        [{"cls": "Mola", "conf": 0.91, "xyxy": [x1, y1, x2, y2]}, ...]

    ★ 这里**不**自己解析 boxes 数组: 把三段(torch 张量 / numpy / list)统一成 numpy 后
      直接交给引擎的 ``from_yolo_results`` —— "类名与坐标怎么对齐"只由引擎说了算。
    """
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    try:
        if len(boxes) == 0:
            return []
    except TypeError:              # 轻量替身没有 __len__ 时不拦, 交给下面的数组转换
        pass

    xyxy = _to_numpy(getattr(boxes, "xyxy", None))
    confidences = _to_numpy(getattr(boxes, "conf", None))
    class_ids = _to_numpy(getattr(boxes, "cls", None))

    labels: list[str] = []
    if class_ids is not None:
        labels = [_class_name(names, int(round(float(value)))) for value in class_ids.reshape(-1)]

    return BeautifyVisualizer.from_yolo_results(
        boxes=xyxy, confidences=confidences, labels=labels,
    )


def detection_rows(detections: Sequence[dict], *, display_name: Any = None,
                   image_name: str = "") -> list[list[Any]]:
    """检测明细 → 表格行 ``[图片, #, 品种, 置信度, x1, y1, x2, y2]``。

    列结构与 :data:`TABLE_HEADERS` 一一对应(单图时"图片"列写文件名, 多图时用来区分来源)。
    坐标与置信度都做了定长舍入, 免得表格里出现 17 位小数。
    """
    rows: list[list[Any]] = []
    for index, detection in enumerate(detections, start=1):
        coords = list(detection.get("xyxy") or [])
        box = [round(float(value), 1) for value in coords] if len(coords) == 4 else [None] * 4
        confidence = detection.get("conf")
        rows.append([
            image_name,
            index,
            _display_name(detection, display_name),
            round(float(confidence), 4) if confidence is not None else None,
            box[0], box[1], box[2], box[3],
        ])
    return rows


def summarize_detections(detections: Sequence[dict], *, display_name: Any = None) -> dict[str, Any]:
    """检出汇总 ``{"total": 总数, "per_class": {品种: 计数}, "classes": 品种数}``。

    品种按"计数降序 → 名字升序"排(与 odp-infer 日志里"各类别检测数"的排法一致)。
    """
    per_class: dict[str, int] = {}
    for detection in detections:
        name = _display_name(detection, display_name)
        per_class[name] = per_class.get(name, 0) + 1
    ordered = dict(sorted(per_class.items(), key=lambda item: (-item[1], item[0])))
    return {"total": len(detections), "per_class": ordered, "classes": len(ordered)}


def format_summary_markdown(summary: dict[str, Any], *, images: int = 0,
                            params: Optional[str] = None,
                            notes: Optional[Sequence[str]] = None) -> str:
    """汇总 → Markdown(总检出数 / 各品种计数 / 本次参数 / 附加提示)。"""
    lines: list[str] = ["### 检出汇总", ""]
    lines.append(
        f"**总检出数：{summary['total']}** ｜ 图片：{images} 张 ｜ 品种：{summary['classes']} 种"
    )

    per_class = summary.get("per_class") or {}
    if per_class:
        lines += ["", "| 品种 | 数量 |", "|---|---|"]
        lines += [f"| {name} | {count} |" for name, count in per_class.items()]
    else:
        lines += ["", "_没有检出任何目标 —— 可以试试调低 conf 阈值。_"]

    if params:
        lines += ["", f"本次参数：{params}"]
    for note in (notes or []):
        lines += ["", note]
    return "\n".join(lines)


def config_source_rows(config: Any, merger: Optional[ConfigMerger],
                       field_names: Sequence[str]) -> list[list[str]]:
    """``[项, 生效值, 来源]`` 三列 —— 来源取自 D5 合并器记下的溯源(YAML / DEFAULT), 不是猜的。"""
    rows: list[list[str]] = []
    for name in field_names:
        metadata = merger.get_metadata(name) if merger is not None else None
        source = metadata.source_label if metadata is not None else ConfigSource.DEFAULT.value
        rows.append([name, _format_value(getattr(config, name, None)), source])
    return rows


def format_config_source_markdown(*, demo_rows: Sequence[Sequence[str]],
                                  infer_rows: Sequence[Sequence[str]],
                                  demo_path: Path, infer_path: Path,
                                  pipeline_path: Path, weight_ref: str,
                                  weight_path: Path, beautify: bool) -> str:
    """界面里的"配置来源"面板: 逐项列出每个默认值的唯一出处(评审用一眼看穿零硬编码)。"""
    lines: list[str] = [
        "### 配置来源（每个可调项的默认值只有一个出处）",
        "",
        f"- 界面旋钮 ← `{_relpath(demo_path)}`（DEFAULT < YAML < CLI 三源合并, 带溯源）",
        f"- 推理参数 ← `{_relpath(infer_path)}`（D5 `build_infer_config` 合并, 与 odp-infer 同一份）",
        f"- 配色 / 标签映射 / 字体 / 美化开关 ← `{_relpath(pipeline_path)}` 的 `visualization` 段",
        f"- 默认权重 ← 名字 `{weight_ref}` → `{_relpath(weight_path)}`（`common/refs.py` 按名解析）",
        f"- 美化绘制 = `{_format_value(beautify)}`（取自上面那份 `visualization.enabled`）",
        "",
        "| 界面项 | 生效值 | 来源 |",
        "|---|---|---|",
    ]
    lines += [f"| {name} | `{value}` | {source} |" for name, value, source in demo_rows]
    lines += ["", "| 推理项 | 生效值 | 来源 |", "|---|---|---|"]
    lines += [f"| {name} | `{value}` | {source} |" for name, value, source in infer_rows]
    lines += [
        "",
        "> `来源 = YAML` → 这一项由配置文件写定；`来源 = DEFAULT` → 配置里没写, 用的是 Pydantic "
        "模型的默认值。两种来源都由 D5 合并器记录, 界面只负责显示。",
    ]
    return "\n".join(lines)


# ============================================================================
# 纯函数 2: 权重按名字解析 + 候选清单（全部走 common/paths.py 与 common/refs.py）
# ============================================================================
def discover_weight_refs() -> list[str]:
    """扫出所有可选权重的**名字**(不含目录、不含后缀) —— 候选清单动态生成, 不写死。"""
    names: list[str] = []
    for directory in WEIGHT_SEARCH_DIRS:
        if not Path(directory).is_dir():
            continue
        for weight in sorted(Path(directory).glob("*.pt")):
            if weight.stem not in names:
                names.append(weight.stem)
    return names


def default_weight_ref() -> str:
    """``demo.yaml`` 的 weight 为空时的兜底: 扫到的第一个权重名(扫不到就给空串)。"""
    discovered = discover_weight_refs()
    return discovered[0] if discovered else ""


def weight_choices(weight_ref: Optional[str] = None) -> list[str]:
    """下拉候选: 当前默认权重排第一, 其余按扫描顺序去重(不写死任何清单)。"""
    ordered: list[str] = []
    for ref in [weight_ref, *discover_weight_refs()]:
        if ref and ref not in ordered:
            ordered.append(ref)
    return ordered


def resolve_weight_ref(ref: Any) -> Path:
    """权重**名字** → 权重路径。只按名字解析, 源码里不出现盘符 / 绝对路径。

    顺序与 ``common/refs.py`` 的语义一致: ``models/trained`` → ``pretrained`` →
    ``checkpoints``; 都没命中时回落到 ``models/trained`` 下的既定落点
    (交给加载时报"文件不存在 + 搜索目录")。显式路径(含分隔符)由 ``resolve_ref`` 原样接受。
    """
    name = str(ref or "").strip()
    if not name:
        raise ValueError(
            "没有可用的权重名: configs/runtime/demo.yaml 的 weight 为空, "
            "且 models/trained 里也没扫到 *.pt。请在 demo.yaml 里填一个权重名, "
            "或用 odp-demo --weight <名字> 传入。"
        )
    for directory in WEIGHT_SEARCH_DIRS:
        candidate = resolve_ref(name, base_dir=directory, default_suffix=".pt")
        if candidate.exists():
            return candidate
    return resolve_trained_model(name)


def load_yolo(weight_path: Any) -> Any:
    """加载 ultralytics 模型(延迟 import: 顶层不把 torch/ultralytics 拖进来)。"""
    from ultralytics import YOLO
    weight_path = Path(weight_path)
    if not weight_path.exists():
        raise FileNotFoundError(
            f"找不到权重文件：{_relpath(weight_path)}\n"
            f"按名字解析的搜索目录：{', '.join(_relpath(d) for d in WEIGHT_SEARCH_DIRS)}"
        )
    LOGGER.info(f"加载权重: {_relpath(weight_path)}")
    return YOLO(str(weight_path))


# ============================================================================
# 配置来源: demo.yaml / infer.yaml / infer_pipeline.yaml（全部复用既有子系统）
# ============================================================================
def load_demo_config(cli_args: Optional[dict] = None):
    """``demo.yaml`` + CLI 的合并(DEFAULT < YAML < CLI), 带溯源。

    ★ 走 ``runtime_config.demo.build_demo_config`` —— 它与 ``build_train/val/infer_config`` /
      ``build_transform/validate/clean_config`` 是**同一条流水线**(都走内部 ``_build_config``):
      同样的优先级(CLI > YAML > DEFAULT)、同样的溯源链(``ConfigMerger``)、
      同样的 ``ValidationError`` 来源增强。
    ★ 这里显式传 ``paths.runtime_config_path("demo")``(SSoT 落点): 配置文件缺失时
      ``YAMLLoader`` 会 fail-fast 并提示「先生成模板: odp-gen-config demo」。

    Returns:
        ``(YOLODemoConfig, ConfigMerger)`` —— config 是生效值, merger 里带每项的来源。
    """
    return build_demo_config(
        yaml_path=runtime_config_path(DEMO_CONFIG_NAME),
        cli_args=cli_args,
    )


def device_value(label: Any) -> Optional[str]:
    """界面上的 device 文本 → 配置值: ``auto`` / 空 → ``None``(交给 ultralytics 自动选)。"""
    text = "" if label is None else str(label).strip()
    if not text or text.lower() in (DEVICE_AUTO, "none", "null"):
        return None
    return text


def device_label(device: Any) -> str:
    """配置值 → 界面文本(``None`` 显示成"自动")。"""
    return DEVICE_AUTO if device is None else str(device)


def load_infer_config(cli_args: Optional[dict] = None):
    """推理参数的默认值与生效值 —— 复用 D5 的 :func:`build_infer_config`(公开 API)。

    与 odp-infer 读的是同一份 ``configs/runtime/infer.yaml``、同一套三源合并与校验;
    界面只是把滑块的值当"CLI 源"传进去。``device`` 上的 "自动" 在这里归一成 ``None``。
    """
    args = dict(cli_args or {})
    if "device" in args:
        args["device"] = device_value(args["device"])
    return build_infer_config(
        yaml_path=runtime_config_path(INFER_CONFIG_NAME),
        cli_args=args,
    )


def pipeline_path(pipeline_yaml: Any = None) -> Path:
    """美化/帧源配置的落点: demo.yaml 给了就按 ``common/refs.py`` 解析, 否则用 SSoT 默认落点。"""
    if pipeline_yaml:
        return resolve_ref(str(pipeline_yaml), base_dir=RUNTIME_CONFIGS_DIR, default_suffix=".yaml")
    return runtime_config_path(PIPELINE_CONFIG_NAME)


def numeric_bounds(config_class: type, field_name: str) -> tuple[Optional[float], Optional[float]]:
    """从 Pydantic 字段的约束里取数值边界(``ge`` / ``gt`` / ``le`` / ``lt``)。

    ★ 用途: 滑块的 min/max 也由配置模型决定 —— 界面里不写 ``0.0`` / ``1.0`` 这种魔法数。
    """
    lower: Optional[float] = None
    upper: Optional[float] = None
    for constraint in config_class.model_fields[field_name].metadata:
        if hasattr(constraint, "ge"):
            lower = constraint.ge
        elif hasattr(constraint, "gt"):
            lower = constraint.gt
        if hasattr(constraint, "le"):
            upper = constraint.le
        elif hasattr(constraint, "lt"):
            upper = constraint.lt
    return lower, upper


def predict_kwargs(config: YOLOInferConfig) -> dict[str, Any]:
    """逐帧 predict 的计算参数 —— 与 ``InferService.predict()`` 的取法逐行同构:
    白名单 + 过滤 None(不盲传整个 config), 再补一个 ``verbose=False``。"""
    kwargs = {
        name: getattr(config, name)
        for name in PREDICT_KEYS
        if getattr(config, name, None) is not None
    }
    kwargs["verbose"] = False
    return kwargs


def split_cli_args(cli_args: Optional[dict] = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """把 odp-demo 的 CLI 值分成两拨: ``demo.yaml`` 的界面字段 / ``infer.yaml`` 的推理旋钮。

    两套配置各有 ``extra="forbid"``, 混着传会直接报"多余字段" —— 这里按字段名显式分流,
    遇到谁都不认识的名字(拼错的参数)当场报错, 不静默丢弃。
    """
    raw = dict(cli_args or {})
    demo_fields = set(YOLODemoConfig.model_fields)
    infer_fields = set(INFER_UI_KNOBS)

    unknown = sorted(set(raw) - demo_fields - infer_fields)
    if unknown:
        raise ValueError(f"未知的 odp-demo 参数：{unknown}（既不是 demo.yaml 的字段, 也不是推理旋钮）")

    demo_cli = {key: value for key, value in raw.items() if key in demo_fields}
    infer_cli = {key: value for key, value in raw.items() if key in infer_fields}
    return demo_cli, infer_cli


def cli_args_from_namespace(ns: argparse.Namespace) -> dict[str, Any]:
    """argparse Namespace → CLI 值 dict(只放非 None 的字段, 排除 CLI 自己的行为开关)。

    与 ``cli/infer.py`` 的 ``_ns_to_cli_args`` 同一纪律: ``None`` = "没给",
    让下层(YAML / Pydantic 默认值)有机会兜底。
    """
    return {
        key: value
        for key, value in vars(ns).items()
        if not key.startswith("_") and key not in CLI_ONLY_SWITCHES and value is not None
    }


# ============================================================================
# 界面要用到的一整套默认值（一次算清, 全部来自配置文件）
# ============================================================================
@dataclass
class DemoSetup:
    """``build_app`` 需要的一切: 生效配置 + 溯源 + 解析好的路径。

    每个字段都能追到具体文件(见模块 docstring 的"默认可调项与其唯一来源")。
    """
    demo_config: YOLODemoConfig
    demo_merger: ConfigMerger
    infer_config: YOLOInferConfig
    infer_merger: ConfigMerger
    pipeline: PipelineConfig
    demo_path: Path
    infer_path: Path
    pipeline_path: Path
    weight_ref: str
    weight_path: Path
    weight_choices: list[str]
    beautify: bool


def build_setup(cli_args: Optional[dict] = None, *, beautify: Optional[bool] = None) -> DemoSetup:
    """把"配置 → 界面默认值"这条链走一遍(每一步都复用既有子系统, 不自己造默认值)。

    1. ``demo.yaml`` + CLI 三源合并 → 界面旋钮(D5 ``ConfigMerger``)
    2. ``infer.yaml`` + CLI 三源合并 → conf / iou / imgsz / device / max_det ...(D5)
    3. ``infer_pipeline.yaml`` → 配色 / 标签映射 / 字体 / 美化开关(D8)
    4. 权重名字 → 路径(``common/refs.py``)
    """
    demo_cli, infer_cli = split_cli_args(cli_args)
    demo_config, demo_merger = load_demo_config(demo_cli)

    weight_ref = str(demo_config.weight or "").strip() or default_weight_ref()

    infer_cli = dict(infer_cli)
    infer_cli["model"] = weight_ref              # 只为了溯源口径一致(真正加载的路径由 refs 解析)
    infer_config, infer_merger = load_infer_config(infer_cli)

    resolved_pipeline_path = pipeline_path(demo_config.pipeline_yaml)
    pipeline = load_pipeline_config(resolved_pipeline_path)
    weight_path = resolve_weight_ref(weight_ref)

    return DemoSetup(
        demo_config=demo_config,
        demo_merger=demo_merger,
        infer_config=infer_config,
        infer_merger=infer_merger,
        pipeline=pipeline,
        demo_path=runtime_config_path(DEMO_CONFIG_NAME),
        infer_path=runtime_config_path(INFER_CONFIG_NAME),
        pipeline_path=resolved_pipeline_path,
        weight_ref=weight_ref,
        weight_path=weight_path,
        weight_choices=weight_choices(weight_ref),
        beautify=pipeline.viz_enabled if beautify is None else bool(beautify),
    )


# ============================================================================
# 推理: 单张图片 → 标注图 + 检测明细（复用 D8 帧源 + D7 绘制）
# ============================================================================
def read_image_bgr(path: Any):
    """单张图片 → BGR ndarray。走 ``frame_source``(与 odp-infer 同一个读图入口、同一套扩展名判定)。"""
    from od_platform.frame_source import create_frame_source   # 延迟 import: cv2 只在这一刻进内存
    with create_frame_source(str(path)) as source:
        frame = next(iter(source), None)
    if frame is None:
        raise ValueError(f"{Path(path).name} 不是可读取的图片(格式不受支持, 或文件已损坏)")
    return frame.image


@dataclass
class ImageInference:
    """单张图片的一次推理结果(纯数据, 与界面无关)。"""
    path: Path
    image_name: str
    annotated: Any = None                       # BGR ndarray(已画框)
    detections: list = field(default_factory=list)
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


class DemoRunner:
    """一次演示请求的推理器: 配置与模型只准备一次, 逐张图片复用。

    ★ 依赖全部由构造函数注入(构造函数里不做 IO): 测试可以塞一个模型替身进来,
      不需要真权重、不需要 GPU(见 ``apps/platform/tests/test_gradio_app.py``)。
    ★ 权重**每个请求加载一次**, 不跨请求缓存 YOLO 实例: gradio 的回调可能并发,
      共享同一个 ultralytics 模型实例没有线程安全保证 —— 演示场景用"每点一次加载一次"
      换确定性(代价是第一次点击多约一秒)。
    """

    def __init__(self, *, weight_ref: str, weight_path: Path, infer_config: YOLOInferConfig,
                 pipeline: PipelineConfig, beautify: bool = True, model: Any = None) -> None:
        self.weight_ref = weight_ref
        self.weight_path = Path(weight_path)
        self.infer_config = infer_config
        self.pipeline = pipeline
        self.model = model
        self.predict_kwargs = predict_kwargs(infer_config)

        # 美化器的构造参数与 inference/service.py 完全一致: 配色/标签映射/字体全来自
        # infer_pipeline.yaml 的 visualization 段。
        # ★ labels 只是 D7 里的记录字段(draw / color_for / display_name 都不读它),
        #   所以这里不为了填它去拉一次 model.names。
        self.do_beautify = bool(beautify) and bool(pipeline.viz_enabled)
        self.visualizer: Optional[BeautifyVisualizer] = None
        if self.do_beautify:
            self.visualizer = BeautifyVisualizer(
                labels=None,
                label_mapping=pipeline.label_mapping or None,
                color_mapping=pipeline.color_mapping or None,
                default_color=pipeline.default_color,
                font_path=pipeline.font_path,
            )
        self._style: Optional[DrawStyle] = None

    # ------------------------------------------------------------------ 构造
    @classmethod
    def prepare(cls, *, weight_ref: Any, conf: Any = None, iou: Any = None, imgsz: Any = None,
                device: Any = None, beautify: Optional[bool] = None,
                pipeline_yaml: Any = None, model: Any = None) -> "DemoRunner":
        """按 D5 三源合并出这次的配置, 解析权重, 构造推理器。"""
        infer_config, _merger = load_infer_config(
            {"conf": conf, "iou": iou, "imgsz": imgsz, "device": device}
        )
        resolved_pipeline = load_pipeline_config(pipeline_path(pipeline_yaml))
        resolved_weight = resolve_weight_ref(weight_ref)
        return cls(
            weight_ref=str(weight_ref),
            weight_path=resolved_weight,
            infer_config=infer_config,
            pipeline=resolved_pipeline,
            beautify=resolved_pipeline.viz_enabled if beautify is None else bool(beautify),
            model=model,
        )

    # ------------------------------------------------------------------ 工具
    def display_name(self, class_name: Any) -> str:
        """类名 → 显示名(套 ``label_mapping``; 美化关闭时保持原名)。"""
        if self.visualizer is None:
            return "" if class_name is None else str(class_name)
        return self.visualizer.display_name(class_name)

    def _ensure_model(self):
        if self.model is None:
            self.model = load_yolo(self.weight_path)
        return self.model

    def annotate(self, image: Any, result: Any, detections: Sequence[dict]):
        """画框: 美化走 D7 的 ``BeautifyVisualizer``; 关掉时退回 YOLO 原生 ``plot()``。

        与 ``inference/service._FrameProcessor.draw()`` 同一套动作 ——
        包括"用第一张图的尺寸定字号/线宽, 之后固定"这条细节。
        """
        if self.visualizer is None:
            return result.plot()
        height, width = image.shape[:2]
        if self._style is None:
            self._style = DrawStyle.from_image_size(height, width, **self.pipeline.style_overrides)
        return self.visualizer.draw(
            image.copy(), detections, style=self._style,
            use_label_mapping=self.pipeline.use_label_mapping,
        )

    # ------------------------------------------------------------------ 主流程
    def run(self, image_path: Any) -> ImageInference:
        """跑一张图。**永不抛** —— 任何异常都打包进 :class:`ImageInference.error`。"""
        path = Path(image_path)
        try:
            image = read_image_bgr(path)
        except Exception as exc:
            LOGGER.warning(f"{path.name} 读图失败: {exc}")
            return ImageInference(path=path, image_name=path.name, error=f"读取图片失败：{exc}")

        try:
            model = self._ensure_model()
        except Exception as exc:
            LOGGER.error(f"加载权重失败: {exc}", exc_info=True)
            return ImageInference(path=path, image_name=path.name, error=f"加载权重失败：{exc}")

        try:
            # ★ 与 inference/service.py 的 _FrameProcessor.infer_batch 同一种调用形式:
            #   传【图片列表】而不是单张, 参数走白名单 predict_kwargs。
            results = model([image], **self.predict_kwargs)
        except Exception as exc:
            LOGGER.error(f"推理失败: {exc}", exc_info=True)
            return ImageInference(path=path, image_name=path.name, error=f"推理失败：{exc}")

        result = results[0] if results else None
        if result is None:
            return ImageInference(path=path, image_name=path.name, error="推理没有返回结果")

        detections = detections_from_yolo_result(result, getattr(model, "names", {}))
        try:
            annotated = self.annotate(image, result, detections)
        except Exception as exc:
            LOGGER.error(f"绘制失败: {exc}", exc_info=True)
            return ImageInference(path=path, image_name=path.name,
                                  detections=detections, error=f"绘制失败：{exc}")

        return ImageInference(path=path, image_name=path.name,
                              annotated=annotated, detections=detections)


def _to_rgb(image: Any):
    """BGR → RGB 的连续副本(gradio 的图片组件按 RGB 理解数组)。"""
    return image[..., ::-1].copy()


def _normalize_files(files: Any) -> list[Path]:
    """gradio ``gr.Files`` 给的是路径(str)或路径列表; 也容忍单张字符串 / Path。"""
    if files is None:
        return []
    if isinstance(files, (str, Path)):
        return [Path(files)]
    return [Path(item) for item in files if item]


def _params_line(runner: DemoRunner) -> str:
    """本次推理的生效参数(逐项来自 D5 合并结果, 不是界面自己拼的默认值)。"""
    config = runner.infer_config
    return (
        f"`conf={config.conf}` · `iou={config.iou}` · `imgsz={config.imgsz}` · "
        f"`device={device_label(config.device)}` · `max_det={config.max_det}` · "
        f"美化={'开' if runner.do_beautify else '关'} · 权重 `{runner.weight_ref}`"
    )


def run_demo(files: Any, *, weight_ref: Any, conf: Any, iou: Any, imgsz: Any, device: Any,
             beautify: bool, max_images: int, pipeline_yaml: Any = None,
             model: Any = None) -> tuple[list, list, str]:
    """界面回调的实体(**不依赖 gradio**, 可直接单元测试)。

    Args:
        files: gradio 上传控件的值(路径或路径列表)
        weight_ref: 权重名(由 common/refs.py 解析)
        conf / iou / imgsz / device: 界面上的推理旋钮(交给 D5 三源合并)
        beautify: 是否美化绘制(默认来自 infer_pipeline.yaml 的 visualization.enabled)
        max_images: 单次最多处理几张(来自 demo.yaml)
        pipeline_yaml: 美化/帧源 yaml(demo.yaml 的 pipeline_yaml)
        model: 测试用 —— 预置的模型替身

    Returns:
        ``(gallery, rows, summary_markdown)``
          gallery: 标注结果图(RGB ndarray 列表, gradio 的 Gallery 吃这个)
          rows:    检出明细(列见 :data:`TABLE_HEADERS`)
          summary: Markdown 汇总(总检出数 / 各品种计数 / 参数 / 失败提示)
    """
    paths = _normalize_files(files)
    if not paths:
        return [], [], "### 还没有图片\n\n请先上传至少一张图片（单张或多张都行）。"

    notes: list[str] = []
    if len(paths) > max_images:
        notes.append(
            f"> 一次最多处理 `{max_images}` 张（`max_images` 来自 "
            f"`configs/runtime/{DEMO_CONFIG_NAME}.yaml`），已截断前 `{max_images}` 张。"
        )
        paths = paths[:max_images]

    try:
        runner = DemoRunner.prepare(
            weight_ref=weight_ref, conf=conf, iou=iou, imgsz=imgsz, device=device,
            beautify=beautify, pipeline_yaml=pipeline_yaml, model=model,
        )
    except Exception as exc:
        LOGGER.error(f"准备推理器失败: {exc}", exc_info=True)
        return [], [], f"### 启动失败\n\n```\n{exc}\n```"

    gallery: list = []
    rows: list[list[Any]] = []
    detections: list[dict] = []
    failures: list[str] = []

    for path in paths:
        outcome = runner.run(path)
        if not outcome.ok:
            failures.append(f"- `{outcome.image_name}`：{outcome.error}")
            continue
        gallery.append(_to_rgb(outcome.annotated))
        rows.extend(detection_rows(outcome.detections,
                                   display_name=runner.display_name,
                                   image_name=outcome.image_name))
        detections.extend(outcome.detections)

    if not gallery:
        return [], [], "### 处理失败\n\n" + "\n".join(failures or ["- 没有任何图片被成功处理"])

    summary = summarize_detections(detections, display_name=runner.display_name)
    LOGGER.info(
        f"检出 {summary['total']} 个目标 / {len(gallery)} 张图, 各品种: {summary['per_class']}"
    )
    markdown = format_summary_markdown(summary, images=len(gallery),
                                       params=_params_line(runner),
                                       notes=notes + failures)
    return gallery, rows, markdown


# ============================================================================
# Gradio 界面（顶层不 import gradio: 这里才第一次碰它）
# ============================================================================
def _knob_info(field: str, value: Any, setup: DemoSetup) -> str:
    """控件下方的"默认值 + 来源"提示 —— 来源取自 D5 合并器记下的溯源。"""
    metadata = setup.infer_merger.get_metadata(field)
    source = metadata.source_label if metadata is not None else ConfigSource.DEFAULT.value
    return f"默认 {_format_value(value)}（来源 {source}：{_relpath(setup.infer_path)}）"


def _intro_markdown(setup: DemoSetup) -> str:
    return (
        "上传图片 → 用与 `odp-infer` **同一套配置与绘制**推理：推理参数来自 D5 的三源合并，"
        "框与标签来自 D7 的 `BeautifyVisualizer`。\n\n"
        f"- 推理参数默认值 ← `{_relpath(setup.infer_path)}`\n"
        f"- 配色 / 标签映射 / 字体 / 美化开关 ← `{_relpath(setup.pipeline_path)}` 的 `visualization` 段\n"
        f"- 界面旋钮（标题 / 端口 / 张数上限 / 默认权重）← `{_relpath(setup.demo_path)}`\n"
    )


def _device_choices(current: str) -> list[str]:
    """device 的候选: "自动" + 最常见的几个取值, 当前值一定在列表里(允许自定义输入)。"""
    choices = [DEVICE_AUTO, "cpu", "0", "0,1", "mps"]
    if current and current not in choices:
        choices.insert(1, current)
    return choices


def build_app(setup: Optional[DemoSetup] = None, *, cli_args: Optional[dict] = None,
              model: Any = None):
    """构造 Gradio Blocks(**不 launch**)。

    Args:
        setup: 已经准备好的默认值集合; 不传就用 ``build_setup(cli_args)`` 现算
        cli_args: odp-demo 的命令行字段(dict), 只在 setup 为 None 时用
        model: 测试用 —— 预置的模型替身(会传给每个请求的 DemoRunner)

    Returns:
        ``gradio.Blocks`` 实例
    """
    gr = _import_gradio()
    setup = setup or build_setup(cli_args)
    demo_config, infer_config = setup.demo_config, setup.infer_config

    conf_lower, conf_upper = numeric_bounds(YOLOInferConfig, "conf")
    iou_lower, iou_upper = numeric_bounds(YOLOInferConfig, "iou")
    imgsz_lower, _imgsz_upper = numeric_bounds(YOLOInferConfig, "imgsz")

    def _float_knob(field: str, value: Any, label: str, lower: Any, upper: Any):
        """区间来自字段约束(ge/le)的滑块; 万一将来约束没了就退回数字输入框, 不写死 min/max。"""
        info = _knob_info(field, value, setup)
        if lower is None or upper is None:
            return gr.Number(value=value, label=label, info=info)
        return gr.Slider(minimum=lower, maximum=upper, value=value,
                         step=(upper - lower) / SLIDER_STEPS, label=label, info=info)

    with gr.Blocks(title=demo_config.title) as app:
        gr.Markdown(f"# {demo_config.title}")
        gr.Markdown(_intro_markdown(setup))

        with gr.Row():
            with gr.Column(scale=1, min_width=340):
                files = gr.Files(label="上传图片（可多选, 单张也行）",
                                 file_count="multiple", file_types=["image"], type="filepath")
                weight = gr.Dropdown(
                    choices=setup.weight_choices, value=setup.weight_ref,
                    allow_custom_value=True, filterable=True,
                    label="权重（按名字解析, 可换）",
                    info=f"当前解析到：{_relpath(setup.weight_path)}",
                )
                conf = _float_knob("conf", infer_config.conf, "conf 置信度阈值", conf_lower, conf_upper)
                iou = _float_knob("iou", infer_config.iou, "iou NMS IoU 阈值", iou_lower, iou_upper)
                imgsz = gr.Number(
                    value=infer_config.imgsz, label="imgsz 推理输入尺寸",
                    info=_knob_info("imgsz", infer_config.imgsz, setup),
                    precision=0,                      # imgsz 在 YOLOInferConfig 里是 int, 不显示小数
                    minimum=imgsz_lower,
                )
                device = gr.Dropdown(
                    choices=_device_choices(device_label(infer_config.device)),
                    value=device_label(infer_config.device),
                    allow_custom_value=True, filterable=True,
                    label="device（auto = 交给 ultralytics 选）",
                    info=_knob_info("device", infer_config.device, setup),
                )
                beautify = gr.Checkbox(
                    value=setup.beautify, label="美化绘制（BeautifyVisualizer）",
                    info=f"默认取自 `{_relpath(setup.pipeline_path)}` 的 `visualization.enabled`"
                         f"（{_format_value(setup.pipeline.viz_enabled)}）",
                )
                run_button = gr.Button("开始检测", variant="primary")

            with gr.Column(scale=2):
                summary = gr.Markdown("### 检出汇总\n\n上传图片后点「开始检测」。")
                gallery = gr.Gallery(label="标注结果（BGR → RGB 原样展示）", columns=2,
                                     object_fit="contain", height="auto")
                table = gr.Dataframe(value=[], headers=TABLE_HEADERS, datatype=TABLE_DATATYPES,
                                     label="检出明细（品种 / 置信度 / 框坐标）",
                                     wrap=True, interactive=False)

        with gr.Accordion("配置来源（每个默认值的唯一出处）", open=False):
            gr.Markdown(format_config_source_markdown(
                demo_rows=config_source_rows(demo_config, setup.demo_merger,
                                            tuple(YOLODemoConfig.model_fields)),
                infer_rows=config_source_rows(infer_config, setup.infer_merger,
                                              INFER_SOURCE_FIELDS),
                demo_path=setup.demo_path,
                infer_path=setup.infer_path,
                pipeline_path=setup.pipeline_path,
                weight_ref=setup.weight_ref,
                weight_path=setup.weight_path,
                beautify=setup.beautify,
            ))

        def _on_run(files_value, weight_value, conf_value, iou_value, imgsz_value,
                    device_text, beautify_value):
            """界面回调: 把控件值交给 run_demo（唯一一处"界面 ↔ 引擎"的接缝）。"""
            return run_demo(
                files_value,
                weight_ref=weight_value,
                conf=conf_value,
                iou=iou_value,
                imgsz=imgsz_value,
                device=device_text,
                beautify=beautify_value,
                max_images=demo_config.max_images,
                pipeline_yaml=demo_config.pipeline_yaml,
                model=model,
            )

        run_button.click(
            fn=_on_run,
            inputs=[files, weight, conf, iou, imgsz, device, beautify],
            outputs=[gallery, table, summary],
        )

    return app


# ============================================================================
# CLI 入口（odp-demo）
# ============================================================================
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="odp-demo",
        description="ODPlatform Gradio 演示界面（配置与绘制复用 odp-infer）",
    )
    parser.add_argument("--server-name", type=str, default=None,
                        help="界面绑定地址（默认取 demo.yaml 的 server_name）")
    parser.add_argument("--server-port", type=int, default=None,
                        help="界面端口（默认取 demo.yaml 的 server_port）")
    parser.add_argument("--share", action="store_true", default=None,
                        help="生成 gradio 公网分享链接（默认取 demo.yaml 的 share）")
    parser.add_argument("--inbrowser", action="store_true", default=None,
                        help="启动后自动打开浏览器（默认取 demo.yaml 的 inbrowser）")
    parser.add_argument("--max-images", type=int, default=None,
                        help="单次最多处理几张图（默认取 demo.yaml 的 max_images）")
    parser.add_argument("--weight", type=str, default=None,
                        help="权重名字（默认取 demo.yaml 的 weight；由 common/refs.py 解析）")
    parser.add_argument("--pipeline-yaml", type=str, default=None, dest="pipeline_yaml",
                        help="美化/帧源 yaml（默认取 demo.yaml 的 pipeline_yaml）")
    parser.add_argument("--conf", type=float, default=None,
                        help="置信度阈值（默认取 infer.yaml 的 conf）")
    parser.add_argument("--iou", type=float, default=None,
                        help="NMS IoU 阈值（默认取 infer.yaml 的 iou）")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="推理输入尺寸（默认取 infer.yaml 的 imgsz）")
    parser.add_argument("--device", type=str, default=None,
                        help="cpu / 0 / 0,1 / mps（默认取 infer.yaml 的 device）")
    parser.add_argument("--no-viz", action="store_true",
                        help="关掉美化绘制, 退回 YOLO 原生 plot()"
                             "（默认取 infer_pipeline.yaml 的 visualization.enabled）")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日志级别")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """odp-demo 入口。返回值即退出码(0 正常 / 1 启动失败 / 2 环境或配置问题)。"""
    ns = _build_parser().parse_args(argv)

    # ★ 与其它 CLI 同一条纪律: 整个进程里唯一装 handler 的地方, 落点走 paths.LOGGING_DIR
    get_logger(base_path=LOGGING_DIR, log_type="demo",
               log_level=getattr(logging, ns.log_level))

    # ★ 可选依赖只在函数内部 import —— 没装 gradio 也能跑到这里, 并给出中文指引
    try:
        gr = _import_gradio()
    except GradioMissingError as exc:
        sys.stderr.write(f"\n{exc}\n")
        return EXIT_NO_GRADIO

    try:
        setup = build_setup(cli_args_from_namespace(ns),
                            beautify=False if ns.no_viz else None)
    except Exception as exc:
        sys.stderr.write(f"\n配置准备失败：{exc}\n")
        return EXIT_USAGE

    demo_config = setup.demo_config
    LOGGER.info(f"gradio {getattr(gr, '__version__', '?')}")
    LOGGER.info(f"演示界面配置(生效值): "
                f"{json.dumps(demo_config.to_audit_snapshot(), ensure_ascii=False)}")
    LOGGER.info(f"推理参数(生效值): conf={setup.infer_config.conf} iou={setup.infer_config.iou} "
                f"imgsz={setup.infer_config.imgsz} device={setup.infer_config.device} "
                f"max_det={setup.infer_config.max_det}")
    LOGGER.info(f"默认权重: {setup.weight_ref} → {_relpath(setup.weight_path)}")
    LOGGER.info(f"美化配置: {_relpath(setup.pipeline_path)}（美化={setup.beautify}）")

    try:
        app = build_app(setup=setup)
    except GradioMissingError as exc:
        sys.stderr.write(f"\n{exc}\n")
        return EXIT_NO_GRADIO
    except Exception as exc:
        LOGGER.error(f"界面构造失败: {exc}", exc_info=True)
        sys.stderr.write(f"\n界面构造失败：{exc}\n")
        return EXIT_LAUNCH_FAILED

    url = f"http://{demo_config.server_name}:{demo_config.server_port}"
    LOGGER.info(f"界面地址: {url}（Ctrl+C 结束）")

    try:
        app.launch(
            server_name=demo_config.server_name,
            server_port=demo_config.server_port,
            share=demo_config.share,
            inbrowser=demo_config.inbrowser,
            show_error=True,
        )
    except Exception as exc:
        LOGGER.error(f"界面启动失败: {exc}", exc_info=True)
        sys.stderr.write(f"\n界面启动失败：{exc}\n")
        return EXIT_LAUNCH_FAILED
    return EXIT_OK


__all__ = [
    "GradioMissingError",
    "DemoSetup",
    "DemoRunner",
    "ImageInference",
    "build_app",
    "build_setup",
    "config_source_rows",
    "default_weight_ref",
    "detection_rows",
    "detections_from_yolo_result",
    "device_label",
    "device_value",
    "discover_weight_refs",
    "format_config_source_markdown",
    "format_summary_markdown",
    "load_demo_config",
    "load_infer_config",
    "main",
    "numeric_bounds",
    "pipeline_path",
    "predict_kwargs",
    "resolve_weight_ref",
    "run_demo",
    "split_cli_args",
    "summarize_detections",
    "weight_choices",
]


if __name__ == "__main__":
    sys.exit(main())