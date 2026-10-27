#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : pipeline_config.py
# @Project   : ODPlatform
# @Function  : 读取 infer_pipeline.yaml(帧源+美化+输出) — D5 管不到的那半边配置
"""帧源捕获 + 美化 + 输出落盘 的配置读取 helper.

★ 核心纪律: 不重新发明校验. 这个 helper 只做"yaml → 子字典 → 喂给现成 pydantic 模型":
  - camera 块  → frame_source.CameraConfig(**...)   (extra=forbid, 自带校验)
  - style 块   → 透传给 visualization.DrawStyle.from_image_size(**...) (拿到帧尺寸后才构造)
  - output 块  → OutputConfig(**...)(本模块自带校验) → 传给 LocalFileSink
  - 颜色 list  → tuple (BGR), 因为美化模块吃 tuple

跟 D5 的关系: D5 的 infer.yaml 管 YOLO predict 参数(build_infer_config 读),
这份 infer_pipeline.yaml 管帧源+美化+输出落盘, 两份互不干涉. service 阶段 1 各读各的再捏一起.

文件缺失不算错误 —— 用默认值(美化开启、无中文映射、摄像头走 CameraConfig 默认、
输出落盘走与原写死行为逐字节一致的默认), 打一条 warning 即可.
基本版只要有模型 + 源就能跑, pipeline yaml 是锦上添花.

★ 输出段 (output) 的溯源: `OutputConfig` 是 pydantic 模型(有 model_fields),
  配合本模块的 `OutputConfigMerger` 就能直接喂 `common/config_log.log_effective_config /
  log_override_chains` —— 跟 D5 的 infer.yaml 一样能打出"每个字段现在什么值、谁给的"。
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

logger = logging.getLogger(__name__)

# ── 输出落盘的"允许取值"清单(供 OutputConfig 校验; sinks 也要用 JPEG 那组) ──
JPEG_IMAGE_FORMATS: tuple[str, ...] = ("jpg", "jpeg")
SUPPORTED_IMAGE_FORMATS: tuple[str, ...] = ("jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff")
SUPPORTED_VIDEO_FORMATS: tuple[str, ...] = ("mp4", "avi", "mov", "mkv")
# video_fps 里表达"跟随源帧率"的写法(与 null / 缺省同义)
AUTO_FPS_TOKENS: tuple[str, ...] = ("auto", "source", "null", "none", "")


def _to_bgr_tuple(value: Any) -> tuple[int, int, int]:
    """yaml 里颜色是 list [B,G,R], 美化模块吃 tuple, 转一下."""
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return (int(value[0]), int(value[1]), int(value[2]))
    raise ValueError(f"颜色必须是 3 元素的 [B, G, R], 收到: {value!r}")


def _no_path_separator(text: str, *, what: str) -> str:
    """文件名相关字段不许带路径分隔符(否则 cv2.imwrite 会写到不存在的子目录)."""
    s = str(text)
    if "/" in s or "\\" in s:
        raise ValueError(f"{what} 不能含路径分隔符 (/ 或 \\), 收到: {text!r}")
    return s


# ============================================================================
# 输出落盘配置 (infer_pipeline.yaml 的 output 段)
# ============================================================================
class OutputConfig(BaseModel):
    """推理产物落盘参数 —— 原先写死在 LocalFileSink 里的那些值.

    ★ 兼容性硬要求: 每个字段的默认值 = 改造前 LocalFileSink 的实际行为,
      所以"output 段缺失 / 字段不填"时产物文件名与字节都与老代码完全一致:
        - 图片: <原图名>.jpg        (filename_prefix="" + image_format="jpg")
        - 视频: output.mp4          (video_basename="output" + video_format="mp4")
        - 编码: mp4v                (video_fourcc="mp4v")
        - 质量: jpg 95              (image_quality=95, 与 OpenCV 未传参时的默认值逐一字节相同)
        - 帧率: 源帧率, 缺失兜底 30 (video_fps=None + video_fps_fallback=30.0)
    """

    model_config = ConfigDict(extra="forbid")

    # ---- 文件名 ----
    filename_prefix: str = Field(
        default="",
        description="产物文件名前缀. 空串=不加前缀(默认, 对齐改造前行为).",
    )
    video_basename: str = Field(
        default="output",
        description="视频文件名主干(不含扩展名). 默认 output → output.mp4.",
    )
    video_format: str = Field(
        default="mp4",
        description="视频容器扩展名(mp4/avi/mov/mkv). 须与 video_fourcc 匹配, 否则 VideoWriter 打不开.",
    )
    image_format: str = Field(
        default="jpg",
        description="图片扩展名(jpg/jpeg/png/bmp/webp/tif/tiff). 默认 jpg, 对齐改造前行为.",
    )
    image_quality: int = Field(
        default=95,
        ge=1,
        le=100,
        description="JPEG 质量 1-100, 只对 jpg/jpeg 生效; 95 等于 OpenCV 默认值(逐字节一致).",
    )
    anonymous_frame_template: str = Field(
        default="frame_{index:06d}",
        description="帧源没给原始文件名时的兜底命名模板, 用 {index} 占位; 默认 frame_000123.",
    )

    # ---- 视频编码 ----
    video_fourcc: str = Field(
        default="mp4v",
        description="视频编码器四字符码(如 mp4v/MJPG/XVID/avc1). 默认 mp4v(OpenCV 自带, 不依赖外部编码器).",
    )
    video_fps: float | None = Field(
        default=None,
        description="视频写入帧率; null/auto/source = 跟随源帧率(默认), 数字 = 固定值.",
    )
    video_fps_fallback: float = Field(
        default=30.0,
        gt=0,
        description="源帧率缺失或为 0 时的兜底帧率. 默认 30.0, 对齐改造前行为.",
    )

    # ---- 落盘策略 ----
    overwrite_existing: bool = Field(
        default=True,
        description="同名产物是否覆盖. true=覆盖(默认, 对齐改造前行为); false=跳过已存在的文件.",
    )
    keep_audit_json: bool = Field(
        default=True,
        description="是否写审计快照. true=写(默认); false=不写, 日志里说明按配置跳过.",
    )
    audit_filename: str = Field(
        default="odp_audit.json",
        description="审计快照文件名(写在本次输出目录里). 默认 odp_audit.json.",
    )

    # ---- 校验 (出错信息必须是中文, 且由调用方降级成 warning + 默认值) ----
    @field_validator("image_format")
    @classmethod
    def _check_image_format(cls, v: str) -> str:
        text = str(v).strip().lstrip(".").lower()
        if text not in SUPPORTED_IMAGE_FORMATS:
            raise ValueError(
                f"不支持的图片格式 {v!r}, 可选: {', '.join(SUPPORTED_IMAGE_FORMATS)}"
            )
        return text

    @field_validator("video_format")
    @classmethod
    def _check_video_format(cls, v: str) -> str:
        text = str(v).strip().lstrip(".").lower()
        if text not in SUPPORTED_VIDEO_FORMATS:
            raise ValueError(
                f"不支持的视频容器 {v!r}, 可选: {', '.join(SUPPORTED_VIDEO_FORMATS)}"
            )
        return text

    @field_validator("video_fourcc")
    @classmethod
    def _check_fourcc(cls, v: str) -> str:
        text = str(v).strip()
        if len(text) != 4 or not text.isascii() or not text.isprintable():
            raise ValueError(
                f"编码器四字符码必须是 4 个可见 ASCII 字符 (如 mp4v/MJPG), 收到: {v!r}"
            )
        return text

    @field_validator("video_fps")
    @classmethod
    def _check_video_fps(cls, v: float | None) -> float | None:
        if v is None:
            return None
        fps = float(v)
        if fps <= 0:
            raise ValueError(f"视频帧率必须 > 0 (要跟随源帧率请写 null/auto), 收到: {v!r}")
        return fps

    @field_validator("filename_prefix", "video_basename", "audit_filename")
    @classmethod
    def _check_plain_name(cls, v: str) -> str:
        return _no_path_separator(v, what="output 的文件名字段")

    @field_validator("video_basename", "audit_filename")
    @classmethod
    def _check_not_empty(cls, v: str) -> str:
        if not str(v).strip():
            raise ValueError(f"不能是空串: {v!r}")
        return str(v).strip()

    @field_validator("anonymous_frame_template")
    @classmethod
    def _check_template(cls, v: str) -> str:
        text = _no_path_separator(v, what="anonymous_frame_template")
        if "{index" not in text:
            raise ValueError(f"兜底命名模板必须含 {{index}} 占位符, 收到: {v!r}")
        try:
            text.format(index=0)
        except (KeyError, IndexError, ValueError) as e:
            raise ValueError(f"兜底命名模板无法用 index 格式化 ({e}), 收到: {v!r}") from e
        return text

    # ---- 给日志/审计读的小工具 ----
    def video_filename(self) -> str:
        """视频产物文件名: <前缀><主干>.<容器>."""
        return f"{self.filename_prefix}{self.video_basename}.{self.video_format}"

    def image_filename(self, stem: str) -> str:
        """图片产物文件名: <前缀><原图主干>.<格式>."""
        return f"{self.filename_prefix}{stem}.{self.image_format}"


@dataclass(frozen=True)
class _SourceRecord:
    """溯源链上的一条记录 (给 runtime_config.report 的渲染器读)."""
    value: Any
    source_label: str


class _FieldMeta:
    """单个字段的溯源链 —— 只实现 report 渲染器实际用到的接口."""

    def __init__(self) -> None:
        self._records: list[_SourceRecord] = []      # oldest-first

    @property
    def source_label(self) -> str:
        return self._records[-1].source_label

    def chain(self) -> list[_SourceRecord]:
        """跟 runtime_config.ConfigMerger 一致: newest-first."""
        return list(reversed(self._records))

    def add(self, value: Any, source_label: str) -> None:
        self._records.append(_SourceRecord(value=value, source_label=source_label))


class OutputConfigMerger:
    """output 段的极简溯源桩.

    只为了把 `output` 段的生效值喂给 common/config_log 的两张表:
    `log_effective_config` 读 meta.source_label, `log_override_chains` 读 meta.chain().
    不做合并(只有 DEFAULT 与 YAML 两个源), 所以不需要 ConfigMerger 那一整套.
    """

    def __init__(self) -> None:
        self._meta: dict[str, _FieldMeta] = {}

    def record(self, name: str, value: Any, source_label: str) -> None:
        self._meta.setdefault(name, _FieldMeta()).add(value, source_label)

    def get_metadata(self, name: str) -> _FieldMeta | None:
        return self._meta.get(name)

    def source_labels(self) -> dict[str, str]:
        """字段 → 生效来源标签 (给审计快照用)."""
        return {name: meta.source_label for name, meta in self._meta.items()}


@dataclass
class PipelineConfig:
    """帧源 + 美化 + 输出落盘 配置的解析结果(纯数据, 不含行为)."""

    # ---- 帧源 ----
    camera_raw: dict[str, Any] = field(default_factory=dict)   # 原始 camera 子字典, 延迟构造 CameraConfig

    # ---- 美化 ----
    viz_enabled: bool = True
    use_label_mapping: bool = True
    label_mapping: dict[str, str] = field(default_factory=dict)
    color_mapping: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    default_color: tuple[int, int, int] = (0, 255, 0)
    font_path: str | None = None
    style_overrides: dict[str, Any] = field(default_factory=dict)   # 透传给 DrawStyle.from_image_size

    # ---- 输出落盘 ----
    # ★ 默认值 = 改造前 LocalFileSink 写死的行为, 所以"没配 output 段"不会改变产物
    output: OutputConfig = field(default_factory=OutputConfig)
    output_merger: OutputConfigMerger = field(default_factory=OutputConfigMerger)

    def build_camera_config(self):
        """延迟构造 CameraConfig —— 不是摄像头源时 create_frame_source 会忽略."""
        from od_platform.frame_source import CameraConfig
        if not self.camera_raw:
            return None
        try:
            return CameraConfig(**self.camera_raw)
        except Exception as e:
            logger.warning(f"camera 配置无效, 走默认: {e}")
            return None

    def to_audit(self) -> dict[str, Any]:
        """给 odp_audit.json 用的纯字典快照."""
        return {
            "viz_enabled": self.viz_enabled,
            "use_label_mapping": self.use_label_mapping,
            "label_mapping_n": len(self.label_mapping),
            "color_mapping_n": len(self.color_mapping),
            "default_color": list(self.default_color),
            "font_path": self.font_path,
            "camera": dict(self.camera_raw),
            "style_overrides_n": len(self.style_overrides),
            # ★ 输出落盘: 生效值 + 每个字段的来源(DEFAULT / YAML / YAML(无效,已回落默认))
            "output": self.output.model_dump(),
            "output_sources": self.output_merger.source_labels(),
        }


# ============================================================================
# output 段解析 (逐字段容错: 缺失/写错类型 → warning + 回落默认值, 不抛)
# ============================================================================
def _load_output_config(raw: Any, merger: OutputConfigMerger) -> OutputConfig:
    """yaml 的 output 子字典 → OutputConfig. 边解析边把来源记进 merger.

    容错纪律 (跟本模块既有风格一致):
      - 整个 output 段缺失 / 是 null  → 全默认值, 只记 DEFAULT;
      - 未知字段                      → warning + 忽略(不影响已认识的字段);
      - 单个字段类型/取值非法          → warning + 仅该字段回落默认值, 其余照常生效.
    """
    defaults = OutputConfig()
    known = list(OutputConfig.model_fields)
    for name in known:
        merger.record(name, getattr(defaults, name), "DEFAULT")

    if raw is None:
        return defaults
    if not isinstance(raw, Mapping):
        logger.warning(
            f"output 段必须是字典, 收到 {type(raw).__name__}: {raw!r}; 全部走默认值"
        )
        return defaults

    for key in raw:
        if key not in known:
            logger.warning(f"output 段里的未知字段 '{key}' 已忽略 (可用字段: {', '.join(known)})")

    provided: dict[str, Any] = {k: raw[k] for k in known if k in raw}

    # video_fps 的"跟随源帧率"写法: null/auto/source → None(与默认同义)
    if isinstance(provided.get("video_fps"), str):
        token = provided["video_fps"].strip().lower()
        if token in AUTO_FPS_TOKENS:
            provided["video_fps"] = None

    while True:
        values = {name: provided.get(name, getattr(defaults, name)) for name in known}
        try:
            cfg = OutputConfig(**values)
        except ValidationError as e:
            bad: dict[str, str] = {}
            for err in e.errors():
                loc = err.get("loc") or ()
                if loc:
                    bad.setdefault(str(loc[0]), str(err.get("msg", "")))
            if not bad:                      # 整模型级错误(理论上进不来), 交给上层
                raise
            for name, msg in bad.items():
                logger.warning(
                    f"output.{name} 取值无效 ({provided.get(name)!r}), "
                    f"回落默认值 {getattr(defaults, name)!r}: {msg}"
                )
                merger.record(name, getattr(defaults, name), "YAML(无效,已回落默认)")
                provided.pop(name, None)
            continue
        if cfg.image_format not in JPEG_IMAGE_FORMATS and "image_quality" in provided:
            logger.info(
                f"output.image_quality={cfg.image_quality} 只对 "
                f"{'/'.join(JPEG_IMAGE_FORMATS)} 生效, 当前 image_format="
                f"{cfg.image_format} 已忽略该字段."
            )
        for name in provided:
            merger.record(name, getattr(cfg, name), "YAML")
        return cfg


def coerce_output_config(value: Any) -> OutputConfig:
    """None / OutputConfig / dict → OutputConfig (sink 构造口的兼容适配).

    旧调用点 `LocalFileSink()` 传 None → 全默认值 = 改造前行为, 不破接口.
    """
    if isinstance(value, OutputConfig):
        return value
    if value is None:
        return OutputConfig()
    if isinstance(value, Mapping):
        return _load_output_config(value, OutputConfigMerger())
    logger.warning(
        f"输出配置类型不认识 ({type(value).__name__}: {value!r}), 走默认值"
    )
    return OutputConfig()


def _default_pipeline_config() -> PipelineConfig:
    """yaml 缺失/解析失败时的兜底配置.

    行为与改造前逐字节一致(输出段全默认值), 但**输出溯源仍记全** ——
    审计快照要能回答"每个输出参数的值从哪来", 而不是留一张空表。
    """
    pc = PipelineConfig()
    pc.output = _load_output_config(None, pc.output_merger)
    return pc


def load_pipeline_config(yaml_path: str | Path | None) -> PipelineConfig:
    """读 infer_pipeline.yaml. None / 文件不存在 → 空配置 + warning, 不抛."""
    if yaml_path is None:
        # 走默认路径: <APP_DIR>/configs/runtime/infer_pipeline.yaml
        # ★ 路径只从 paths.py 出(SSoT 纪律): 原先写死的 "configs/runtime/..." 是 CWD 相对,
        #   在仓库根跑 odp-infer 就会找不到文件而静默退回默认配置。
        from od_platform.common.paths import RUNTIME_CONFIGS_DIR

        yaml_path = RUNTIME_CONFIGS_DIR / "infer_pipeline.yaml"
    else:
        yaml_path = Path(yaml_path)

    if not yaml_path.exists():
        logger.warning(f"pipeline yaml 不存在, 用默认配置(含输出落盘默认): {yaml_path}")
        return _default_pipeline_config()

    try:
        with yaml_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"pipeline yaml 解析失败, 用默认配置: {e}")
        return _default_pipeline_config()

    pc = PipelineConfig()

    # ---- 帧源 ----
    fs = raw.get("frame_source", {}) or {}
    pc.camera_raw = fs.get("camera", {}) or {}

    # ---- 美化 ----
    vz = raw.get("visualization", {}) or {}
    pc.viz_enabled = bool(vz.get("enabled", True))
    pc.use_label_mapping = bool(vz.get("use_label_mapping", True))
    pc.label_mapping = dict(vz.get("label_mapping", {}) or {})

    raw_colors = vz.get("color_mapping", {}) or {}
    pc.color_mapping = {k: _to_bgr_tuple(v) for k, v in raw_colors.items()}

    if "default_color" in vz:
        pc.default_color = _to_bgr_tuple(vz["default_color"])

    pc.font_path = vz.get("font_path")
    pc.style_overrides = dict(vz.get("style", {}) or {})

    # ---- 输出落盘 (缺失 = 全默认 = 改造前行为) ----
    pc.output = _load_output_config(raw.get("output"), pc.output_merger)

    logger.info(f"pipeline 配置加载: {yaml_path}, "
                f"美化={pc.viz_enabled}, 标签映射={len(pc.label_mapping)}, "
                f"颜色映射={len(pc.color_mapping)}")
    logger.info(f"输出落盘配置: 图片={pc.output.image_filename('<原图名>')}, "
                f"视频={pc.output.video_filename()}, "
                f"编码器={pc.output.video_fourcc}, "
                f"帧率={'跟随源' if pc.output.video_fps is None else pc.output.video_fps}, "
                f"覆盖同名={pc.output.overwrite_existing}, "
                f"审计快照={pc.output.audit_filename if pc.output.keep_audit_json else '不写'}")
    return pc