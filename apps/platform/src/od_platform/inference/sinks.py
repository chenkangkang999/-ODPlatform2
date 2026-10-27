#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : sinks.py
# @Project   : ODPlatform
# @Function  : 推理结果输出适配器 (OutputSink 抽象 + LocalFileSink + NullSink)
"""推理结果输出适配器.

把"画好的 annotated 帧"送到不同目的地的抽象层:

  - OutputSink     : 抽象基类 (open/write/close 三个方法)
  - LocalFileSink  : 本地视频 / 图片 (★ CLI 默认行为, 等价于老式 cv2.imwrite/VideoWriter)
  - NullSink       : 不写任何东西 (web 流推 / 只跑统计 / 不存盘 时用)

★ 一切皆配置: `LocalFileSink` 的文件名 / 前缀 / 扩展名 / 视频编码器 /
  jpg 质量 / 写入帧率 / 同名覆盖策略, 全部取自 `OutputConfig`
  (即 `configs/runtime/infer_pipeline.yaml` 的 `output` 段, 由
  `inference/pipeline_config.py` 解析). 本模块不再写死任何"影响产物"的取值。
  不传配置 = `OutputConfig` 默认值 = 改造前写死的行为(文件名与字节逐一相同),
  所以旧调用点 `LocalFileSink()` 语义不变。

业务端在自己仓库继承 OutputSink 实现自定义 sink:
  - S3Sink         : 写对象存储
  - WebSocketSink  : 实时推流
  - QtSignalSink   : 桥接到 Qt 信号 (本仓库 apps/desktop 里有实现, 见阶段 11)

实现纪律 (与 D8 InferService 的"永不抛"一脉相承):
  - write() 内部 try/except 包住, 单帧写入失败 logger.warning + 跳过, 不抛
  - close() 幂等 (允许多次调用)
  - open() 由 pipeline 在首批帧到达后调用 (此时 source_type 已知)
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from od_platform.frame_source import SourceType

from .pipeline_config import JPEG_IMAGE_FORMATS, OutputConfig, coerce_output_config

logger = logging.getLogger(__name__)


# ============================================================================
# 抽象基类
# ============================================================================
class OutputSink(ABC):
    """推理结果输出适配器.

    生命周期: open() → write() × N → close().

    实现要求:
      - write 永不抛 (出错 logger.warning + 跳过这一帧)
      - close 幂等 (多次调不出错)
      - 线程亲缘: write 在 _Renderer 线程被调, open/close 在主线程
    """

    @abstractmethod
    def open(self, output_dir: Path, source_type: SourceType) -> None:
        """初始化. 由 ThreadedPipeline 在首批帧到达后调用.

        Args:
            output_dir:  推理输出根目录 (部分 sink 可能不用)
            source_type: 帧源类型 (决定按视频流还是单图处理)
        """

    @abstractmethod
    def write(self, frame, annotated: np.ndarray) -> None:
        """写一帧画好的图.

        Args:
            frame:     frame_source 给出的 Frame 对象 (含 image / info 元数据)
            annotated: 已画过框的 BGR 图像
        """

    @abstractmethod
    def close(self) -> None:
        """收尾. 由 ThreadedPipeline 在退出时调用. 必须幂等."""


# ============================================================================
# 内置实现 1: LocalFileSink (CLI 默认: 视频写配置的容器, 图片逐帧写配置的格式)
# ============================================================================
class LocalFileSink(OutputSink):
    """本地文件 sink — 视频源写视频文件, 图片源逐帧写图片文件.

    落盘取值全部来自 `OutputConfig`(infer_pipeline.yaml 的 output 段):

      产物文件名  = <filename_prefix><video_basename>.<video_format>   (视频)
                    <filename_prefix><原图名主干>.<image_format>        (图片)
      兜底命名    = anonymous_frame_template (帧源没给文件名时)
      视频编码    = video_fourcc (四字符码) + video_fps / video_fps_fallback
      图片质量    = image_quality (仅 jpg/jpeg)
      同名文件    = overwrite_existing (false → 跳过, 不覆盖)

    实现细节:
      - 视频源 (VIDEO / CAMERA): lazy 打开 cv2.VideoWriter, 首帧拿到尺寸再建
      - 图片源 (IMAGE / IMAGES): 每帧 cv2.imwrite 一张
      - 不传 output_config = 全默认值 = 改造前的写死行为
    """

    def __init__(self, output_config: OutputConfig | Mapping[str, Any] | None = None) -> None:
        # ★ 向后兼容: `LocalFileSink()` 依旧合法, 且语义 = 改造前行为
        self.cfg: OutputConfig = coerce_output_config(output_config)
        self.output_dir: Path | None = None
        self._is_stream: bool = False
        self._video = None           # cv2.VideoWriter, lazy
        self._video_warned: bool = False
        self._skip_video: bool = False
        self._count: int = 0

    def open(self, output_dir: Path, source_type: SourceType) -> None:
        self.output_dir = output_dir
        self._is_stream = source_type in (SourceType.VIDEO, SourceType.CAMERA)

    # ---- 命名: 全部由配置推导, 这里不再有写死的文件名/扩展名 ----
    def video_path(self) -> Path:
        """视频产物路径 (<前缀><主干>.<容器>)."""
        return self.output_dir / self.cfg.video_filename()

    def image_path(self, frame) -> Path:
        """图片产物路径 (<前缀><原图名主干>.<格式>)."""
        src = getattr(frame.info, "filename", None)
        if not src:
            src = self.cfg.anonymous_frame_template.format(
                index=getattr(frame.info, "frame_index", 0)
            )
        return self.output_dir / self.cfg.image_filename(Path(str(src)).stem)

    def _resolve_fps(self, frame) -> float:
        """写入帧率: 配置给了固定值就用它, 否则跟随源帧率, 源缺失再兜底.

        ★ 默认(video_fps=None + video_fps_fallback=30.0)与改造前
          `float(frame.info.fps) if ... else 30.0` 完全一致。
        """
        if self.cfg.video_fps is not None:
            return float(self.cfg.video_fps)
        src = getattr(frame.info, "fps", None)
        try:
            src_fps = float(src) if src else 0.0
        except (TypeError, ValueError):
            src_fps = 0.0
        return src_fps if src_fps > 0 else float(self.cfg.video_fps_fallback)

    def write(self, frame, annotated) -> None:
        import cv2
        try:
            if self._is_stream:
                self._write_video(cv2, frame, annotated)
            else:
                self._write_image(cv2, frame, annotated)
            self._count += 1
        except Exception as e:
            logger.warning(f"LocalFileSink.write 失败, 跳过: {e}")

    def _write_video(self, cv2, frame, annotated) -> None:
        if self._skip_video:
            return
        if self._video is None:
            out = self.video_path()
            if not self.cfg.overwrite_existing and out.exists():
                self._skip_video = True
                logger.warning(
                    f"视频已存在且 output.overwrite_existing=false, 本次不写: {out}"
                )
                return
            h, w = annotated.shape[:2]
            fps = self._resolve_fps(frame)
            fourcc = cv2.VideoWriter_fourcc(*self.cfg.video_fourcc)
            self._video = cv2.VideoWriter(str(out), fourcc, fps, (w, h))
            logger.info(
                f"视频 writer:   {out.name} | fourcc={self.cfg.video_fourcc} | "
                f"容器={self.cfg.video_format} | fps={fps} | 尺寸={w}x{h}"
            )
        if (not self._video_warned and hasattr(self._video, "isOpened")
                and not self._video.isOpened()):
            # ★ 不静默降级: 打不开是"产物会缺失"级别的事故, 必须可见
            self._video_warned = True
            logger.error(
                f"cv2.VideoWriter 未能打开 (fourcc={self.cfg.video_fourcc}, "
                f"容器={self.cfg.video_format}); 本机可能不支持该组合, 视频写不出来. "
                f"改 infer_pipeline.yaml 的 output.video_fourcc / video_format."
            )
        self._video.write(annotated)

    def _write_image(self, cv2, frame, annotated) -> None:
        out = self.image_path(frame)
        if not self.cfg.overwrite_existing and out.exists():
            logger.warning(
                f"图片已存在且 output.overwrite_existing=false, 跳过: {out.name}"
            )
            return
        params = self._imwrite_params(cv2)
        ok = (cv2.imwrite(str(out), annotated, params) if params
              else cv2.imwrite(str(out), annotated))
        if not ok:
            logger.warning(
                f"cv2.imwrite 返回 False, 未写出: {out.name} "
                f"(格式 {self.cfg.image_format}, 质量参数 {params})"
            )

    def _imwrite_params(self, cv2) -> list[int]:
        """编码参数: jpg/jpeg 才吃 image_quality, 其余格式用 OpenCV 默认.

        ★ image_quality 默认 95 == OpenCV 未传参时的默认值, 已实测逐字节相同,
          所以默认路径的产物与改造前完全一致。
        """
        if self.cfg.image_format in JPEG_IMAGE_FORMATS:
            return [int(cv2.IMWRITE_JPEG_QUALITY), int(self.cfg.image_quality)]
        return []

    def close(self) -> None:
        if self._video is not None:
            try:
                self._video.release()
            except Exception as e:
                logger.warning(f"LocalFileSink.close release 失败 (已吞): {e}")
            finally:
                self._video = None


# ============================================================================
# 内置实现 2: NullSink (不写任何东西)
# ============================================================================
class NullSink(OutputSink):
    """什么也不写.

    用途:
      - web 流推: annotated 已通过 WebSocketSink 推走, 后端无需落盘
      - desktop: annotated 已通过 QtSignalSink 给 UI, 不需要本地文件
      - 性能基准测试: 排除 IO 干扰
      - --no-save 模式下的 CLI
    """

    def open(self, output_dir: Path, source_type: SourceType) -> None:
        pass

    def write(self, frame, annotated) -> None:
        pass

    def close(self) -> None:
        pass