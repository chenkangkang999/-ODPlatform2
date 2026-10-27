# apps/desktop/qt_sink.py
from __future__ import annotations

import logging
import numpy as np
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from od_platform.frame_source import SourceType
from od_platform.inference import OutputSink

logger = logging.getLogger(__name__)


class QtSignalSink(QObject, OutputSink):
    """OutputSink → Qt 信号桥.

    每帧推理完, 由 _Renderer 线程调 write(...) → 转成 Qt 信号
    → Qt 跨线程派发 (QueuedConnection) → 主线程 UI 槽函数 setPixmap.

    ★ Signal 用 object 而不是 np.ndarray:
       Qt 对 numpy 类型注册不一定可靠 (PySide6 不同版本行为不同),
       用 object 兜底, 槽函数自己 isinstance(frame, np.ndarray) 兜底验证.
    """

    frame_ready = Signal(object)        # 发: annotated (BGR, ndarray, contiguous)
    write_count = 0

    def __init__(self, parent: QObject | None = None) -> None:
        QObject.__init__(self, parent)
        OutputSink.__init__(self)

    # ---- OutputSink 三方法 ----
    def open(self, output_dir: Path, source_type: SourceType) -> None:
        logger.info(f"QtSignalSink.open: dir={output_dir}, source_type={source_type}")

    def write(self, frame, annotated: np.ndarray) -> None:
        try:
            # ★ 必须连续, QImage 直接拿底层 buffer 时不连续会乱
            if not annotated.flags["C_CONTIGUOUS"]:
                annotated = np.ascontiguousarray(annotated)
            self.frame_ready.emit(annotated)
            self.write_count += 1
            if self.write_count <= 3:
                logger.info(f"QtSignalSink.write[{self.write_count}]: "
                            f"shape={annotated.shape}, dtype={annotated.dtype}")
        except Exception as e:
            logger.warning(f"QtSignalSink.write 失败 (已吞): {e}")

    def close(self) -> None:
        logger.info(f"QtSignalSink.close: 共发了 {self.write_count} 帧")