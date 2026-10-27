# apps/desktop/infer_worker.py
from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from od_platform.inference import (
    CancelToken,
    FrameEvent,
    InferHooks,
    InferResult,
    ProgressEvent,
    infer_yolo,
)

logger = logging.getLogger(__name__)


class InferWorker(QObject):
    """跑 service.predict 的工作对象. 通过 moveToThread 放到 QThread 里.

    ★ 接缝集成:
      - sink         : 由 main 线程 new 后传进来 (Qt 信号必须在主线程构造), 见 main.py
      - cancel_token : 这里持有, GUI Stop 按钮 click → self.cancel_token.cancel()
      - hooks        : 在 start_infer 里组装, 把 service 事件转发成 Qt 信号
    """

    # 转 Qt 信号给 UI 监听
    progress_changed = Signal(int, int, float)            # frame_idx, total_frames|-1, fps_loop
    inference_finished = Signal(object)                    # InferResult
    inference_failed = Signal(str)                         # error message

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.cancel_token = CancelToken()
        self._sink = None                                  # 由 main 注入

    def set_sink(self, sink) -> None:
        """main 线程构造好 sink 后调 (sink 内部的 Signal 必须在主线程构造)."""
        self._sink = sink

    @Slot(str, str)
    def start_infer(self, model: str, source: str) -> None:
        """跑推理. 阻塞调用 (在 worker thread 里), 通过 hooks/signals 回报状态."""
        logger.info(f"InferWorker.start_infer: model={model}, source={source}")

        # 重置 cancel_token, 一个 worker 可以跑多次推理
        self.cancel_token = CancelToken()

        # ★ 组装 hooks: service 事件 → Qt 信号
        hooks = InferHooks(
            on_progress=self._on_progress,
            on_complete=self._on_complete,
            on_error=self._on_error,
            progress_interval_frames=15,        # 每 15 帧推一次 (~0.5 秒, 桌面 UI 不需要更频)
        )

        try:
            result: InferResult = infer_yolo(
                cli_args={
                    "source": source,
                    "model": model,
                    "show": False,              # 桌面端不要 cv2.imshow
                    "save": False,              # annotated 已通过 sink 给 UI 了, 不需要本地副本
                },
                output_sink=self._sink,
                hooks=hooks,
                cancel_token=self.cancel_token,
            )
            # service 永不抛, 所以这里只可能是正常返回
            # on_complete 已经在 hooks 里 emit 过了, 这里不重复
            logger.info(f"InferWorker 推理结束: success={result.success}")
        except Exception as e:
            # 兜底防御: 万一 service 哪天破纪律抛了
            logger.error(f"InferWorker 捕获到 service 异常: {e}", exc_info=True)
            self.inference_failed.emit(str(e))

    @Slot()
    def cancel(self) -> None:
        """GUI Stop 按钮 → 触发取消信号."""
        logger.info("InferWorker.cancel: 发取消信号 (< 2s 内 pipeline 会响应)")
        self.cancel_token.cancel()

    # ---- hooks → Qt 信号 ----
    def _on_progress(self, evt: ProgressEvent) -> None:
        total = evt.total_frames if evt.total_frames is not None else -1
        self.progress_changed.emit(evt.frame_idx, total, evt.fps_loop)

    def _on_complete(self, result: InferResult) -> None:
        self.inference_finished.emit(result)

    def _on_error(self, exc: Exception) -> None:
        self.inference_failed.emit(str(exc))