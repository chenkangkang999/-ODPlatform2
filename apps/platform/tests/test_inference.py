# apps/platform/tests/test_inference.py
"""D7/D8 inference 子系统测试.

覆盖讲义自带的验证脚本:
  - §1.4 pipeline_config: yaml 解析 / CameraConfig 延迟构造 / 缺失文件降级
  - §2.3 overlay       : FPSCounter 双口径 / Metrics 5 维 snapshot + speed 平均
  - §3.4 三个接缝       : CancelToken / InferHooks(业务异常必吞) / OutputSink
  - §6.3 CLI           : cli_args 映射 (None 不进 dict) / 缺 source 退出码
  - §4.2 四级流水线      : reader→predict→render(/sink) 端到端 + 落盘

⚠ 前置桥接 (仅当真实包缺失时生效):
  D7 讲义声明 frame_source / visualization 包与 common.{log_rename,model_path,
  config_log} + paths.CHECKPOINTS_DIR "均已就位", 但本仓库当前尚未落地这些模块
  (另有讲义/由上层补齐). 为了让本测试今天就能验证 D7 自身代码, 这里在
  sys.modules 里注入【最小替身】, 且只在真正缺失时注入 —— 真包一旦落地,
  替身自动失效, 测试直接跑真实现.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest

# ============================================================================
# 前置桥接 (只在缺失时注入)
# ============================================================================
import od_platform  # noqa: E402
import od_platform.common.paths as _paths  # noqa: E402


def _missing(modname: str) -> bool:
    if modname in sys.modules:
        return False
    try:
        return importlib.util.find_spec(modname) is None
    except (ImportError, ModuleNotFoundError, ValueError):
        return True


def _make_module(name: str, **attrs) -> None:
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m


def _install_prereq_bridge() -> None:
    if not hasattr(_paths, "CHECKPOINTS_DIR"):
        _paths.CHECKPOINTS_DIR = _paths.MODELS_DIR / "checkpoints"

    if _missing("od_platform.common.log_rename"):
        _make_module("od_platform.common.log_rename",
                     rename_log_to_save_dir=lambda *a, **kw: None)

    if _missing("od_platform.common.model_path"):
        def resolve_model_path(name, search_dirs=None, **kw):
            p = Path(str(name))
            if p.is_file():
                return p
            for d in (search_dirs or []):
                cand = Path(d) / p.name
                if cand.is_file():
                    return cand
            raise FileNotFoundError(name)
        _make_module("od_platform.common.model_path", resolve_model_path=resolve_model_path)

    if _missing("od_platform.common.config_log"):
        _make_module("od_platform.common.config_log",
                     log_effective_config=lambda *a, **kw: None,
                     log_override_chains=lambda *a, **kw: None)

    if _missing("od_platform.frame_source"):
        import cv2
        from enum import Enum
        from pydantic import BaseModel, ConfigDict

        class SourceType(str, Enum):
            IMAGE = "image"
            IMAGE_FOLDER = "image_folder"
            VIDEO = "video"
            CAMERA = "camera"
            STREAM = "stream"

        IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})

        class CameraConfig(BaseModel):
            model_config = ConfigDict(extra="forbid")
            camera_id: int = 0
            width: int = 640
            height: int = 480
            fps: int = 30
            backend: str = "auto"

        class _Folder:
            def __init__(self, paths, stride=1):
                self._paths = list(paths)[::max(1, stride)]

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def get_source_type(self):
                return SourceType.IMAGE_FOLDER

            def __iter__(self):
                for i, p in enumerate(self._paths):
                    img = cv2.imread(str(p))
                    if img is None:
                        continue
                    yield types.SimpleNamespace(
                        image=img,
                        info=types.SimpleNamespace(frame_index=i, filename=Path(p).name,
                                                   fps=None, source=str(p)))

        def create_frame_source(source, *, stride=1):
            p = Path(str(source))
            imgs = [q for q in sorted(p.iterdir())
                    if q.is_file() and q.suffix.lower() in IMAGE_EXTENSIONS]
            if not imgs:
                raise FileNotFoundError(str(source))
            return _Folder(imgs, stride)

        _make_module("od_platform.frame_source", SourceType=SourceType,
                     CameraConfig=CameraConfig, IMAGE_EXTENSIONS=IMAGE_EXTENSIONS,
                     create_frame_source=create_frame_source)

    if _missing("od_platform.visualization"):
        import cv2
        from dataclasses import dataclass

        @dataclass
        class DrawStyle:
            box_thickness: int = 2
            text_scale: float = 0.5

            @classmethod
            def from_image_size(cls, h, w, **overrides):
                return cls(**{k: v for k, v in overrides.items()
                              if k in ("box_thickness", "text_scale")})

        class BeautifyVisualizer:
            def __init__(self, labels=None, label_mapping=None, color_mapping=None,
                         default_color=(0, 255, 0), font_path=None):
                self.label_mapping = dict(label_mapping or {})
                self.color_mapping = dict(color_mapping or {})
                self.default_color = tuple(default_color)

            @staticmethod
            def from_yolo_results(boxes=None, confidences=None, labels=None):
                boxes = np.asarray(boxes if boxes is not None else np.zeros((0, 4)))
                labels = list(labels or [])
                return [{"xyxy": boxes[i].tolist(), "conf": 0.0,
                         "label": labels[i] if i < len(labels) else "?"}
                        for i in range(len(boxes))]

            def draw(self, image, dets, *, style=None, use_label_mapping=True):
                out = image.copy()
                for d in dets:
                    x1, y1, x2, y2 = [int(v) for v in d["xyxy"]]
                    name = d["label"]
                    disp = self.label_mapping.get(name, name) if use_label_mapping else name
                    color = self.color_mapping.get(name, self.default_color)
                    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(out, disp, (x1, max(12, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                return out

        _make_module("od_platform.visualization",
                     BeautifyVisualizer=BeautifyVisualizer, DrawStyle=DrawStyle)


_install_prereq_bridge()

from od_platform.cli.infer import _build_parser, _ns_to_cli_args, main  # noqa: E402
from od_platform.frame_source import SourceType  # noqa: E402
from od_platform.inference import (  # noqa: E402
    CancelToken,
    FrameEvent,
    InferHooks,
    InferService,
    InferStats,
    LocalFileSink,
    NullSink,
    OutputSink,
    ProgressEvent,
    infer_yolo,
    load_pipeline_config,
)
from od_platform.inference.overlay import FPSCounter, Metrics  # noqa: E402

PIPE_YAML = _paths.APP_DIR / "configs" / "runtime" / "infer_pipeline.yaml"


# ============================================================================
# §1.4 pipeline_config
# ============================================================================
def test_pipeline_config_parses_bundled_yaml():
    pc = load_pipeline_config(str(PIPE_YAML))
    assert pc.viz_enabled is True
    assert len(pc.label_mapping) == 8 and pc.label_mapping["person"] == "人员"
    assert len(pc.color_mapping) == 5
    assert pc.color_mapping["person"] == (0, 255, 0)      # list → tuple(BGR)
    assert pc.default_color == (0, 255, 0)
    assert pc.camera_raw == {"width": 1280, "height": 720, "fps": 30}
    assert pc.to_audit()["color_mapping_n"] == 5


def test_pipeline_config_missing_file_falls_back_to_defaults():
    pc = load_pipeline_config(str(PIPE_YAML.parent / "definitely_missing.yaml"))
    assert pc.viz_enabled is True and pc.camera_raw == {}
    assert pc.build_camera_config() is None


def test_pipeline_config_camera_deferred_construction():
    pc = load_pipeline_config(str(PIPE_YAML))
    cam = pc.build_camera_config()
    assert (cam.camera_id, cam.width, cam.height, cam.fps, cam.backend) == \
        (0, 1280, 720, 30, "auto")


def test_pipeline_config_rejects_bad_color(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("visualization:\n  color_mapping:\n    person: [1, 2]\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_pipeline_config(str(bad))


# ============================================================================
# §2.3 overlay
# ============================================================================
def test_fps_counter_window_and_inst():
    fps = FPSCounter(window_size=5)
    for ms in [10, 12, 11, 9, 10]:
        fps.update(ms)
    assert abs(fps.fps - 96.2) < 1.0
    assert abs(fps.inst - 100.0) < 1.0
    assert FPSCounter().fps == 0.0 and FPSCounter().inst == 0.0


def test_metrics_snapshot_and_speed_avg():
    m = Metrics()
    m.capture.update(33.3)
    m.add_speed({"preprocess": 1.5, "inference": 8.0, "postprocess": 2.0})
    m.add_speed({"preprocess": 1.6, "inference": 8.5, "postprocess": 2.1})
    m.render.update(28.0)
    m.loop.update(50.0)
    snap = m.snapshot()
    assert snap["capture_fps"] == 30.0
    assert snap["infer_fps"] == 121.2 and snap["loop_fps"] == 20.0
    assert m.speed_avg_ms() == {"preprocess": 1.55, "inference": 8.25, "postprocess": 2.05}
    assert Metrics().speed_avg_ms() == {}


# ============================================================================
# §3.4 三个接缝
# ============================================================================
def test_cancel_token_basic_and_idempotent():
    t = CancelToken()
    assert not t.is_cancelled()
    t.cancel()
    assert t.is_cancelled()
    t.cancel(); t.cancel()          # 幂等
    assert t.wait(0.01) is True


def test_cancel_token_cross_thread_visible():
    t = CancelToken()
    seen = threading.Event()

    def watcher():
        for _ in range(100):
            if t.is_cancelled():
                seen.set()
                return
            time.sleep(0.005)

    threading.Thread(target=watcher).start()
    time.sleep(0.05)
    t.cancel()
    assert seen.wait(0.5)


def test_infer_hooks_empty_shortcircuits():
    h = InferHooks()
    assert h.on_frame is None and h.progress_interval_frames == 30
    fe = FrameEvent(0, np.zeros((1, 1, 3), np.uint8), np.zeros((1, 1, 3), np.uint8), 0)
    pe = ProgressEvent(10, 100, 1.0, 30.0, 60.0, 5)
    h.fire_frame(fe); h.fire_progress(pe); h.fire_complete("R"); h.fire_error(RuntimeError())


def test_infer_hooks_invokes_business_callbacks():
    calls = []
    h = InferHooks(on_frame=lambda e: calls.append(("frame", e.frame_idx)),
                   on_complete=lambda r: calls.append(("done", r)))
    fe = FrameEvent(0, np.zeros((1, 1, 3), np.uint8), np.zeros((1, 1, 3), np.uint8), 0)
    h.fire_frame(fe); h.fire_complete("OK")
    assert calls == [("frame", 0), ("done", "OK")]


def test_infer_hooks_swallows_business_exceptions():
    """核心纪律: 业务回调抛异常不能炸 pipeline."""
    def bad(*a, **kw):
        raise RuntimeError("user code bug")

    h = InferHooks(on_frame=bad, on_progress=bad, on_complete=bad, on_error=bad)
    fe = FrameEvent(0, np.zeros((1, 1, 3), np.uint8), np.zeros((1, 1, 3), np.uint8), 0)
    pe = ProgressEvent(1, None, 0.1, 1.0, 1.0, 0)
    h.fire_frame(fe); h.fire_progress(pe); h.fire_complete("x"); h.fire_error(Exception())


def test_output_sink_is_abstract():
    with pytest.raises(TypeError):
        OutputSink()  # type: ignore[abstract]


def test_null_sink_noop_and_idempotent(tmp_path):
    sink = NullSink()
    sink.open(tmp_path, SourceType.VIDEO)
    for _ in range(5):
        sink.write(types.SimpleNamespace(), np.zeros((10, 10, 3), np.uint8))
    sink.close(); sink.close()
    assert list(tmp_path.iterdir()) == []


def test_local_file_sink_writes_jpgs_for_image_source(tmp_path):
    sink = LocalFileSink()
    sink.open(tmp_path, SourceType.IMAGE)
    for i in range(3):
        frame = types.SimpleNamespace(
            info=types.SimpleNamespace(filename=f"test_{i}.png", frame_index=i))
        sink.write(frame, np.full((50, 50, 3), 128, np.uint8))
    sink.close()
    assert sorted(p.name for p in tmp_path.glob("*.jpg")) == \
        ["test_0.jpg", "test_1.jpg", "test_2.jpg"]


def test_local_file_sink_writes_mp4_for_video_source(tmp_path):
    sink = LocalFileSink()
    sink.open(tmp_path, SourceType.VIDEO)
    for i in range(8):
        frame = types.SimpleNamespace(
            info=types.SimpleNamespace(fps=30.0, frame_index=i, filename=None))
        sink.write(frame, np.full((100, 200, 3), i * 30, np.uint8))
    sink.close()
    assert (tmp_path / "output.mp4").exists()


# ============================================================================
# §6.3 CLI
# ============================================================================
def test_cli_arg_mapping_skips_cli_only_switches():
    ns = _build_parser().parse_args(
        ["--source", "imgs", "--model", "yolo11n.pt", "--conf", "0.5",
         "--no-save", "--no-viz", "--warmup", "1"])
    assert _ns_to_cli_args(ns) == \
        {"source": "imgs", "model": "yolo11n.pt", "conf": 0.5, "save": False}


def test_cli_missing_source_returns_exit_code_1(capsys, tmp_path, monkeypatch):
    """纪律 A: service 永不抛 → CLI 友好报错 + 退出码 1."""
    monkeypatch.chdir(tmp_path)          # 让 CLI 的 logs/ 落在临时目录, 不污染仓库
    assert main(["--log-level", "ERROR"]) == 1
    assert "未指定推理输入源" in capsys.readouterr().err


# ============================================================================
# §4.2 四级流水线端到端 (reader → predict → render → sink)
# ============================================================================
class _FakeProcessor:
    """替掉 GPU predict + 美化绘制, 只验证流水线装配/线程/落盘契约."""

    def __init__(self):
        self.batches = []

    def infer_batch(self, images):
        self.batches.append(len(images))
        results = [types.SimpleNamespace(
            boxes=None, speed={"preprocess": 1.0, "inference": 2.0, "postprocess": 0.5})
            for _ in images]
        return results, [[] for _ in images], [0] * len(images), 0.01

    def draw(self, image, result, labels, n):
        return image.copy()


def _write_images(folder: Path, n: int = 6, size=(64, 96)) -> None:
    import cv2
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        img = rng.integers(0, 255, (size[0], size[1], 3), dtype=np.uint8)
        assert cv2.imwrite(str(folder / f"img_{i}.jpg"), img)


def test_threaded_pipeline_end_to_end_writes_artifacts(tmp_path):
    from od_platform.inference.pipeline import ThreadedPipeline

    imgs = tmp_path / "imgs"
    _write_images(imgs, 6)
    out = tmp_path / "out"
    out.mkdir()

    proc = _FakeProcessor()
    stats = InferStats()
    sink = LocalFileSink()
    pipeline = ThreadedPipeline(
        processor=proc, source=str(imgs), camera_config=None, output_dir=out,
        output_sink=sink, batch_size=2, save=True, show=False, show_info=False,
        window_name="test", warmup_frames=0,
    )
    interrupted = pipeline.run(stats)

    assert interrupted is False
    assert stats.frames == 6
    assert sum(proc.batches) == 6
    assert stats.loop_fps > 0 and stats.capture_fps > 0
    assert stats.speed_ms == {"preprocess": 1.0, "inference": 2.0, "postprocess": 0.5}
    assert sorted(p.name for p in out.glob("*.jpg")) == [f"img_{i}.jpg" for i in range(6)]


def test_threaded_pipeline_cancel_token_and_hooks(tmp_path):
    from od_platform.inference.pipeline import ThreadedPipeline

    imgs = tmp_path / "imgs"
    _write_images(imgs, 8)
    out = tmp_path / "out"
    out.mkdir()

    token = CancelToken()
    seen = []

    def on_progress(evt):
        seen.append(evt.frame_idx)
        token.cancel()

    hooks = InferHooks(on_progress=on_progress, progress_interval_frames=2)
    stats = InferStats()
    pipeline = ThreadedPipeline(
        processor=_FakeProcessor(), source=str(imgs), camera_config=None,
        output_dir=out, output_sink=NullSink(), batch_size=1, save=False,
        show=False, show_info=False, window_name="test", warmup_frames=0,
        hooks=hooks, cancel_token=token,
    )
    assert pipeline.run(stats) is True          # 被用户中断
    assert stats.frames == 2 < 8
    assert seen == [2]
    assert list(out.iterdir()) == []            # NullSink 不落盘


# ============================================================================
# 纪律 A: InferService 永不抛
# ============================================================================
def test_infer_service_never_raises_without_source():
    res = InferService().predict(cli_args={"model": "yolo11n.pt"}, rename_log=False)
    assert res.success is False
    assert "未指定推理输入源" in (res.error or "")


def test_infer_yolo_never_raises_without_source():
    res = infer_yolo(cli_args={"model": "yolo11n.pt"}, rename_log=False)
    assert res.success is False and res.stats == {}
    assert json.dumps(res.stats) == "{}"