# apps/platform/tests/test_inference.py
"""D7/D8 inference 子系统测试.

覆盖讲义自带的验证脚本:
  - §1.4 pipeline_config: yaml 解析 / CameraConfig 延迟构造 / 缺失文件降级
  - §1.5 output 段配置   : 默认值=改造前行为 / 改配置真生效 / 类型写错逐字段容错
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
import logging
import sys
import tempfile
import threading
import time
import types
from contextlib import contextmanager
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
from od_platform.inference.pipeline_config import (  # noqa: E402
    OutputConfig,
    coerce_output_config,
)
from od_platform.runtime_config.report import render_effective, render_overrides  # noqa: E402

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


# ============================================================================
# §1.5 输出落盘配置 (infer_pipeline.yaml 的 output 段)
# ----------------------------------------------------------------------------
# 两条硬要求各有一组钉子:
#   1) 配置缺失/字段缺失 → 默认值 = 改造前 LocalFileSink 写死的行为(文件名与字节一致)
#   2) 配置改了就真生效 → 产物名 / 编码器 / 质量 / 帧率随之变化, 且能被 cv2 读回
# ============================================================================
def test_output_config_defaults_are_legacy_behavior():
    """默认值必须逐字段等于改造前写死在 LocalFileSink 里的那些值."""
    cfg = OutputConfig()
    assert cfg.video_filename() == "output.mp4"          # 原: output.mp4
    assert cfg.image_filename("fish_0") == "fish_0.jpg"  # 原: <stem>.jpg
    assert cfg.video_fourcc == "mp4v"                    # 原: fourcc(*"mp4v")
    assert cfg.video_fps is None and cfg.video_fps_fallback == 30.0   # 原: 源帧率 else 30.0
    assert cfg.image_quality == 95                       # OpenCV 未传参时的默认质量
    assert cfg.overwrite_existing is True                # 原: cv2.imwrite/VideoWriter 直接覆盖
    assert cfg.keep_audit_json is True                   # 原: 必写审计快照
    assert cfg.audit_filename == "odp_audit.json"
    assert cfg.anonymous_frame_template.format(index=7) == "frame_000007"   # 原: frame_{:06d}


def test_pipeline_config_output_section_absent_uses_legacy_defaults(tmp_path):
    """output 段整段缺失 → 全默认值, 且来源表标 DEFAULT(不是静默空表)."""
    p = tmp_path / "no_output.yaml"
    p.write_text("visualization:\n  enabled: true\n", encoding="utf-8")
    pc = load_pipeline_config(str(p))
    assert pc.output == OutputConfig()
    labels = pc.output_merger.source_labels()
    assert labels["image_format"] == "DEFAULT" and labels["video_fourcc"] == "DEFAULT"
    eff = "\n".join(render_effective(pc.output, pc.output_merger))
    assert "(来源: DEFAULT)" in eff


def test_pipeline_config_output_overrides_and_provenance(tmp_path):
    """写进 yaml 的字段生效, 且能查到"值从哪来" + 覆盖链."""
    p = tmp_path / "out.yaml"
    p.write_text(
        "output:\n"
        "  filename_prefix: \"odp_\"\n"
        "  image_format: \".PNG\"\n"          # 前导点 + 大写都要被规范化
        "  image_quality: 60\n"
        "  video_basename: \"clip\"\n"
        "  video_format: \"avi\"\n"
        "  video_fourcc: \"MJPG\"\n"
        "  video_fps: \"auto\"\n"            # auto = 跟随源帧率(None)
        "  video_fps_fallback: 15\n"
        "  overwrite_existing: false\n"
        "  keep_audit_json: false\n"
        "  audit_filename: \"infer_audit.json\"\n",
        encoding="utf-8",
    )
    pc = load_pipeline_config(str(p))
    o = pc.output
    assert (o.filename_prefix, o.image_format, o.image_quality) == ("odp_", "png", 60)
    assert (o.video_basename, o.video_format, o.video_fourcc) == ("clip", "avi", "MJPG")
    assert o.video_fps is None and o.video_fps_fallback == 15.0
    assert o.overwrite_existing is False and o.keep_audit_json is False
    assert o.video_filename() == "odp_clip.avi"
    assert o.image_filename("fish") == "odp_fish.png"

    labels = pc.output_merger.source_labels()
    assert labels["image_format"] == "YAML" and labels["audit_filename"] == "YAML"
    eff = "\n".join(render_effective(o, pc.output_merger))
    assert "(来源: YAML)" in eff and "video_fourcc" in eff
    chains = "\n".join(render_overrides(o, pc.output_merger))
    assert "image_quality" in chains            # 95(DEFAULT) → 60(YAML)

    audit = pc.to_audit()
    assert audit["output"]["video_fourcc"] == "MJPG"
    assert audit["output_sources"]["image_format"] == "YAML"


def test_pipeline_config_output_bad_values_fall_back_with_warning(tmp_path):
    """类型/取值写错 → 逐个 warning + 只回落后该字段, 合法字段不受连坐."""
    p = tmp_path / "bad_output.yaml"
    p.write_text(
        "output:\n"
        "  image_quality: \"高清\"\n"          # 不是数字
        "  video_fourcc: \"mp4\"\n"            # 只有 3 个字符
        "  video_fps: -5\n"                    # <= 0
        "  image_format: \"gif\"\n"            # 不在允许清单
        "  filename_prefix: \"a/b\"\n"         # 带路径分隔符
        "  anonymous_frame_template: \"frame_{idx}\"\n"   # 缺 {index}
        "  overwrite_existing: \"maybe\"\n"    # 不是布尔
        "  unknown_knob: 1\n"                  # 未知字段
        "  video_basename: \"clip\"\n",        # ★ 唯一合法: 必须仍然生效
        encoding="utf-8",
    )
    with _capture_logs("od_platform.inference.pipeline_config") as logs:
        pc = load_pipeline_config(str(p))
    o = pc.output
    assert (o.image_quality, o.video_fourcc, o.video_fps) == (95, "mp4v", None)
    assert (o.image_format, o.filename_prefix) == ("jpg", "")
    assert o.anonymous_frame_template == "frame_{index:06d}"
    assert o.overwrite_existing is True
    assert o.video_basename == "clip"          # 合法字段照常生效
    assert pc.output_merger.source_labels()["video_basename"] == "YAML"
    assert pc.output_merger.source_labels()["video_fourcc"] == "YAML(无效,已回落默认)"
    text = "\n".join(logs)
    assert "output.image_quality" in text and "output.video_fourcc" in text
    assert "output.video_fps" in text and "未知字段 'unknown_knob'" in text


def test_pipeline_config_output_section_not_dict(tmp_path):
    """output 段不是字典 → warning + 全默认, 不抛."""
    p = tmp_path / "weird.yaml"
    p.write_text("output: 123\n", encoding="utf-8")
    with _capture_logs("od_platform.inference.pipeline_config") as logs:
        pc = load_pipeline_config(str(p))
    assert pc.output == OutputConfig()
    assert any("output 段必须是字典" in m for m in logs)


def test_pipeline_config_missing_file_records_output_provenance(tmp_path):
    """整个 yaml 缺失时, 输出配置仍要落到默认值且溯源表不空(审计要可查)."""
    pc = load_pipeline_config(str(tmp_path / "definitely_missing.yaml"))
    assert pc.output == OutputConfig()
    assert pc.output_merger.source_labels()["video_fourcc"] == "DEFAULT"


def test_coerce_output_config_accepts_none_config_and_dict():
    """sink 构造口的兼容适配: 旧调用点传 None 仍合法, 也接受 dict/OutputConfig."""
    assert coerce_output_config(None) == OutputConfig()
    assert coerce_output_config({"image_format": "png"}).image_format == "png"
    assert coerce_output_config({"image_format": "nope"}).image_format == "jpg"   # 容错回默认
    assert coerce_output_config(42) == OutputConfig()
    cfg = OutputConfig(image_format="png")
    assert coerce_output_config(cfg) is cfg


@contextmanager
def _capture_logs(logger_name: str):
    """直接挂在目标 logger 上收日志.

    仓库的 `get_logger` 会关掉 `odp_platform` 的 propagate(且 CLI 测试可能把级别设成
    ERROR), 所以 pytest 的 caplog(挂在 root) 收不到这些 warning; 这里显式接管。
    """
    records: list[str] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record.getMessage())

    target = logging.getLogger(logger_name)
    prev_level = target.level
    handler = _Collector()
    target.addHandler(handler)
    target.setLevel(logging.WARNING)
    try:
        yield records
    finally:
        target.removeHandler(handler)
        target.setLevel(prev_level)


def _frame(filename, index=0, fps=None):
    return types.SimpleNamespace(
        info=types.SimpleNamespace(filename=filename, frame_index=index, fps=fps))


def _noise(h=60, w=80, seed=7):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def test_local_file_sink_names_and_format_come_from_config(tmp_path):
    """filename_prefix / image_format / 兜底模板 → 产物名随配置变, 且能读回."""
    import cv2

    cfg = coerce_output_config({
        "filename_prefix": "odp_", "image_format": "png",
        "anonymous_frame_template": "anon_{index:05d}",
    })
    sink = LocalFileSink(cfg)
    sink.open(tmp_path, SourceType.IMAGE)
    img = _noise()
    sink.write(_frame("fish_0.png"), img)
    sink.write(_frame(None, index=42), img)
    sink.close()

    assert sorted(p.name for p in tmp_path.iterdir()) == \
        ["odp_anon_00042.png", "odp_fish_0.png"]      # 前缀对"兜底命名"同样生效
    back = cv2.imread(str(tmp_path / "odp_fish_0.png"))
    assert back is not None and list(back.shape) == [60, 80, 3]
    assert np.array_equal(back, img)          # PNG 无损: 像素逐一相同


def test_local_file_sink_video_codec_container_and_fps_from_config(tmp_path):
    """video_fourcc / video_format / video_basename / video_fps → 编码器与帧率随配置变."""
    import cv2

    cfg = coerce_output_config({
        "filename_prefix": "odp_", "video_basename": "clip", "video_format": "avi",
        "video_fourcc": "MJPG", "video_fps": 12.0,
    })
    sink = LocalFileSink(cfg)
    sink.open(tmp_path, SourceType.VIDEO)
    img = _noise()
    for i in range(8):
        sink.write(_frame(None, index=i, fps=25.0), (img + i).astype(np.uint8))
    sink.close()

    out = tmp_path / "odp_clip.avi"
    assert out.exists()
    cap = cv2.VideoCapture(str(out))
    try:
        assert cap.isOpened()
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        fi = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((fi >> (8 * k)) & 0xFF) for k in range(4))
        ok, frame = cap.read()
    finally:
        cap.release()
    assert n == 8
    assert abs(fps - 12.0) < 0.5              # 固定 12, 不是源帧率 25
    assert fourcc == "MJPG"                   # 编码器真的换了
    assert ok and list(frame.shape) == [60, 80, 3]


def test_local_file_sink_default_fps_follows_source_then_fallback(tmp_path):
    """video_fps=null/缺省 = 跟随源帧率; 源没有帧率时才用 video_fps_fallback."""
    import cv2

    def _fps_of(d: Path, name: str) -> float:
        cap = cv2.VideoCapture(str(d / name))
        try:
            return float(cap.get(cv2.CAP_PROP_FPS))
        finally:
            cap.release()

    img = _noise()

    d1 = tmp_path / "follow_source"
    d1.mkdir()
    sink = LocalFileSink(coerce_output_config({"video_fps": None}))
    sink.open(d1, SourceType.VIDEO)
    for i in range(4):
        sink.write(_frame(None, index=i, fps=25.0), img)
    sink.close()
    assert abs(_fps_of(d1, "output.mp4") - 25.0) < 0.5      # 跟随源

    d2 = tmp_path / "fallback_30"
    d2.mkdir()
    sink = LocalFileSink()
    sink.open(d2, SourceType.VIDEO)
    for i in range(4):
        sink.write(_frame(None, index=i, fps=None), img)
    sink.close()
    assert abs(_fps_of(d2, "output.mp4") - 30.0) < 0.5      # 老行为: 兜底 30

    d3 = tmp_path / "fallback_15"
    d3.mkdir()
    sink = LocalFileSink(coerce_output_config({"video_fps_fallback": 15}))
    sink.open(d3, SourceType.VIDEO)
    for i in range(4):
        sink.write(_frame(None, index=i, fps=None), img)
    sink.close()
    assert abs(_fps_of(d3, "output.mp4") - 15.0) < 0.5      # 兜底可配


def test_local_file_sink_image_quality_changes_bytes(tmp_path):
    """image_quality 真的改变 jpg 字节(体积/哈希), 且仍能正常读回."""
    import cv2

    img = _noise(seed=11)
    sizes = {}
    for q in (95, 40):
        d = tmp_path / f"q{q}"
        d.mkdir()
        sink = LocalFileSink(coerce_output_config({"image_quality": q}))
        sink.open(d, SourceType.IMAGE)
        sink.write(_frame("fish.jpg"), img)
        sink.close()
        p = d / "fish.jpg"
        back = cv2.imread(str(p))
        assert back is not None and list(back.shape) == [60, 80, 3]
        sizes[q] = p.stat().st_size
    assert sizes[40] < sizes[95]                       # 质量低 → 体积小
    assert (tmp_path / "q95" / "fish.jpg").read_bytes() != \
        (tmp_path / "q40" / "fish.jpg").read_bytes()


def test_local_file_sink_overwrite_existing_false_skips(tmp_path):
    """overwrite_existing=false → 同名不覆盖(保留第一次的字节); 默认 true → 覆盖."""
    import cv2

    first = np.full((40, 40, 3), 10, np.uint8)
    second = np.full((40, 40, 3), 200, np.uint8)

    d = tmp_path / "keep"
    d.mkdir()
    sink = LocalFileSink(coerce_output_config({"overwrite_existing": False}))
    sink.open(d, SourceType.IMAGE)
    sink.write(_frame("same.png"), first)
    h1 = (d / "same.jpg").read_bytes()
    sink.write(_frame("same.png"), second)
    sink.close()
    assert (d / "same.jpg").read_bytes() == h1
    assert cv2.imread(str(d / "same.jpg")).mean() < 50          # 还是 10 那张

    d2 = tmp_path / "replace"
    d2.mkdir()
    sink = LocalFileSink()
    sink.open(d2, SourceType.IMAGE)
    sink.write(_frame("same.png"), first)
    sink.write(_frame("same.png"), second)
    sink.close()
    assert cv2.imread(str(d2 / "same.jpg")).mean() > 150        # 第二次覆盖生效


# ============================================================================
# §4.3 service 透传: output 段 → sink / 审计快照
# ----------------------------------------------------------------------------
# 用替身换掉 YOLO 与流水线, 只验证"配置有没有真的透传下去"(不跑真推理, 秒级).
# ============================================================================
def _patch_service_for_lightweight_run(monkeypatch, tmp_path) -> dict:
    from od_platform.inference import pipeline as pipeline_mod
    from od_platform.inference import service as svc

    captured: dict = {}

    class _FakeYOLO:
        def __init__(self, path):
            self.names = {0: "fish"}

    class _RecorderPipeline:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self, stats):
            return False

    monkeypatch.setattr(svc, "YOLO", _FakeYOLO)
    monkeypatch.setattr(svc, "resolve_model_path", lambda *a, **k: tmp_path / "fake.pt")
    monkeypatch.setattr(svc, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(pipeline_mod, "ThreadedPipeline", _RecorderPipeline)
    return captured


def test_service_passes_output_config_to_local_file_sink(monkeypatch, tmp_path):
    captured = _patch_service_for_lightweight_run(monkeypatch, tmp_path)
    pipe_yaml = tmp_path / "pipe.yaml"
    pipe_yaml.write_text(
        "output:\n  filename_prefix: \"odp_\"\n  video_fourcc: \"MJPG\"\n"
        "  video_format: \"avi\"\n",
        encoding="utf-8",
    )
    res = InferService().predict(
        pipeline_yaml=str(pipe_yaml),
        cli_args={"source": str(tmp_path / "x.jpg"), "model": "yolo11n.pt", "save": True},
        rename_log=False,
    )
    assert res.success is True
    sink = captured["output_sink"]
    assert isinstance(sink, LocalFileSink)
    assert sink.cfg.filename_prefix == "odp_" and sink.cfg.video_fourcc == "MJPG"
    assert sink.cfg.video_filename() == "odp_output.avi"


def test_service_audit_json_is_configurable(monkeypatch, tmp_path):
    """keep_audit_json=false 不写审计快照; audit_filename 决定文件名."""
    _patch_service_for_lightweight_run(monkeypatch, tmp_path)

    off = tmp_path / "off.yaml"
    off.write_text("output:\n  keep_audit_json: false\n", encoding="utf-8")
    res = InferService().predict(
        pipeline_yaml=str(off),
        cli_args={"source": str(tmp_path / "x.jpg"), "model": "yolo11n.pt", "save": True},
        rename_log=False,
    )
    assert res.success is True and res.audit_path is None
    assert list(res.output_dir.glob("*.json")) == []

    custom = tmp_path / "custom.yaml"
    custom.write_text("output:\n  audit_filename: \"infer_audit.json\"\n", encoding="utf-8")
    res2 = InferService().predict(
        pipeline_yaml=str(custom),
        cli_args={"source": str(tmp_path / "x.jpg"), "model": "yolo11n.pt", "save": True},
        rename_log=False,
    )
    assert res2.audit_path is not None and res2.audit_path.name == "infer_audit.json"
    payload = json.loads(res2.audit_path.read_text(encoding="utf-8"))
    assert payload["pipeline"]["output"]["audit_filename"] == "infer_audit.json"
    assert payload["pipeline"]["output_sources"]["audit_filename"] == "YAML"
    assert payload["result_summary"]["sink"] == "LocalFileSink"


def test_service_keeps_caller_provided_sink_untouched(monkeypatch, tmp_path):
    captured = _patch_service_for_lightweight_run(monkeypatch, tmp_path)
    mine = NullSink()
    res = InferService().predict(
        cli_args={"source": str(tmp_path / "x.jpg"), "model": "yolo11n.pt", "save": True},
        output_sink=mine,
        rename_log=False,
    )
    assert res.success is True
    assert captured["output_sink"] is mine