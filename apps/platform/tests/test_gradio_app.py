# apps/platform/tests/test_gradio_app.py
"""odp-demo（visualization/gradio_app + runtime_config/demo）测试.

★ 关键约束: **CI 不装 gradio**（它只是 pyproject 里 ``[ui]`` 这组可选依赖）。
  因此本文件守三条纪律:

  1. 顶层不 import gradio —— 被测模块自己也不许在顶层 import（用 AST 钉住,
     见 :func:`test_top_level_never_imports_gradio`），并在子进程里把
     ``sys.modules["gradio"] = None`` 再 import 一次, 证明"没装 gradio 也能 import";
  2. 纯函数（检测明细 → 表格行 / 汇总 / 权重按名解析 / 配置三源合并）在没有 gradio 时
     照常可测 —— 这些测试连 gradio 是否存在都不问;
  3. 未安装 gradio 时 ``main()`` 必须给出中文提示并以非 0 退出码结束。

★ "一切皆配置"的回归钉子: 这里的默认值断言**不写** 0.25 / 0.7 / 640 这类字面量,
  而是反过来证明"生效值 == 配置文件里写的值"(见
  :func:`test_infer_defaults_come_from_infer_yaml` / :func:`test_demo_defaults_come_from_demo_yaml`)。
"""
from __future__ import annotations

import ast
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from od_platform.common.paths import runtime_config_path
from od_platform.visualization import gradio_app


# ============================================================================
# 假替身: 不进 GPU、不加载真权重, 只模拟 ultralytics Results 的形状
# ============================================================================
class _TorchLike:
    """模拟 torch 张量的 ``.cpu().numpy()`` 三段式(证明 _to_numpy 的鸭子类型适配)。"""

    def __init__(self, array):
        self._array = np.asarray(array)

    def cpu(self):
        return self

    def numpy(self):
        return self._array


class _FakeBoxes:
    """只提供 ultralytics ``result.boxes`` 会用到的三段数据。"""

    def __init__(self, xyxy, conf, cls, *, torch_like: bool = False):
        wrap = _TorchLike if torch_like else (lambda value: np.asarray(value, dtype=float))
        self.xyxy = wrap(xyxy)
        self.conf = wrap(conf)
        self.cls = wrap(cls)

    def __len__(self) -> int:
        return len(np.asarray(self.conf))


class _FakeResult:
    """够用的 Results 替身: ``boxes`` + ``plot()``。"""

    def __init__(self, boxes=None, *, plot_value=None):
        self.boxes = boxes
        self._plot_value = plot_value

    def plot(self):
        if self._plot_value is not None:
            return self._plot_value
        return np.full((8, 8, 3), 7, dtype=np.uint8)


class _FakeModel:
    """可调用模型替身: 记录收到的参数, 返回事先备好的 Results。"""

    def __init__(self, result, names=None):
        self.result = result
        self.names = names if names is not None else {0: "Mola", 1: "Tilapia"}
        self.calls: list[tuple] = []

    def __call__(self, images, **kwargs):
        self.calls.append((images, kwargs))
        return [self.result]


def _write_image(tmp_path: Path, name: str = "fish.png") -> Path:
    """真写一张 PNG 出来(读图走 frame_source, 必须有真文件)。"""
    import cv2
    path = tmp_path / name
    cv2.imwrite(str(path), np.full((64, 80, 3), 128, dtype=np.uint8))
    return path


# ============================================================================
# 1) gradio 只在函数内部 import —— 三条独立证据
# ============================================================================
def test_top_level_never_imports_gradio():
    """AST 级钉子: 模块顶层的 import 里不许出现 gradio(也不许 from gradio import ...)。"""
    source = Path(gradio_app.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    top_level_modules: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_modules += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            top_level_modules.append(node.module or "")

    offenders = [name for name in top_level_modules
                 if name == "gradio" or name.startswith("gradio.")]
    assert offenders == [], f"模块顶层不该 import gradio, 但发现: {offenders}"


def test_importable_without_gradio_in_fresh_interpreter():
    """真证据: 新进程里把 gradio 屏蔽掉再 import —— 模块照样进得来。

    ``sys.modules["gradio"] = None`` 会让 ``import gradio`` 抛 ImportError,
    等价于"这个环境没装 gradio"。
    """
    code = (
        "import sys; sys.modules['gradio'] = None; "
        "import od_platform.visualization.gradio_app as m; "
        "print(m.__file__)"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, f"没装 gradio 时 import 失败:\n{proc.stderr}"
    assert proc.stdout.strip().endswith("gradio_app.py")


def test_pure_helpers_work_without_gradio(monkeypatch):
    """纯函数在没有 gradio 的进程里照常可测(屏蔽 gradio 后直接调)。"""
    monkeypatch.setitem(sys.modules, "gradio", None)

    rows = gradio_app.detection_rows(
        [{"cls": "Mola", "conf": 0.87123, "xyxy": [1.0, 2.0, 30.0, 40.0]}]
    )
    assert rows == [["", 1, "Mola", 0.8712, 1.0, 2.0, 30.0, 40.0]]

    summary = gradio_app.summarize_detections([{"cls": "Mola"}, {"cls": "Mola"}])
    assert summary == {"total": 2, "per_class": {"Mola": 2}, "classes": 1}


def test_main_without_gradio_gives_chinese_hint_and_nonzero(monkeypatch, capsys, tmp_path):
    """没装 gradio: main() 打中文提示 + 返回非 0 退出码, 且日志走 paths.LOGGING_DIR + log_type="demo"。"""
    logged: dict = {}
    monkeypatch.setattr(gradio_app, "LOGGING_DIR", tmp_path)
    monkeypatch.setattr(gradio_app, "get_logger", lambda **kwargs: logged.update(kwargs))
    monkeypatch.setitem(sys.modules, "gradio", None)

    exit_code = gradio_app.main([])

    captured = capsys.readouterr()
    assert exit_code != 0
    assert exit_code == gradio_app.EXIT_NO_GRADIO
    assert "gradio" in captured.err
    assert "pip install" in captured.err
    assert "odp-demo" in captured.err
    assert logged["log_type"] == "demo"
    assert logged["base_path"] == tmp_path


def test_build_app_reports_missing_gradio_with_chinese_message(monkeypatch):
    """没装 gradio 时 build_app() 抛的是 GradioMissingError(带中文指引), 而不是 AttributeError。"""
    monkeypatch.setitem(sys.modules, "gradio", None)
    with pytest.raises(gradio_app.GradioMissingError) as excinfo:
        gradio_app.build_app()
    assert "pip install" in str(excinfo.value)


def test_half_uninstalled_gradio_is_treated_as_missing(monkeypatch, capsys, tmp_path):
    """回归钉子: pip 卸载 gradio 后会残留一个**空的命名空间包**(import 成功、没有 Blocks)。

    这种"半卸载"状态也必须走"没装 gradio"的分支(中文提示 + 退出码 2),
    不能一路崩到 ``gr.Blocks`` 的 AttributeError。
    """
    stub = types.ModuleType("gradio")          # import 得进来, 但里面什么组件都没有
    monkeypatch.setitem(sys.modules, "gradio", stub)
    monkeypatch.setattr(gradio_app, "LOGGING_DIR", tmp_path)
    monkeypatch.setattr(gradio_app, "get_logger", lambda **kwargs: None)

    exit_code = gradio_app.main([])

    captured = capsys.readouterr()
    assert exit_code == gradio_app.EXIT_NO_GRADIO
    assert "gradio" in captured.err
    assert "残缺" in captured.err


def test_help_works_without_gradio(capsys, monkeypatch):
    """``--help`` 不依赖 gradio(参数解析在 import gradio 之前)。"""
    monkeypatch.setitem(sys.modules, "gradio", None)
    with pytest.raises(SystemExit) as excinfo:
        gradio_app.main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    for flag in ("--server-port", "--weight", "--conf", "--iou", "--imgsz",
                 "--device", "--no-viz", "--max-images"):
        assert flag in out


# ============================================================================
# 2) 纯函数: 检测明细 → 表格行 / 汇总 / Markdown
# ============================================================================
def test_detection_rows_uses_display_name_and_handles_missing_values():
    display = {"Mola": "翻车鱼"}.get
    rows = gradio_app.detection_rows(
        [
            {"cls": "Mola", "conf": 0.5, "xyxy": [0.0, 1.0, 2.0, 3.0]},
            {"cls": "Mola"},                        # 没有 conf / 没有坐标 —— 不许炸
        ],
        display_name=display,
        image_name="a.jpg",
    )
    assert rows[0] == ["a.jpg", 1, "翻车鱼", 0.5, 0.0, 1.0, 2.0, 3.0]
    assert rows[1] == ["a.jpg", 2, "翻车鱼", None, None, None, None, None]


def test_detection_rows_matches_table_schema():
    """表格列数必须与 TABLE_HEADERS / TABLE_DATATYPES 对齐(界面靠这个固定表头)。"""
    rows = gradio_app.detection_rows([{"cls": "Mola", "conf": 0.5, "xyxy": [0, 0, 1, 1]}])
    assert len(rows[0]) == len(gradio_app.TABLE_HEADERS) == len(gradio_app.TABLE_DATATYPES)


def test_summarize_detections_sorts_by_count_then_name():
    detections = [{"cls": "Tilapia"}, {"cls": "Mola"}, {"cls": "Tilapia"}, {"cls": "Aba"}]
    summary = gradio_app.summarize_detections(detections)
    assert summary["total"] == 4
    assert summary["classes"] == 3
    assert list(summary["per_class"].items()) == [("Tilapia", 2), ("Aba", 1), ("Mola", 1)]


def test_format_summary_markdown_contains_totals_and_empty_hint():
    summary = gradio_app.summarize_detections([{"cls": "Mola", "conf": 0.9}])
    text = gradio_app.format_summary_markdown(summary, images=1, params="`conf=0.5`",
                                             notes=["> 提示"])
    assert "总检出数：1" in text
    assert "| Mola | 1 |" in text
    assert "`conf=0.5`" in text
    assert "> 提示" in text

    empty = gradio_app.format_summary_markdown(
        gradio_app.summarize_detections([]), images=1)
    assert "总检出数：0" in empty
    assert "没有检出任何目标" in empty


def test_config_source_rows_takes_source_from_merger():
    config, merger = gradio_app.load_demo_config({"server_port": 7999})
    rows = {row[0]: row for row in gradio_app.config_source_rows(
        config, merger, ("server_port", "title"))}

    assert rows["server_port"][1] == "7999"
    assert rows["server_port"][2] == "CLI"          # 来源是合并器记的, 不是猜的
    assert rows["title"][2] == "YAML"


def test_format_config_source_markdown_lists_every_source_file():
    setup = gradio_app.build_setup()
    text = gradio_app.format_config_source_markdown(
        demo_rows=gradio_app.config_source_rows(setup.demo_config, setup.demo_merger,
                                                tuple(gradio_app.YOLODemoConfig.model_fields)),
        infer_rows=gradio_app.config_source_rows(setup.infer_config, setup.infer_merger,
                                                 gradio_app.INFER_SOURCE_FIELDS),
        demo_path=setup.demo_path,
        infer_path=setup.infer_path,
        pipeline_path=setup.pipeline_path,
        weight_ref=setup.weight_ref,
        weight_path=setup.weight_path,
        beautify=setup.beautify,
    )
    assert "配置来源" in text
    assert "configs" in text and "runtime" in text
    for name in gradio_app.INFER_SOURCE_FIELDS:
        assert f"| {name} |" in text


# ============================================================================
# 3) 纯函数: ultralytics Results → 检测明细
# ============================================================================
def test_detections_from_yolo_result_with_numpy():
    boxes = _FakeBoxes([[10, 20, 30, 40]], [0.9], [1.0])
    detections = gradio_app.detections_from_yolo_result(
        SimpleNamespace(boxes=boxes), {0: "Mola", 1: "Tilapia"})

    assert detections == [{"cls": "Tilapia", "conf": 0.9, "xyxy": [10.0, 20.0, 30.0, 40.0]}]


def test_detections_from_yolo_result_with_torch_like_tensors():
    """torch 张量风格(``.cpu().numpy()``)也要能解析 —— 引擎里就是这么拿的。"""
    boxes = _FakeBoxes([[1, 2, 3, 4]], [0.25], [0.0], torch_like=True)
    detections = gradio_app.detections_from_yolo_result(SimpleNamespace(boxes=boxes), {0: "Mola"})
    assert detections == [{"cls": "Mola", "conf": 0.25, "xyxy": [1.0, 2.0, 3.0, 4.0]}]


def test_detections_from_yolo_result_empty_and_missing_boxes():
    assert gradio_app.detections_from_yolo_result(SimpleNamespace(boxes=None), {}) == []
    assert gradio_app.detections_from_yolo_result(SimpleNamespace(boxes=_FakeBoxes([], [], [])), {}) == []
    # 名字表里没有这个 id: 退化成 id 字符串, 不炸
    boxes = _FakeBoxes([[0, 0, 1, 1]], [0.5], [7.0])
    assert gradio_app.detections_from_yolo_result(SimpleNamespace(boxes=boxes), {})[0]["cls"] == "7"
    # model.names 也可能是 list
    assert gradio_app.detections_from_yolo_result(
        SimpleNamespace(boxes=boxes), ["Mola"])[0]["cls"] == "7"


# ============================================================================
# 4) 权重按名字解析（源码里不出现盘符 / 绝对路径）
# ============================================================================
def test_resolve_weight_ref_searches_trained_then_pretrained(tmp_path, monkeypatch):
    trained = tmp_path / "trained"
    pretrained = tmp_path / "pretrained"
    trained.mkdir()
    pretrained.mkdir()
    monkeypatch.setattr(gradio_app, "WEIGHT_SEARCH_DIRS", (trained, pretrained))

    pretrained.joinpath("yolo11n.pt").write_bytes(b"x")
    assert gradio_app.resolve_weight_ref("yolo11n") == pretrained / "yolo11n.pt"

    trained.joinpath("yolo11n.pt").write_bytes(b"x")     # 自训练产物优先于预训练权重
    assert gradio_app.resolve_weight_ref("yolo11n") == trained / "yolo11n.pt"
    assert gradio_app.resolve_weight_ref("yolo11n.pt") == trained / "yolo11n.pt"
    assert gradio_app.resolve_weight_ref("  yolo11n  ") == trained / "yolo11n.pt"


def test_resolve_weight_ref_falls_back_to_trained_dir(tmp_path, monkeypatch):
    """名字扫不到时回落到 models/trained 下的既定落点(交给加载时报"文件不存在")。"""
    empty = tmp_path / "trained"
    empty.mkdir()
    monkeypatch.setattr(gradio_app, "WEIGHT_SEARCH_DIRS", (empty,))
    monkeypatch.setattr(gradio_app, "resolve_trained_model",
                        lambda ref: empty / f"{ref}.pt")

    resolved = gradio_app.resolve_weight_ref("not-exist")
    assert resolved == empty / "not-exist.pt"


def test_resolve_weight_ref_rejects_empty_name(monkeypatch):
    monkeypatch.setattr(gradio_app, "WEIGHT_SEARCH_DIRS", ())
    with pytest.raises(ValueError, match="权重名"):
        gradio_app.resolve_weight_ref("")


def test_discover_and_choose_weight_refs(tmp_path, monkeypatch):
    trained = tmp_path / "trained"
    trained.mkdir()
    for name in ("b-model.pt", "a-model.pt"):
        trained.joinpath(name).write_bytes(b"x")
    monkeypatch.setattr(gradio_app, "WEIGHT_SEARCH_DIRS", (trained,))

    assert gradio_app.discover_weight_refs() == ["a-model", "b-model"]
    assert gradio_app.default_weight_ref() == "a-model"
    # 当前默认权重排第一, 且不重复
    assert gradio_app.weight_choices("b-model") == ["b-model", "a-model"]


def test_default_weight_ref_of_repo_prefers_demo_yaml_value():
    """本仓库的真实默认: demo.yaml 的 weight 能被解析到 models/trained 下(名字 → 路径)。"""
    setup = gradio_app.build_setup()
    assert setup.weight_ref == gradio_app.load_demo_config()[0].weight
    assert setup.weight_path.name == f"{setup.weight_ref}.pt"
    assert "trained" in str(setup.weight_path).replace("\\", "/")


# ============================================================================
# 5) 配置来源: demo.yaml / infer.yaml 三源合并 + 溯源（零硬编码的证明）
# ============================================================================
def test_demo_defaults_come_from_demo_yaml():
    """界面旋钮的默认值 == demo.yaml 里写的值, 且溯源标成 YAML。"""
    raw = yaml.safe_load(runtime_config_path("demo").read_text(encoding="utf-8"))
    config, merger = gradio_app.load_demo_config()

    for name in ("title", "server_name", "server_port", "share", "inbrowser",
                 "max_images", "weight"):
        assert getattr(config, name) == raw[name], name
        metadata = merger.get_metadata(name)
        assert metadata is not None and metadata.source_label == "YAML", name


def test_infer_defaults_come_from_infer_yaml():
    """推理旋钮的默认值 == infer.yaml 里写的值(不是界面代码里的字面量)。"""
    raw = yaml.safe_load(runtime_config_path("infer").read_text(encoding="utf-8"))
    config, merger = gradio_app.load_infer_config()

    for name in ("conf", "iou", "imgsz", "max_det", "batch"):
        assert getattr(config, name) == raw[name], name
        assert merger.get_metadata(name).source_label == "YAML", name

    # infer.yaml 里 device: null → 合并器不记来源(None 不参与覆盖), 生效值就是模型默认的 None
    assert config.device is None
    assert merger.get_metadata("device") is None
    assert gradio_app.device_label(config.device) == gradio_app.DEVICE_AUTO


def test_demo_config_cli_overrides_yaml_and_keeps_chain():
    config, merger = gradio_app.load_demo_config({"server_port": 7999})
    assert config.server_port == 7999
    assert merger.get_metadata("server_port").source_label == "CLI"
    # 覆盖链: 新值(CLI) ← 旧值(YAML) ← 默认值(DEFAULT)
    chain = merger.get_metadata("server_port").chain()
    assert [node.source_label for node in chain] == ["CLI", "YAML", "DEFAULT"]


def test_infer_config_cli_overrides_and_device_auto_maps_to_none():
    config, _merger = gradio_app.load_infer_config(
        {"conf": 0.5, "iou": 0.6, "imgsz": 512, "device": gradio_app.DEVICE_AUTO})
    assert (config.conf, config.iou, config.imgsz) == (0.5, 0.6, 512)
    assert config.device is None


def test_split_cli_args_separates_demo_and_infer_fields():
    demo_cli, infer_cli = gradio_app.split_cli_args(
        {"server_port": 7999, "weight": "yolo11n", "conf": 0.4, "device": "cpu"})
    assert demo_cli == {"server_port": 7999, "weight": "yolo11n"}
    assert infer_cli == {"conf": 0.4, "device": "cpu"}


def test_split_cli_args_rejects_unknown_field():
    with pytest.raises(ValueError, match="未知的 odp-demo 参数"):
        gradio_app.split_cli_args({"conf": 0.4, "typo_field": 1})


def test_cli_args_from_namespace_skips_none_and_switches():
    ns = gradio_app._build_parser().parse_args(
        ["--weight", "yolo11n", "--no-viz", "--log-level", "DEBUG"])
    args = gradio_app.cli_args_from_namespace(ns)
    assert args == {"weight": "yolo11n"}          # None 的字段 + CLI 开关都不进配置合并


def test_build_setup_merges_every_source():
    setup = gradio_app.build_setup({"server_port": 7999, "max_images": 3,
                                    "conf": 0.5, "device": "cpu"})
    assert setup.demo_config.server_port == 7999
    assert setup.demo_config.max_images == 3
    assert setup.infer_config.conf == 0.5
    assert setup.infer_config.device == "cpu"
    assert setup.infer_config.model == setup.weight_ref        # 权重名也进了 D5 合并(溯源口径一致)
    assert setup.pipeline_path == runtime_config_path(gradio_app.PIPELINE_CONFIG_NAME)
    assert setup.beautify is setup.pipeline.viz_enabled
    assert setup.weight_ref in setup.weight_choices


def test_build_setup_honours_no_viz_switch():
    setup = gradio_app.build_setup(beautify=False)
    assert setup.beautify is False                     # infer_pipeline.yaml 的 enabled 被显式关掉
    assert setup.pipeline.viz_enabled is True          # 配置本身没变


def test_numeric_bounds_come_from_pydantic_constraints():
    from od_platform.runtime_config import YOLOInferConfig
    assert gradio_app.numeric_bounds(YOLOInferConfig, "conf") == (0.0, 1.0)
    assert gradio_app.numeric_bounds(YOLOInferConfig, "iou") == (0.0, 1.0)
    lower, upper = gradio_app.numeric_bounds(YOLOInferConfig, "imgsz")
    assert lower is not None and upper is None         # imgsz 只有下界 → 界面用数字输入框


def test_predict_kwargs_follow_service_whitelist():
    from od_platform.runtime_config import YOLOInferConfig
    config, _merger = gradio_app.load_infer_config()
    kwargs = gradio_app.predict_kwargs(config)

    assert set(kwargs) <= set(gradio_app.PREDICT_KEYS) | {"verbose"}
    assert kwargs["verbose"] is False
    assert "classes" not in kwargs                     # None 的参数不进白名单结果
    assert kwargs["conf"] == config.conf and kwargs["imgsz"] == config.imgsz
    assert not any(name in kwargs for name in ("batch", "workers", "cache", "rect"))


# ============================================================================
# 6) 端到端: 假模型跑通 run_demo（读图 → 推理 → 画框 → 表格/汇总）
# ============================================================================
def test_run_demo_with_fake_model_draws_and_summarizes(tmp_path):
    image_path = _write_image(tmp_path)
    boxes = _FakeBoxes([[1, 2, 30, 40]], [0.9], [0.0])
    model = _FakeModel(_FakeResult(boxes))

    gallery, rows, markdown = gradio_app.run_demo(
        [str(image_path)], weight_ref="yolo11n", conf=None, iou=None, imgsz=None,
        device=None, beautify=True, max_images=4, model=model)

    # 标注图 = 原图尺寸的 RGB 数组, 且真的被画过(纯灰底 std=0, 画了框就不为 0)
    assert len(gallery) == 1
    assert gallery[0].shape == (64, 80, 3)
    assert float(np.asarray(gallery[0]).std()) > 0

    assert rows == [[image_path.name, 1, "Mola", 0.9, 1.0, 2.0, 30.0, 40.0]]
    assert "总检出数：1" in markdown and "| Mola | 1 |" in markdown

    # 模型收到的 predict 参数: 白名单 + 来自 infer.yaml 的 conf/...
    images, kwargs = model.calls[0]
    assert len(images) == 1 and images[0].shape == (64, 80, 3)   # 引擎同款"传图片列表"
    assert kwargs["conf"] == gradio_app.load_infer_config()[0].conf
    assert kwargs["verbose"] is False


def test_run_demo_without_viz_uses_yolo_plot(tmp_path):
    image_path = _write_image(tmp_path)
    plotted = np.full((64, 80, 3), 11, dtype=np.uint8)          # plot() 的返回值(BGR)
    model = _FakeModel(_FakeResult(_FakeBoxes([[1, 2, 3, 4]], [0.5], [0.0]), plot_value=plotted))

    gallery, _rows, markdown = gradio_app.run_demo(
        [str(image_path)], weight_ref="yolo11n", conf=None, iou=None, imgsz=None,
        device=None, beautify=False, max_images=4, model=model)

    assert len(gallery) == 1
    assert gallery[0][..., 0].tolist() == np.full((64, 80), 11).tolist()   # BGR→RGB 翻转
    assert "美化=关" in markdown


def test_run_demo_without_files_returns_hint():
    gallery, rows, markdown = gradio_app.run_demo(
        None, weight_ref="yolo11n", conf=None, iou=None, imgsz=None, device=None,
        beautify=True, max_images=4)
    assert (gallery, rows) == ([], [])
    assert "还没有图片" in markdown


def test_run_demo_truncates_to_max_images(tmp_path):
    images = [_write_image(tmp_path, "a.png"), _write_image(tmp_path, "b.png")]
    model = _FakeModel(_FakeResult(_FakeBoxes([[1, 2, 3, 4]], [0.5], [0.0])))

    gallery, rows, markdown = gradio_app.run_demo(
        [str(p) for p in images], weight_ref="yolo11n", conf=None, iou=None, imgsz=None,
        device=None, beautify=True, max_images=1, model=model)

    assert len(gallery) == 1 and len(rows) == 1
    assert "最多处理" in markdown and "`1`" in markdown


def test_run_demo_reports_unreadable_file(tmp_path):
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not a png")
    model = _FakeModel(_FakeResult(_FakeBoxes([], [], [])))

    gallery, rows, markdown = gradio_app.run_demo(
        [str(broken)], weight_ref="yolo11n", conf=None, iou=None, imgsz=None,
        device=None, beautify=True, max_images=4, model=model)

    assert (gallery, rows) == ([], [])
    assert "处理失败" in markdown and "读取图片失败" in markdown


def test_demo_runner_reports_missing_weight(tmp_path):
    """权重名解析不到时, 错误信息里要能看见"搜索目录"(界面不跑去联网下载)。"""
    runner = gradio_app.DemoRunner(
        weight_ref="no-such-weight",
        weight_path=tmp_path / "no-such-weight.pt",
        infer_config=gradio_app.load_infer_config()[0],
        pipeline=gradio_app.build_setup().pipeline,
        beautify=True,
    )
    outcome = runner.run(_write_image(tmp_path, "c.png"))
    assert not outcome.ok
    assert "找不到权重文件" in outcome.error
    assert outcome.detections == []


# ============================================================================
# 7) 界面本体: 装了 gradio 才跑（CI 上自动跳过）
# ============================================================================
def _require_gradio() -> None:
    """装了**且可用**才跑界面本体测试。

    ★ 不用 ``pytest.importorskip("gradio")``: pip 卸载后 site-packages 里可能残留一个空的
      gradio/ 命名空间包, ``import gradio`` 会"成功", importorskip 就不跳过了 ——
      这里复用被测模块自己的探针, 把"半卸载"也一并 skip 掉。
    """
    try:
        gradio_app._import_gradio()
    except gradio_app.GradioMissingError as exc:
        pytest.skip(f"gradio 不可用, 跳过界面本体测试: {exc}")


def test_build_app_smoke():
    """build_app 能构造出 Blocks(不 launch —— 端口绑定留给真机验证)。"""
    _require_gradio()
    app = gradio_app.build_app()
    assert type(app).__name__ == "Blocks"
    assert hasattr(app, "launch")


def test_build_app_accepts_setup_injection():
    _require_gradio()
    setup = gradio_app.build_setup({"server_port": 7998})
    app = gradio_app.build_app(setup=setup)
    assert type(app).__name__ == "Blocks"