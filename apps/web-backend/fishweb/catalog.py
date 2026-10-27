"""Read repository facts, keeping documented results distinct from local measurements."""
import csv
import importlib.util
import json
import os
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
STATE = ROOT / "runs/web"
MODEL_ROOT = ROOT / "models"
DATASET_YAML = ROOT / "apps/platform/configs/datasets/fish.yaml"
ENGINE_PYTHON = os.environ.get("FISH_ENGINE_PYTHON", sys.executable)
DOCUMENTED_METRICS = {
    "map50": 0.9073, "map50_95": 0.7442, "precision": 0.8268, "recall": 0.8874,
    "recommended_conf": 0.4, "f1_at_recommended_conf": 0.8608,
    "source": "README.md · 评估指标说明 · 真实 fish 评估结果",
    "source_url": "https://github.com/chenkangkang999/-ODPlatform2#评估指标说明",
    "provenance": "repository_documentation", "measured_locally": False,
    "split": "val", "images": 458, "instances": 661,
    "model": "YOLO11n", "epochs": 10, "imgsz": 512, "batch": 8,
    "note": "仓库作者记录的基线；本机尚未复测。目标精确率0.90未达到，0.4为F1最优参考阈值。",
}


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return default


def local_integration():
    """Use only the local importer completion marker, never a browser-supplied path."""
    path = ROOT / "runs/web/dataset/integration.json"
    doc = read_json(path, {})
    private_yaml = ROOT / "runs/web/dataset/fish.yaml"
    if (not isinstance(doc, dict) or doc.get("identity_ready") is not True
            or doc.get("dataset_yaml") != "runs/web/dataset/fish.yaml"
            or private_yaml.is_symlink() or not private_yaml.is_file()
            or not contained(private_yaml, ROOT)):
        return {}
    return doc


def dataset_yaml_path():
    return ROOT / "runs/web/dataset/fish.yaml" if local_integration() else DATASET_YAML


def dataset_config():
    return yaml.safe_load(dataset_yaml_path().read_text(encoding="utf-8"))


def class_names():
    names = dataset_config()["names"]
    return list(names.values()) if isinstance(names, dict) else names


def contained(path, base):
    try:
        return Path(path).resolve().is_relative_to(Path(base).resolve())
    except (OSError, ValueError):
        return False


def model_catalog():
    result = []
    if MODEL_ROOT.exists():
        for path in sorted(MODEL_ROOT.rglob("*.pt")):
            if not contained(path, MODEL_ROOT) or not path.is_file():
                continue
            relative = path.relative_to(MODEL_ROOT).as_posix()
            result.append({"id": relative, "name": path.name, "path": "models/" + relative,
                           "kind": "fish" if "fish" in path.name.lower() or relative.startswith("trained/") else "other",
                           "bytes": path.stat().st_size,
                           "verification": "推理前会核验模型的33类标签与fish配置完全一致"})
    return result


def resolve_model(model_id):
    if not isinstance(model_id, str) or model_id not in {m["id"] for m in model_catalog()}:
        raise ValueError("请选择 models 目录下已有的本地 .pt 权重；不接受上传权重或外部路径。")
    path = (MODEL_ROOT / model_id).resolve()
    if not contained(path, MODEL_ROOT):
        raise ValueError("权重路径超出 models 目录。")
    return path


def model_configuration(models=None):
    models = model_catalog() if models is None else models
    try:
        config = yaml.safe_load((ROOT / "apps/platform/configs/runtime/demo.yaml").read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        config = {}
    weight = str(config.get("weight") or "").strip()
    filename = Path(weight).name
    if filename and not filename.lower().endswith(".pt"):
        filename += ".pt"
    available = bool(filename and any(model["name"] == filename for model in models))
    message = ("已找到项目配置的本地鱼类权重，可上传素材进行识别。" if available else
               "项目已配置鱼类模型，但本机尚缺对应的 .pt 文件：" + filename + "。") if filename else "请接入本项目训练的33类鱼类 .pt 权重。"
    return {"configured_weight": weight, "expected_path": "models/trained/" + filename if filename else "",
            "file_available": available, "fish_candidates": sum(model["kind"] == "fish" for model in models),
            "message": message}


def _mtime(path):
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def _dataset_root(config):
    root = (ROOT / config["path"]).resolve()
    # Web only serves/scans the project's local data, not arbitrary YAML paths.
    return root if contained(root, ROOT / "data") else None


def _direct_files(directory, suffixes):
    if directory.is_symlink() or not directory.is_dir() or not contained(directory, ROOT / "data"):
        return []
    return sorted((p for p in directory.iterdir()
                   if p.suffix.lower() in suffixes and not p.is_symlink() and p.is_file()), key=lambda p: p.name)


@lru_cache(maxsize=4)
def _scan_dataset(root_string, class_count, cache_key):
    """Directory stamps catch new splits; the time bucket refreshes label edits."""
    root = Path(root_string) if root_string else None
    local_counts = {}
    distributions = [0] * class_count
    images = []
    for split in ("train", "val", "test"):
        files = _direct_files(root / "images" / split, {".jpg", ".jpeg", ".png", ".webp"}) if root else []
        local_counts[split] = len(files)
        images.extend(files[:6])
        labels = _direct_files(root / "labels" / split, {".txt"}) if root else []
        for label in labels:
            try:
                for line in label.read_text(encoding="utf-8").splitlines():
                    fields = line.split()
                    if len(fields) == 5 and fields[0].isdigit() and int(fields[0]) < class_count:
                        distributions[int(fields[0])] += 1
            except (OSError, UnicodeError):
                pass
    return local_counts, distributions, images


def _local_scan(config, class_count):
    root = _dataset_root(config)
    directories = [root / kind / split for kind in ("images", "labels") for split in ("train", "val", "test")] if root else []
    stamps = tuple(_mtime(path) for path in directories)
    key = (str(dataset_yaml_path()), _mtime(dataset_yaml_path()), stamps, int(time.monotonic() // 30))
    return _scan_dataset(str(root) if root else "", class_count, key)


def local_validation():
    integration = local_integration()
    validation = {"documented": "15/15 PASS", "measured_locally": False}
    relative = integration.get("validation_report")
    if not isinstance(relative, str):
        return validation
    report_path = ROOT / relative
    if not contained(report_path, ROOT / "runs/data_validation") or report_path.is_symlink():
        return validation
    report = read_json(report_path, {})
    if (not isinstance(report, dict) or report.get("overall_severity") != "PASS"
            or Path(str(report.get("yaml_path", ""))).resolve() != dataset_yaml_path().resolve()):
        return validation
    results = report.get("results", [])
    checks = sum(isinstance(row, dict) and row.get("severity") == "PASS" for row in results)
    if results and checks == len(results):
        validation.update(measured_locally=True, overall="PASS", checks_passed=checks,
                          checks_total=len(results), source=relative, created_at=integration.get("created_at"))
    return validation


def dataset_info():
    config = dataset_config()
    names = class_names()
    counts = dict(config["odp_meta"]["split"]["counts"])
    counts.update(total=sum(counts.values()), instances=6034, classes=len(names))
    local_counts, distributions, images = _local_scan(config, len(names))
    has_data = all(local_counts.get(k, 0) > 0 for k in ("train", "val", "test"))
    return {"name": "BengalDeltaFish", "dataset_id": "fish", "classes": [
        {"id": i, "name": n, "instances": distributions[i] if has_data else None}
        for i, n in enumerate(names)], "counts": counts, "split": config["odp_meta"]["split"],
        "fingerprint": config["odp_meta"]["contract_fingerprint"],
        "license": "CC BY 4.0", "format": "YOLO", "provenance": "repository_documentation",
        "source": "apps/platform/configs/datasets/fish.yaml + README.md",
        "local": {"available": has_data, "counts": local_counts, "total": sum(local_counts.values()),
                  "instances": sum(distributions), "dataset_yaml": dataset_yaml_path().relative_to(ROOT).as_posix() if contained(dataset_yaml_path(), ROOT) else None,
                  "message": "已接入本地鱼类数据" if has_data else "仓库不包含数据文件；需放入 data/processed/fish 并保留原有划分和血缘章。"},
        "validation": local_validation() if has_data else {"documented": "15/15 PASS", "measured_locally": False},
        "samples": [{"name": p.name, "url": "/api/dataset/sample/" + str(i) + "/"} for i, p in enumerate(images)],
    }


def sample_paths():
    return _local_scan(dataset_config(), len(class_names()))[2]


def training_summary():
    """Portable, sanitized evidence from the supplied archive, never a new local run."""
    summary = read_json(ROOT / "docs/fish_training_run.json")
    if (not isinstance(summary, dict) or summary.get("schema_version") != 1
            or summary.get("provenance") != "provided_training_archive"
            or summary.get("measured_locally") is not False
            or not isinstance(summary.get("id"), str)
            or not isinstance(summary.get("metrics"), dict)
            or not isinstance(summary.get("curves"), list)):
        return None
    return summary


def experiments():
    found = []
    provided = training_summary()
    for report in sorted((ROOT / "runs/detect_eval").glob("*/report.json"), reverse=True)[:30]:
        doc = read_json(report, {})
        if not isinstance(doc, dict) or not doc:
            continue
        # The platform evaluates a materialized _dataset_view/data.yaml. Its
        # original dataset identity is retained under dataset.source_yaml.
        original_yaml = (doc.get("dataset") or {}).get("source_yaml", doc.get("data_yaml", ""))
        if str(original_yaml).replace("\\", "/").rsplit("/", 1)[-1].lower() != "fish.yaml":
            continue
        found.append({"id": report.parent.name, "kind": "evaluation", "metrics": doc.get("overall", {}),
                      "per_class": doc.get("per_class", []), "curves": doc.get("pr_curves", []),
                      "conf_sweep": doc.get("conf_sweep", []), "warnings": doc.get("warnings", []),
                      "source": str(report.relative_to(ROOT)), "measured_locally": True})
    for results in sorted((ROOT / "runs/training").glob("*/results.csv"), reverse=True)[:30]:
        try:
            with results.open(encoding="utf-8-sig") as handle:
                rows = [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(handle)]
            audit = read_json(results.parent / "odp_audit.json", {})
            if "fish" not in json.dumps(audit, ensure_ascii=False).lower():
                continue
            imported = provided and results.parent.name == provided["id"]
            found.append({"id": results.parent.name, "kind": "training", "curves": rows,
                          "metrics": provided["metrics"] if imported else (rows[-1] if rows else {}),
                          "source": str(results.relative_to(ROOT)),
                          "provenance": "provided_training_archive" if imported else "local_training",
                          "source_label": provided["source_label"] if imported else "本机训练记录",
                          "measured_locally": not bool(imported)})
        except (OSError, ValueError):
            continue
    if provided and not any(row["kind"] == "training" and row["id"] == provided["id"] for row in found):
        # New clones retain the real history without committing runs or model binaries.
        found.append({"id": provided["id"], "kind": "training", "curves": provided["curves"],
                      "metrics": provided["metrics"], "source": "docs/fish_training_run.json",
                      "provenance": provided["provenance"], "source_label": provided["source_label"],
                      "measured_locally": False})
    return {"experiments": found, "documented_metrics": DOCUMENTED_METRICS,
            "training_summary": provided,
            "source": "local_reports" if any(row["measured_locally"] for row in found) else
                      ("provided_training_archive" if provided else "repository_documentation")}


def llm_configured():
    return all(os.environ.get(key, "").strip() for key in ("FISH_LLM_BASE_URL", "FISH_LLM_API_KEY", "FISH_LLM_MODEL"))


@lru_cache(maxsize=2)
def engine_available(python):
    if Path(python).resolve() == Path(sys.executable).resolve():
        return all(importlib.util.find_spec(name) is not None for name in ("ultralytics", "torch", "cv2"))
    try:
        probe = subprocess.run([python, "-c", "import importlib.util; print(all(importlib.util.find_spec(n) is not None for n in ('ultralytics','torch','cv2')))"],
                               capture_output=True, text=True, timeout=10, check=False,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return probe.returncode == 0 and probe.stdout.strip() == "True"
    except (OSError, subprocess.TimeoutExpired):
        return False


def status():
    engine = engine_available(ENGINE_PYTHON)
    models = model_catalog()
    return {"project": "FishScope · 淡水鱼智能检测平台", "dataset": "fish", "models": models,
            "model_configuration": model_configuration(models),
            "engine": {"available": engine, "python": ENGINE_PYTHON,
                       "message": "本地YOLO环境就绪" if engine else "需安装平台核心依赖后运行推理"},
            "llm": {"configured": llm_configured(), "mode": "llm" if llm_configured() else "local",
                    "model": os.environ.get("FISH_LLM_MODEL", "") if llm_configured() else "本地规则解释",
                    "disclosure": "仅发送检测统计与问题给配置的大模型服务，不发送视频原文件。"},
            "gradio": {"available": importlib.util.find_spec("gradio") is not None,
                       "url": "http://127.0.0.1:7860", "command": "odp-demo"},
            "local_only": True}
