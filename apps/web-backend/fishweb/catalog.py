"""Read repository facts, keeping documented results distinct from local measurements."""
import csv
import importlib.util
import json
import os
import subprocess
import sys
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


def dataset_config():
    return yaml.safe_load(DATASET_YAML.read_text(encoding="utf-8"))


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


def dataset_info():
    config = dataset_config()
    names = class_names()
    counts = dict(config["odp_meta"]["split"]["counts"])
    counts.update(total=sum(counts.values()), instances=6034, classes=len(names))
    local_counts = {}
    distributions = [0] * len(names)
    images = []
    root = ROOT / config["path"]
    for split in ("train", "val", "test"):
        directory = root / "images" / split
        files = [p for p in directory.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} and contained(p, root)]
        local_counts[split] = len(files)
        images.extend(files[:6])
        for label in (root / "labels" / split).glob("*.txt"):
            if not contained(label, root):
                continue
            try:
                for line in label.read_text(encoding="utf-8").splitlines():
                    fields = line.split()
                    if len(fields) == 5 and fields[0].isdigit() and int(fields[0]) < len(names):
                        distributions[int(fields[0])] += 1
            except OSError:
                pass
    has_data = all(local_counts.get(k, 0) > 0 for k in ("train", "val", "test"))
    return {"name": "BengalDeltaFish", "dataset_id": "fish", "classes": [
        {"id": i, "name": n, "instances": distributions[i] if has_data else None}
        for i, n in enumerate(names)], "counts": counts, "split": config["odp_meta"]["split"],
        "fingerprint": config["odp_meta"]["contract_fingerprint"],
        "license": "CC BY 4.0", "format": "YOLO", "provenance": "repository_documentation",
        "source": "apps/platform/configs/datasets/fish.yaml + README.md",
        "local": {"available": has_data, "counts": local_counts, "total": sum(local_counts.values()),
                  "message": "已找到本地划分" if has_data else "仓库不包含数据文件；需放入 data/processed/fish 并保留原有划分和血缘章。"},
        "validation": {"documented": "15/15 PASS", "measured_locally": False},
        "samples": [{"name": p.name, "url": "/api/dataset/sample/" + str(i) + "/"} for i, p in enumerate(images)],
    }


def sample_paths():
    paths = []
    root = ROOT / dataset_config()["path"]
    for split in ("train", "val", "test"):
        paths.extend([p for p in (root / "images" / split).glob("*")
                      if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} and contained(p, root)][:6])
    return paths


def experiments():
    found = []
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
            found.append({"id": results.parent.name, "kind": "training", "curves": rows,
                          "metrics": rows[-1] if rows else {}, "source": str(results.relative_to(ROOT)),
                          "measured_locally": True})
        except (OSError, ValueError):
            continue
    return {"experiments": found, "documented_metrics": DOCUMENTED_METRICS,
            "source": "local_reports" if found else "repository_documentation"}


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
    return {"project": "FishScope · 淡水鱼智能检测平台", "dataset": "fish", "models": model_catalog(),
            "engine": {"available": engine, "python": ENGINE_PYTHON,
                       "message": "本地YOLO环境就绪" if engine else "需安装平台核心依赖后运行推理"},
            "llm": {"configured": llm_configured(), "mode": "llm" if llm_configured() else "local",
                    "model": os.environ.get("FISH_LLM_MODEL", "") if llm_configured() else "本地规则解释",
                    "disclosure": "仅发送检测统计与问题给配置的大模型服务，不发送视频原文件。"},
            "gradio": {"available": importlib.util.find_spec("gradio") is not None,
                       "url": "http://127.0.0.1:7860", "command": "odp-demo"},
            "local_only": True}
