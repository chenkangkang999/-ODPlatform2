"""Bounded local background jobs; durable summaries and explicit cancellation."""
import json
import os
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from .catalog import ENGINE_PYTHON, ROOT, STATE, class_names, dataset_info, read_json, resolve_model

POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fish-job")
LOCK = threading.RLock()
JOBS = {}
PROCESSES = {}
CANCELLED = set()
MAX_UPLOAD_BYTES = 250 * 1024 * 1024
ACTIVE = {"queued", "running"}


def now():
    return datetime.now(timezone.utc).isoformat()


def job_directory(job_id):
    return STATE / job_id


def persist(job):
    directory = job_directory(job["id"])
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "job.tmp"
    temporary.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(directory / "job.json")


def load_jobs():
    with LOCK:
        for file in STATE.glob("*/job.json"):
            job = read_json(file)
            if not isinstance(job, dict) or file.parent.name != job.get("id"):
                continue
            if job.get("status") in ACTIVE:
                job.update(status="interrupted", error="服务重新启动，任务已中断；请重新提交。", updated_at=now())
                persist(job)
            JOBS[job["id"]] = job


load_jobs()


def update(job_id, **fields):
    with LOCK:
        job = JOBS[job_id]
        job.update(fields, updated_at=now())
        persist(job)


def log(job_id, line):
    with LOCK:
        lines = JOBS[job_id]["logs"]
        lines.append(str(line).strip())
        JOBS[job_id]["logs"] = lines[-150:]
        persist(JOBS[job_id])


def get_job(job_id):
    with LOCK:
        job = JOBS.get(job_id)
        return json.loads(json.dumps(job)) if job else None


def all_jobs():
    with LOCK:
        return sorted([json.loads(json.dumps(j)) for j in JOBS.values()], key=lambda j: j["created_at"], reverse=True)[:50]


def new_job(kind, params):
    with LOCK:
        if any(job["status"] in ACTIVE for job in JOBS.values()):
            raise ValueError("已有任务正在处理，请等待完成或取消后再提交。")
        job_id = uuid.uuid4().hex
        job = {"id": job_id, "type": kind, "status": "queued", "progress": 0,
               "created_at": now(), "updated_at": now(), "params": params,
               "logs": [], "error": None, "result": None}
        JOBS[job_id] = job
        persist(job)
    return job


def numeric(params, key, default, lower, upper, integer=False):
    try:
        value = float(params.get(key, default))
        if not lower <= value <= upper or (integer and not value.is_integer()):
            raise ValueError()
        return int(value) if integer else value
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{key} 必须在 {lower} 到 {upper} 之间。") from None


def device_value(params):
    value = str(params.get("device", "cpu")).strip()
    if value not in {"cpu", "0", "auto"}:
        raise ValueError("device 仅支持 cpu、0 或 auto。")
    return "" if value == "auto" else value


def inference_params(params):
    model_id = str(params.get("model", ""))
    if not model_id:
        raise ValueError("尚未选择鱼类权重。请把同学训练的33类 fish .pt 权重放到 models/trained，再刷新模型列表。")
    resolve_model(model_id)
    result = {"model": model_id, "conf": numeric(params, "conf", .4, .01, .99),
              "iou": numeric(params, "iou", .7, .1, .95),
              "imgsz": numeric(params, "imgsz", 640, 320, 1280, True),
              "stride": numeric(params, "stride", 1, 1, 30, True), "device": device_value(params)}
    if result["imgsz"] % 32:
        raise ValueError("imgsz 必须是32的倍数。")
    return result


def create_inference(upload, params):
    params = inference_params(params)
    suffix = Path(upload.name).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        raise ValueError("请上传 JPG、PNG、WEBP 图片或 MP4、MOV、AVI、MKV、WEBM 视频。")
    if not upload.size or upload.size > MAX_UPLOAD_BYTES:
        raise ValueError("文件大小必须大于0且不超过250 MB。")
    job = new_job("inference", params)
    job_id = job["id"]
    source = job_directory(job_id) / ("input" + suffix)
    try:
        written = 0
        with source.open("wb") as handle:
            for chunk in upload.chunks():
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise ValueError("文件超过250 MB限制。")
                handle.write(chunk)
        update(job_id, filename=Path(upload.name).name)
        POOL.submit(run_worker, job_id, "inference", {**params, "source": str(source)})
    except Exception:
        update(job_id, status="failed", error="文件保存失败。")
        source.unlink(missing_ok=True)
        raise
    return get_job(job_id)


def create_training(params):
    if not dataset_info()["local"]["available"]:
        raise ValueError("本机缺少 fish 训练/验证/测试数据，不能启动训练。请先接入冻结数据集及血缘文件。")
    model = resolve_model(params.get("model", ""))
    parsed = {"model": model.relative_to(ROOT / "models").as_posix(),
              "epochs": numeric(params, "epochs", 10, 1, 300, True),
              "batch": numeric(params, "batch", 8, 1, 64, True),
              "imgsz": numeric(params, "imgsz", 640, 320, 1280, True), "device": device_value(params)}
    if parsed["imgsz"] % 32:
        raise ValueError("imgsz 必须是32的倍数。")
    job = new_job("train", parsed)
    POOL.submit(run_worker, job["id"], "train", parsed)
    return get_job(job["id"])


def run_worker(job_id, kind, params):
    with LOCK:
        if job_id in CANCELLED:
            return
    update(job_id, status="running", progress=1)
    directory = job_directory(job_id)
    config = directory / "worker.json"
    config.write_text(json.dumps(params, ensure_ascii=False), encoding="utf-8")
    command = [ENGINE_PYTHON, str(Path(__file__).with_name("worker.py")), kind, str(config)]
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(ROOT / "apps/platform/src") + os.pathsep + env.get("PYTHONPATH", "")
    try:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        with LOCK:
            PROCESSES[job_id] = process
            cancel_requested = job_id in CANCELLED
        if cancel_requested:
            terminate_process(process)
        for line in process.stdout:
            line = line.rstrip()
            if line.startswith("FISH_PROGRESS "):
                try:
                    update(job_id, progress=min(99, max(1, float(line.split()[1]))))
                except (ValueError, IndexError):
                    pass
            else:
                log(job_id, line[-2000:])
        code = process.wait()
        if job_id in CANCELLED:
            update(job_id, status="cancelled", error="用户已取消任务。")
        elif code:
            update(job_id, status="failed", error="任务执行失败，请查看日志；不会生成虚构检测结果。")
        else:
            result = read_json(directory / "result.json", {})
            if kind == "inference":
                result["detections_url"] = f"/api/jobs/{job_id}/files/detections.json"
                for key in ("annotated", "thumbnail"):
                    if result.get(key):
                        result[key + "_url"] = f"/api/jobs/{job_id}/files/{result[key]}"
            update(job_id, status="completed", progress=100, result=result)
    except Exception as exc:
        log(job_id, f"{type(exc).__name__}: {exc}")
        update(job_id, status="failed", error="本地运行环境无法启动任务，请查看日志。")
    finally:
        with LOCK:
            PROCESSES.pop(job_id, None)


def terminate_process(process):
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=False)
    else:
        process.terminate()


def cancel(job_id):
    with LOCK:
        if job_id not in JOBS:
            raise KeyError(job_id)
        if JOBS[job_id]["status"] not in ACTIVE:
            raise ValueError("该任务已经结束。")
        CANCELLED.add(job_id)
        process = PROCESSES.get(job_id)
        update(job_id, status="cancelled", error="用户已取消任务。")
    if process:
        terminate_process(process)
    return get_job(job_id)
