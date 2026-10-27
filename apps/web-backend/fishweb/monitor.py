"""Real local resource sampling; no simulated metrics or configuration contents."""
from __future__ import annotations

import copy
import json
import math
import re
import subprocess
import threading
import time
from datetime import datetime, timezone

import psutil
from django.http import JsonResponse
from django.views.decorators.http import require_GET

from . import catalog, jobs

REFRESH_SECONDS = 5
GPU_REFRESH_SECONDS = 60
_LOCK = threading.RLock()
_CACHE = None
_CACHE_TIME = None
_GPU_CACHE = {"available": None, "count": None, "devices": [], "message": "正在检查本地GPU状态。"}
_GPU_ATTEMPT = None
_GPU_PENDING = False

_GPU_PROBE = """import json
try:
    import torch
    available = bool(torch.cuda.is_available())
    devices = []
    if available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            free_memory, total_memory = torch.cuda.mem_get_info(index)
            devices.append({'name': str(properties.name), 'total_memory_bytes': int(total_memory),
                            'used_memory_bytes': int(total_memory-free_memory),
                            'free_memory_bytes': int(free_memory)})
    print(json.dumps({'available': available, 'count': len(devices), 'devices': devices,
                      'message': 'CUDA GPU可用。' if available else '当前引擎没有可用CUDA GPU，可使用CPU。'}))
except Exception:
    print(json.dumps({'available': None, 'count': None, 'devices': [], 'message': 'GPU状态暂不可读。'}))
"""


def _number(value, *, percent=False):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number) or number < 0 or (percent and number > 100):
        return None
    return round(number, 1) if percent else int(number)


def _safe_text(value):
    """Only bounded job evidence is exposed; credential-shaped values are masked."""
    text = str(value or "")[:1500]
    text = re.sub(r"(?i)\bBearer\s+[^\s,;]+", "Bearer [redacted]", text)
    text = re.sub(r"(?i)\b((?:[A-Z_]*API[_-]?KEY|TOKEN|AUTHORIZATION|PASSWORD|SECRET))\s*[:=]\s*[\"']?[^\s,;\"']+",
                  r"\1=[redacted]", text)
    text = re.sub(r"\b(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})",
                  "[redacted]", text)
    return text[:500]


def _read_gpu_state():
    try:
        result = subprocess.run([catalog.ENGINE_PYTHON, "-c", _GPU_PROBE], capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=12,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=False)
        data = json.loads(result.stdout.strip().splitlines()[-1])
        if result.returncode or not isinstance(data, dict) or not isinstance(data.get("devices"), list):
            raise ValueError("Invalid GPU probe result")
        return {"available": data.get("available"), "count": data.get("count"),
                "devices": data["devices"], "message": str(data.get("message", ""))[:150]}
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return {"available": None, "count": None, "devices": [], "message": "GPU状态暂不可读。"}


def _finish_gpu_probe():
    global _GPU_CACHE, _GPU_PENDING
    result = _read_gpu_state()
    with _LOCK:
        _GPU_CACHE = result
        _GPU_PENDING = False


def _gpu_state():
    global _GPU_ATTEMPT, _GPU_PENDING
    now = time.monotonic()
    with _LOCK:
        if not _GPU_PENDING and (_GPU_ATTEMPT is None or now - _GPU_ATTEMPT >= GPU_REFRESH_SECONDS):
            _GPU_ATTEMPT = now
            _GPU_PENDING = True
            threading.Thread(target=_finish_gpu_probe, name="fish-gpu-probe", daemon=True).start()
        return copy.deepcopy(_GPU_CACHE)


def _task_state():
    with jobs.LOCK:
        records = [{"id": item["id"], "type": item.get("type"), "status": item.get("status"),
                    "progress": item.get("progress"), "updated_at": item.get("updated_at"),
                    "error": _safe_text(item.get("error")) or None,
                    "latest_logs": [_safe_text(line) for line in (item.get("logs") or [])[-5:]]}
                   for item in jobs.JOBS.values()]
    counts = {key: 0 for key in ("queued", "running", "completed", "failed", "cancelled", "interrupted")}
    for item in records:
        if item["status"] in counts:
            counts[item["status"]] += 1
    counts["total"] = len(records)
    records.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
    return counts, records[:5]


def _sample():
    warnings = []
    cpu = {"percent": None, "logical_count": None}
    memory = {"used_bytes": None, "total_bytes": None, "percent": None}
    disk = {"used_bytes": None, "total_bytes": None, "free_bytes": None, "percent": None}
    process = {"pid": None, "uptime_seconds": None, "rss_bytes": None}
    runtime = {"engine_available": None, "dataset_available": None, "dataset_images": None,
               "model_candidates": None, "llm_configured": None}
    try:
        # A short real window also works when the development server creates a
        # new request thread; psutil's first interval=None value would be 0.0.
        cpu.update(percent=_number(psutil.cpu_percent(interval=0.1), percent=True),
                   logical_count=_number(psutil.cpu_count(logical=True)))
        if cpu["percent"] is None:
            warnings.append("CPU采样值暂不可用。")
    except (psutil.Error, OSError, ValueError):
        warnings.append("CPU状态暂不可读。")
    try:
        value = psutil.virtual_memory()
        memory.update(used_bytes=_number(value.used), total_bytes=_number(value.total),
                      percent=_number(value.percent, percent=True))
    except (psutil.Error, OSError, ValueError):
        warnings.append("内存状态暂不可读。")
    try:
        value = psutil.disk_usage(str(catalog.ROOT))
        disk.update(used_bytes=_number(value.used), total_bytes=_number(value.total),
                    free_bytes=_number(value.free), percent=_number(value.percent, percent=True))
    except (psutil.Error, OSError, ValueError):
        warnings.append("项目所在磁盘状态暂不可读。")
    try:
        current = psutil.Process()
        process.update(pid=current.pid, uptime_seconds=_number(time.time() - current.create_time()),
                       rss_bytes=_number(current.memory_info().rss))
    except (psutil.Error, OSError, ValueError):
        warnings.append("Web进程状态暂不可读。")
    try:
        dataset = catalog.dataset_info()
        runtime.update(engine_available=catalog.engine_available(catalog.ENGINE_PYTHON),
                       dataset_available=dataset["local"]["available"], dataset_images=dataset["local"]["total"],
                       model_candidates=len(catalog.model_catalog()), llm_configured=catalog.llm_configured())
    except (OSError, ValueError, KeyError, TypeError):
        warnings.append("数据或引擎状态暂不可读，请刷新工作空间。")
    counts, recent = _task_state()
    return {"sampled_at": datetime.now(timezone.utc).isoformat(), "refresh_seconds": REFRESH_SECONDS,
            "source": "local_system", "cpu": cpu, "memory": memory, "disk": disk, "process": process,
            "gpu": _gpu_state(), "jobs": counts, "runtime": runtime, "recent_jobs": recent, "warnings": warnings}


def snapshot():
    global _CACHE, _CACHE_TIME
    now = time.monotonic()
    with _LOCK:
        if _CACHE is None or _CACHE_TIME is None or now - _CACHE_TIME >= REFRESH_SECONDS:
            _CACHE = _sample()
            _CACHE_TIME = time.monotonic()
        return copy.deepcopy(_CACHE)


@require_GET
def monitor(request):
    return JsonResponse(snapshot(), json_dumps_params={"ensure_ascii": False})
