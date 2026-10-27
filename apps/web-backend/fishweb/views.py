import json
import mimetypes
import os
import re
from pathlib import Path
from urllib.parse import urlparse

from django.http import FileResponse, HttpResponse, JsonResponse, StreamingHttpResponse
from django.views.decorators.http import require_GET, require_http_methods, require_POST

from . import assistant, catalog, jobs


def error(message, status=400, code="invalid_request"):
    return JsonResponse({"error": str(message), "code": code}, status=status, json_dumps_params={"ensure_ascii": False})


class LocalOnlyMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # 本机回环限制：云端部署（如创空间容器，平台反代后 REMOTE_ADDR 不是 127.0.0.1）
        # 设置环境变量 FISH_PUBLIC=1 即放开；不设置时仍只允许本机预览。
        if os.environ.get("FISH_PUBLIC") != "1" and request.META.get("REMOTE_ADDR", "127.0.0.1") not in {"127.0.0.1", "::1"}:
            return error("此服务仅供本机预览。", 403)
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            if request.headers.get("Sec-Fetch-Site") == "cross-site":
                return error("仅接受同源请求。", 403)
            origin = request.headers.get("Origin")
            referer = request.headers.get("Referer")
            source = origin or referer
            if source and urlparse(source).netloc != request.get_host():
                return error("仅接受同源请求。", 403)
        response = self.get_response(request)
        response["X-Content-Type-Options"] = "nosniff"
        response["Referrer-Policy"] = "same-origin"
        return response


def payload(request):
    try:
        data = json.loads(request.body)
    except (ValueError, UnicodeDecodeError):
        raise ValueError("请求体需要合法JSON。") from None
    if not isinstance(data, dict):
        raise ValueError("请求体需要JSON对象。")
    return data


@require_GET
def status(request):
    return JsonResponse(catalog.status(), json_dumps_params={"ensure_ascii": False})


@require_GET
def dataset(request):
    return JsonResponse(catalog.dataset_info(), json_dumps_params={"ensure_ascii": False})


@require_GET
def experiments(request):
    return JsonResponse(catalog.experiments(), json_dumps_params={"ensure_ascii": False})


@require_http_methods(["GET", "POST"])
def job_list(request):
    if request.method == "GET":
        return JsonResponse({"jobs": jobs.all_jobs()}, json_dumps_params={"ensure_ascii": False})
    try:
        data = payload(request)
        if data.get("type") != "train":
            raise ValueError("此入口仅接受type=train；上传推理请使用/api/inference/。")
        return JsonResponse(jobs.create_training(data), status=202, json_dumps_params={"ensure_ascii": False})
    except ValueError as exc:
        return error(exc)


@require_GET
def job_detail(request, job_id):
    job = jobs.get_job(job_id)
    return JsonResponse(job, json_dumps_params={"ensure_ascii": False}) if job else error("未找到任务。", 404)


@require_POST
def cancel(request, job_id):
    try:
        return JsonResponse(jobs.cancel(job_id), json_dumps_params={"ensure_ascii": False})
    except KeyError:
        return error("未找到任务。", 404)
    except ValueError as exc:
        return error(exc)


@require_POST
def inference(request):
    try:
        upload = request.FILES.get("file")
        if upload is None or len(request.FILES) != 1:
            raise ValueError("请上传一个图片或视频文件，字段名为file。")
        return JsonResponse(jobs.create_inference(upload, request.POST), status=202,
                            json_dumps_params={"ensure_ascii": False})
    except ValueError as exc:
        return error(exc, code="missing_fish_weights" if not request.POST.get("model") else "invalid_request")


@require_POST
def chat(request):
    try:
        data = payload(request)
        return JsonResponse(assistant.answer(data.get("message"), data.get("job_id"), data.get("history")),
                            json_dumps_params={"ensure_ascii": False})
    except ValueError as exc:
        return error(exc)


def ranged_file(request, path, content_type=None):
    size = path.stat().st_size
    content_type = content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    range_header = request.headers.get("Range", "")
    if not range_header:
        response = FileResponse(path.open("rb"), content_type=content_type)
        response["Accept-Ranges"] = "bytes"
        return response
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header)
    if not match or not size:
        return HttpResponse(status=416, headers={"Content-Range": f"bytes */{size}"})
    first, last = match.groups()
    if not first and not last:
        return HttpResponse(status=416, headers={"Content-Range": f"bytes */{size}"})
    start = int(first) if first else max(0, size - int(last))
    end = min(int(last), size - 1) if first and last else size - 1
    if start >= size or start > end:
        return HttpResponse(status=416, headers={"Content-Range": f"bytes */{size}"})

    def pieces():
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = handle.read(min(65536, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    return StreamingHttpResponse(pieces(), status=206, content_type=content_type,
                                 headers={"Accept-Ranges": "bytes", "Content-Range": f"bytes {start}-{end}/{size}",
                                          "Content-Length": str(end - start + 1)})


@require_GET
def job_file(request, job_id, name):
    if name not in {"annotated.mp4", "annotated.jpg", "thumbnail.jpg", "detections.json"}:
        return error("不可访问该文件。", 404)
    job = jobs.get_job(job_id)
    if not job or job.get("status") != "completed" or job.get("type") != "inference":
        return error("该任务没有可用结果。", 404)
    path = jobs.job_directory(job_id) / name
    if not path.is_file() or not catalog.contained(path, catalog.STATE):
        return error("结果文件不存在。", 404)
    return ranged_file(request, path)


@require_GET
def dataset_sample(request, index):
    paths = catalog.sample_paths()
    if not 0 <= index < len(paths):
        return error("样本不存在。", 404)
    return ranged_file(request, paths[index])


@require_GET
def frontend(request, path=""):
    dist = catalog.ROOT / "apps/web-frontend/dist"
    candidate = dist / path
    if path and candidate.is_file() and catalog.contained(candidate, dist):
        return ranged_file(request, candidate)
    index = dist / "index.html"
    if not index.is_file():
        return HttpResponse("前端尚未构建，请在 apps/web-frontend 执行 npm install 和 npm run build。", status=503, content_type="text/plain; charset=utf-8")
    return ranged_file(request, index, "text/html; charset=utf-8")
