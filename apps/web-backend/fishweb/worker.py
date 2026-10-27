"""Isolated engine process. No browser-visible credentials or arbitrary shell execution."""
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parents[1]
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(ROOT / "apps/platform/src"))


def aggregate_frames(frames):
    counts = Counter()
    confidence_sum = Counter()
    peaks = Counter()
    total = 0
    for frame in frames:
        concurrent = Counter()
        for detection in frame["detections"]:
            label = detection["label"]
            counts[label] += 1
            concurrent[label] += 1
            confidence_sum[label] += detection["confidence"]
            total += 1
        for label, number in concurrent.items():
            peaks[label] = max(peaks[label], number)
    return {"total_detections": total, "species_count": len(counts),
            "class_counts": [{"label": label, "count": count,
                              "mean_confidence": round(confidence_sum[label] / count, 4),
                              "peak_per_frame": peaks[label]} for label, count in counts.most_common()],
            "counting_note": "视频统计为抽样帧中的检出框次，同一条鱼可能被重复计数；没有目标跟踪，不能视为独立鱼的数量。"}


def validate_fish_labels(names, expected):
    actual = list(names.values()) if isinstance(names, dict) else list(names)
    if actual != list(expected):
        raise ValueError("权重标签与fish配置中的33个品种不一致。请使用本项目训练的鱼类检测权重；通用COCO权重不能替代。")


def detection_rows(result):
    rows = []
    if result.boxes is None:
        return rows
    for box in result.boxes:
        class_id = int(box.cls.item())
        rows.append({"class_id": class_id, "label": result.names[class_id],
                     "confidence": round(float(box.conf.item()), 5),
                     "xyxy": [round(float(value), 2) for value in box.xyxy[0].tolist()]})
    return rows


def ffmpeg_binary():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def infer(config, output):
    import cv2
    import numpy as np
    from ultralytics import YOLO
    from fishweb.catalog import class_names, resolve_model

    weight = resolve_model(config["model"])
    model = YOLO(str(weight))
    validate_fish_labels(model.names, class_names())
    source = Path(config["source"])
    is_video = source.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}
    started = time.monotonic()
    kwargs = {key: config[key] for key in ("conf", "iou", "imgsz", "device")}
    kwargs.update(verbose=False, max_det=300)
    if not kwargs["device"]:
        kwargs.pop("device")
    frames = []
    notes = []
    fps = 0
    frame_total = 1
    frame_index = 0
    writer = None
    capture = None
    annotated = "annotated.jpg"
    try:
        if is_video:
            capture = cv2.VideoCapture(str(source))
            if not capture.isOpened():
                raise ValueError("视频无法解码，请转换为常见MP4后重试。")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if not 0 < fps <= 240:
                raise ValueError("视频帧率无效。")
            frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if frame_total <= 0 or frame_total > 18000 or width * height > 4096 * 2160:
                raise ValueError("视频最多18000帧，分辨率不超过4096×2160，请裁剪压缩后重试。")
            # Every output frame corresponds to one sampled input frame; adjust FPS to preserve time.
            writer = cv2.VideoWriter(str(output / "annotated_raw.avi"), cv2.VideoWriter_fourcc(*"MJPG"),
                                     fps / config["stride"], (width, height))
            if not writer.isOpened():
                raise RuntimeError("本机无法创建视频编码器。")
        while True:
            if is_video:
                ok, frame = capture.read()
                if not ok:
                    break
            else:
                frame = cv2.imdecode(np.fromfile(source, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError("图片无法解码；扩展名必须与实际文件内容相符。")
                if frame.shape[0] * frame.shape[1] > 4096 * 4096:
                    raise ValueError("图片分辨率过大，请缩小到4096×4096以内。")
            if frame_index % config["stride"] == 0:
                result = model.predict(frame, **kwargs)[0]
                rows = detection_rows(result)
                frames.append({"frame": frame_index, "time_seconds": round(frame_index / fps, 3) if fps else 0,
                               "detections": rows})
                rendered = result.plot()
                if len(frames) == 1:
                    cv2.imencode(".jpg", rendered)[1].tofile(output / "thumbnail.jpg")
                if writer:
                    writer.write(rendered)
                else:
                    cv2.imencode(".jpg", rendered)[1].tofile(output / annotated)
                if len(frames) > 6000:
                    raise ValueError("最多分析6000个抽样帧，请增大抽帧间隔或裁剪视频。")
                print("FISH_PROGRESS", 2 + 85 * (frame_index + 1) / max(frame_total, 1), flush=True)
            frame_index += 1
            if not is_video:
                break
    finally:
        if capture:
            capture.release()
        if writer:
            writer.release()
    if not frames:
        raise ValueError("未能从视频读取有效帧。")
    if is_video:
        print("FISH_PROGRESS 90", flush=True)
        annotated = "annotated.mp4"
        subprocess.run([ffmpeg_binary(), "-y", "-i", str(output / "annotated_raw.avi"), "-an",
                        "-c:v", "libx264", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        str(output / annotated)], capture_output=True, check=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        (output / "annotated_raw.avi").unlink(missing_ok=True)
        notes.append("输出为无声H.264视频；按抽帧间隔调整播放帧率，时间尺度保持一致。")
    summary = aggregate_frames(frames)
    if not is_video:
        summary["counting_note"] = "图片统计为该单帧中的检测框数量；框可能存在误检或漏检，请结合原图复核。"
    summary.update({"frames_analyzed": len(frames), "frames_read": frame_index, "source_frames": frame_total,
                    "duration_seconds": round(frame_index / fps, 3) if fps else 0,
                    "processing_seconds": round(time.monotonic() - started, 2), "stride": config["stride"],
                    "model": config["model"], "conf": config["conf"], "iou": config["iou"],
                    "media_type": "video" if is_video else "image", "annotated": annotated,
                    "thumbnail": "thumbnail.jpg", "notes": notes, "source": "local_yolo",
                    "preview_detections": [{"frame": f["frame"], "time_seconds": f["time_seconds"], **d}
                                           for f in frames[:30] for d in f["detections"]][:100]})
    (output / "detections.json").write_text(json.dumps({"summary": summary, "frames": frames}, ensure_ascii=False), encoding="utf-8")
    return summary


def train(config, output):
    import yaml
    from fishweb.catalog import dataset_yaml_path, resolve_model

    # The platform CLI parses --batch as float, while its configuration treats
    # floats as GPU fractions. Preserve the complete runtime baseline and carry
    # the Web's integer batch through a private YAML instead of that CLI option.
    baseline = ROOT / "apps/platform/configs/runtime/train.yaml"
    runtime = yaml.safe_load(baseline.read_text(encoding="utf-8"))
    if not isinstance(runtime, dict):
        raise ValueError("平台训练配置必须为 YAML 对象。")
    runtime["batch"] = config["batch"]
    output.mkdir(parents=True, exist_ok=True)
    private_config = (output / "train.runtime.yaml").resolve()
    private_config.write_text(yaml.safe_dump(runtime, allow_unicode=True, sort_keys=False), encoding="utf-8")
    command = [sys.executable, "-m", "od_platform.cli.train_model", "--config", str(private_config),
               "--data", str(dataset_yaml_path()),
               "--model", str(resolve_model(config["model"])), "--epochs", str(config["epochs"]),
               "--imgsz", str(config["imgsz"]),
               "--experiment-name", "fish-web-" + output.name[:8]]
    if config["device"]:
        command += ["--device", config["device"]]
    print("调用现有 odp-train，保留平台的数据身份核验、审计和归档。", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    return {"message": "训练完成，指标读取 runs/training 下真实产物，归档权重在 models/trained。",
            "source": "od_platform.cli.train_model"}


if __name__ == "__main__":
    try:
        from fishweb.env import load_local_env

        load_local_env()
        config_path = Path(sys.argv[2])
        params = json.loads(config_path.read_text(encoding="utf-8"))
        result = (infer if sys.argv[1] == "inference" else train)(params, config_path.parent)
        (config_path.parent / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", flush=True)
        sys.exit(1)
