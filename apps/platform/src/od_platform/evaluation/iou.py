#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : iou.py
# @Project   : ODPlatform
# @Function  : 基于匹配的 IoU 分析（官方 mAP 口径之外的补充口径）

"""显式的 **IoU 指标** —— 把"框定得有多准"这件事单独说清楚。

★ 为什么要在 mAP 之外再算一遍 IoU

    :mod:`od_platform.evaluation.metrics` 给的三件套是 ultralytics 的官方口径:

        · ``map50``     —— 只在 IoU=0.5 这一个门限上算 AP;
        · ``map75``     —— 只在 IoU=0.75 这个门限上算 AP;
        · ``map50_95``  —— 在 IoU=0.5:0.05:0.95 共 10 个门限上取平均。

    这三个数**确实**是"沿 IoU 轴看"的, 但它们回答的是"能不能把目标找到",
    回答不了"框贴得有多紧"。两件事在实际项目里要分开说:

        · 框的位置偏半个身位 —— mAP50 可能仍然很高（0.5 门限很宽松）,
          但下游的测距/跟踪/裁剪都会受影响;
        · 因此需要一个**直接看匹配框 IoU 数值分布**的口径。

★ 本模块口径（必须与官方口径分清, 报告里也分开写）

    1. 预测框来自 ``model.predict(conf=settings.conf, iou=settings.iou, ...)``
       —— 阈值一律取自配置, 不在本模块里另写死一个数, 这样"这次按什么阈值算的"
       永远能在报告的 ``settings`` 与 ``iou_analysis.conf`` 里查到。
       ★ 但要小心一个真实的坑: ``eval.yaml`` 的默认 ``conf`` 是 **0.001**（那是给
         val 用的极低阈值, 为了让 P-R 曲线完整）。用它跑匹配, NMS 后会吐出**几百个
         低置信度框**, 于是"未匹配的预测(FP)"暴涨, 匹配口径的 P 会低到没有意义
         （真跑 synth 时 FP=783、P=0.0101）。**IoU 分布本身不受影响**（配上的还是那
         几个高分框）, 但要让匹配口径的 P/R 有意义, 请把 ``--conf`` 提到 conf 扫描
         推荐的那个阈值再跑一遍。本模块会在 ``note`` 里把这件事写明白。
    2. **同类**（``cls`` 相同）的预测与真值两两求 IoU, 按 IoU **降序贪心一对一**
       匹配 —— 一个预测只能配一个真值, 一个真值也只能被配一次。
       这就是常见的"匹配口径"TP/FP/FN。
    3. 只有 ``IoU >= IOU_MATCH_THRESHOLD``(0.5) 的候选才算匹配得上;
       没配上的预测算 FP, 没配上的真值算 FN。
    4. 在**匹配对**上统计 IoU 的 平均 / 中位数 / 最小 / 最大, 以及 0.50–1.00 的
       五档直方图 —— 这就是"框定得多准"的直接读数。

    ★ 匹配口径的 P/R/F1 **不等于**官方 mAP 口径的 P/R。前者是"单个 IoU 门限上的
      一张点估计", 后者是对 P-R 曲线积分。两者都报, 但绝不相加、不相除。

★ 本模块的可测性

    :func:`box_iou` / :func:`greedy_match` / :func:`bucket_of` 都是纯函数
    （输入是普通 list/tuple）, 不需要权重、不需要 GPU、不需要真实图片 ——
    判据可以完全离线单测。只有 :func:`analyze_iou` 才碰框架与磁盘。
"""
from __future__ import annotations

import logging
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional, Sequence, Tuple

from od_platform.common.progress import ProgressReporter
from od_platform.evaluation.dataset import DatasetView, list_split_images
from od_platform.evaluation.metrics import f1_of

logger = logging.getLogger(__name__)

# 判定"这个框算配上了"的 IoU 门限。0.5 是检测领域的通行门限, 与 AP50 同源。
IOU_MATCH_THRESHOLD: float = 0.5

# 低于这个 conf 就属于"val 级极低阈值": 匹配口径的 P 会被大量低置信度预测拉垮,
# 报告里要如实标注（IoU 分布本身不受影响）。这个数只影响"要不要加那句说明"。
IOU_LOW_CONF_HINT: float = 0.01

# IoU 直方图分档。上界一律开区间, 最后一档含 1.0（完全重合要落在最后一档里）。
IOU_BUCKETS: Tuple[Tuple[str, float, float], ...] = (
    ("0.50-0.60", 0.50, 0.60),
    ("0.60-0.70", 0.60, 0.70),
    ("0.70-0.80", 0.70, 0.80),
    ("0.80-0.90", 0.80, 0.90),
    ("0.90-1.00", 0.90, 1.00),
)

# 浮点比较的容差: 0.6 这种"正好落在档位边界"的值不该因为二进制误差掉到下一档
_EPS: float = 1e-9

Box = Tuple[float, float, float, float]


# ---------------------------------------------------------------- 纯函数

def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """两个 xyxy 框的 IoU。

    · 无交集 → 0.0（不是负数）;
    · 退化成点/线的框 → 面积按 0 处理, 并集为 0 时返回 0.0（不做 0/0）。
    """
    ax1, ay1, ax2, ay2 = (float(v) for v in a[:4])
    bx1, by1, bx2, by2 = (float(v) for v in b[:4])

    inter_w = min(ax2, bx2) - max(ax1, bx1)
    inter_h = min(ay2, by2) - max(ay1, by1)
    if inter_w <= 0.0 or inter_h <= 0.0:
        return 0.0
    intersection = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= _EPS:
        return 0.0
    return intersection / union


def normalized_to_xyxy(cx: float, cy: float, width: float, height: float,
                       image_width: float, image_height: float) -> Box:
    """YOLO 的归一化 ``cx cy w h`` → 绝对 ``xyxy``。

    ★ 这一步是必须的: 标签是归一化的, 而框架给的预测框是绝对像素,
      不换算就没法求交。
    """
    box_w = float(width) * float(image_width)
    box_h = float(height) * float(image_height)
    center_x = float(cx) * float(image_width)
    center_y = float(cy) * float(image_height)
    return (center_x - box_w / 2.0, center_y - box_h / 2.0,
            center_x + box_w / 2.0, center_y + box_h / 2.0)


def bucket_of(value: float) -> Optional[str]:
    """IoU 落在哪一档; 低于第一档下界或高于 1.0 时返回 None（不进直方图）。"""
    number = float(value)
    for name, lower, upper in IOU_BUCKETS:
        if lower - _EPS <= number < upper - _EPS:
            return name
    # 最后一档的右端点(1.0)要包含进来: 完全重合是最该被记下的一种情况
    if number >= IOU_BUCKETS[-1][2] - _EPS:
        return IOU_BUCKETS[-1][0]
    return None


def greedy_match(predictions: Sequence[Dict[str, Any]],
                 ground_truths: Sequence[Dict[str, Any]], *,
                 iou_threshold: float = IOU_MATCH_THRESHOLD
                 ) -> Tuple[List[Dict[str, Any]], List[int], List[int]]:
    """按 IoU 降序的贪心一对一匹配（只允许同类配对）。

    Args:
        predictions: 每项形如 ``{"cls": int, "conf": float, "box": (x1,y1,x2,y2)}``
        ground_truths: 同上（``conf`` 可缺）
        iou_threshold: 低于它的候选不算匹配

    Returns:
        ``(匹配对, 未匹配的预测下标, 未匹配的真值下标)``

    ★ 为什么先排序再贪心: 若按"预测顺序"依次认领, 一个位置刚好的高分框可能被
      前面的劣质框抢走真值。降序保证"最像的先配", 这也是常见实现的做法。

    ★ 排序键里带上下标是为了**结果可复现**: IoU 完全并列时按下标定序,
      同一份输入永远得到同一份匹配。
    """
    used_pred: set = set()
    used_gt: set = set()

    candidates: List[Tuple[float, int, int]] = []
    for pred_index, pred in enumerate(predictions):
        for gt_index, gt in enumerate(ground_truths):
            if int(pred.get("cls", -1)) != int(gt.get("cls", -1)):
                continue                      # 类别不同的框不算匹配, 只算 FP+FN
            value = box_iou(pred["box"], gt["box"])
            if value < iou_threshold:
                continue
            candidates.append((value, pred_index, gt_index))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    pairs: List[Dict[str, Any]] = []
    for value, pred_index, gt_index in candidates:
        if pred_index in used_pred or gt_index in used_gt:
            continue                          # 一对一: 谁都不许被用第二次
        used_pred.add(pred_index)
        used_gt.add(gt_index)
        pairs.append({"pred_index": pred_index, "gt_index": gt_index, "iou": value})

    unmatched_pred = [i for i in range(len(predictions)) if i not in used_pred]
    unmatched_gt = [i for i in range(len(ground_truths)) if i not in used_gt]
    return pairs, unmatched_pred, unmatched_gt


def parse_label_file(path: Path, *, image_width: float, image_height: float,
                     n_classes: int) -> List[Dict[str, Any]]:
    """读一个 YOLO 标签文件, 换成绝对坐标的框。

    坏行跳过而不抛 —— 一个坏标签不该让整场 IoU 分析失败（那是 D4 质检的职责）。
    """
    rows: List[Dict[str, Any]] = []
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("标签读不了, 记为 0 个真值: %s (%s)", path, exc)
        return rows

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            cx, cy, width, height = (float(value) for value in parts[1:5])
        except (TypeError, ValueError):
            continue
        if not 0 <= class_id < max(int(n_classes), 0):
            continue
        rows.append({
            "cls": class_id,
            "box": normalized_to_xyxy(cx, cy, width, height, image_width, image_height),
        })
    return rows


def _ratio(numerator: int, denominator: int) -> Optional[float]:
    """比值; 分母为 0 时返回 None（不返回 0 —— 0 是"算出来很差"）。"""
    if denominator <= 0:
        return None
    return numerator / denominator


def summarize_iou(pairs: Sequence[Dict[str, Any]], *, n_predictions: int,
                  n_ground_truth: int, iou_threshold: float = IOU_MATCH_THRESHOLD,
                  decimals: int = 4) -> Dict[str, Any]:
    """把匹配结果整理成报告里的 IoU 块（纯函数, 便于离线单测）。"""
    values = [float(pair["iou"]) for pair in pairs]
    matched = len(values)
    false_positive = int(n_predictions) - matched
    false_negative = int(n_ground_truth) - matched

    histogram: List[Dict[str, Any]] = []
    for name, _lower, _upper in IOU_BUCKETS:
        count = sum(1 for value in values if bucket_of(value) == name)
        histogram.append({
            "bucket": name,
            "count": count,
            # 占比的分母是**匹配对**数: 它回答"配上的这些框里有多少落在这一档"
            "share": round(count / matched, decimals) if matched else None,
        })

    precision = _ratio(matched, int(n_predictions))
    recall = _ratio(matched, int(n_ground_truth))
    f1 = f1_of(precision, recall)

    def _round(value: Optional[float]) -> Optional[float]:
        return None if value is None else round(float(value), decimals)

    return {
        "status": "ok",
        "match_threshold": float(iou_threshold),
        "n_images": None,                     # 由 analyze_iou 填（纯函数不知道图数）
        "n_predictions": int(n_predictions),
        "n_ground_truth": int(n_ground_truth),
        "matched_pairs": matched,
        "tp": matched,                        # 匹配口径下 TP 即"配上的对"
        "fp": false_positive,
        "fn": false_negative,
        "precision": _round(precision),
        "recall": _round(recall),
        "f1": _round(f1),
        "iou": {
            "mean": _round(mean(values)) if values else None,
            "median": _round(median(values)) if values else None,
            "min": _round(min(values)) if values else None,
            "max": _round(max(values)) if values else None,
        },
        "histogram": histogram,
        "note": ("匹配口径: 同类预测/真值按 IoU 降序贪心一对一, "
                 f"门限 IoU>={iou_threshold}; 与官方 mAP 口径不同, 两者不可混用或相加"),
    }


# ---------------------------------------------------------------- 端到端

def _predictions_of(result: Any) -> List[Dict[str, Any]]:
    """把一次 ``predict`` 的返回值里的框抽成朴素 list（不碰框架私有属性）。"""
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    classes = getattr(boxes, "cls", None)
    confidences = getattr(boxes, "conf", None)
    coordinates = getattr(boxes, "xyxy", None)
    if classes is None or coordinates is None:
        return []
    try:
        class_list = [int(v) for v in classes]
        conf_list = [float(v) for v in confidences] if confidences is not None else []
        coord_list = [[float(v) for v in row] for row in coordinates]
    except (TypeError, ValueError) as exc:      # pragma: no cover - 取决于框架版本
        logger.warning("预测框不可解析, 本张图按 0 个预测处理: %s", exc)
        return []

    rows: List[Dict[str, Any]] = []
    for index, class_id in enumerate(class_list):
        if index >= len(coord_list):
            continue
        box = coord_list[index]
        if len(box) < 4:
            continue
        rows.append({
            "cls": class_id,
            "conf": conf_list[index] if index < len(conf_list) else None,
            "box": (box[0], box[1], box[2], box[3]),
        })
    return rows


def analyze_iou(model: Any, view: DatasetView, settings: Any) -> Dict[str, Any]:
    """对视图里的每一张图做匹配式 IoU 分析, 返回可进报告的字典。

    Args:
        model: 有 ``.predict(**kwargs)`` 的模型（真实 YOLO 或测试替身）
        view: :class:`~od_platform.evaluation.dataset.DatasetView`
        settings: :class:`~od_platform.evaluation.settings.EvalSettings`

    Returns:
        含 ``n_predictions / matched_pairs / tp / fp / fn / precision / recall / f1
        / iou{mean,median,min,max} / histogram / per_image / pairs`` 的字典。

    ★ 阈值取自 ``settings.conf`` / ``settings.iou``（都进报告, 可追溯）。
      注意 ``eval.yaml`` 的默认 ``conf`` 是 0.001, 那是给 val 用的极低阈值 —— 用它跑
      匹配会让 FP 暴涨、匹配口径的 P 失去意义（IoU 分布不受影响）。本函数会在
      ``note`` 里如实标注这件事, 并保留 ``conf`` 字段供核对。
    """
    images = list_split_images(view.images_dir)
    if not images:
        logger.warning("视图里一张图都没有, IoU 分析跳过: %s", view.images_dir)
        return {"status": "无图片", "images_dir": str(view.images_dir)}

    logger.info("开始 IoU 匹配分析: %d 张图 conf=%s iou=%s device=%s",
                len(images), settings.conf, settings.iou, settings.device)

    kwargs: Dict[str, Any] = {
        "source": [str(path) for path in images],
        "conf": settings.conf,
        "iou": settings.iou,
        "imgsz": settings.imgsz,
        "device": settings.device,
        "max_det": settings.max_det,
        "verbose": False,
        "stream": True,
    }
    results = model.predict(**kwargs)

    progress = ProgressReporter("IoU 匹配分析", len(images), logger=logger, unit="张图")

    pair_rows: List[Dict[str, Any]] = []
    per_image: List[Dict[str, Any]] = []
    all_ious: List[float] = []
    total_predictions = total_ground_truth = 0
    processed = 0

    for index, result in enumerate(results, 1):
        processed = index
        progress.update(index)

        path = Path(str(getattr(result, "path", "") or ""))
        shape = getattr(result, "orig_shape", None)
        if shape is None or len(shape) < 2:
            # 没有原图尺寸就无法把归一化标签换算成绝对坐标 —— 如实跳过, 不猜
            logger.warning("结果里没有 orig_shape, 无法对该图做 IoU 匹配: %s", path)
            continue
        image_height, image_width = float(shape[0]), float(shape[1])

        predictions = _predictions_of(result)
        label_path = view.labels_dir / f"{path.stem}.txt"
        ground_truths = parse_label_file(label_path, image_width=image_width,
                                        image_height=image_height,
                                        n_classes=view.n_classes)

        pairs, unmatched_pred, unmatched_gt = greedy_match(predictions, ground_truths)
        total_predictions += len(predictions)
        total_ground_truth += len(ground_truths)

        for pair in pairs:
            pred = predictions[pair["pred_index"]]
            gt = ground_truths[pair["gt_index"]]
            class_id = int(gt["cls"])
            all_ious.append(float(pair["iou"]))
            pair_rows.append({
                "image": path.name,
                "class_id": class_id,
                "class_name": (view.class_names[class_id]
                               if 0 <= class_id < len(view.class_names) else f"class_{class_id}"),
                "iou": settings.round(pair["iou"]),
                "confidence": settings.round(pred.get("conf")),
                "gt_x1": settings.round(gt["box"][0]), "gt_y1": settings.round(gt["box"][1]),
                "gt_x2": settings.round(gt["box"][2]), "gt_y2": settings.round(gt["box"][3]),
                "pred_x1": settings.round(pred["box"][0]), "pred_y1": settings.round(pred["box"][1]),
                "pred_x2": settings.round(pred["box"][2]), "pred_y2": settings.round(pred["box"][3]),
            })

        per_image.append({
            "image": path.name,
            "n_ground_truth": len(ground_truths),
            "n_predictions": len(predictions),
            "matched_pairs": len(pairs),
            "fp": len(unmatched_pred),
            "fn": len(unmatched_gt),
            "mean_iou": settings.round(mean([p["iou"] for p in pairs])) if pairs else None,
        })

    progress.finish(done=processed)

    summary = summarize_iou(pair_rows, n_predictions=total_predictions,
                            n_ground_truth=total_ground_truth,
                            iou_threshold=IOU_MATCH_THRESHOLD,
                            decimals=settings.decimals)
    summary["n_images"] = len(images)
    summary["conf"] = settings.conf
    summary["nms_iou"] = settings.iou
    if float(settings.conf) <= IOU_LOW_CONF_HINT:
        # 如实说清"这个 P 为什么低": 低阈值下 NMS 后的框绝大多数是低置信度的,
        # 它们当然配不上真值 —— 那是阈值造成的, 不是框定得不准。
        summary["note"] += (f"；本次 conf={settings.conf} 属 val 级极低阈值, "
                            f"未匹配的 {summary['fp']} 个预测里绝大多数是低置信度框, "
                            "所以匹配口径的 P 偏低、不代表交付质量; "
                            "IoU 分布不受影响。要看部署阈值的匹配口径请提高 --conf")
    summary["per_image"] = per_image
    summary["pairs"] = pair_rows
    logger.info("IoU 匹配分析完成: %d 个匹配对 / %d 个预测 / %d 个真值, 平均 IoU=%s",
                summary["matched_pairs"], total_predictions, total_ground_truth,
                summary["iou"]["mean"])
    return summary