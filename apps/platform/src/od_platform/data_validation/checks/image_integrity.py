# apps/platform/src/od_platform/data_validation/checks/image_integrity.py
"""check: image_integrity —— 每张图像都能真正解码吗(截断 / 损坏 / 伪装扩展名)。

D4 此前的检查只看"文件在不在、标签对不对", 没有一项碰过图像本体:
0 字节的 jpg、下载中断的半张图、把文本改成 .png 后缀的假图, 都能顺着
pair_existence(文件存在)与 label_format(标签合法)一路 PASS 进训练集, 直到
YOLO 在 dataloader 里抛错才暴露。本 check 把这类坏图提前拦在数据质检阶段。

★ 判定: 用 Pillow 的 ``Image.open(p).load()`` 强制解码 —— 只 open 不 load 抓不到
  截断文件(JPEG header 完好、像素数据缺失), 必须让解码器真正读完像素。
★ 代价: 全量解码是 I/O + CPU 密集操作, 大数据集上会跑一阵, 所以接入
  ProgressReporter 按比例打点(与 manifest_lineage 同一套口径)。
★ 阈值: 坏图占比 >= image_integrity_error_ratio(默认 0.0 = 一张都不许坏)→ ERROR;
  低于该线但有坏图 → WARNING(用户把线调高时才会走到这一档)。
  展示上限复用 details_preview_limit。
★ Pillow 缺失时不炸 import(注册表会 import 所有 check, 一项缺依赖不该拖垮全部),
  而是降级为 INFO 跳过。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from od_platform.common.constants import (
    DETAILS_PREVIEW_LIMIT, IMAGE_INTEGRITY_ERROR_RATIO,
)
from od_platform.common.progress import ProgressReporter
from od_platform.data_validation.registry import (
    CheckContext, CheckResult, CheckSeverity, check,
)

_NAME = "image_integrity"
logger = logging.getLogger(__name__)


def _load_image_module():
    """延迟 import Pillow: 没装时返回 None 而不是让注册表 import 失败。"""
    try:
        from PIL import Image
    except ImportError:                                  # pragma: no cover
        return None
    return Image


def decode_error(path: Path, image_cls=None) -> Optional[str]:
    """能完整解码返回 None; 否则返回一行错误摘要(不抛异常)。

    Args:
        path: 图像文件路径
        image_cls: 注入用(测试可传假实现); None 时使用 Pillow 的 ``Image``

    Returns:
        None = 解码成功; 字符串 = 失败原因(``类型: 消息``)
    """
    Image = image_cls if image_cls is not None else _load_image_module()
    if Image is None:
        return "PillowUnavailable: 未安装 Pillow"
    try:
        with Image.open(path) as im:
            im.load()          # ★ 必须 load: 只 open 不读像素抓不到截断文件
        return None
    except Exception as exc:   # 任何解码异常都算坏图, 不能让整轮质检崩
        return f"{type(exc).__name__}: {exc}"


@check(_NAME)
def validate_image_integrity(ctx: CheckContext) -> CheckResult:
    snap = ctx.snapshot
    image_cls = _load_image_module()
    if image_cls is None:
        return CheckResult(_NAME, CheckSeverity.INFO,
                           "未安装 Pillow, 跳过图像解码校验",
                           {"reason": "pillow_missing"})
    # ★ 阈值来自 D4 配置(ctx.settings); 读不到回落 constants(唯一默认值出处)
    max_detail = ctx.setting("details_preview_limit", DETAILS_PREVIEW_LIMIT)
    error_ratio = ctx.setting("image_integrity_error_ratio", IMAGE_INTEGRITY_ERROR_RATIO)

    pairs = [(split, img) for split in snap.splits
             for img in snap.images_per_split[split]]
    total = len(pairs)
    if total == 0:
        return CheckResult(_NAME, CheckSeverity.PASS, "没有可校验的图像",
                           {"total_images": 0})

    corrupt: List[Dict[str, str]] = []
    per_split: Dict[str, int] = {}
    progress = ProgressReporter(f"D4 {_NAME}", total, logger=logger, unit="张图")
    for index, (split, img) in enumerate(pairs, 1):
        err = decode_error(img, image_cls)
        if err is not None:
            corrupt.append({"split": split, "image": str(img), "error": err})
            per_split[split] = per_split.get(split, 0) + 1
        progress.update(index)
    progress.finish()

    ratio = len(corrupt) / total
    details = {
        "total_images": total,
        "n_corrupt": len(corrupt),
        "corrupt_ratio": round(ratio, 4),
        "corrupt_per_split": per_split,
        "corrupt_preview": corrupt[:max_detail],
        "thresholds": {"error": error_ratio},
    }
    if corrupt and ratio >= error_ratio:
        return CheckResult(
            _NAME, CheckSeverity.ERROR,
            f"{len(corrupt)}/{total} 张图无法解码({ratio:.1%})"
            " —— 训练会在 dataloader 崩溃, 必须替换或剔除",
            details)
    if corrupt:
        return CheckResult(
            _NAME, CheckSeverity.WARNING,
            f"{len(corrupt)}/{total} 张图无法解码({ratio:.1%})", details)
    return CheckResult(_NAME, CheckSeverity.PASS,
                       f"全部 {total} 张图均可完整解码", details)