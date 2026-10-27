# scripts/demo_data/make_synth_demo.py
"""生成 YOLO 风格【可解码】演示数据集 data/raw/synth/(40 张真实 JPEG)。

为什么需要它: D3 讲义生成的 data/raw/demo/ 里 .jpg 是空字节文件("空图:全流程不读像素"),
ultralytics 解不了这种图, 拿它训练会报损坏图像。本脚本用 numpy + PIL 画出带随机噪声
背景与随机色块的【真实 JPEG】, 并写出与之匹配的 YOLO 归一化标签 —— 用来真机验证
D6 训练链路端到端可跑(GPU/产物/验章/归档)。

⚠️ 这些是【合成图, 仅用于验证链路可训练, 指标无意义】—— 色块与噪声没有真实语义,
   1 个 epoch 的 mAP 只证明"跑通了", 不证明"学得好"。

幂等: 固定种子 SEED, 每次都覆盖同名 images/*.jpg 与 labels/*.txt, 重跑结果一致。

复现: python scripts/demo_data/make_synth_demo.py
数据目录 gitignored, 本脚本本身留在仓库里 —— 它是可复现的演示数据来源。
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from od_platform.common.paths import RAW_DATA_DIR

SEED = 20260727          # 固定种子: 每次生成同一批图, 演示可复现
N_IMAGES = 40
SIZE = 320               # 320x320, 够小好训, 又足够让 ultralytics 解出框图
CLASSES = ["red_block", "green_block", "blue_block"]
CLASS_COLORS = [(220, 40, 40), (40, 200, 80), (50, 90, 230)]

ROOT = RAW_DATA_DIR / "synth"
IMG_DIR = ROOT / "images"
LBL_DIR = ROOT / "labels"
IMG_DIR.mkdir(parents=True, exist_ok=True)
LBL_DIR.mkdir(parents=True, exist_ok=True)


def _make_image(rng: np.random.Generator, boxes):
    """噪声背景 + 每类色块: 像素是真内容, 不是空文件。"""
    # 背景: 深灰底 + 逐像素噪声(真实像素 -> JPEG 可解码)
    bg = rng.integers(20, 90, size=(SIZE, SIZE, 3), dtype=np.uint8)
    img = Image.fromarray(bg, mode="RGB")
    draw = ImageDraw.Draw(img)
    for cls_id, (x0, y0, x1, y1) in boxes:
        draw.rectangle([x0, y0, x1, y1], fill=CLASS_COLORS[cls_id])
        # 再叠一点块内噪声, 免得色块是纯色(纯色在极小 imgsz 下也照样可学)
        patch = rng.integers(0, 40, size=(y1 - y0, x1 - x0, 3), dtype=np.uint8)
        px = np.asarray(img)[y0:y1, x0:x1].astype(np.int16)
        px = np.clip(px + patch - 20, 0, 255).astype(np.uint8)
        arr = np.asarray(img).copy()
        arr[y0:y1, x0:x1] = px
        img = Image.fromarray(arr, mode="RGB")
        draw = ImageDraw.Draw(img)
    return img


def main() -> None:
    rng = np.random.default_rng(SEED)
    for i in range(N_IMAGES):
        stem = f"s_{i:03d}"
        n_boxes = int(rng.integers(1, 4))                 # 每图 1~3 个框
        boxes, labels = [], []
        for _ in range(n_boxes):
            cls_id = int(rng.integers(0, len(CLASSES)))
            w = int(rng.integers(48, 112))                # 框边长 48~111 px
            h = int(rng.integers(48, 112))
            x0 = int(rng.integers(0, SIZE - w))
            y0 = int(rng.integers(0, SIZE - h))
            x1, y1 = x0 + w, y0 + h
            boxes.append((cls_id, (x0, y0, x1, y1)))
            # YOLO 标签: 5 列 <cls> <cx> <cy> <w> <h>, 全部归一化到 [0,1]
            labels.append(f"{cls_id} {(x0 + x1) / 2 / SIZE:.6f} {(y0 + y1) / 2 / SIZE:.6f} "
                          f"{w / SIZE:.6f} {h / SIZE:.6f}")

        _make_image(rng, boxes).save(IMG_DIR / f"{stem}.jpg", format="JPEG", quality=92)
        (LBL_DIR / f"{stem}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")

    print(f"可训练演示数据就绪: {ROOT}  ({N_IMAGES} 张真实 JPEG, {SIZE}x{SIZE}, "
          f"类别: {','.join(CLASSES)})")
    print("下一步: python -m od_platform.cli.transform_data --dataset synth "
          f"--format yolo --classes {' '.join(CLASSES)}")


if __name__ == "__main__":
    main()