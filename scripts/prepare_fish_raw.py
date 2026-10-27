#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""把**外部**的 Roboflow 风格目标检测数据集, 整理成 ODPlatform 的 raw 布局。

背景
----
`odp-transform` 对 YOLO 格式的约定是「一图一标签、按 stem 配对」, 且落点是
`data/raw/<name>/{images,labels}/`(**扁平**, 物化时按 `glob("<stem>.*")` 取图,
不递归)。而 Roboflow 导出的目录是三层:

    <dataset>/data.yaml            # train: ../train/images, nc, names
    <dataset>/train/{images,labels}/
    <dataset>/valid/{images,labels}/
    <dataset>/test/{images,labels}/

本脚本把这三种 split **合并成一个池子**并拍平, 交给 `odp-transform` 重新划分 ——
这样 train/val/test 由流水线用固定种子一次性冻结、写进划分契约(D3 决定 5),
而不是沿用导出方那份没有留痕的划分。原有的三份 split 目录只作为来源, 不会保留。

用法(路径不写死, 从命令行给)::

    python scripts/prepare_fish_raw.py --src "<外部数据集根, 含 data.yaml>" --name fish

选参
----
* `--name`      落点 `data/raw/<name>/`(默认 fish)
* `--copy/--move` 默认 copy(原始数据只读; move 只在你确认源目录可弃时用)
* `--dry-run`   只统计不落盘
"""
from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

# 让脚本在"还没装包"时也能找到 src/ ; 装了包则走已安装版本
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "apps" / "platform" / "src"))

import yaml  # noqa: E402

from od_platform.common.paths import RAW_DATA_DIR  # noqa: E402

SPLIT_DIRS = ("train", "valid", "test", "val")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _load_names(dataset_root: Path) -> list[str]:
    """从 data.yaml 读类名顺序。YOLO 格式不含类名, 顺序即 class_id, 不能猜。"""
    candidates = sorted(dataset_root.glob("*.yaml")) + sorted(dataset_root.glob("*.yml"))
    if not candidates:
        raise FileNotFoundError(f"{dataset_root} 下找不到 data.yaml")
    data = yaml.safe_load(candidates[0].read_text(encoding="utf-8")) or {}
    names = data.get("names")
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=lambda k: int(k))]
    if isinstance(names, (list, tuple)):
        return [str(n) for n in names]
    raise ValueError(f"{candidates[0]} 里没有可用的 names")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="整理外部 Roboflow 数据集到 data/raw/<name>/{images,labels}")
    p.add_argument("--src", required=True, help="外部数据集根目录(含 data.yaml)")
    p.add_argument("--name", default="fish", help="落点数据集名: data/raw/<name>/")
    p.add_argument("--move", action="store_true", help="移动而非复制(默认复制, 源目录只读)")
    p.add_argument("--dry-run", action="store_true", help="只统计, 不落盘")
    args = p.parse_args(argv)

    src = Path(args.src).expanduser().resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"源目录不存在: {src}")
    names = _load_names(src)

    splits: dict[str, tuple[list[Path], list[Path]]] = {}
    for split in SPLIT_DIRS:
        img_dir, lbl_dir = src / split / "images", src / split / "labels"
        if not img_dir.is_dir():
            continue
        images = sorted(q for q in img_dir.iterdir()
                        if q.is_file() and q.suffix.lower() in IMAGE_SUFFIXES)
        labels = sorted(lbl_dir.glob("*.txt")) if lbl_dir.is_dir() else []
        splits[split] = (images, labels)

    if not splits:
        raise FileNotFoundError(f"{src} 下没有 train/valid/test 的 images 目录")

    # ---- 先体检: 配对、重名、类号范围 ----
    seen: dict[str, str] = {}
    collisions: list[tuple[str, str, str]] = []
    n_images = n_labels = 0
    class_hist: Counter[int] = Counter()
    unpaired: list[str] = []
    out_of_range: list[tuple[str, int]] = []
    empty_labels = 0

    for split, (images, labels) in splits.items():
        label_by_stem = {q.stem: q for q in labels}
        for image in images:
            n_images += 1
            if image.stem in seen:
                collisions.append((image.stem, seen[image.stem], split))
                continue
            seen[image.stem] = split
            label = label_by_stem.get(image.stem)
            if label is None:
                unpaired.append(f"{split}/{image.name}")
                continue
            n_labels += 1
            text = label.read_text(encoding="utf-8")
            lines = [ln for ln in text.splitlines() if ln.strip()]
            if not lines:
                empty_labels += 1
            for line in lines:
                try:
                    cls_id = int(float(line.split()[0]))
                except (ValueError, IndexError):
                    out_of_range.append((f"{split}/{label.name}", -1))
                    continue
                class_hist[cls_id] += 1
                if not 0 <= cls_id < len(names):
                    out_of_range.append((f"{split}/{label.name}", cls_id))

    print(f"源目录      : {src}")
    print(f"类名({len(names)}) : {names}")
    for split, (images, labels) in splits.items():
        print(f"  {split:<6}: images={len(images):<5} labels={len(labels):<5}")
    print(f"合并后      : images={n_images} labels={n_labels} "
          f"实例={sum(class_hist.values())} 空标签={empty_labels}")
    if collisions:
        print(f"⚠️ 跨 split 同名图片 {len(collisions)} 个(会被跳过): {collisions[:5]}")
    if unpaired:
        print(f"⚠️ 缺标签的图 {len(unpaired)} 个(会被跳过): {unpaired[:5]}")
    if out_of_range:
        print(f"⚠️ 类号越界/非法 {len(out_of_range)} 处: {out_of_range[:5]}")
    print("类分布(类号:实例数):")
    for idx, name in enumerate(names):
        print(f"  {idx:>2} {name:<14}: {class_hist.get(idx, 0)}")

    if args.dry_run:
        print("\n[dry-run] 未落盘。")
        return 0

    # ---- 落盘: 拍平进 data/raw/<name>/{images,labels} ----
    dst = RAW_DATA_DIR / args.name
    dst_images, dst_labels = dst / "images", dst / "labels"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    op = shutil.move if args.move else shutil.copy2
    written = skipped = 0
    for split, (images, labels) in splits.items():
        label_by_stem = {q.stem: q for q in labels}
        for image in images:
            if collisions and image.stem != seen.get(image.stem) and image.stem in {c[0] for c in collisions}:
                skipped += 1
                continue
            target_image = dst_images / image.name
            op(str(image), str(target_image))
            written += 1
            label = label_by_stem.get(image.stem)
            if label is not None:
                op(str(label), str(dst_labels / label.name))

    print(f"\n✅ 已落盘 {written} 张图(跳过 {skipped}) -> {dst}")
    print(f"   图: {dst_images}")
    print(f"   签: {dst_labels}")
    print(f"   下一步: odp-transform --dataset {args.name} --format yolo "
          f"--classes {' '.join(names)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())