# apps/platform/src/od_platform/data_pipeline/orchestrator.py
"""数据流水线编排:检查原料 → 转换 → 扫描 → 报告 → 划分 → 契约 → 物化 → yaml。

编排者不实现任何一步的细节,只做接线与次序;每步的失败语义由子系统自己定。
产物分家:processed/(生产)+ runs/(审计 manifest)+ configs/(yaml 合格证)。

★ 阈值来源(D3 配置化):全部走 TransformSettings(configs/runtime/transform.yaml,
  由 odp-gen-config transform 生成;三源合并 CLI > YAML > DEFAULT)。
  编排层自己【不】再 import 任何阈值常量 —— 覆盖率闸门 / 稀有类预警 / 划分默认值
  的"唯一默认值出处"是 common/constants.py,由 TransformSettings 引用一次。
"""
from __future__ import annotations

import logging
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from od_platform.common import paths, stamp
from od_platform.common.constants import AnnotationFormat, IMAGE_EXTENSIONS, Task
from od_platform.common.lineage import (
    SplitManifest, TOOL_VERSION, compute_data_fingerprint,
)
from od_platform.common.run_context import RunContext
from od_platform.common.progress import ProgressReporter
from od_platform.data_pipeline.convert.registry import ConvertOptions
from od_platform.data_pipeline.convert.service import convert_data_to_yolo
from od_platform.data_pipeline.report import (
    analyze_balance, format_report, warn_if_rare,
)
from od_platform.data_pipeline.split.materializer import (
    SplitOutputDirs, SplitSourceDirs, materialize,
)
from od_platform.data_pipeline.split.split_service import split_dataset
from od_platform.data_pipeline.split.yaml_writer import write_dataset_yaml
from od_platform.runtime_config.transform import TransformSettings, build_transform_config

logger = logging.getLogger(__name__)

_SPLITS = ("train", "val", "test")


class DatasetPipeline:
    """一条命令的完整实现:odp-transform 的全部逻辑。"""

    def __init__(
        self,
        dataset: str,
        annotation_format: str,
        *,
        task: str = Task.DETECT,
        train_rate: Optional[float] = None,
        val_rate: Optional[float] = None,
        classes: Optional[List[str]] = None,
        random_state: Optional[int] = None,
        split_strategy: Optional[str] = None,
        settings: Optional[TransformSettings] = None,
        run: Optional[RunContext] = None,
    ):
        """★ 阈值/比例一律读 settings, 不再直接 import 常量.

        Args:
            settings: D3 配置(TransformSettings)。None 时用 build_transform_config()
                      的默认值 —— 既有调用点(DatasetPipeline("demo", "pascal_voc"))
                      行为完全不变。
            train_rate / val_rate / random_state / split_strategy / classes:
                      【显式覆盖】, 优先级高于 settings(给"不走 yaml 的调用点"留的后门)。
                      None = 没表态 = 用 settings 里的值。
        """
        self.dataset = dataset
        self.annotation_format = annotation_format
        self.task = task

        # 生效配置 = 传入的 settings(或默认) + 显式覆盖(覆盖值走一次完整校验)
        if settings is None:
            # 默认配置: build_transform_config() 与 build_train_config 同构,
            # 返回 (config, merger); 编排层只要 config, 溯源在 CLI 那边打。
            settings, _merger = build_transform_config()
        self.settings: TransformSettings = settings
        explicit = {
            "train_rate":    train_rate,
            "val_rate":      val_rate,
            "classes":       classes,
            "random_state":  random_state,
            "split_strategy": split_strategy,
        }
        given = {name: value for name, value in explicit.items() if value is not None}
        if given:
            self.settings = self.settings.model_validate(
                {**self.settings.model_dump(), **given})

        # 从生效配置取值(保留原有实例属性名: 外部/测试仍在读 pipeline.train_rate)
        self.train_rate = self.settings.train_rate
        self.val_rate = self.settings.val_rate
        self.classes = self.settings.classes
        self.random_state = self.settings.random_state
        self.split_strategy = self.settings.split_strategy

        self._run_ctx = run        # 存成私有名:实例属性不能遮蔽 run() 方法
        self.raw_root = paths.raw_dataset_root(dataset)
        self.raw_images = self.raw_root / "images"
        self.raw_annotations = self.raw_root / "annotations"

    def _annotation_dir(self) -> Path:
        """源标注目录:YOLO 格式的标注在 labels/,其余在 annotations/。"""
        if self.annotation_format == AnnotationFormat.YOLO:
            return self.raw_root / "labels"
        return self.raw_annotations

    # ---- 前置检查 ----

    def _check_raw(self) -> None:
        if not self.raw_root.exists():
            raise FileNotFoundError(f"数据集不存在: {self.raw_root}")
        if not self.raw_images.exists():
            raise FileNotFoundError(f"缺少图像目录: {self.raw_images}")
        ann_dir = self._annotation_dir()
        if not ann_dir.exists():
            raise FileNotFoundError(f"缺少标注目录: {ann_dir}")
        self._check_coverage()

    def _check_coverage(self) -> None:
        """覆盖率闸门:标注数/图像数 过低时直接拒跑。

        ★ 进度反馈: 这里要先数一遍图像总数, 而总数正是"要数出来的东西"——所以走
          "总量未知"模式(total=0, 每 UNKNOWN_TOTAL_STEP 项打一个点, 不编百分比)。
          大数据集上这是几秒到几十秒的静默等待, 接上进度后用户能看到它在动。
        """
        progress = ProgressReporter("清点图像", 0, logger=logger, unit="张图")
        n_images = 0
        for path in self.raw_images.rglob("*"):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                n_images += 1
                progress.update(n_images)
        progress.finish(done=n_images)
        if n_images == 0:
            raise FileNotFoundError(f"图像目录为空: {self.raw_images}")
        if self.annotation_format == AnnotationFormat.COCO:
            logger.debug("COCO 为单 json 格式,跳过 stem 级覆盖率检查")
            return
        ann_dir = self._annotation_dir()
        n_annos = len(list(ann_dir.glob("*.*"))) if ann_dir.exists() else 0
        coverage = n_annos / n_images
        # ★ 两个阈值都来自 TransformSettings(唯一默认值出处: common/constants.py)
        hard = self.settings.coverage_hard_threshold
        soft = self.settings.coverage_soft_threshold
        if coverage < hard:
            raise ValueError(
                f"图像-标注覆盖率 {coverage:.1%} 低于硬阈值 "
                f"{hard:.0%}(图 {n_images} / 标注 {n_annos})。"
                f"疑似标注导出中断或路径配错 —— 请修源头,不要带病训练。")
        if coverage < soft:
            logger.warning(
                "图像-标注覆盖率 %.1f%% 低于软阈值 %.0f%%,继续但请确认是有意为之",
                coverage * 100, soft * 100)

    # ---- 扫描:转换产物 → 内存结构 ----

    def _scan(self, staging: Path, classes: List[str]):
        """扫描源图 + 读取转换出的标签(两段都是逐文件长循环, 各接一条进度)。"""
        image_index: Dict[str, Path] = {}
        candidates = sorted(self.raw_images.rglob("*"))
        progress_images = ProgressReporter("扫描图像目录", len(candidates),
                                           logger=logger, unit="个条目")
        for position, path in enumerate(candidates, 1):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                image_index[path.stem] = path
            progress_images.update(position)
        progress_images.finish(done=len(candidates))

        labels_per_image: Dict[str, List[str]] = defaultdict(list)
        label_bytes: Dict[str, bytes] = {}
        label_files = sorted(staging.glob("*.txt"))
        progress_labels = ProgressReporter("读取标签", len(label_files),
                                           logger=logger, unit="个标签文件")
        for position, txt in enumerate(label_files, 1):
            label_bytes[txt.stem] = txt.read_bytes()
            for line in txt.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    idx = int(line.split()[0])
                    labels_per_image[txt.stem].append(classes[idx])
            progress_labels.update(position)
        progress_labels.finish(done=len(label_files))
        return image_index, labels_per_image, label_bytes

    # ---- 主流程 ----

    def run(self) -> Dict[str, object]:
        if self._run_ctx is not None:
            return self._execute(self._run_ctx)
        with RunContext("data_pipeline") as run:
            return self._execute(run)

    def _execute(self, run: RunContext) -> Dict[str, object]:
        self._check_raw()
        out = SplitOutputDirs.for_dataset(self.dataset)

        with tempfile.TemporaryDirectory(prefix="odp_pipe_") as staging_dir:
            staging = Path(staging_dir)
            ann_in = self._annotation_dir()
            options = ConvertOptions(task=self.task, classes=self.classes)
            classes = convert_data_to_yolo(ann_in, staging, self.annotation_format, options)

            image_index, labels_per_image, label_bytes = self._scan(staging, classes)

            report = analyze_balance(
                dict(labels_per_image), classes,
                min_split_rate=min(self.train_rate, self.val_rate,
                                   self.settings.test_rate),
                warn_min_expected=self.settings.rare_warn_min_expected)
            for line in format_report(report):
                logger.info(line)
            warn_if_rare(report)

            stems = sorted(set(image_index) & set(label_bytes))
            splits = split_dataset(
                stems, self.train_rate, self.val_rate, self.random_state,
                strategy=self.split_strategy,
                labels_per_image={s: labels_per_image.get(s, []) for s in stems})

            manifest = SplitManifest.build(
                dataset=self.dataset, strategy=self.split_strategy,
                seed=self.random_state,
                ratios=(self.train_rate, self.val_rate, self.settings.test_rate),
                names=classes, splits=splits, label_bytes=label_bytes,
                created_at=run.created_at)
            manifest_path = run.artifact_path("manifest.json")
            manifest.write(manifest_path)
            manifest_ref = manifest_path.relative_to(paths.ROOT_DIR).as_posix()

            source = SplitSourceDirs(images=self.raw_images, labels=staging)
            counts = materialize(manifest, source, out)

            yaml_path = paths.dataset_yaml_path(self.dataset)
            write_dataset_yaml(
                yaml_path, dataset_root=out.root, classes=classes, manifest=manifest,
                dataset_name=self.dataset, source_format=self.annotation_format,
                task=self.task, manifest_ref=manifest_ref)

            # ★ 盖 transform 章: 我(D3)划分了这批数据。
            #   data_fingerprint 从【我刚建好的 samples】算 —— 一手证据, 不抄任何人。
            data_fp = compute_data_fingerprint(manifest.samples)
            stamp.write_stamp(
                stage_file=stamp.STAMP_TRANSFORM,
                data_fingerprint=data_fp,
                run_id=run.run_id,
                tool_version=TOOL_VERSION,          # odp-transform/0.4
                payload={
                    "contract_fingerprint": manifest.contract_fingerprint,
                    "manifest_ref": manifest_ref,
                    "n_samples": len(manifest.samples),
                    "created_at": run.created_at,
                },
            )

        logger.info("数据集 %r 完成: %s", self.dataset, counts)
        return {
            "counts": counts,
            "yaml": str(yaml_path),
            "manifest_path": str(manifest_path),
            "run_dir": str(run.run_dir),
            "contract_fingerprint": manifest.contract_fingerprint,
            "data_fingerprint": data_fp,
        }