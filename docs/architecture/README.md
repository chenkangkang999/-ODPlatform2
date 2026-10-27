# docs/architecture —— 架构决策记录（ADR）

存放所有 ADR（Architecture Decision Record）。一篇 ADR 只回答一个问题：
**当时为什么这么定、代价是什么、什么条件下该重新评估**。

代码只记录"做了什么"，ADR 记录"为什么"，两者缺一不可。

## 已有记录

| 编号 | 主题 | 一句话摘要 | 状态 |
|---|---|---|---|
| [ADR-001](ADR-001-monorepo.md) | 采用 Monorepo + `apps/` 布局 | 今天多打一层目录，换来未来多端扩展零成本；配套 marker file + 双层 `pyproject.toml`。 | ✅ Accepted |
| [ADR-002](ADR-002-dataset-yaml-paths.md) | 数据集 yaml 的路径约定 | 写入端 `path` 写【相对仓库根】（可移植、CI 守卫拦得住盘符），读取端统一在 `common/dataset_yaml.py` 解析（绝对原样 / 相对先 yaml 目录再仓库根），喂 ultralytics 前落一份绝对路径运行期副本；初版的"绝对路径 + 多候选探测"已被本篇取代。 | ✅ Accepted |
| [ADR-003](ADR-003-runtime-config-precedence.md) | 运行配置三源合并与优先级 | 优先级固定 `CLI > YAML > DEFAULT`（DEFAULT 自动注入最前），默认值唯一真相是 Pydantic；每个字段的来源必须可观测（来源报告 / 覆盖链 / 审计 JSON / 增强报错）。`infer_pipeline.yaml` 不在合并链内。 | ✅ Accepted |
| [ADR-004](ADR-004-inference-output-and-viz.md) | 推理输出与可视化 | 美化绘制已真正落地且样式可配（`infer_pipeline.yaml` 的 `visualization` 段）；输出落盘参数（文件名 / 编码器 / 质量 / 帧率 / 覆盖与审计开关）已提到同文件的 `output` 段，由 `OutputConfig` 解析、`LocalFileSink` 消费，默认值与原写死行为逐字节一致。 | ✅ Accepted（第 2 条已落地） |
| [ADR-005](ADR-005-progress-retention-eval-and-label-dir-fix.md) | 进度反馈与 `runs/` 保留；评估的两套 IoU 口径；标签目录推导缺陷修正 | 长循环用"按比例打点的日志"而非 `tqdm`；`runs/` 用"保最新 N 个 **且** N 天内不动"的双条件策略、`lineage` 硬豁免；`odp-eval` 把**官方 mAP 口径**与**匹配口径**并列但绝不混用；`resolve_labels_dir` 由"只认 Roboflow 布局"改为三候选探测，修掉 D3 布局下真值框恒为 0 的静默缺陷。 | ✅ Accepted |
| [ADR-006](ADR-006-everything-is-config.md) | 一切皆配置 —— 用配置层消灭硬编码 | 凡是"改它会让同一输入得到不同产物"的取值，都必须走「Pydantic Settings + 三源合并（CLI > YAML > DEFAULT）+ 来源溯源 + `odp-gen-config` 模板」；默认值只允许存在一处（引用 `common/constants.py`）；路径只有 `paths.py` 一个真相；安全白名单 / 协议常量 / 数学边界**明确不配置**并写明理由。 | ✅ Accepted |

## 命名约定

`ADR-<三位编号>-<短横线小写主题>.md`，编号只增不改；决策变更时**新增一条 ADR**，
并在旧记录里标注被取代（不要直接改写历史决策）。

## 与代码的对应关系

| ADR | 主要落点 |
|---|---|
| ADR-001 | `apps/`、`.odp-workspace`、根 `pyproject.toml`、`apps/platform/pyproject.toml` |
| ADR-002 | `common/dataset_yaml.py`、`data_pipeline/split/yaml_writer.py`、`data_validation/snapshot.py`、`data_validation/checks/_lineage.py`、`evaluation/dataset.py`、`training/service.py` |
| ADR-003 | `runtime_config/{base,train,val,infer,loaders,merger,generator,registry,report}.py`、`common/config_log.py`、`common/config_utils.py`、`cli/train_model.py`、`cli/infer.py` |
| ADR-004 | `visualization/__init__.py`、`inference/{service,sinks,overlay,pipeline_config}.py`、`configs/runtime/infer_pipeline.yaml` |
| ADR-005 | `common/{progress,retention}.py`、`cli/clean_runs.py`、`evaluation/{settings,dataset,metrics,iou,report,service}.py`、`cli/evaluate_model.py` |
| ADR-006 | `runtime_config/{base,transform,validate,train,val,infer,clean,loaders,merger,registry,generator,report}.py`、`apps/platform/configs/runtime/*.yaml`、`common/{paths,constants,dataset_yaml,config_log,config_utils}.py`、`.github/workflows/ci.yml` 的绝对路径守卫 |