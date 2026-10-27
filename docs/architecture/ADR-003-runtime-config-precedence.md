# ADR-003：运行配置三源合并 —— 优先级 CLI > YAML > DEFAULT，且必须可观测

| 项目 | 内容 |
|---|---|
| 状态 | ✅ Accepted |
| 决策日期 | 2026-10-27 |

## 背景

同一份代码要在不同机器上跑：开发机（有 GPU）、CI（Linux、无 GPU）、产线机（可能只有 CPU）。
配置既要能表达"这次怎么跑"，又不能把某台机器的特征固化进仓库。

现状：

- 运行配置在 `apps/platform/configs/runtime/*.yaml`（进 git）：
  `train.yaml`、`val.yaml`、`infer.yaml`，字段由 Pydantic v2 模型定义并带默认值；
- 命令行参数可临时覆盖；
- `infer_pipeline.yaml`（帧源 + 美化）是**另一条读取路径**，见下文第 3 条。

一个具体的麻烦是"我明明改了配置怎么没生效"。这类问题的排查成本最高，
因为**配置的生效路径是隐形的**：同一个字段可能来自 Pydantic 默认值、YAML、或 CLI，
而起作用的那一处不会自己说出来。

## 备选方案

- **方案 A：只留 YAML，不接受 CLI 覆盖。**
  简单，但每次试参数都要改文件，还会把 git 工作区弄脏。
- **方案 B：只留 CLI，不要 YAML。**
  命令行会变得极长，且无法把"这次实验的配方"沉淀成可评审的文件。
- **方案 C：YAML 为底 + CLI 覆盖 + Pydantic 默认值兜底，并逐字段打印来源。**（本决策）

## 决定

采用 **方案 C**，并把三条约束写进代码而不是文档：

1. **优先级 = `sources` list 的顺序（后面的覆盖前面的），DEFAULT 永远自动注入到最前。**

   `runtime_config/merger.py` 的 `ConfigMerger.merge()` 接受
   `sources=[(source_id, dict), ...]`，`_do_merge()` 先把
   `_extract_defaults(config_class)` 放进列表头。`source_id` 接受 `ConfigSource` 枚举
   （`DEFAULT` / `YAML` / `CLI`）**或任意字符串**（例如 `"ENV"`），所以加一层环境变量
   不需要扩枚举。

   各 `build_*_config()` 固定拼成：**YAML < extra_sources < CLI**。

2. **默认值的唯一真相是 Pydantic 的 `Field(default=...)`；`None` 等同于"没给"。**

   - `_extract_defaults()` 用 `model_fields` 反射默认值，**默认值为 `None` 的字段不进合并**
     （让 Pydantic 自己处理）；
   - `loaders._drop_none()` 与 `merger._apply_source()` 都跳过 `None`，
     但**保留 `False` / `0` / `''`** —— 这是"用户显式填的假"，必须生效；
   - `cli/train_model.py` 里所有配置字段的 argparse `default` **一律是 `None`**，
     注释写明理由：argparse 再给默认值就会永远覆盖 YAML。

3. **可观测性是硬要求，不是可选功能。** 合并器提供三个出口：

   | 出口 | 位置 | 内容 |
   |---|---|---|
   | **字段级溯源链表** | `ConfigMetadata.overridden_from` | 每个字段一条链，新节点指向被覆盖的上一个节点 |
   | **给人看的来源报告** | `get_source_report()` | 按来源分组列出**最终生效**的字段值 |
   | **覆盖链报告** | `get_conflict_report()` | `epochs: 100 (DEFAULT) → 200 (YAML) → 300 (CLI)` |
   | **机器读的审计日志** | `to_audit_log()` | `fields_count_total` / `fields_by_source` / `overridden_count` / `overridden_fields`（**只装"谁给的"，不装字段值**，避免敏感数据进 ELK） |
   | **增强的报错** | `_enhance_validation_error()` | Pydantic `ValidationError` 的 `msg` 尾部追加 `[来源: 50(CLI) ← 200(YAML) ← 100(DEFAULT)]` |

   `chain_str()` 用 `←` 连接、"当前 ← 上一个 ← 更早"，符合"从结果倒推原因"的阅读习惯。
   训练侧由 `runtime_config/report.py` 的 `render_effective` / `render_overrides`
   经 `common/config_log.log_config_report()` 打进日志；推理侧把
   `merger.to_audit_log()` 写进 `runs/detect_infer/<name>/odp_audit.json` 的 `merger` 键。

4. **配置里不写机器相关值。** 约定如下（模板里就是这么发的）：

   | 字段 | 写法 | 理由 |
   |---|---|---|
   | `device` | 保持 `null`（自动：有 CUDA 用 GPU，否则 CPU） | 写死卡号换机器即失效；要临时固定用 `--device 0` |
   | `model` | 给名字或相对引用（`yolo11n.pt`），由 `refs.resolve_*` 解析 | 避免绝对路径 |
   | `batch` / `imgsz` | 写"通常合适"的值（模板 `16` / `640`），机器差异用 CLI 覆盖 | 8G 显存与 24G 显存需求不同 |
   | `workers` | 模板 `8`，Windows 有问题时改 `0` 或 `4` | 平台差异大，不宜写死 |

   确实需要"本机专属"的固定值时，用**环境变量或 CLI**（`extra_sources` 就是为此留的缝），
   不要改仓库里的 yaml。

5. **`infer_pipeline.yaml` 走的是另一条读取路径，本条 ADR 只覆盖前半边。**
   `inference/pipeline_config.load_pipeline_config()` 是手写解析（`yaml.safe_load` +
   逐字段取值），**不经过 `ConfigMerger`**，因此它没有三源合并、也没有逐字段溯源。
   它只负责"帧源 + 美化"这一半，与 D5 的 `infer.yaml` 互不重叠。
   文件不存在时返回空配置 + warning，不抛 —— 基本版只要有模型 + 源就能跑。

6. **敏感字段自动 mask。** `_display_value()` 对
   `config_class.sensitive_field_names()` 里的键返回 `SENSITIVE_MASK`，
   来源报告与覆盖报告里新旧两处都 mask。

## 理由

1. 仓库里的配置是**团队的默认共识**，不是某台机器的快照。
2. 三层结构让"默认 → 本次调整 → 一次性覆盖"各得其所，试参数不必弄脏 git。
3. 把溯源做成数据结构（链表）而不是日志字符串，才能同时喂给三种消费者：
   人（报告）、机器（审计 JSON）、报错信息（ValidationError）。
   只做日志的话，"改配置没生效"依然要靠猜。
4. 只记录"谁给的"而不记录值，让审计日志可以安全地外送。

## 后果

- **正面**
  - 同一份仓库可在开发机（GPU）与 CI（CPU）上各跑各的，配置不需要分叉；
  - 字段生效路径一目了然；冲突有完整覆盖链；非法值直接指出是哪一层给错的；
  - 加一层配置来源（环境变量 / etcd）只需在 `sources` list 里插一项，不动枚举。
- **负面**
  - `device: null` 时，如果 torch 是 `+cpu` 构建，会**静默用 CPU**。
    缓解手段是每次运行开头打印的"运行环境概览"（`odp-init` 会打印
    `CUDA可用 / GPU数量 / 显存`），把"静默"变成"可见"；
    README 也把 CUDA 版 torch 的安装单列一节。
  - `get_logger()` 是**幂等**的：整个进程只配一次日志，第二次调用即使传了不同的
    `log_level` / `temp_log` 也不再生效 —— 这是有意的（重复挂 handler 会让同一条日志打印 N 遍）。
    各 CLI 入口因此都从 `paths.LOGGING_DIR`（绝对路径）出发，
    **不写 CWD 相对的 `"logs"`** —— 那会让日志落在"你在哪个目录敲命令"上
    （`cli/infer.py` 里就留着这条注释）。
  - 字段变多后 Pydantic 模型成为单一事实源，改字段要同时更新配置模板（`odp-gen-config`）。
- **中性**
  - `infer_pipeline.yaml` 没有溯源能力（第 5 条）。要么接受这个不对称，
    要么将来把它也接进 `ConfigMerger`。

## 撤销条件

- 需要按环境分层（dev / prod 两套默认值）时，引入 profile 机制并由新 ADR 取代；
- 若支持"机器档案"（把本机能力写进 `.odp-meta/` 而非仓库配置），需要补充那一层；
- 若 `infer_pipeline.yaml` 被接进 `ConfigMerger`，本 ADR 第 5 条需改写。