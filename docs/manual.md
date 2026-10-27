# ODPlatform 使用手册

面向使用者的操作手册。环境搭建与设计理由见 [README](../README.md) 与 [ADR](architecture/)；
这里只讲"怎么做、产物在哪、出问题怎么办"。

- 命令入口由 `apps/platform/pyproject.toml` 的 `[project.scripts]` 提供，共 9 条。
- 示例统一用 Windows PowerShell 写法：`& .\.venv\Scripts\odp-xxx.exe ...`。
- 未特别说明时，**当前工作目录都是仓库根**（`D:\ODPlatform2`）。
- 参数表来自本机 `.venv` 里实际安装的命令的 `--help` 输出（2026-10-27）。

---

## 一、上手三步

### 1.1 准备环境

```powershell
# 普通包走阿里源
& .\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel -i https://mirrors.aliyun.com/pypi/simple/

# ★ torch / torchvision 必须走 PyTorch 官方 cu126 索引（GPU 版）
& .\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 其余依赖 + 可编辑安装 platform 端
& .\.venv\Scripts\python.exe -m pip install colorlog psutil pyyaml pydantic ultralytics opencv-python pytest pytest-cov ruff mypy pre-commit -i https://mirrors.aliyun.com/pypi/simple/
& .\.venv\Scripts\python.exe -m pip install -e .\apps\platform

# 确认 GPU
& .\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

一次性脚本版：`scripts/setup_venv.ps1`（做的事完全相同）。

本机实测输出：`torch 2.14.1+cu126 cuda 12.6 avail True`。

### 1.2 建目录、看环境

```powershell
& .\.venv\Scripts\odp-init.exe
```

输出包含"运行环境概览"：操作系统 / 主机名 / Python版本 / PyTorch版本 / Ultralytics版本 /
CPU型号与核心数 / 总内存与可用内存 / **CUDA可用 / GPU数量 / GPU_0 / GPU_0_显存**。
`CUDA可用: False` 说明 torch 是 `+cpu` 构建或驱动不匹配，先解决再谈性能。

### 1.3 放数据

把原始数据放到 `data/raw/<数据集名>/`。**标注格式决定标注目录名**：

| 格式 | 图像目录 | 标注目录 | 备注 |
|---|---|---|---|
| `pascal_voc` | `images/` | `annotations/`（`*.xml`） | 类别名可从数据自带信息取 |
| `coco` | `images/` | `annotations/`（`*.json`） | 单 json，跳过 stem 级覆盖率检查 |
| `yolo` | `images/` | **`labels/`**（`*.txt`） | **类别名必须 `--classes` 显式给** |

目录必须是**扁平**的：一张图 `images/<stem>.<ext>` 配一份标签 `labels/<stem>.txt`
（物化时按 `glob("<stem>.*")` 取图，不递归）。

#### 1.3.1 外部 Roboflow 数据集的接入（本次业务数据集就是这么来的）

Roboflow 导出的目录是三层的：

```
<dataset>/data.yaml            # train: ../train/images, nc, names, roboflow 元信息
<dataset>/train/{images,labels}/
<dataset>/valid/{images,labels}/
<dataset>/test/{images,labels}/
```

它自带的 `train/valid/test` 三份划分**不沿用** —— 本平台的划分在 `odp-transform`
这一步用固定种子一次性冻结并写进划分契约，训练只读那份冻结的划分。
用仓库自带的整理脚本把三份 split 合并拍平（**源路径从命令行给，不写死盘符**）：

```powershell
& .\.venv\Scripts\python.exe scripts\prepare_fish_raw.py --src "<外部数据集根，含 data.yaml>" --name fish
# --dry-run 先体检：逐 split 计数、类分布、重名、缺签、类号越界
```

脚本会打印体检结果（本数据集：4560 图 / 4560 签 / 6034 实例 / 33 类、无重名无缺签），
然后落到 `data/raw/fish/{images,labels}/`，并直接提示下一步的 `odp-transform` 命令。

---

## 二、命令详解

## 2.1 `odp-init` —— 建目录 + 环境快照

### 用途

按 `common/paths.get_dirs_to_initialize()` 的**唯一目录清单**创建运行时目录，并打印环境概览与
`data/raw/` 状态。`paths.py` 决定要哪些目录，`init_project.py` 只负责创建。

### 参数

**本命令没有 argparse 参数，也没有 `--help`。** 任何附带参数都会被忽略，命令直接执行初始化
（因为它对应 `od_platform.cli.init_project:initialize_project`，签名是 `()`）。

### 示例

```powershell
& .\.venv\Scripts\odp-init.exe
```

### 产物

| 落点 | 说明 |
|---|---|
| 15 项目录 | 见下表 |
| `apps/platform/logging/init_project/init-project_<YYYYMMDD-HHMMSS>-<pid>.log` | 运行日志 |

要创建的 15 项目录（来自 `paths.py`）：

```
data/                     runs/                     models/
models/pretrained/        models/trained/           data/raw/
data/processed/           apps/platform/configs/    apps/platform/logging/
apps/platform/tests/      scripts/                  docs/
.odp-meta/logs/           apps/platform/configs/datasets/
apps/platform/configs/runtime/
```

> 注意 `models/checkpoints/` **不在**这份清单里 —— 它只是 `odp-infer` 的权重搜索目录之一，
> 用到时按需自建。

### 失败模式与排查

| 现象 | 原因 | 处理 |
|---|---|---|
| 输出里 15 行全是"已存在" | 正常 —— 幂等，不会重复建 | 无 |
| `data\raw 为空 → 请放入至少一个数据集` | 还没放数据 | 放入 `data/raw/<name>/` 后重跑或直接进下一步 |
| `创建失败 <dir>: ...` 后进程退出码 1 | 目录创建失败（权限 / 占用） | `init_project.py` 是"第一次失败立刻退出"，可能留下半初始化状态，修好权限后重跑即可补齐 |
| 中文乱码 | 控制台代码页不是 UTF-8 | `chcp 65001` |

## 2.2 `odp-reset` —— 撤销 init 的运行时产物

### 用途

清理 `odp-init` 造的**运行时产物**。默认 dry-run；真删必须 `--yes`，且要交互式输入大写 `RESET`
（`--force` 跳过交互）。

**安全边界**（`common/paths.get_dirs_to_reset()`）——会被清理的只有：

- `data/processed/`
- `runs/`
- `models/trained/`
- `apps/platform/logging/`

**绝不清理**：`data/raw/`（原始数据）、`models/pretrained/`（下载的权重）、
以及任何进 git 的目录（代码 / 配置 / 文档 / 测试 / `scripts/`），
外加载 `PROTECTED_DIRS` 黑名单作为最后一道安全网（`is_protected()`）。
`odp-reset` 自己的审计日志写在 `.odp-meta/logs/reset_project/`，不会被自己清掉。

### 参数

`usage: odp-reset [-h] [--yes] [--force] [--dry-run]`

| 参数 | 说明 |
|---|---|
| `--yes` | 真正执行删除（默认是 dry-run） |
| `--force` | 跳过交互式确认（仅当 `--yes` 时有效） |
| `--dry-run` | 显式声明 dry-run |

`--dry-run` 与 `--yes` 同时给出时**以 `--dry-run` 为准**（只打印不删除）并给出警告。

### 示例

```powershell
& .\.venv\Scripts\odp-reset.exe                    # 默认 dry-run（推荐先跑这个）
& .\.venv\Scripts\odp-reset.exe --yes              # 真删除，需要输入 RESET
& .\.venv\Scripts\odp-reset.exe --yes --force      # CI / 脚本调用，跳过交互
& .\.venv\Scripts\odp-reset.exe --dry-run          # 显式 dry-run
```

### 产物

无新增产物（除了审计日志）。dry-run 会打印一张"目录 / 文件数 / 大小"表并给出合计。

### 失败模式与排查

| 现象 | 原因 | 处理 |
|---|---|---|
| 提示"这是 dry-run（默认行为）" | 没给 `--yes` | 确认清单无误后加 `--yes` |
| 输入不是大写的 `RESET` | 交互确认失败 | 命令取消并返回 1；重新运行并精确输入 `RESET` |
| `⛔ 拒绝处理受保护目录` | `get_dirs_to_reset()` 与 `is_protected()` 冲突 | 属于配置异常，不要绕过；检查 `paths.py` 的清单 |
| 删除时 `PermissionError` | Windows 上 git checkout 出的只读文件 | 已有 `_on_rm_error` 回调自动 `chmod` 后重试；仍失败时看日志里具体路径 |

## 2.3 `odp-transform` —— 原始数据 → 可训练成品

### 用途

把 `data/raw/<name>/` 的原始标注转成 YOLO 格式，做类别平衡报告、按比例划分、
写契约指纹、物化到 `data/processed/`，最后生成数据集 yaml。
转换前有**覆盖率闸门**：标注数/图像数的比值低于 `COVERAGE_HARD_THRESHOLD`（0.5）直接拒跑。

```
usage: odp-transform [-h] --dataset DATASET --format {pascal_voc,coco,yolo}
                     [--task {detect,segment}] [--train-rate TRAIN_RATE]
                     [--val-rate VAL_RATE] [--seed SEED]
                     [--split-strategy {random,stratified_multilabel}]
                     [--classes [CLASSES ...]]
```

### 参数

| 参数 | 必填 | 默认 | 说明 |
|---|---|---|---|
| `--dataset` | ✅ | — | 数据集名（读 `data/raw/<name>`） |
| `--format` | ✅ | — | `pascal_voc` / `coco` / `yolo` |
| `--task` | | `detect` | `detect` / `segment` |
| `--train-rate` | | `0.8` | 训练集比例 |
| `--val-rate` | | `0.1` | 验证集比例（其余给 test） |
| `--seed` | | `42` | 随机种子（`DEFAULT_RANDOM_STATE`） |
| `--split-strategy` | | `random` | `random` / `stratified_multilabel` |
| `--classes` | | `None` | 类别白名单（顺序即 `class_id`）；不传 = 按数据自带；传空 = 全排除 |

默认值来自 `common/constants.py`（`DEFAULT_TRAIN_RATE=0.8`、`DEFAULT_VAL_RATE=0.1`、
`DEFAULT_RANDOM_STATE=42`、`DEFAULT_SPLIT_STRATEGY="random"`）。

### 示例

```powershell
# Pascal VOC
& .\.venv\Scripts\odp-transform.exe --dataset <name> --format pascal_voc

# YOLO：类别名必须显式给，顺序即 class_id
& .\.venv\Scripts\odp-transform.exe --dataset <name> --format yolo --classes 类1 类2 类3

# 自定义划分 + 分层策略
& .\.venv\Scripts\odp-transform.exe --dataset <name> --format coco --train-rate 0.7 --val-rate 0.15 --seed 42 --split-strategy stratified_multilabel
```

### 产物

| 落点 | 说明 |
|---|---|
| `data/processed/<name>/images/{train,val,test}/` | 物化后的图像 |
| `data/processed/<name>/labels/{train,val,test}/` | YOLO 归一化标签 |
| `apps/platform/configs/datasets/<name>.yaml` | 训练入口 + `odp_meta` 合格证（`contract_fingerprint`、`manifest_path`、`split.rates/counts`） |
| `runs/data_pipeline/<run_id>/manifest.json` | 样本级清单（每个样本的 split 与标签 sha256） |
| `runs/lineage/<data_fingerprint>/transform.stamp.json` | transform 章 |
| `apps/platform/logging/transform/transform_<时间戳>.log` | 运行日志（含类别平衡报告表） |

`<run_id>` 形如 `20261027-115547`；同一秒内重复运行会自增后缀 `-2`、`-3`（`RunContext`）。

### 失败模式与排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `数据集不存在: .../data/raw/<name>` | 名字写错或没放数据 | 核对 `--dataset` 与目录名 |
| `缺少标注目录: .../labels` | yolo 格式必须用 `labels/`，不是 `annotations/` | 改名或换 `--format` |
| `图像-标注覆盖率 x% 低于硬阈值 50%` | 导出中断 / 路径配错 | **修源头，不要带病训练** |
| 覆盖率低于软阈值 90% | — | 只 WARNING 并继续；确认是有意为之 |
| `稀有类 'xxx': 在最小 split 的期望样本量低于 3.0` | 该类样本太少 | 只预警不拦截；补数据或换划分策略 |
| 日志里类别名乱码 | 源数据类别文本编码不是 UTF-8 | 统一源文件编码为 UTF-8 |

### 退出码

`0` 成功 / `1` 数据或参数错误（`FileNotFoundError`、`ValueError`）/ `2` argparse 用法错误。

## 2.4 `odp-validate` —— 数据质检（训练前的验收）

### 用途

对数据集做 **15 项只读 check**，把结果按"最严"聚合成总评与退出码。
它**不改任何数据**；只写报告与章。

```
usage: odp-validate [-h] [--task {detect,segment}] [--run-id RUN_ID]
                    [--no-report]
                    yaml_path
```

### 参数

| 参数 | 必填 | 说明 |
|---|---|---|
| `yaml_path` | ✅ | 数据集 yaml。**相对当前工作目录**解析 |
| `--task` | | `detect` / `segment`；不传则取 yaml 里 `odp_meta.task` |
| `--run-id` | | 指定报告 id（默认时间戳） |
| `--no-report` | | 不写 json/csv 报告 |

### 示例

```powershell
# 在仓库根：必须是完整相对路径
& .\.venv\Scripts\odp-validate.exe apps\platform\configs\datasets\<name>.yaml

# 只打印，不落报告
& .\.venv\Scripts\odp-validate.exe apps\platform\configs\datasets\<name>.yaml --no-report
```

> `--help` 里的示例写的是 `configs/datasets/demo.yaml` —— 那是**相对 `apps/platform/`** 的写法。
> 在仓库根按原样敲不会报"文件不存在"，而是 15 项里 `yaml_schema` 报
> `yaml 无法加载`、`fingerprint_match` / `manifest_lineage` 报 `yaml 缺少 odp_meta.manifest_path`，
> 总评 ERROR、退出码 2。这是最容易踩的一个坑。

### 15 项 check 与严重度

严重度四级：`ERROR`（必须修）> `WARNING`（可训但建议修）> `INFO`（提示）> `PASS`（通过）。
`get_all_checks()` 按名字排序返回：

| # | check | 看什么 |
|---|---|---|
| 1 | `annotation_coverage` | 有有效标注的图占比 |
| 2 | `bbox_within_image` | 框是否越出图像边界 |
| 3 | `box_geometry` | 退化框 / 极端宽高比 |
| 4 | `class_balance` | 各类实例数失衡比 |
| 5 | `class_presence` | 每个类是否在各 split 都出现 |
| 6 | `duplicate_annotations` | 完全重复的标注行 |
| 7 | `fingerprint_match` | manifest 自洽 + yaml 指纹三方一致 |
| 8 | `label_format` | 标签行结构与取值范围 |
| 9 | `manifest_lineage` | 逐样本核验：在契约说的 split 里、内容 sha256 未变 |
| 10 | `orphan_labels` | 孤儿标签（有标签没图） |
| 11 | `pair_existence` | 图像与标签配对 |
| 12 | `split_presence` | 各 split 是否存在（train 缺失即 ERROR） |
| 13 | `split_uniqueness` | 同一图是否跨 split（泄漏） |
| 14 | `stem_collision` | 同名多扩展名冲突 |
| 15 | `yaml_schema` | yaml 能否加载、结构是否合法 |

参考实现里的阈值全部集中在 `common/constants.py`（`PAIR_MISSING_ERROR_RATIO`、
`BBOX_OUT_ERROR_RATIO`、`SPLIT_MIN_IMAGES`、`ORPHAN_LABELS_WARN_RATIO` …），check 里不写魔法数。

### 产物

| 落点 | 说明 |
|---|---|
| `runs/data_validation/<run_id>/report.json` | `run_id` / `yaml_path` / `overall_severity` / `duration_seconds` / `results[{name,severity,summary,details}]`，以及与上次报告有历史可比时追加的 `comparison` 块 |
| `runs/data_validation/<run_id>/results.csv` | 表头 `run_id,check,severity,summary`（`utf-8-sig`） |
| `runs/data_validation/<run_id>/compare.json` | 历史对比结论（`--no-report` 时不产出；无可比历史时写明"首次质检"） |
| `runs/lineage/<data_fingerprint>/validate.stamp.json` | validate 章。**仅在 yaml 能定位到 manifest 时盖** —— 非 `odp-transform` 产出的数据集不盖 |

### 历史对比（`compare.json`）

单次报告只回答"现在有没有问题"，回答不了"**这一次比上一次好了还是坏了**"：
刚修的那个问题真的消失了吗？有没有顺手引入新的？历史对比就是把这两件事说清楚。

两层口径：

| 层 | 怎么判 | 回答什么 |
|---|---|---|
| **check 级** | 用 severity 的变化判定 | "整项修好了 / 整项新坏了 / 仍然有问题" —— severity 是所有 check 的共同语言，不依赖任何 check 的 `details` 结构 |
| **条目级** | 把 `details` 里逐条列出的证据抽成稳定 key 做集合差集 | "整项还没修好，但其中 3 条已经消失了" —— check 级看不见的问题 |

条目身份是 **(check 名, 条目 key)**，severity 不参与条目匹配 ——
否则同一批条目在整项降级（ERROR→WARNING）时会被误报成"全消失又全新出现"。

两条如实标注：

- **历史报告靠 `yaml_path` 精确匹配**（报告里没记数据集名）。同一份数据换个路径会被当成
  不同数据集 —— 这是有意的：路径变了，数据就不是同一批。
- 条目 key 来自各 check 的 preview，而 preview 是**截断**的（各 check 有各自的上限）。
  所以条目级差集是"已列出的那些条目"的差集，不是全量；check 级判定不受影响。

`compare.json` 的键：`has_history` / `previous_run_id` / `current_run_id` /
`previous_overall` / `current_overall` / `improved` / `previous_counts` / `current_counts` /
`trend` / `newly_failing` / `fixed` / `still_failing` / `added_checks` / `removed_checks` /
`new_items` / `disappeared_items` / `persisting_items`。

`INFO` **不算"问题"**（与 `registry` 的定义一致：`passed = severity in (PASS, INFO)`）——
把 INFO 当问题会让每份不均衡数据集的报告都常年飘红。

### 退出码

`ERROR → 2` / `WARNING → 1` / `PASS·INFO → 0`。

### 失败模式与排查

| 现象 | 原因 | 处理 |
|---|---|---|
| 15 项全 `yaml 无法加载`，退出码 2 | yaml 路径相对仓库根写错（少写了 `apps\platform\`） | 用完整相对路径或绝对路径 |
| 日志里 `没有同数据集的历史质检报告, 本次为首次质检(无历史可比)` | 这是第一次质检这份 yaml（或换了路径） | 不是错误；`compare.json` 里 `has_history` 为假 |
| `日志里 历史对比失败, 跳过: ...` | 历史报告损坏/不可读 | 对比失败**不影响质检结论**（护栏吞掉异常），报告照常产出 |
| `fingerprint_match: yaml 缺少 odp_meta.manifest_path` | 这份 yaml 不是 `odp-transform` 产出 | 跑 `odp-transform` 生成合格证；外部 yaml 无法对账是设计使然 |
| `manifest 自身指纹对不上 —— 内容被改过` | `manifest.json` 被手改 | 重跑 `odp-transform`（会换新指纹） |
| `split_uniqueness` 报泄漏 | 同一图出现在多个 split | 重跑 `odp-transform`；不要手工搬文件 |
| 单 check 报 `check 自身崩溃: ...` | `service._safe_run_one` 的护栏把异常降级为 ERROR | 看 `details.exception`，这是 check 的 bug，不是数据问题 |

## 2.5 `odp-gen-config` —— 生成运行配置模板

### 用途

从 Pydantic 配置类**反射**生成带注释的 YAML 模板（值即默认值）。字段分组与元数据来自
`BaseConfig.get_field_groups()` / `get_field_metadata()`，生成器里不另维护一份字段表。

```
usage: odp-gen-config [-h] [-o OUTPUT] [--overwrite] [--no-backup]
                      {train,val,infer}
```

### 参数

| 参数 | 必填 | 默认 | 说明 |
|---|---|---|---|
| `{train,val,infer}` | ✅ | — | 要生成的配置名（choices 直接取自 `CONFIG_REGISTRY` 的键） |
| `-o` / `--output` | | `configs/runtime/<name>.yaml` | 输出路径（实际是 `paths.runtime_config_path(name)`，即 `apps/platform/configs/runtime/<name>.yaml`） |
| `--overwrite` | | 关 | 覆盖已有文件（默认不覆盖） |
| `--no-backup` | | 关 | 覆盖时不备份原文件（默认备份成 `<name>.yaml.bak.<时间戳>`） |

### 示例

```powershell
& .\.venv\Scripts\odp-gen-config.exe train
& .\.venv\Scripts\odp-gen-config.exe infer --overwrite                 # 覆盖前自动备份
& .\.venv\Scripts\odp-gen-config.exe val --overwrite --no-backup       # 显式关掉备份
& .\.venv\Scripts\odp-gen-config.exe train -o .\my_train.yaml
```

### 产物

| 落点 | 说明 |
|---|---|
| `apps/platform/configs/runtime/<name>.yaml` | 模板本体 |
| `apps/platform/configs/runtime/<name>.yaml.bak.<YYYYMMDD_HHMMSS>` | 仅 `--overwrite` 且文件原已存在时产生 |

配置模板尾部会附一段 FAQ，写明 **CLI > YAML > DEFAULT** 的优先级
（见 [ADR-003](architecture/ADR-003-runtime-config-precedence.md)）。

### 失败模式与排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `- 文件已存在, 未覆盖 ...`（退出码 0） | 默认安全行为，**不是错误** | 确认要重建就加 `--overwrite` |
| 找不到生成的配置文件 | 用了 `-o` 指到别处 | 看命令打印的 `✓ 已生成: <路径>` |
| yaml 里出现 `xxx.bak.*` 文件 | 覆盖备份，**已被 `.gitignore` 忽略**（`*.bak.*`） | 可手删 |

## 2.6 `odp-train` —— 训练

### 用途

一次训练 = 顺序 + 验章 + 记录 + 归档：

1. 读运行配置（YAML + CLI 三源合并，打印每个字段的来源）
2. **数据身份核验**（yaml / manifest / transform 章 / validate 章对账），不通过拒绝开训
3. 解析模型引用 → 本地有就用绝对路径，没有交给 ultralytics 自动下载
4. 调用 ultralytics 训练（`project=runs/training`、`name=<run_id>`）
5. 抽指标并记日志
6. 盖 train 章 + 写审计快照 `odp_audit.json`
7. 归档 `best.pt` 到 `models/trained/`

```
usage: odp-train [-h] [--config CONFIG] [--no-archive] [--skip-validate]
                 [--data DATA] [--model MODEL] [--epochs EPOCHS]
                 [--batch BATCH] [--imgsz IMGSZ] [--device DEVICE]
                 [--experiment-name EXPERIMENT_NAME]
```

### 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--config` | `train` | 运行配置引用：`'train'` / `'train.yaml'` / 路径 |
| `--no-archive` | 关 | 不归档 `best.pt` |
| `--skip-validate` | 关 | 跳过数据身份核验（明知故犯：审计会标记） |
| `--data` | `None` | 数据集引用：`'<name>'` / 路径。名字会经 `refs.resolve_dataset_yaml` 解析成 `apps/platform/configs/datasets/<name>.yaml` |
| `--model` | `None`（代码里兜底 `yolov8n.pt`） | 模型引用：`'yolov8n'` / `'yolov8n.pt'` / 路径 |
| `--epochs` | `None` | 训练轮数（不传就用 `train.yaml` 的值，模板默认 100） |
| `--batch` | `None` | int 或 float（`<1` 表示显存占比；`-1` 表示自动） |
| `--imgsz` | `None` | 输入尺寸 |
| `--device` | `None` | `'0'` / `'0,1'` / `'cpu'` |
| `--experiment-name` | `None` | 给人看的实验名（进日志与审计，**不进目录名**） |

> **默认值的唯一真相是 Pydantic 的 `Field(default=...)`**：argparse 侧所有配置字段的 default
> 一律是 `None`（"我没表态"），不会永远覆盖 YAML 里的值。

### 示例

```powershell
& .\.venv\Scripts\odp-train.exe --data <name> --model yolo11n --epochs 100
& .\.venv\Scripts\odp-train.exe --data <name> --model yolo11n --device 0 --batch 16 --imgsz 640
& .\.venv\Scripts\odp-train.exe --data <name> --model yolo11n --experiment-name baseline_v1
& .\.venv\Scripts\odp-train.exe --data <name> --model yolo11n --no-archive --skip-validate
```

### 产物

| 落点 | 说明 |
|---|---|
| `runs/training/<run_id>/` | ultralytics 现场：`args.yaml`、`results.csv`、`results.png`、`Box*.png`、`confusion_matrix*.png`、`labels.jpg`、`train_batch0.jpg`、`val_batch*`、`weights/best.pt`、`weights/last.pt` |
| `runs/training/<run_id>/odp_audit.json` | `run_id` / `created_at` / `artifacts` / `experiment_name` / `data_yaml` / `data_identity` / `config` / `metrics` / `config_sources` |
| `models/trained/<stem>-best.pt` | 归档权重（", 复制而非移动" —— 现场保持完整） |
| `runs/lineage/<data_fingerprint>/train.stamp.json` | train 章 |
| `apps/platform/logging/train/<stem>.log` | 日志，`<stem>` = `train-<run_id>-<dataset>-<model>` |

`<stem>` 规则出自 `common/naming.py` 的 `run_stem()`（阶段-`run_id`-数据集-模型）。
仓库现有的一例：`models/trained/train-20261027-115300-synth-yolo11n-best.pt`。

显存经验值（8GB，本机 RTX 4060 Laptop）：`yolo11n/s` + `batch 16` + `imgsz 640`；
换 m/l 要降 batch 或 imgsz，或用 `--batch 0.7` 这种显存占比写法。

### 失败模式与排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `数据身份核验未通过, 拒绝开训: ...` | 没跑 `odp-validate`，或数据/manifest/yaml 被改过 | 先 `odp-validate`；确实要跳过用 `--skip-validate` |
| `现场里没有 best.pt, 跳过归档` | `save=False` 或训练未产出权重 | 这是**常 warning 而非错误**；检查 `train.yaml` 的 `save` |
| `YAML 配置文件不存在: ...` + 修复指引 | `--config` 指向的 `train.yaml` 不在 `configs/runtime/` | `odp-gen-config train` 先生成，或校正 `--config` |
| `ValidationError ... [来源: ...]` | 字段值非法（合并器已把溯源链嵌进报错信息） | 按 `[来源: x(YAML) ← y(DEFAULT)]` 定位到具体哪一层给了坏值 |
| Windows 上 DataLoader 卡住 / 多进程报错 | DataLoader worker 在 Windows 的已知问题 | 把 `configs/runtime/train.yaml` 的 `workers` 改成 `0` 或 `4` |
| 训练很慢、`odp-init` 里 `CUDA可用: False` | torch 是 `+cpu` 构建 | 按 1.1 重装 cu126 版 |
| 显存不足 OOM | batch / imgsz / 模型过大 | 降 `--batch`、降 `--imgsz`，或换更小的模型 |
| 训练被 Ctrl-C | — | 进程返回失败并把现场保留在 `runs/training/<run_id>/`，日志里有 `现场保留:` |

### 退出码

`0` 成功 / `1` 失败（`TrainResult.success` 为假）。

## 2.7 `odp-eval` —— 独立评估已训好的模型

### 用途

`odp-train` 只在训练过程中顺带打印指标，无法"拿一个已训好的权重单独跑评估、出结构化报告"。
`odp-eval` 提供独立入口、独立配置（`configs/runtime/eval.yaml`）、独立报告，
让论文与答辩里的数字**随时能重跑出来**，每次跑的是同一件事。

流程：数据集 yaml → 只读安全的数据集视图（硬链接，不碰 `data/raw`）→
`model.val(conf=0.001)` → 每类指标 + P-R 曲线 → 逐阈值再 val 一次（置信度扫描）→
推荐阈值 → **匹配式 IoU 分析（另跑一遍 predict）** → 如实警告 → 报告三种出口。

```
usage: odp-eval [-h] --weights WEIGHTS --data DATA [--yaml YAML]
                [--split {val,test,train}] [--imgsz IMGSZ] [--batch BATCH]
                [--device DEVICE] [--workers WORKERS] [--max-det MAX_DET]
                [--conf CONF] [--iou IOU] [--classes NAME [NAME ...]]
                [--exclude-classes NAME [NAME ...]] [--conf-sweep]
                [--no-conf-sweep] [--no-iou-analysis] [--limit LIMIT]
                [--run-id RUN_ID] [--output-dir OUTPUT_DIR] [--no-report]
                [--log-level {DEBUG,INFO,WARNING,ERROR}]
```

### 参数

| 参数 | 必填 | 默认（来自 `eval.yaml` / `EvalSettings`） | 说明 |
|---|---|---|---|
| `--weights` | ✅ | — | 权重文件（如 `models/trained/xxx/weights/best.pt`） |
| `--data` | ✅ | — | 数据集 yaml（Roboflow 导出的也可以，相对路径会自动解析） |
| `--yaml` | | `apps/platform/configs/runtime/eval.yaml` | 评估配置路径 |
| `--split` | | `val` | `val` / `test` / `train` |
| `--imgsz` | | `640` | 必须是 32 的倍数，范围 `[32, 1536]` |
| `--batch` | | `8` | 批大小；`-1` = 自动 |
| `--device` | | `cpu` | `cpu` / `0` / `0,1`（默认 CPU，不抢训练的卡） |
| `--workers` | | `2` | 数据加载进程数 |
| `--max-det` | | `300` | 每图最多保留多少框 |
| `--conf` | | `0.001` | val 用的置信度下限（越低 P-R 曲线越完整） |
| `--iou` | | `0.7` | **NMS 的 IoU 阈值**（不是 mAP 的 IoU 门限） |
| `--classes` | | 无 | 额外出一套"只算这些类"的平均口径 |
| `--exclude-classes` | | 无 | 额外出一套"排除这些类"的平均口径 |
| `--conf-sweep` / `--no-conf-sweep` | | 按配置（默认开） | 是否做置信度扫描（不做可省 N 倍 val 时间） |
| `--no-iou-analysis` | | 按配置（`EvalSettings.iou_analysis` 默认开） | 跳过匹配式 IoU 分析（不做 IoU 分布/直方图，省一遍 predict） |
| `--limit` | | `0` | 只评估前 N 张；**仅供链路冒烟，指标不代表整个 split** |
| `--run-id` | | 时间戳 | 报告 id |
| `--output-dir` | | `runs/detect_eval/<run_id>/` | 报告目录 |
| `--no-report` | | 关 | 只打印，不写 json/csv |
| `--log-level` | | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

`--classes` 与 `--exclude-classes` **不能同时给**（口径必须唯一，否则 `ValueError`）。
拼错的类名不会静默失效 —— `_warn_unknown_classes` 会警告"数据集之外的类名（不会起作用）"。

### 示例

```powershell
# 基本
& .\.venv\Scripts\odp-eval.exe --weights models\trained\<stem>-best.pt --data apps\platform\configs\datasets\<name>.yaml

# 用 GPU、限定 split、指定报告 id
& .\.venv\Scripts\odp-eval.exe --weights models\trained\<stem>-best.pt --data apps\platform\configs\datasets\<name>.yaml --device 0 --split test --run-id baseline_v1

# 额外出一套"排除小类"的平均口径
& .\.venv\Scripts\odp-eval.exe --weights models\trained\<stem>-best.pt --data apps\platform\configs\datasets\<name>.yaml --exclude-classes <小类名>

# 只要一次 val、不要扫描（省时间）
& .\.venv\Scripts\odp-eval.exe --weights models\trained\<stem>-best.pt --data apps\platform\configs\datasets\<name>.yaml --no-conf-sweep

# 不要扫描也不要 IoU 匹配分析（最快，只出 mAP 与每类指标）
& .\.venv\Scripts\odp-eval.exe --weights models\trained\<stem>-best.pt --data apps\platform\configs\datasets\<name>.yaml --no-conf-sweep --no-iou-analysis

# 想要"有意义的匹配口径 P/R"：把 conf 提到推荐阈值再跑
& .\.venv\Scripts\odp-eval.exe --weights models\trained\<stem>-best.pt --data apps\platform\configs\datasets\<name>.yaml --conf 0.25 --no-conf-sweep

# 只跑 20 张确认链路通
& .\.venv\Scripts\odp-eval.exe --weights models\trained\<stem>-best.pt --data apps\platform\configs\datasets\<name>.yaml --limit 20 --no-conf-sweep
```

### 匹配式 IoU 分析（与官方 mAP 口径并列）

`evaluation/iou.py` 给的是**另一套口径**，回答"框贴得有多紧"，与 mAP 回答的
"能不能把目标找到"分开说。要点：

| 项 | 说明 |
|---|---|
| 预测来源 | `model.predict(conf=settings.conf, iou=settings.iou, imgsz, device, max_det, stream=True)`；阈值都取自配置，报告里可追溯 |
| 匹配规则 | **只允许同类**，按 IoU **降序贪心一对一**；排序键带下标，结果可复现 |
| 匹配门限 | `IOU_MATCH_THRESHOLD = 0.5`（模块常量，无 CLI 开关） |
| TP/FP/FN | 配上的对 = TP；没配上的预测 = FP；没配上的真值 = FN |
| IoU 读数 | 匹配对的 `mean` / `median` / `min` / `max` |
| 直方图 | `0.50-0.60`、`0.60-0.70`、`0.70-0.80`、`0.80-0.90`、`0.90-1.00` 五档；`share` 的分母是**匹配对数** |
| 逐图明细 | `per_image`：每图的 `n_ground_truth` / `n_predictions` / `matched_pairs` / `fp` / `fn` / `mean_iou` |
| 匹配对明细 | `pairs`（同时落成 `iou_pairs.csv`） |

⚠️ 两条必须记住的口径纪律：

1. **匹配口径的 P/R/F1 ≠ 官方 mAP 口径的 P/R**（前者是单门限上的点估计，后者是对 P-R 曲线积分）。
   两者都报，绝不相加、不相除；报告渲染里也把两块分开写。
2. **`conf` 太低会让匹配口径的 P 失真**：`eval.yaml` 默认 `conf = 0.001` 是给 val 用的
   （为让 P-R 曲线完整）。用它跑匹配，NMS 后几百个低置信度框全成 FP，P 会低到没有意义。
   **IoU 分布本身不受影响。** 当 `conf <= 0.01` 时 `iou_analysis.note` 会写明这件事
   （`IOU_LOW_CONF_HINT = 0.01`）；要看部署阈值的匹配口径请提高 `--conf`。

### 产物

`runs/detect_eval/<run_id>/`（目录名 = `<task_label>_eval`，`task_label` 默认 `detect`）：

| 文件 / 目录 | 说明 |
|---|---|
| `report.json` | 完整结构。顶层键：`run_id` `weights` `data_yaml` `split` `duration_seconds` `overall` `per_class` `stratified` `conf_sweep` `recommended_threshold` `warnings` `dataset` `settings` `speed_ms` `pr_curves` `iou_analysis` |
| `metrics.csv` | 每类一行。表头 `run_id,class_id,class_name,evaluated,instances,AP50,AP50_95,precision,recall` |
| `conf_sweep.csv` | 每阈值一行。表头 `run_id,conf,precision,recall,f1` |
| `iou_pairs.csv` | 每个匹配对一行。列 `run_id,image,class_id,class_name,iou,confidence,gt_x1,gt_y1,gt_x2,gt_y2,pred_x1,pred_y1,pred_x2,pred_y2`。只写匹配对；FP/FN 清单在 `report.json` 的 `iou_analysis.per_image` 里。`--no-iou-analysis` 时不产出 |
| `_dataset_view/data.yaml` | 规范化数据集 yaml（只用相对路径，`train`/`val` 都指向视图，`test` 额外写） |
| `_dataset_view/images/` `_dataset_view/labels/` | 只读视图；优先硬链接（零额外字节），跨卷退化为复制并在 `dataset.copied_files` 记数 |

CLI 最后会打印一行摘要（`report.summary_line()`），形如
`mAP50=0.xxxx  mAP50-95=0.xxxx  验证集 N 张 / M 个真值框`；
有警告时再打印 `注意: 有 K 条警告, 详见报告与日志`。

指标定义、口径差别与推荐阈值规则见 [README 的「评估指标说明」](../README.md#评估指标说明)
与 [ADR-005](architecture/ADR-005-progress-retention-eval-and-label-dir-fix.md)。

### 失败模式与排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `权重文件不存在: ...` → 退出码 2 | `--weights` 路径错 | 用 `models\trained\<stem>-best.pt` |
| `数据集 yaml ... 里的 'val' 路径解析不出来` | split 目录在所有候选位置都不存在 | 报错里附了全部候选路径，逐条对照；Roboflow 的 `../xxx` 会被自动剥前缀重试 |
| `验证集只有 N 张图（低于 100 张）` | 样本太少，mAP 抖动大于结论 | 不是错误；报数时给区间，别把小数点后一位当精确值 |
| `以下类别的真值框少于 100 个: ...` | 小类 AP 不稳定 | 同时报"排除小类后的平均"那套口径 |
| `有 K 个类没有真值（...）` | 这些类在本 split 一个真值框都没有 | 报告里标 `—` 而不是 0；确认是不是 split 选错了 |
| `本次只评估了前 N 张图（--limit）` | 用了 `--limit` | 指标不代表整个 split，只能用于验证链路 |
| `数据集视图里有 N 个文件未能硬链接（跨卷）` | 视图目录与数据集不在同一卷 | 只是多占磁盘；把 `--output-dir` 指到同一卷可避免 |
| conf 阈值扫出全 0 / `f1: null` | 模型在该数据上确实没检出（或置信度全低于最低阈值） | 先看 `metrics.csv` 的每类 `instances` 与 `AP50`，再确认 split 与权重是不是配套的 |
| `iou_analysis` 里 P 极低、FP 很大 | 用了 val 级极低 `conf`（默认 0.001），低置信度框全成 FP | 这是**阈值造成的**，不是框定得不准（IoU 分布不受影响）；提高 `--conf` 再跑 |
| 报告里没有 `iou_analysis` / 没有 `iou_pairs.csv` | 用了 `--no-iou-analysis` 或配置里 `iou_analysis: false` | 去掉该开关即可 |
| `结果里没有 orig_shape, 无法对该图做 IoU 匹配` | 该张图的预测结果缺原图尺寸，归一化标签换不成绝对坐标 | 该图被如实跳过（不猜），其余图不受影响 |
| `exclude_classes 与 include_classes 不能同时给` | `--classes` 与 `--exclude-classes` 同时给了 | 二选一 |

### 退出码

`0` 成功 / `1` 用法或参数错误（配置有问题、参数有问题）/ `2` 评估失败（找不到权重或数据集、
模型加载失败、口径冲突）。

## 2.8 `odp-infer` —— 推理

### 用途

端到端推理：帧源（摄像头 / 视频 / 单图 / 图片夹）→ 逐帧预测 → 美化绘制 + HUD →
输出到 sink（本地文件 / 不落盘）→ 统计与审计。

```
usage: odp-infer [-h] [--source SOURCE] [--model MODEL] [--yaml YAML]
                 [--pipeline-yaml PIPELINE_YAML] [--name EXPERIMENT_NAME]
                 [--conf CONF] [--iou IOU] [--imgsz IMGSZ] [--max-det MAX_DET]
                 [--classes CLASSES [CLASSES ...]] [--device DEVICE]
                 [--batch BATCH] [--show | --no-show] [--save | --no-save]
                 [--no-viz] [--no-hud] [--warmup WARMUP]
                 [--log-level {DEBUG,INFO,WARNING,ERROR}]
```

### 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--source` | `None` | 输入源：摄像头号（`0`/`1`）/ 视频文件 / 图片或目录 / RTSP。不传则用 `infer.yaml` |
| `--model` | `None` | 模型文件名（`best.pt` / `yolo11n.pt`）。不传则用 `infer.yaml` |
| `--yaml` | `None` | `configs/runtime/infer.yaml` 路径 |
| `--pipeline-yaml` | `None` | 帧源 + 美化 + 输出落盘 yaml 路径，默认 `apps/platform/configs/runtime/infer_pipeline.yaml` |
| `--name` | `None` | 输出子目录名（重复自增：`predict` / `predict2` / …） |
| `--conf` | 按配置（`infer.yaml` 模板里是 `0.25`） | 置信度阈值（0~1） |
| `--iou` | 按配置（模板 `0.7`） | NMS IoU 阈值（0~1） |
| `--imgsz` | 按配置（模板 `640`） | 推理输入尺寸 |
| `--max-det` | 按配置（模板 `300`） | 单图最大检测数 |
| `--classes` | `None` | 只保留这些**类别 ID**（整数） |
| `--device` | 按配置（模板 `null` = 自动） | `cpu` / `0` / `0,1` / `mps` |
| `--batch` | 按配置（模板 `16`） | 批大小（视频 / 图片夹） |
| `--show` / `--no-show` | 按配置（模板 `show: false`） | 弹窗显示 / 关掉弹窗（互斥） |
| `--save` / `--no-save` | 按配置（模板 `save: true`） | 结果存盘 / 不存盘（互斥） |
| `--no-viz` | 关 | 关掉美化，用 YOLO 原生 `plot()` |
| `--no-hud` | 关 | 画面不叠 HUD 信息面板 |
| `--warmup` | `0` | 启动丢弃前 N 帧（摄像头帧率不稳） |
| `--log-level` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

> `--no-viz`、`--no-hud`、`--warmup`、`--yaml`、`--pipeline-yaml`、`--log-level`
> **不进配置合并**（它们是 CLI 自己的行为开关，不是 `YOLOInferConfig` 字段）。
> 其余字段只取非 `None` 的值交给合并器，让 Pydantic 默认值兜底。

### 示例

```powershell
# 图片
& .\.venv\Scripts\odp-infer.exe --source .\test.jpg --model models\pretrained\yolo11n.pt --no-show --save

# 视频，输出子目录叫 line_01
& .\.venv\Scripts\odp-infer.exe --source D:\videos\line.mp4 --model models\trained\<stem>-best.pt --no-show --save --name line_01

# 图片目录
& .\.venv\Scripts\odp-infer.exe --source .\frames\ --model models\trained\<stem>-best.pt --no-show --save

# 摄像头弹窗
& .\.venv\Scripts\odp-infer.exe --source 0 --model yolo11n.pt --show

# RTSP 流
& .\.venv\Scripts\odp-infer.exe --source rtsp://user:pass@host/stream --model yolo11n.pt --no-show --save

# 换原生绘制 + 不要 HUD
& .\.venv\Scripts\odp-infer.exe --source .\test.jpg --model yolo11n.pt --no-viz --no-hud --save
```

### 产物

`runs/<task>_infer/<name>/`（`task` 来自 `infer.yaml` 的 `task`，默认 `detect`；`name` 默认 `predict`）：

| 文件 | 说明 |
|---|---|
| `odp_audit.json` | 顶层键：`mode` `config` `merger` `pipeline` `stats` `result_summary`；`pipeline` 里含 `output`（生效值）与 `output_sources`（来源） |
| `output.mp4` | 视频 / 摄像头源。文件名 / 容器 / 编码器由 `output` 段决定（默认 `output.mp4` + `mp4v`） |
| `<原图名>.jpg` | 图片源，每帧一张；前缀与扩展名由 `output` 段决定（默认无前缀 + `jpg`） |
| — | 用 `--no-save` 时不落盘（走 `NullSink`） |

### 输出落盘配置（`infer_pipeline.yaml` 的 `output` 段）

落盘参数**不写死在代码里**（`inference/sinks.py` 只负责消费配置）。
字段、默认值、原先写死的位置：

| 字段 | 默认值 | 原写死位置 | 说明 |
|---|---|---|---|
| `filename_prefix` | `""` | （新增能力） | 产物文件名前缀，空串=不加 |
| `video_basename` | `"output"` | `sinks.py` `"output.mp4"` | 视频文件名主干（不含扩展名） |
| `video_format` | `"mp4"` | `sinks.py` `"output.mp4"` | 视频容器：`mp4` / `avi` / `mov` / `mkv` |
| `image_format` | `"jpg"` | `sinks.py` `f"{stem}.jpg"` | 图片扩展名：`jpg` / `jpeg` / `png` / `bmp` / `webp` / `tif` / `tiff` |
| `image_quality` | `95` | `cv2.imwrite` 未传参（OpenCV 默认即 95） | JPEG 质量 1–100，**仅 jpg/jpeg 生效** |
| `anonymous_frame_template` | `"frame_{index:06d}"` | `sinks.py` `f"frame_{frame.info.frame_index:06d}"` | 帧源没给原文件名时的兜底命名，`{index}`=帧序号 |
| `video_fourcc` | `"mp4v"` | `sinks.py` `VideoWriter_fourcc(*"mp4v")` | 编码器四字符码：`mp4v` / `MJPG` / `XVID` / `avc1` … |
| `video_fps` | `null`（= `auto` / `source`） | `sinks.py` `float(frame.info.fps)` | `null`=跟随源帧率；数字=固定帧率 |
| `video_fps_fallback` | `30.0` | `sinks.py` `... else 30.0` | 源帧率缺失或为 0 时的兜底 |
| `overwrite_existing` | `true` | `cv2.imwrite` / `VideoWriter` 的隐式覆盖 | `false`=同名跳过（日志里会说明跳过了哪个文件） |
| `keep_audit_json` | `true` | `service.py` 无条件写 | `false`=不写审计快照（日志里说明是按配置跳过） |
| `audit_filename` | `"odp_audit.json"` | `service.py` `output_dir / "odp_audit.json"` | 审计快照文件名 |

**默认值 = 改造前 `LocalFileSink` 的实际行为**：`output` 段整段删掉、或字段不填，
产物文件名与字节都与改前完全一致（中文注释/示例见该 yaml 的 `output` 段）。
改配置后，`infer_pipeline.yaml` 的生效值与来源会以“配置参数信息 / 配置覆盖情况”两张表
打进本次推理日志（`common/config_log`），并写进 `odp_audit.json` 的 `pipeline.output` /
`pipeline.output_sources`。

示例（换编码器 + 加前缀 + 不覆盖旧产物）：

```yaml
output:
  filename_prefix: "line01_"
  video_format: "avi"
  video_fourcc: "MJPG"      # 容器与编码器要配对，否则 VideoWriter 打不开(日志会报 error)
  image_quality: 80
  overwrite_existing: false
```

> 写错类型/取值不会让整次推理失败：`inference/pipeline_config.py` 逐个字段
> 打 warning 并只让该字段回落默认值（未知字段也会 warning 后忽略）。

日志：`apps/platform/logging/infer/infer_<时间戳>.log`。
`cli/infer.py` 用的是 `get_logger(base_path=LOGGING_DIR, log_type="infer")`
—— **绝对路径（`paths.LOGGING_DIR`）**，不写 CWD 相对的 `"logs"`，
所以日志不会落在"你在哪个目录敲命令"上。
`common/log_rename.py` 里"存盘时把日志搬进本次输出目录"目前是**桩**（`pass`），不做实际改名。

推理统计打印内容：处理帧数、检测总数、平均延迟（ms/帧）、
帧率（捕获 / 推理 / 渲染 / loop / 当前）、模型 `speed(ms)`、各类别检测数。

### 失败模式与排查

| 现象 | 原因 | 处理 |
|---|---|---|
| 触发联网下载 / `Download failure` | `--model` 给的名字不在搜索目录里 | 搜索目录只有 `models/checkpoints/` 与 `models/pretrained/`，且不补 `.pt` 后缀；自训练权重在 `models/trained/`，**必须给全路径** |
| 服务器上卡住 | `--show` 弹窗 | 加 `--no-show` |
| 画面没有框 | 用了 `--no-viz` 但美化本身有问题？先分辨：美化默认走 `BeautifyVisualizer.draw()`，`--no-viz` 走原生 `plot()` | 两边都试一次；`infer_pipeline.yaml` 的 `visualization.enabled: false` 等价于关美化 |
| 输出视频特别大 | 默认编码器 `mp4v` 压缩率低 | 改 `infer_pipeline.yaml` 的 `output.video_fourcc`（如 `MJPG` / `avc1`，注意与 `video_format` 配对），不必改代码；见 [ADR-004](architecture/ADR-004-inference-output-and-viz.md) |
| `cv2.VideoWriter 未能打开`（error） | `output.video_fourcc` 与 `output.video_format` 不配对，或本机不支持该编码器 | 按日志里的 fourcc/容器提示改成配对组合（如 `MJPG`+`avi`、`mp4v`+`mp4`），或恢复默认 |
| `output.xxx 取值无效 (...)，回落默认值`（warning） | `output` 段某个字段类型/取值写错 | 只影响该字段（其余字段照常生效）；按提示改成合法值，或删掉该字段用默认 |
| `output 段里的未知字段 'xxx' 已忽略`（warning） | 字段名拼错，或用了本版本没有的字段 | 对照手册的 `output` 字段表改名 |
| `pipeline yaml 不存在, 用默认配置: ...` | 配置文件不在或 `--pipeline-yaml` 指错 | 不是致命错误；确认路径或直接跑默认 |
| `camera 配置无效, 走默认: ...` | `infer_pipeline.yaml` 的 `frame_source.camera` 字段非法 | `CameraConfig` 是 `extra=forbid`，按提示删掉多余字段 |
| 中文标签乱码 | `label_mapping` 用了中文，但 cv2 的 Hershey 字体不支持中文 | Hershey 字体不读外部字体文件；需要中文标签要先解决字体渲染，`font_path` 字段目前只是预留 |

### 退出码

`0` 成功 / `1` 失败（失败时 stderr 打印 `推理失败: <error>` 与日志路径）。

## 2.9 `odp-clean` —— 按保留策略清理 `runs/`

### 用途

`runs/` 会随时间累积（推理输出、质检报告、划分 manifest、训练现场）。
`odp-clean` 提供**可解释、可预测**的保留策略：

> 在每个受管子目录里保留：**最新的 `keep_latest` 个**，以及**修改时间在 `max_age_days` 之内的**；
> 其余才列入清理计划。**两个条件都满足才删** —— 宁可留着。

于是"跑得多"不会把最近的产物删掉，"跑得久"也不会把最新的删掉。
`runs/lineage/` 是**硬编码豁免**（不是"默认参数恰好没选到"）：删了数据身份核验就对不上账，
而且无法重建（重跑 D3 会换 manifest 指纹）。

```
usage: odp-clean [-h] [--yes] [--keep-latest KEEP_LATEST]
                 [--max-age-days MAX_AGE_DAYS] [--protect [PROTECT ...]]
                 [--summary] [--runs-dir RUNS_DIR]
```

### 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--yes`（别名 `--apply`） | 关 | 真的执行删除（不加则只预演） |
| `--keep-latest` | `5` | 每个子目录至少保留最新的 N 个 |
| `--max-age-days` | `30` | 这个天数内的一律不动 |
| `--protect` | 无 | 额外豁免的子目录名（空格分隔） |
| `--summary` | 关 | 只打印现状概览，不计算清理计划 |
| `--runs-dir` | 仓库 `runs/` | 运行现场根目录（便于测试 / 离线巡检） |

策略参数由**代码默认值**（`common/retention.py` 的 `DEFAULT_KEEP_LATEST` /
`DEFAULT_MAX_AGE_DAYS`）与命令行提供，**不读取 runtime yaml**。

### 示例

```powershell
& .\.venv\Scripts\odp-clean.exe --summary                       # 只看现状
& .\.venv\Scripts\odp-clean.exe                                 # 预演（默认不删）
& .\.venv\Scripts\odp-clean.exe --keep-latest 3 --max-age-days 7
& .\.venv\Scripts\odp-clean.exe --yes                           # 真删
& .\.venv\Scripts\odp-clean.exe --yes --protect detect_infer    # 额外豁免某类现场
& .\.venv\Scripts\odp-clean.exe --runs-dir D:\tmp\runs --summary # 巡检别的目录
```

### 产物

- `--summary` / 预演：不写任何文件。
- `--yes`：无新产物，只删；同时写日志 `apps/platform/logging/clean/`。

现状概览的真实输出（本机仓库，2026-10-27）：

```
运行现场根目录: D:\ODPlatform2\runs
当前总占用: 54.4 MB
  data_pipeline        20 个现场        0.1 MB
  data_validation       4 个现场        0.0 MB
  detect               44 个现场        0.0 MB
  detect_eval           1 个现场        0.3 MB
  detect_infer         17 个现场       32.4 MB
  lineage               2 个现场        0.0 MB   [豁免]
  training              4 个现场       21.7 MB
```

### 失败模式与排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `策略参数无效: ...` → 退出码 1 | `--keep-latest < 1` 或 `--max-age-days < 0` | 按提示修正 |
| `[预演] 没有删除任何东西。` | 没给 `--yes` | 确认清单后加 `--yes` |
| `没有需要清理的现场目录（策略内全部保留）` | 都还在保留窗口内 | 调小 `--keep-latest` / `--max-age-days` |
| 某些现场被"跳过" | 逐个候选都先过围栏校验（`_assert_within`）；符号链接一律拒绝递归删除 | 看日志里的 `跳过 ...: 原因` |

### 退出码

`0` 成功 / `1` 用法或策略参数错误。

---

## 2.10 `odp-demo` —— Gradio 演示界面

### 用途

把「上传一张图 → 看框、看检出表、看汇总」这条路走通，供答辩 / 演示 / 人工抽查用：

上传图片（单张或多张）→ 逐张推理 → 展示**标注后的结果图**（`BeautifyVisualizer` 画的框 +
`label_mapping` 后的品种名）→ 展示**检出明细表**（品种 / 置信度 / 框坐标）与**汇总**
（总检出数、各品种计数）→ 可随时换权重、调 `conf` / `iou` / `imgsz` / `device`、开关美化绘制。

它**不重写任何推理逻辑**，复用关系如下：

| 环节 | 复用的东西 | 与谁同源 |
|---|---|---|
| 推理参数默认值 | `runtime_config.build_infer_config()`（D5 三源合并 + Pydantic 校验） | `odp-infer` 读的同一份 `configs/runtime/infer.yaml` |
| 读图 | `frame_source.create_frame_source()`（D8 帧源） | `odp-infer` 的帧源 |
| 画框 / 标签 | `visualization.BeautifyVisualizer` + `DrawStyle.from_image_size()`（D7 的交付本体） | `inference/service._FrameProcessor.draw()` 的同一套动作 |
| 配色 / 标签映射 / 字体 / 美化开关 | `inference.pipeline_config.load_pipeline_config()` | `configs/runtime/infer_pipeline.yaml` 的 `visualization` 段 |
| 权重名字 → 路径 | `common/refs.py`（`models/trained` → `pretrained` → `checkpoints`） | 全局路径 SSoT 纪律 |
| 界面旋钮 / 端口 / 张数上限 | `configs/runtime/demo.yaml`（`YOLODemoConfig`，可用 `odp-gen-config demo` 生成） | D5 的三源合并 + 溯源 |

> **为什么没有直接调 `InferService.predict()`？** 它的两个业务接缝都拿不到「每个框的类别 /
> 置信度 / 坐标」：`sinks.OutputSink.write(frame, annotated)` 只给画好的图；
> `hooks.InferHooks` 的 `FrameEvent.detections` 在 `inference/pipeline.py` 里写死为 `None`。
> 走 service 只能拿「标注图 + 每类计数」，拿不到本命令要求的**检出表**；而且 service 每次调用
> 都会在 `runs/<task>_infer/<name>/` 下新建输出目录（名字默认还自增：`predict` / `predict2` / …）
> —— 对一个"点一下就推一次"的界面来说，那是每点一次留一份运行现场。所以这里保留**同一套配置与
> 绘制**，只把最外层驱动换成单张图直调（`model([BGR 图], **predict_kwargs)` →
> `BeautifyVisualizer.draw()`，`predict_kwargs` 的取法与 `InferService.predict()` 逐行同构）。
> 结果与 `odp-infer --source <同一张图> --model <同名权重>` 等价，且**不产 `runs/` 现场**。

### 安装（可选依赖）

`gradio` 是**可选依赖组 `ui`**，引擎本体（`odp-infer` / `odp-train` / …）不需要它：

```powershell
# 在仓库根执行
& .\.venv\Scripts\python.exe -m pip install -e "./apps/platform[ui]"
# 国内源
& .\.venv\Scripts\python.exe -m pip install -e "./apps/platform[ui]" -i https://mirrors.aliyun.com/pypi/simple/
```

CI 不装 gradio 也能跑 pytest：`visualization/gradio_app.py` **顶层不 import gradio**
（`main()` 里才延迟 import），纯函数（检出明细 → 表格行 / 汇总、权重按名解析、配置三源合并）
在没有 gradio 的环境里照常可测；未安装 gradio 时 `main()` 打印中文指引并以退出码 `2` 结束。

### 参数

```
usage: odp-demo [-h] [--server-name SERVER_NAME] [--server-port SERVER_PORT]
                [--share] [--inbrowser] [--max-images MAX_IMAGES]
                [--weight WEIGHT] [--pipeline-yaml PIPELINE_YAML]
                [--conf CONF] [--iou IOU] [--imgsz IMGSZ] [--device DEVICE]
                [--no-viz] [--log-level {DEBUG,INFO,WARNING,ERROR}]
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--server-name` | 按 `demo.yaml` 的 `server_name`（模板 `127.0.0.1`） | 界面绑定地址（`0.0.0.0` = 局域网可访问） |
| `--server-port` | 按 `demo.yaml` 的 `server_port`（模板 `7860`） | 界面端口 |
| `--share` | 按 `demo.yaml` 的 `share`（模板 `false`） | 生成 gradio 公网分享链接（经第三方服务，慎用） |
| `--inbrowser` | 按 `demo.yaml` 的 `inbrowser`（模板 `false`） | 启动后自动打开浏览器 |
| `--max-images` | 按 `demo.yaml` 的 `max_images`（模板 `8`） | 单次最多处理几张图（超出截断并在结果区提示） |
| `--weight` | 按 `demo.yaml` 的 `weight`（模板 `train-20261027-124755-fish-yolo11n-best`） | 权重**名字**（可带/不带 `.pt`），由 `common/refs.py` 解析；界面下拉里可换 |
| `--pipeline-yaml` | 按 `demo.yaml` 的 `pipeline_yaml`（模板 `null` = 与 `odp-infer` 同一个落点） | 美化 / 帧源 yaml |
| `--conf` | 按 `infer.yaml` 的 `conf`（模板 `0.25`） | 置信度阈值 |
| `--iou` | 按 `infer.yaml` 的 `iou`（模板 `0.7`） | NMS IoU 阈值 |
| `--imgsz` | 按 `infer.yaml` 的 `imgsz`（模板 `640`） | 推理输入尺寸 |
| `--device` | 按 `infer.yaml` 的 `device`（模板 `null` = 自动） | `cpu` / `0` / `0,1` / `mps`；界面上的 `auto` 等价于 `null` |
| `--no-viz` | 关（**不覆盖配置**） | 关掉美化绘制，退回 YOLO 原生 `plot()`；要不要美化默认由 `infer_pipeline.yaml` 的 `visualization.enabled` 说了算 |
| `--log-level` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

命令行参数与其他命令同一纪律：**只把非 `None` 的值当"CLI 源"**交给配置合并
（`CLI > YAML > DEFAULT`）；`--log-level` / `--no-viz` 是 CLI 自己的行为开关、不进配置。

#### 默认值的唯一来源（界面代码里没有硬编码的默认值）

| 界面可调项 | 默认值的唯一来源 |
|---|---|
| `conf` / `iou` / `imgsz` / `device` | `configs/runtime/infer.yaml` —— 经 D5 `build_infer_config()` 三源合并；界面把控件值当"CLI 源"传进去 |
| `max_det` / `batch` / `classes` / `augment` / `agnostic_nms` / `retina_masks` | 同上（界面**不暴露**这些项，由配置生效，与 `odp-infer` 一致） |
| 标题 / 绑定地址 / 端口 / `share` / `inbrowser` / 单次张数上限 / 默认权重名 / 美化 yaml 路径 | `configs/runtime/demo.yaml` ← `runtime_config/demo.py` 的 `YOLODemoConfig` 的 `Field(default=...)`（可用 `odp-gen-config demo` 重新生成） |
| 配色 / 标签映射 / 字体 / 美化开关 | `configs/runtime/infer_pipeline.yaml` 的 `visualization` 段（界面不另写一套配色） |
| 权重文件路径 | 名字 → `models/trained` → `models/pretrained` → `models/checkpoints`（`common/refs.py` 按名解析，源码里不出现盘符 / 绝对路径） |
| 权重下拉候选 | 动态扫描上面三个目录的 `*.pt`（不写死清单） |
| 滑块 `min` / `max` | `YOLOInferConfig` 字段的 Pydantic 约束（`ge` / `le`）反射而来（步长 = 区间 / 100，纯界面刻度） |
| 日志落点 | `common/logging_utils.get_logger(base_path=paths.LOGGING_DIR, log_type="demo")` |

界面上「配置来源」折叠面板会把上面这张表按**生效值 + 来源（`YAML` / `DEFAULT`）**实时列出来 ——
来源取自 D5 合并器记下的溯源链，不是猜的。

> `infer_pipeline.yaml` 的 `output` 段（落盘文件名 / 编码器 / 审计快照）对 `odp-demo`
> **不生效**：这个界面只在浏览器里出结果，不落盘。

### 示例

```powershell
# 用 demo.yaml 的默认值起界面（默认权重 = 已训好的 fish 权重）
& .\.venv\Scripts\odp-demo.exe

# 换端口 / 换权重 / 指定设备
& .\.venv\Scripts\odp-demo.exe --server-port 7999 --weight train-20261027-121740-synth-yolo11n-best --device 0

# 临时改这次界面的推理参数默认值（界面上还能再拖）
& .\.venv\Scripts\odp-demo.exe --conf 0.4 --imgsz 512

# 对比一下"美化绘制"与 YOLO 原生绘制
& .\.venv\Scripts\odp-demo.exe --no-viz

# 重新生成界面配置模板（覆盖前自动备份）
& .\.venv\Scripts\odp-gen-config.exe demo --overwrite
```

### 产物

- **不产 `runs/` 现场**：结果只回到浏览器（图片 + 表格 + 汇总），不落盘、不写审计快照。
- 日志：`apps/platform/logging/demo/demo_<时间戳>.log`（走 `paths.LOGGING_DIR`，绝对路径，
  不随工作目录漂移）。启动时会把生效的界面配置（`to_audit_snapshot()`）与推理参数写进日志，
  便于复盘「这次界面是什么配置起的」。
- 权重、配置、数据一律只读；演示过程不修改它们。

### 失败模式与排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `odp-demo 需要可选依赖 gradio…`（退出码 2） | 没装 `[ui]` 可选依赖 | `pip install -e "./apps/platform[ui]"` |
| `配置准备失败：…`（退出码 2） | `demo.yaml` 缺失，或字段名写错（`extra="forbid"` 会当场报错） | `odp-gen-config demo` 生成模板；字段名照模板 |
| `找不到权重文件：…`（结果区红字） | `--weight` / `demo.yaml` 的名字在三个搜索目录里都没有 | 用界面下拉里的候选名；或先 `odp-train` 训一个 |
| `界面启动失败：…`（退出码 1） | 端口被占用 | 换 `--server-port` |
| 结果区 `处理失败 / 读取图片失败` | 上传的不是受支持格式或文件损坏 | 换 `jpg/png/bmp/webp` |
| 只处理了前 N 张 | 触到 `max_images` 上限 | 改 `demo.yaml` 的 `max_images` 或分批上传 |
| 标注图上没有框 | `infer_pipeline.yaml` 的 `visualization.enabled` 被改成了 `false`，或界面取消勾选了「美化绘制」 | 检查该配置项；勾回「美化绘制」 |
| 中文标签乱码 | 与 `odp-infer` 同一条：cv2 的 Hershey 字体不支持中文，且 `label_mapping` 只映射 COCO 名（fish 的 33 个品种名不在其中，所以默认显示英文品种名） | 品种名保持英文；要中文标签需先解决字体渲染（`font_path` 目前只是预留） |
| `pipeline yaml 不存在, 用默认配置: …` | 配置不在或 `--pipeline-yaml` 指错 | 非致命；确认路径或直接跑默认 |

### 退出码

`0` 正常（界面被 Ctrl+C 结束后返回）/ `1` 界面启动失败（端口占用等）/
`2` 未安装 gradio、配置缺失或非法、参数用法错误。

---

## 三、产物与追溯一览

```
runs/
├── data_pipeline/<run_id>/manifest.json              # 谁、用什么配方切了哪些样本
├── data_validation/<run_id>/report.json | results.csv | compare.json   # 质检结论 + 与上次的对比
├── training/<run_id>/                                # 训练现场
│   ├── odp_audit.json                                # 配置快照 + 各字段来源 + 指标 + 数据身份
│   ├── args.yaml / results.csv / results.png / Box*.png / confusion_matrix*.png
│   └── weights/{best.pt,last.pt}
├── detect_eval/<run_id>/                             # 评估报告
│   ├── report.json / metrics.csv / conf_sweep.csv / iou_pairs.csv
│   └── _dataset_view/{data.yaml,images/,labels/}
├── detect_infer/<name>/                              # 推理现场
│   ├── odp_audit.json                                # 名字/开关可配(output 段)
│   └── output.mp4 | <图片名>.jpg                      # 名字/格式/编码器可配(output 段)
├── lineage/<data_fingerprint>/                       # ★ 数据流转卡（odp-clean 永不动）
│   ├── transform.stamp.json     (odp-transform 盖)
│   ├── validate.stamp.json      (odp-validate 盖)
│   └── train.stamp.json         (odp-train 盖)
└── detect/val*/                                      # ultralytics 自己建的 val 现场
```

**章的两条铁律**（`common/stamp.py`）：

1. 每一方**只写自己那一联**，只主张自己亲自取证的事实。盖章时的 `data_fingerprint`
   必须是本环节**自己算的**，绝不读别人的章抄过来 —— 转述不是验证。
   `stamp` 模块因此**故意不提供**"读回来改一改再存"的接口。
2. 目录名 = `data_fingerprint`。**数据一变，指纹变，指向空目录，上游的章自动查不到 =
   自动失效**，不需要额外的作废机制；数据没变则章还在，下游不必重复劳动。

章文件内容：`stage` / `data_fingerprint` / `run_id` / `tool_version` + 各环节的 payload。
当前版本：`TOOL_VERSION = "odp-transform/0.4"`、`VALIDATOR_VERSION = "odp-validate/0.4"`、
`TRAINER_VERSION = "odp-train/0.4"`。

## 四、排障对照表（跨命令）

| 症状 | 原因 | 处理 |
|---|---|---|
| 控制台中文/符号乱码 | 代码页不是 UTF-8 | `chcp 65001`；或设 `$env:PYTHONIOENCODING='utf-8'` |
| `CUDA可用: False` / 推理训练很慢 | torch 装成了 `+cpu` 构建 | 按 1.1 换 cu126 构建（注意驱动版本上限：cu126 要求驱动 ≥ 560） |
| `odp-validate` 15 项全 `yaml 无法加载` | yaml 路径少了 `apps\platform\` 前缀 | 用完整相对路径或绝对路径 |
| `Download failure ...` / 模型下载失败 | `--model` 的名字本地找不到，转去联网 | 给全路径，或把权重放进 `models/pretrained/` |
| `odp-transform` 报缺标注目录 | yolo 格式用了 `annotations/` | 必须是 `labels/` |
| `odp-transform` 报覆盖率过低 | 导出中断 / 路径配错 | 修源头，不要带病训练 |
| 训练被拦"数据身份核验未通过" | 没跑 `odp-validate`，或数据被改 | 先 `odp-validate`；确实要跳过用 `--skip-validate` |
| 训练时 Windows 卡住/多进程错 | DataLoader worker | 把 `train.yaml` 的 `workers` 设为 `0` 或 `4` |
| 显存不足（OOM） | batch / imgsz / 模型过大 | 降 `--batch`（或写 `0.7` 这类占比）与 `--imgsz` |
| `--device 0` 报错 | 本机无 CUDA 或卡号不对 | 用 `--device cpu` 或去掉该参数（自动选卡） |
| `runs/` 吃掉磁盘 | 历史现场累积 | `odp-clean`（先不加 `--yes` 看预演） |
| 日志在哪 | 各命令不同，但**都在绝对路径下**（不随工作目录漂移） | `odp-init/transform/validate/train/eval/infer/clean` → `apps/platform/logging/<log_type>/`；`odp-reset` → `.odp-meta/logs/reset_project/` |

## 五、日常检查清单

- [ ] `odp-init` 输出里 `CUDA可用: True`、显存正常
- [ ] 数据放在 `data/raw/<name>/`，标注目录符合所选 `--format` 的约定
- [ ] `odp-transform` 日志里没有覆盖率硬阈值告警
- [ ] `odp-validate` 退出码为 0（无 ERROR）
- [ ] `runs/` 所在盘有足够空间（推理视频产物动辄数百 MB）
- [ ] 训练后 `models/trained/<stem>-best.pt` 与 `runs/training/<run_id>/odp_audit.json` 都存在
- [ ] `odp-eval` 报告里看过 `warnings` 再引用指标数字
- [ ] 跑 `odp-clean` 之前先不加 `--yes` 看预演，**确认 `lineage` 标着 `[豁免]`**
- [ ] 跑 `odp-reset` 之前先不加 `--yes` 看计划，**确认 `data/raw/` 不在其中**

---

## 六、一切皆配置（配置层速查）

本仓库的硬性纪律：**凡是"改它会让同一份输入得到不同产物"的取值，都必须能从配置改。**
完整决策与边界见 [ADR-006](architecture/ADR-006-everything-is-config.md)。

### 6.1 九套模板与它们的命令

| yaml 模板（`apps/platform/configs/runtime/`） | 谁读它 | 怎么覆盖 | 怎么重新生成 |
|---|---|---|---|
| `transform.yaml` | `odp-transform` | `odp-transform --yaml <名/路径>` + `--train-rate/--val-rate/--seed/--split-strategy` | `odp-gen-config transform` |
| `validate.yaml` | `odp-validate` | `odp-validate --yaml <名/路径>` | `odp-gen-config validate` |
| `clean.yaml` | `odp-clean` | `odp-clean --yaml <名/路径>` + `--keep-latest/--max-age-days/--protect` | `odp-gen-config clean` |
| `train.yaml` | `odp-train` | `odp-train --config <路径>` + `--epochs/--batch/--imgsz/--device`… | `odp-gen-config train` |
| `val.yaml` | 验证流程 | `odp-gen-config val` 生成后按需引用 | `odp-gen-config val` |
| `eval.yaml` | `odp-eval` | `odp-eval --yaml <名/路径>` + `--conf/--iou/--split/--conf-sweep`… | （随仓库提供） |
| `infer.yaml` | `odp-infer` | `odp-infer --yaml <名/路径>` + CLI | `odp-gen-config infer` |
| `infer_pipeline.yaml` | `odp-infer`（帧源 + 美化 + **输出落盘**） | `odp-infer --pipeline-yaml <名/路径>` | （随仓库提供） |
| `demo.yaml` | `odp-demo` | `odp-demo` 的 CLI | `odp-gen-config demo` |

### 6.2 改一个阈值该改哪里（三条纪律）

1. **优先级 `CLI > YAML > DEFAULT`**：临时试一下用 CLI，长期生效改 yaml，**不要改源码**；
2. **默认值只有一处**：每个 `runtime_config/<子系统>.py` 的字段默认值直接引用
   `common/constants.py`（例：`coverage_hard_threshold: float = C.COVERAGE_HARD_THRESHOLD`）。
   想确认某个阈值的"出厂值"，grep 那个常量即可；
3. **改完要知道谁生效**：启动日志里有"配置来源报告"与"覆盖链"
   （`epochs : 100(DEFAULT) → 200(YAML) → 50(CLI)`）；本次运行的生效配置与来源也写进
   `runs/.../odp_audit.json`。

### 6.3 两个常用实验（验证"配置真的生效"）

```powershell
# ① 把覆盖率硬闸门提到 0.99 → 同一份数据必须被拒（退出码 1）
#    编辑 apps/platform/configs/runtime/transform.yaml 的 coverage_hard_threshold: 0.99
& .\.venv\Scripts\odp-transform.exe --dataset demo --format pascal_voc --split-strategy stratified_multilabel
#    期望：日志出现"覆盖率 ... 低于硬阈值"，退出码 1

# ② 把出界框判 ERROR 的门限调低 → 同一个越界框的严重度升级
#    编辑 validate.yaml 的 bbox_out_error_ratio: 0.0
& .\.venv\Scripts\odp-validate.exe apps\platform\configs\datasets\demo.yaml
#    期望：bbox_within_image 由 WARNING 变 ERROR，退出码由 1 变 2
```

### 6.4 明确**不**配置的东西（看到它们别奇怪）

- 销毁类操作的白名单 / 保护目录（`paths.get_dirs_to_reset()` / `PROTECTED_DIRS`）——
  安全设计不能由用户可写的配置放开；
- 协议与框架常量：`cv2` 的 `CAP_PROP_*` / FOURCC 语义、`hashlib.sha256`、契约版本号 `TOOL_VERSION`；
- 数学边界与容差：`0/1` 取值范围、浮点比较容差、IoU 分桶边界；
- 哨兵语义：如 `total<=0` 表示"总量未知"。

判据：**改它会让"同一输入得到不同产物" → 进配置；改它会让"安全承诺或数据契约失效" → 留在代码。**