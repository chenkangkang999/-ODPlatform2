# ODPlatform

通用**目标检测开发平台** —— 从原始标注到可部署权重的完整链路：
**转换 → 质检 → 训练 → 评估 → 推理**，每一步都留指纹与审计快照。

- **Monorepo**：`apps/` 下预留多个端，当前只有 `apps/platform/` 是真实实现的代码端。
- **src 布局**：包源码在 `apps/platform/src/od_platform/`，可编辑安装为 `od-platform`。
- **marker file**：仓库根有 `.odp-workspace`，`common/paths.py` 从模块位置向上找到它来定位仓库根
  —— 路径常量集中在一处，**不写死盘符**。

## 当前业务场景与数据集状态

本平台当前面向的业务场景是**淡水鱼品种细分类检测**（fine-grained detection：既定位鱼体、又区分品种）。

业务数据集已接入：**BengalDeltaFish**（孟加拉三角洲淡水鱼检测数据集，Roboflow 导出、CC BY 4.0、YOLO 格式）。

| 项目 | 实测值 |
|---|---|
| 数据集名（`--dataset`） | `fish` → `data/raw/fish/`、`data/processed/fish/`、`apps/platform/configs/datasets/fish.yaml` |
| 图片 / 标签 / 实例 | **4560 / 4560 / 6034**（一图一签，无缺签、无孤儿标签） |
| **类别数** | **33 个淡水鱼品种** |
| 类别（`names` 顺序即 `class_id`） | `Batashi, Bele, Chanda, Chapila, Chela, Foli, Ilish, Kajuli, Kayakanta, Koi, Magur, Mola, Mrigel, Pabda, Pangas, Puti, Rui, Rupchada, Shol, Sorputi, Taki, Tarabaim, Telapia, Tengra, Veda, Vetki, ayer, boyal, chitol, kalbaus, katol, kholshe, shing` |
| 划分（`stratified_multilabel`，seed 42，一次性冻结） | train **3652** / val **458** / test **450** |
| 契约指纹 | `e0272ec1daf7d634455dc5f453428f4791913cad6e0da5916e6415279b638727` |
| `odp-validate` 结果 | **15/15 项 PASS，总评 PASS**（`data_fingerprint=1dc1c08a…`） |

> 说明：原始 Roboflow 导出自带 `train/valid/test` 三份划分，但本平台**不沿用**它 ——
> 划分在 `odp-transform` 这一步用固定种子一次性产出、落盘并写进划分契约（D0 决定 5：
> 训练只读那份冻结的划分）。合并原始三份 split 的整理脚本见
> [`scripts/prepare_fish_raw.py`](scripts/prepare_fish_raw.py)（源路径走命令行参数，不写死盘符）。

> ⚠️ 仓库里另有 `data/raw/{demo,demococo,demoyolo,broken,synth}` —— 那是**链路演示/验证用数据**，
> 不是业务数据集（其中 `demo` 的图片是 0 字节空文件，**不能用来训练**；`synth` 是合成噪声图，
> 仅用于验证训练链路可跑，指标无意义）。

平台与具体数据集解耦：**把任意数据集按下面的目录约定放进 `data/raw/<name>/` 即可**，
类别名由 `odp-transform --classes` 或数据自带信息决定，不存在写死的类别表。

## 目录结构

| 目录 / 文件 | 说明 | 进 git |
|---|---|---|
| `apps/platform/src/od_platform/` | 核心引擎包源码（子系统见下表） | ✅ |
| `apps/platform/configs/datasets/` | **数据集 yaml**（`odp-transform` 产出，含 `odp_meta` 合格证） | ✅ |
| `apps/platform/configs/runtime/` | **运行配置**（`train/val/infer/infer_pipeline/eval`） | ✅ |
| `apps/platform/tests/` | 端内 pytest（`pyproject.toml` 的 `testpaths`） | ✅ |
| `apps/platform/logging/` | 业务工具日志（`init_project/transform/validate/train/eval/infer/clean`） | ❌ gitignored |
| `apps/platform/pyproject.toml` | 包 `od-platform` 定义 + `[project.scripts]` 命令入口 | ✅ |
| `apps/web-backend/` | Web 后端**占位**（仅 README，V1.1 启动） | ✅ |
| `apps/desktop/` | 桌面端**占位**（README + `infer_worker.py` / `qt_sink.py` 两个 Qt 集成文件，未构成可运行应用） | ✅ |
| `data/raw/<name>/` | **原始数据**：只读，`odp-reset` 绝不碰 | ❌ |
| `data/processed/<name>/` | **派生数据集**：`images/{train,val,test}` + `labels/{train,val,test}` | ❌ |
| `models/pretrained/` | 下载/放置的预训练权重（输入） | ❌（保留 README） |
| `models/trained/` | 训练归档权重 `*-best.pt`（可再生产） | ❌（保留 README） |
| `models/checkpoints/` | 推理时的权重搜索目录（**`odp-init` 不创建它**，用到时自建） | ❌ |
| `runs/` | 每次运行的现场目录（manifest / 报告 / 权重 / 推理输出 / 章） | ❌（保留 README） |
| `runs/lineage/<data_fingerprint>/` | **数据流转卡**：`transform.stamp.json` / `validate.stamp.json` / `train.stamp.json` | ❌ |
| `tests/` | 跨端（E2E）测试的家 | ✅ |
| `scripts/` | 工程脚本（`setup_venv.ps1`、`demo_data/`、`reset_project.py`…） | ✅ |
| `docs/` | 本目录：README、使用手册、ADR | ✅ |
| `apps/platform/logging/infer/` | `odp-infer` 的日志（`get_logger(base_path=LOGGING_DIR, log_type="infer")`，**绝对路径**，不随工作目录漂移） | ❌ |
| `.odp-meta/logs/` | 元工具（`odp-reset`）的审计日志 —— 不会被它自己清理 | ❌ |
| `.odp-workspace` | workspace marker file，**不要删** | ✅ |
| `pyproject.toml`（根） | 只有 `[tool.ruff]` / `[tool.mypy]` / `[tool.pytest.ini_options]`，不描述任何包 | ✅ |
| `.pre-commit-config.yaml` | ruff-format + ruff --fix + pytest | ✅ |
| `.github/workflows/ci.yml` | CI：3.10/3.11/3.12 × 演示数据 × ruff × pytest | ✅ |

### 子系统一览（`apps/platform/src/od_platform/`）

| 子系统 | 职责 |
|---|---|
| `common/` | 路径/日志/字符串/系统/性能/常量/血缘指纹/引用解析/注册表工具/运行上下文/命名/盖章/进度/保留策略/日志改名/模型路径/配置日志/配置工具 |
| `data_pipeline/` | `odp-transform`：转换（pascal_voc / coco / yolo）→ 扫描报告 → 划分 → 契约指纹 → 物化 → 写 yaml |
| `data_validation/` | `odp-validate`：15 项只读 check，按最严聚合成退出码 |
| `runtime_config/` | Pydantic v2 配置模型 + 三源合并（DEFAULT / YAML / CLI）+ YAML 模板生成 |
| `training/` | `odp-train`：身份核验 → 训练 → 指标 → 盖章 → 审计 → 归档 |
| `inference/` | `odp-infer`：帧源 → 逐帧预测 → 美化/HUD → 输出 sink → 审计 |
| `frame_source/` | 帧源抽象：摄像头 / 视频 / 图片 / 图片夹，含重连、线程化、异步包装 |
| `visualization/` | 检测框绘制（`BeautifyVisualizer`，含分辨率自适应 `DrawStyle`） |
| `evaluation/` | `odp-eval`：只读数据集视图 → val → 每类指标 → 置信度扫描 → 报告 |
| `cli/` | 各命令的 argparse 入口与退出码 |

## 环境准备

当前机器：Windows 11，Python **3.12.4**，`.venv` 已存在且 `od-platform` 已可编辑安装。

本文写作时在本机 `.venv` 上执行：

```powershell
& .\.venv\Scripts\python.exe -c "import sys, torch, ultralytics, pydantic, od_platform; print('python', sys.version.split()[0]); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available()); print('ultralytics', ultralytics.__version__); print('pydantic', pydantic.VERSION); print('od_platform', od_platform.__file__)"
```

真实输出：

```
python 3.12.4
torch 2.14.1+cu126 cuda 12.6 avail True
ultralytics 8.4.189
pydantic 2.14.1
od_platform D:\ODPlatform2\apps\platform\src\od_platform\__init__.py
```

### 从零重建环境（GPU 版 torch）

仓库自带 `scripts/setup_venv.ps1`，它做的正是下面这套：

```powershell
# 1) 创建虚拟环境（已存在则复用）
python -m venv .\.venv

$PY = ".\.venv\Scripts\python.exe"

# 2) 构建工具走阿里源
& $PY -m pip install --upgrade pip setuptools wheel -i https://mirrors.aliyun.com/pypi/simple/

# 3) ★ torch / torchvision 必须走 PyTorch 官方 cu126 索引（GPU 版）
& $PY -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 4) 其余依赖走阿里源
& $PY -m pip install colorlog psutil pyyaml pydantic ultralytics opencv-python pytest pytest-cov ruff mypy pre-commit -i https://mirrors.aliyun.com/pypi/simple/

# 5) 把 platform 端可编辑安装进这个 .venv（命令入口 odp-* 由这一步产生）
& $PY -m pip install -e .\apps\platform
```

**为什么 torch 要单独一条索引**：`apps/platform/pyproject.toml` 里写的是 `torch>=2.0.0`
——它**不区分 CPU/CUDA 构建**。普通包走阿里源没问题，但 torch 如果也从普通源装，
很可能拿到 `+cpu` 构建，**GPU 会静默失效**（不报错，只是慢很多）。
`cu126` 要求驱动 ≥ 560；驱动更老要换 `cu121` / `cu118`。

校验 GPU 是否真的可用：

```powershell
& .\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
& .\.venv\Scripts\odp-init.exe    # 会打印「运行环境概览」，含 CUDA可用 / GPU数量 / 显存
```

本机为 RTX 4060 Laptop 8GB；`odp-init` 的输出里 GPU 行为
`GPU_0 : NVIDIA GeForce RTX 4060 Laptop GPU`、`GPU_0_显存 : 8.00 GB`。

## 快速开始（8 步）

> 所有命令都在**仓库根** `D:\ODPlatform2` 下执行。命令统一写 `& .\.venv\Scripts\odp-xxx.exe`。

### 本次业务数据集（`fish`，33 类淡水鱼）的完整命令序列

```powershell
# ① 把外部 Roboflow 数据集整理成 raw 布局（源路径从命令行给，不写死盘符）
& .\.venv\Scripts\python.exe scripts\prepare_fish_raw.py --src "<外部数据集根，含 data.yaml>" --name fish

# ② 转换 + 划分 + 物化（yolo 格式必须显式给类名，顺序即 class_id）
$names = "Batashi Bele Chanda Chapila Chela Foli Ilish Kajuli Kayakanta Koi Magur Mola Mrigel Pabda Pangas Puti Rui Rupchada Shol Sorputi Taki Tarabaim Telapia Tengra Veda Vetki ayer boyal chitol kalbaus katol kholshe shing".Split(" ")
& .\.venv\Scripts\odp-transform.exe --dataset fish --format yolo --split-strategy stratified_multilabel --classes @names

# ③ 质检（应得 15/15 PASS，并盖 D4 章）
& .\.venv\Scripts\odp-validate.exe apps\platform\configs\datasets\fish.yaml

# ④ 生成运行配置模板（train/val/infer/eval）
& .\.venv\Scripts\odp-gen-config.exe train

# ⑤ GPU 训练
& .\.venv\Scripts\odp-train.exe --data fish --epochs 15 --imgsz 640 --batch 0.2 --model yolo11n

# ⑥ 评估（mAP50 / mAP50-95 / IoU 匹配分析 / 置信度扫描）
& .\.venv\Scripts\odp-eval.exe --weights models\trained\<run>-fish-yolo11n-best.pt --data apps\platform\configs\datasets\fish.yaml --device 0 --conf-sweep

# ⑦ 推理（图片夹 / 视频 / 摄像头）
& .\.venv\Scripts\odp-infer.exe --source data\processed\fish\images\test --model models\trained\<run>-fish-yolo11n-best.pt --device 0 --save --no-show
```

实测结果（本机 RTX 4060 Laptop 8GB）见下文「评估指标」一节；
`data/raw/`、`data/processed/`、`models/*.pt`、`runs/` 都**不进 git**（`.gitignore`）。

### 0) 初始化目录 + 看环境

```powershell
& .\.venv\Scripts\odp-init.exe
```

`odp-init` **没有 argparse 参数，也没有 `--help`**（直接执行初始化）。
它按 `common/paths.get_dirs_to_initialize()` 的 15 项目录清单建目录，并打印环境概览、数据目录状态。

产物 / 落点：目录本体；日志写在 `apps/platform/logging/init_project/init-project_<时间戳>.log`。

### 1) 放入原始数据

```
data/raw/<name>/images/            # Pascal VOC 与 COCO
data/raw/<name>/annotations/       # VOC 的 *.xml，或 COCO 的 *.json
data/raw/<name>/labels/            # YOLO 的 *.txt（注意是 labels/ 不是 annotations/）
```

三种格式的目录约定不同（见 `data_pipeline/orchestrator.py` 的 `_annotation_dir()`）：

| `--format` | 图像目录 | 标注目录 |
|---|---|---|
| `pascal_voc` | `images/` | `annotations/`（`*.xml`） |
| `coco` | `images/` | `annotations/`（`*.json`） |
| `yolo` | `images/` | **`labels/`**（`*.txt`） |

### 2) 转换 + 划分 + 物化

```powershell
& .\.venv\Scripts\odp-transform.exe --dataset <name> --format pascal_voc
# yolo 格式没有类别名，必须显式给，顺序即 class_id：
& .\.venv\Scripts\odp-transform.exe --dataset <name> --format yolo --classes 类1 类2 类3
```

产物 / 落点：

- `data/processed/<name>/images/{train,val,test}/` 与 `labels/{train,val,test}/`
- `apps/platform/configs/datasets/<name>.yaml`（含 `odp_meta`：契约指纹、manifest 路径、划分计数）
- `runs/data_pipeline/<run_id>/manifest.json`（样本级清单）
- `runs/lineage/<data_fingerprint>/transform.stamp.json`
- 日志 `apps/platform/logging/transform/transform_<时间戳>.log`

退出码：`0` 成功 / `1` 数据或参数错误 / `2` argparse 用法错误。

### 3) 质检

```powershell
& .\.venv\Scripts\odp-validate.exe apps\platform\configs\datasets\<name>.yaml
```

⚠️ **相对路径的坑**：位置参数 `yaml_path` 是**相对当前工作目录**解析的。
`--help` 里写的示例是 `configs/datasets/demo.yaml`（那是相对 `apps/platform/` 的写法）。
在仓库根运行时必须写 `apps\platform\configs\datasets\<name>.yaml`；写错不会报"文件不存在"，
而是 15 项 check 全部报 `yaml 无法加载` 并以退出码 2 结束。

产物 / 落点：

- `runs/data_validation/<run_id>/report.json`、`runs/data_validation/<run_id>/results.csv`
- `runs/data_validation/<run_id>/compare.json` —— **与同一份 `yaml_path` 上一次质检报告的历史对比**
  （`--no-report` 时不产出；没有历史可比时写明"首次质检"）
- `runs/lineage/<data_fingerprint>/validate.stamp.json`（仅在 yaml 能定位到 manifest 时盖）
- 退出码：`ERROR→2` / `WARNING→1` / `PASS·INFO→0`

历史对比回答的是"**这一次比上一次好了还是坏了**"—— 单次报告回答不了。
它有两层口径（`data_validation/compare.py`）：

- **check 级**：用 severity 的变化判定"整项修好了 / 整项新坏了 / 仍然有问题"；
- **条目级**：把各 check `details` 里逐条列出的证据抽成稳定 key 做集合差集，
  回答"整项还没修好、但其中 3 条已经消失了"这种 check 级看不见的问题。

`compare.json` 的键：`has_history` `previous_run_id` `current_run_id` `previous_overall`
`current_overall` `improved` `previous_counts` `current_counts` `trend` `newly_failing`
`fixed` `still_failing` `added_checks` `removed_checks` `new_items` `disappeared_items`
`persisting_items`。`report.json` 里也会带上同一个 `comparison` 块。

两条如实标注：历史报告靠 **`yaml_path` 精确匹配**（路径变了就算另一批数据），
而条目级差集受各 check 的 preview **截断上限**限制（check 级判定不受影响）。

### 4) 生成运行配置

```powershell
& .\.venv\Scripts\odp-gen-config.exe train     # 或 val / infer
```

产物 / 落点：`apps/platform/configs/runtime/<name>.yaml`。
文件已存在时**默认不覆盖**（打印提示并以 0 退出）；要覆盖加 `--overwrite`，
覆盖前会备份成 `<name>.yaml.bak.<YYYYMMDD_HHMMSS>`。

### 5) 训练

```powershell
& .\.venv\Scripts\odp-train.exe --data <name> --model yolo11n --epochs 100
```

`--data <name>` 会被 `common/refs.resolve_dataset_yaml` 解析成
`apps/platform/configs/datasets/<name>.yaml`；也可以直接给路径。

训练前会做**数据身份核验**（yaml / manifest / transform 章 / validate 章对账），
不通过就拒绝开训（`--skip-validate` 可强跳，审计会标记）。

产物 / 落点：

- `runs/training/<run_id>/` —— ultralytics 现场：`args.yaml`、`results.csv`、`results.png`、
  `Box*.png`、`confusion_matrix*.png`、`weights/best.pt`、`weights/last.pt`
- `runs/training/<run_id>/odp_audit.json` —— 配置快照 + 各字段来源 + 指标 + 数据身份
- `models/trained/<stem>-best.pt` —— 归档权重（`--no-archive` 可关）
- `runs/lineage/<data_fingerprint>/train.stamp.json`
- 日志 `apps/platform/logging/train/<stem>.log`，其中
  `stem = train-<run_id>-<dataset>-<model>`（规则见 `common/naming.py`）
- 退出码：`0` 成功 / `1` 失败

### 6) 独立评估

```powershell
& .\.venv\Scripts\odp-eval.exe --weights models\trained\<stem>-best.pt --data apps\platform\configs\datasets\<name>.yaml
```

产物 / 落点：`runs/detect_eval/<run_id>/`

- `report.json` —— 完整结构（含 `per_class` / `stratified` / `conf_sweep` / `recommended_threshold` / `warnings` / `pr_curves` / `iou_analysis`）
- `metrics.csv` —— 每类一行（表头 `run_id,class_id,class_name,evaluated,instances,AP50,AP50_95,precision,recall`）
- `conf_sweep.csv` —— 每个阈值一行（表头 `run_id,conf,precision,recall,f1`）
- `iou_pairs.csv` —— 每个匹配上的"预测-真值"对一行（列见「评估指标说明」；`--no-iou-analysis` 时不产出）
- `_dataset_view/` —— 本次评估实际用的只读数据集视图（硬链接 + 规范化 `data.yaml`）

`<run_id>` 默认是时间戳（`--run-id` 可指定）；`--output-dir` 可把报告写到别处。
退出码：`0` 成功 / `1` 用法或参数错误 / `2` 评估失败。

### 7) 推理

```powershell
# 图片 / 视频 / 图片目录
& .\.venv\Scripts\odp-infer.exe --source <图片或视频或目录> --model models\trained\<stem>-best.pt --no-show --save
# 摄像头（会弹窗）
& .\.venv\Scripts\odp-infer.exe --source 0 --model yolo11n.pt --show
```

产物 / 落点：`runs/detect_infer/<name>/`，`<name>` 默认 `predict`，重名自动变成 `predict2`、`predict3`…

- `odp_audit.json` —— `mode / config / merger / pipeline / stats / result_summary`
  （`pipeline.output` 是输出落盘的生效值，`pipeline.output_sources` 是每个字段的来源）
- 视频/摄像头源：默认 `output.mp4`；图片源：每帧一张 `<原图名>.jpg`
  —— 文件名前缀 / 主干 / 扩展名 / 视频编码器 / jpg 质量 / 写入帧率 / 覆盖策略
  都由 `infer_pipeline.yaml` 的 `output` 段决定（不填 = 与原默认行为**逐字节一致**）
- 日志 `apps/platform/logging/infer/infer_<时间戳>.log`（绝对路径；
  `common/log_rename.py` 的"搬进本次输出目录"目前是**桩**，不做实际改名）
- 退出码：`0` 成功 / `1` 失败

### 8) 清理运行现场（可选）

```powershell
& .\.venv\Scripts\odp-clean.exe --summary     # 只看现状
& .\.venv\Scripts\odp-clean.exe               # 预演（默认不删）
& .\.venv\Scripts\odp-clean.exe --yes         # 真删
```

`runs/lineage/` **永远豁免**（删了数据身份就对不上账，且无法重建）。

## 命令总表

| 命令 | 作用 | 关键参数 | 主要产物路径 |
|---|---|---|---|
| `odp-init` | 建 15 项运行时目录 + 打印环境概览与数据目录状态 | **无参数** | 目录本体；`apps/platform/logging/init_project/*.log` |
| `odp-reset` | 撤销 init 的运行时产物（默认 dry-run） | `--yes` `--force` `--dry-run` | 清理 `data/processed/`、`runs/`、`models/trained/`、`apps/platform/logging/`；审计日志 `.odp-meta/logs/reset_project/` |
| `odp-transform` | 原始标注 → 划分 → 物化 → 数据集 yaml | `--dataset` `--format` `--task` `--yaml` `--train-rate` `--val-rate` `--seed` `--split-strategy` `--classes` | `data/processed/<name>/`、`configs/datasets/<name>.yaml`、`runs/data_pipeline/<run_id>/manifest.json`、`runs/lineage/<fp>/transform.stamp.json` |
| `odp-validate` | 15 项只读质检 + 与上次报告的历史对比，按最严聚合退出码 | `<yaml_path>`（位置） `--yaml`（阈值配置） `--task` `--run-id` `--no-report` | `runs/data_validation/<run_id>/{report.json,results.csv,compare.json}`、`runs/lineage/<fp>/validate.stamp.json` |
| `odp-gen-config` | 从 Pydantic 模型反射生成 YAML 模板（**默认值唯一出处**） | `{train,val,infer,demo,transform,validate,clean}`（位置） `-o/--output` `--overwrite` `--no-backup` | `apps/platform/configs/runtime/<name>.yaml`（+ 覆盖时 `<name>.yaml.bak.<时间戳>`） |
| `odp-train` | 身份核验 → 训练 → 盖章 → 审计 → 归档 | `--config` `--data` `--model` `--epochs` `--batch` `--imgsz` `--device` `--experiment-name` `--no-archive` `--skip-validate` | `runs/training/<run_id>/`、`models/trained/<stem>-best.pt`、`runs/lineage/<fp>/train.stamp.json` |
| `odp-eval` | 独立评估已训好模型 + IoU 匹配分析 + 置信度扫描 | `--weights` `--data`（必填）`--yaml` `--split` `--imgsz` `--batch` `--device` `--workers` `--max-det` `--conf` `--iou` `--classes` `--exclude-classes` `--conf-sweep/--no-conf-sweep` `--no-iou-analysis` `--limit` `--run-id` `--output-dir` `--no-report` `--log-level` | `runs/detect_eval/<run_id>/{report.json,metrics.csv,conf_sweep.csv,iou_pairs.csv,_dataset_view/}` |
| `odp-infer` | 图片/视频/目录/摄像头/RTSP 推理 | `--source` `--model` `--yaml` `--pipeline-yaml` `--name` `--conf` `--iou` `--imgsz` `--max-det` `--classes` `--device` `--batch` `--show/--no-show` `--save/--no-save` `--no-viz` `--no-hud` `--warmup` `--log-level` | `runs/detect_infer/<name>/{odp_audit.json,output.mp4|<图片名>.jpg}` |
| `odp-clean` | 按保留策略清理 `runs/`（默认预演） | `--yes/--apply` `--yaml` `--keep-latest` `--max-age-days` `--protect` `--summary` `--runs-dir` | 无新产物；`--yes` 时写 `apps/platform/logging/clean/` |
| `odp-demo` | Gradio 演示界面：上传图片 → 推理 → 标注图 + 检出表 + 汇总（配置与绘制全部复用 `odp-infer`） | `--server-name` `--server-port` `--share` `--inbrowser` `--max-images` `--weight` `--pipeline-yaml` `--conf` `--iou` `--imgsz` `--device` `--no-viz` `--log-level`（需可选依赖 `[ui]`：`pip install -e "./apps/platform[ui]"`） | 无新产物（不写 `runs/`）；日志 `apps/platform/logging/demo/` |

各命令的完整参数表与示例见 **[docs/manual.md](docs/manual.md)**。

## 一切皆配置（零硬编码）

本仓库的硬性纪律：**凡是"改它会让同一份输入得到不同产物"的取值，都必须能从配置改。**
完整决策与边界见 [ADR-006](docs/architecture/ADR-006-everything-is-config.md)，这里只给结论与用法。

### 配置层清单（**9 套模板 / 13 个模型**）

| 子系统 | 配置模型（默认值的唯一出处） | yaml 模板 | 覆盖途径 |
|---|---|---|---|
| 数据流水线 | `runtime_config/transform.py` | `transform.yaml` | `odp-transform --yaml` + CLI（`--train-rate/--val-rate/--seed/--split-strategy`） |
| 数据质检 | `runtime_config/validate.py` | `validate.yaml` | `odp-validate --yaml` |
| runs 保留策略 | `runtime_config/clean.py` | `clean.yaml` | `odp-clean --yaml` + CLI（`--keep-latest/--max-age-days/--protect`） |
| 训练 | `runtime_config/train.py` | `train.yaml` | `odp-train --config` + CLI（`--epochs/--batch/--imgsz/--device`…） |
| 验证 | `runtime_config/val.py` | `val.yaml` | `odp-gen-config val` 生成后按需引用 |
| 评估 | `runtime_config/val.py` + `evaluation/settings.py` | `eval.yaml` | `odp-eval --yaml` + CLI（`--conf/--iou/--split/--conf-sweep`…） |
| 推理 | `runtime_config/infer.py` + `inference/pipeline_config.py` | `infer.yaml` + `infer_pipeline.yaml`（含 `output:` 段） | `odp-infer --yaml/--pipeline-yaml` + CLI |
| 演示界面 | `runtime_config/demo.py` | `demo.yaml` | `odp-demo` 的 CLI（`--server-port/--weight/--conf/--imgsz`…） |

三套新模板都由 `odp-gen-config` 生成（带中文注释与示例），**不要手写**：

```powershell
& .\.venv\Scripts\odp-gen-config.exe transform   # 覆盖率闸门 / 稀有类阈值 / 划分比例与种子
& .\.venv\Scripts\odp-gen-config.exe validate    # 13 个质检阈值
& .\.venv\Scripts\odp-gen-config.exe clean       # 保留个数 / 最大保留天数 / 豁免清单
& .\.venv\Scripts\odp-gen-config.exe demo        # 演示界面的服务器与默认阈值
```

### 三条纪律

1. **优先级固定 `CLI > YAML > DEFAULT`**：CLI 只暴露最常用的开关，深水区交给 yaml；
2. **默认值只允许存在一处**：每个模型的字段默认值直接引用 `common/constants.py`
   （如 `coverage_hard_threshold: float = C.COVERAGE_HARD_THRESHOLD`）—— "改一处忘一处"因此可被 grep 验证；
3. **每个生效值都能查到来源**：启动打印"配置来源报告 + 覆盖链"（`epochs: 100(DEFAULT) → 200(YAML) → 50(CLI)`），
   生效配置写进 `runs/.../odp_audit.json`，Pydantic 报错也带来源链。

### 明确**不**配置的例外（防止"为了配置而配置"）

- **销毁类操作的白名单 / 保护目录**（`paths.get_dirs_to_reset()` / `PROTECTED_DIRS`）：
  安全设计不能由用户可写的配置放开 —— 那不是配置化，是把保险丝换成铜丝；
- **协议与框架常量**：`cv2` 的 `CAP_PROP_*` / FOURCC 语义、`hashlib.sha256` 的选择、
  契约版本号 `TOOL_VERSION`（它参与指纹配方，改动 = 换契约，必须走代码评审）；
- **数学边界与容差**：`0/1` 取值范围、浮点比较容差、IoU 分桶边界（这是定义不是偏好）；
- **哨兵语义**：如 `total<=0` 表示"总量未知"。

判据一句话：**改它会让"同一输入得到不同产物" → 进配置；改它会让"安全承诺或数据契约失效" → 留在代码。**

### 守卫（防回退）

| 守卫 | 位置 | 拦什么 |
|---|---|---|
| 硬编码绝对路径守卫 | `.github/workflows/ci.yml` | 盘符/绝对路径进仓库（测试文件豁免） |
| "模板 == 代码默认值"契约测试 | `apps/platform/tests/runtime_config/**` | 模板与默认值漂移（曾当场抓到 `1e-06` 浮点回环问题） |
| `extra="forbid"` 拼写防线 | 每个 Settings 模型 | yaml 拼错字段名（`epchs=300` 构造时就报错） |
| 阈值真生效用例 | `apps/platform/tests/**` | 改了配置不生效 / 生效了破坏默认行为 |

## 评估指标说明

`odp-eval` 的指标分四块，口径不同、**不要混着说**。

### 1) IoU —— 同一个词，三种用途，绝不能混着说

IoU（Intersection over Union，交并比）= 预测框与真值框交集面积 ÷ 并集面积。本平台里它出现三次，
含义完全不同：

| 用途 | 参数 / 位置 | 取值 | 含义 |
|---|---|---|---|
| **NMS 去重阈值** | `--iou`（`EvalSettings.iou`） | `0.7` | 抑制重叠冗余框。**它不参与精度计算**，调大调小只改变"留下几个框" |
| **mAP 的 TP 判定门限** | 由 mAP 口径本身决定，**没有 CLI 开关** | `0.5`（mAP50）/ `0.5:0.05:0.95` 共 10 档取平均（mAP50-95） | 预测框与真值框 IoU ≥ 门限才算命中 |
| **匹配口径的门限** | `evaluation/iou.py` 的 `IOU_MATCH_THRESHOLD`，**同样没有 CLI 开关** | `0.5` | 贪心一对一匹配时，"这个框算配上了"的最低 IoU |

`evaluation/settings.py` 里对 `iou` 字段的注释就是第一条：
`iou: float = 0.7  # NMS 的 IoU 阈值（不是 mAP 的 IoU 门限）`。

### 1b) 匹配式 IoU 分析 —— 与官方 mAP 口径并列的另一套读数

mAP50 / mAP75 / mAP50-95 确实"沿 IoU 轴看"，但它们回答的是**"能不能把目标找到"**，
回答不了**"框贴得有多紧"**。框的位置偏半个身位时 mAP50 可能仍然很高（0.5 门限很宽松），
但下游的测距 / 跟踪 / 裁剪都会受影响。所以 `evaluation/iou.py` 额外给一套**匹配口径**：

- 预测框来自 `model.predict(conf=settings.conf, iou=settings.iou, …)` ——
  阈值一律取自配置，报告里能在 `settings` 与 `iou_analysis.conf` / `iou_analysis.nms_iou` 查到；
- **只允许同类**（`cls` 相同）预测与真值两两求 IoU；
- 按 IoU **降序贪心一对一**匹配（一个预测只配一个真值，一个真值只配一次；
  排序键带下标，保证 IoU 并列时结果可复现）。这就是常见的"匹配口径 TP / FP / FN"；
- 只有 `IoU >= IOU_MATCH_THRESHOLD`（0.5）的候选才算匹配得上；没配上的预测算 FP，没配上的真值算 FN；
- 在**匹配对**上统计 IoU 的 平均 / 中位数 / 最小 / 最大，以及 `0.50-0.60 … 0.90-1.00` **五档直方图**
  —— 这才是"框定得多准"的直接读数。

> ⚠️ **匹配口径的 P / R / F1 不等于官方 mAP 口径的 P / R。**
> 前者是"单个 IoU 门限上的一张点估计"，后者是对 P-R 曲线积分。两者都报，
> **绝不相加、不相除**（`iou.py` 的 `note` 与报告渲染里都写明了这一条）。

> ⚠️ **`eval.yaml` 的默认 `conf` 是 `0.001`** —— 那是给 val 用的极低阈值（为了让 P-R 曲线完整）。
> 用它跑匹配，NMS 后会吐出几百个低置信度框，"未匹配的预测（FP）"暴涨，
> 于是匹配口径的 P 会低到没有意义。**IoU 分布本身不受影响**（配上的还是那几个高分框）。
> 要让匹配口径的 P/R 有意义，请把 `--conf` 提到置信度扫描推荐的那个阈值再跑一遍。
> `iou_analysis.note` 会在 `conf <= 0.01` 时把这件事写明白。
> 用 `--no-iou-analysis` 可整块跳过（省一遍 predict）。

### 2) mAP50 与 mAP50-95

- **mAP50**：先把 IoU 门限固定为 **0.5**，算每个类的 AP（P-R 曲线下面积），再对**有真值的类**取平均。
- **mAP50-95**：在 IoU = `0.50, 0.55, 0.60, …, 0.95` 这 **10 个门限**上各算一遍 mAP50 再平均。
  门限越严，同一份预测的命中越少，所以 mAP50-95 恒 ≤ mAP50。
- 报告里还有 `map75`（IoU=0.75 的 mAP），口径与上面一致。
- **平均的分母是"有真值的类"**：`evaluation/metrics.py` 的 `mean_ap` 与框架的 `map50`
  用的是同一口径，因此"不排除任何类"时 `stratified` 的第一块与 `overall.map50` 应当相等。
  没有真值的类**不参与平均**，也不会被记成 0。

### 3) 每类 AP · P · R，以及"无真值"

`report.json` 的 `per_class` 每项含
`class_id / class_name / evaluated / instances / AP50 / AP50_95 / precision / recall`。

- `AP50`、`AP50_95` —— 该类的平均精度（门限见上）。
- `precision`（P，精确率）= 报出来的框里有多少是对的；`recall`（R，召回率）= 真值框里有多少被抓到。
- `instances` —— 该类在本 split 的**真值框数**，由 `evaluation/dataset.py` 的 `count_instances`
  直接从标签文件数出来（不读框架私有字段）。
- `evaluated: false` + `note: "没有真值（—）, 未参与评测"` —— 该类在本 split **一个真值框都没有**。
  报告里显示成破折号 `—` 而不是 `0`，因为"没评测"和"评测得很差"是两件事。

### 4) 置信度扫描与推荐阈值

推理部署要选一个 `conf` 阈值。`odp-eval` 的扫描是**逐阈值各跑一次 val**，
`conf_thresholds` 默认 `(0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7)`。

> 为什么不是从 P-R 曲线推出来？因为框架给的 PR 曲线横轴是**召回率**，不是置信度 ——
> 没有 "conf → R" 的对应关系，推不出来。代价就是 N 倍 val 时间，所以既可配置，
> 也能用 `--no-conf-sweep` 关掉（`evaluation/service.py` 与 `metrics.conf_sweep_rows` 都写明了这一点）。

**推荐规则**（`metrics.recommend_threshold`，参数 `target_precision` 默认 `0.90`）：

1. 在**精度 ≥ target_precision** 的阈值里，取**召回率最大**的那个（而不是取最大 F1）；
2. 并列时取精度更高的；再并列取阈值更低的；
3. 若**没有任何阈值达标**，不硬推荐 —— 返回 `status: "目标精度不可达"`，
   并给出 F1 最高的阈值作为参考，`reason` 里明确写"不构成「达标」结论"。

### 5) 本次业务数据集（`fish`，33 类淡水鱼）的实测数值

下面这组数字是**本仓库真跑出来的**（RTX 4060 Laptop 8GB / `yolo11n` / `imgsz=512` /
`batch=8` / 10 epoch / `stratified_multilabel` 冻结划分：train 3652 · val 458 · test 450）。

训练（`odp-train --data fish --model yolo11n --epochs 10`，耗时 0.114 h）：

| 指标 | 数值 |
|---|---|
| **mAP50** | **0.9095** |
| **mAP50-95** | **0.7454** |
| P / R（官方口径） | 0.8270 / 0.8872 |
| 推理速度 | 2.28 ms/图（总 4.07 ms/图含前后处理），10 epoch ≈ 7 分钟 |

评估（`odp-eval --weights models/trained/train-<run>-fish-yolo11n-best.pt --data apps/platform/configs/datasets/fish.yaml --device 0 --imgsz 512 --batch 8 --conf-sweep`，val 458 张 / 661 真值框 / 33 类）：

| 口径 | 数值 |
|---|---|
| **mAP50 / mAP50-95**（官方，`conf=0.001` 走完整 P-R 曲线） | **0.9073 / 0.7442** |
| 官方 P / R（全类平均） | 0.8268 / 0.8874 |
| 推荐置信度阈值 | `conf=0.4` → P=0.8349 · R=0.8884 · **F1=0.8608**（`target_precision=0.90` 不可达，如实标"目标精度不可达"，给 F1 最优点作参考） |
| **IoU 匹配分析**（`conf=0.001`，门限 IoU≥0.5） | 预测 3664 / 真值 661 / **匹配 645 对**；**IoU 平均 0.8961 · 中位 0.9158 · 最小 0.5054 · 最大 0.9879** |
| 匹配口径 P/R/F1（`conf=0.001`） | 0.1760 / 0.9758 / 0.2983 —— 低 P 是极低阈值吐框所致，**不代表框不准** |
| **部署阈值口径**（加 `--conf 0.4` 重跑） | 预测 753 / **匹配 586 对**（TP=586 FP=167 FN=75）；**IoU 平均 0.9012 · 中位 0.9187**；匹配口径 **P=0.7782 R=0.8865 F1=0.8289** |

每类 AP@0.5:0.95 的**最难/最易**（`report.json` 的 `per_class` 有全部 33 类）：
最好 `kholshe 0.925` · `Veda 0.907` · `Chapila 0.879`；最难 `Rui 0.387` · `Tengra 0.496` · `Ilish 0.555`。
冷启动推理演示：
- 单图 `odp-infer --conf 0.4` → 检出 **1 条鱼，品种 `Mola`**（`runs/detect_infer/fish_demo/`）；
- 整个 test split（450 张）`odp-infer --source data/processed/fish/images/test --conf 0.4 --save` →
  **733 个检出，覆盖全部 33 个品种**，总耗时 **5.60 s**（推理 163.7 FPS、平均 12.04 ms/帧），
  产物在 `runs/detect_infer/fish_test_batch/`（450 张标注图 + `odp_audit.json`）。

> ⚠️ 这组数字的含义：10 epoch 的 `yolo11n` 只是把链路跑通、给出可复现的基线；
> 要冲更高 mAP 应加 epoch、换 `yolo11s/m`、或提高 `imgsz`。**划分与数据指纹已冻结**
> （`contract_fingerprint=e0272ec1…`、`data_fingerprint=1dc1c08a…`），所以换超参后指标仍可比。

### 产物

| 文件 | 内容 |
|---|---|
| `runs/detect_eval/<run_id>/report.json` | 顶层键：`run_id` `weights` `data_yaml` `split` `duration_seconds` `overall` `per_class` `stratified` `conf_sweep` `recommended_threshold` `warnings` `dataset` `settings` `speed_ms` `pr_curves` `iou_analysis` |
| `runs/detect_eval/<run_id>/metrics.csv` | `run_id,class_id,class_name,evaluated,instances,AP50,AP50_95,precision,recall`（`utf-8-sig`，Excel 直接打开中文不乱码） |
| `runs/detect_eval/<run_id>/conf_sweep.csv` | `run_id,conf,precision,recall,f1`；`None` 写空单元格而不是 0 |
| `runs/detect_eval/<run_id>/iou_pairs.csv` | 每个**匹配上**的"预测-真值"对一行，列：`run_id,image,class_id,class_name,iou,confidence,gt_x1,gt_y1,gt_x2,gt_y2,pred_x1,pred_y1,pred_x2,pred_y2`（FP/FN 清单在 `report.json` 的 `iou_analysis.per_image` 里按图给出） |
| `runs/detect_eval/<run_id>/_dataset_view/` | `data.yaml`（只用相对路径）+ `images/` + `labels/`；优先硬链接，跨卷才退化为复制并在报告里记数 |

报告里还有一块 **`warnings`（"必须说在前面的警告"）**，触发线来自 `EvalSettings`：
验证集张数低于 `min_images_for_reliable_map`（默认 100）、某类真值框低于
`min_instances_for_reliable_ap`（默认 100）、类别框数比超过 `imbalance_ratio_warn`（默认 10）、
某类无真值、用了 `--limit`、以及视图文件未能硬链接。这些是**如实呈现**，不是错误。

### 数据集视图为什么要存在

`odp-eval` **不把 `data/raw` 直接交给 ultralytics**。原因写在 `evaluation/dataset.py` 的模块 docstring 里：

1. `data/raw` 是只读区，而 ultralytics 在 val 时会往数据集目录写 `labels.cache`；
2. Roboflow 导出的相对路径（`train: ../train/images`）常常是坏的，字面解析会跑到数据集外面；
3. ultralytics 强制要求 `train:` 与 `val:` 两个键都在，val-only 也必须给 `train:`。

所以它在 `runs/detect_eval/<run_id>/_dataset_view/` 下用**硬链接**搭一个只读视图，
并写一份只用相对路径的规范化 `data.yaml`。

## 开发工作流

```powershell
# 可编辑安装（含开发依赖）
& .\.venv\Scripts\python.exe -m pip install -e .\apps\platform[dev]

# 代码风格 + import 排序（配置在根 pyproject.toml）
& .\.venv\Scripts\ruff.exe check .
& .\.venv\Scripts\ruff.exe format --check .

# 类型检查
& .\.venv\Scripts\mypy.exe apps\platform\src

# 测试（testpaths = apps/platform/tests + tests）
& .\.venv\Scripts\pytest.exe -q
```

根 `pyproject.toml` 只放共享工具配置（`[tool.ruff]` / `[tool.mypy]` / `[tool.pytest.ini_options]`）
——**它没有 `[build-system]` 也没有 `[project]`**，因为仓库根不描述任何包。

pre-commit（`.pre-commit-config.yaml`）挂了三个 hook：

| hook | 来源 | 作用 |
|---|---|---|
| `ruff-format` | `astral-sh/ruff-pre-commit` rev `v0.6.9` | 格式化 |
| `ruff`（`--fix`） | 同上 | lint + 自动修 |
| `pytest` | `local` / `language: system` | `python -m pytest apps/platform/tests tests -q` |

```powershell
& .\.venv\Scripts\pre-commit.exe install
& .\.venv\Scripts\pre-commit.exe run --all-files
```

> pre-commit 配置里 `default_language_version.python` 写的是 `python3.10`；
> `ruff` 的 `target-version` 也是 `py310`。本机 `.venv` 是 3.12，`odp-init` 打印的
> `Python版本` 为 `3.12.4`。

## CI

`.github/workflows/ci.yml`，在 push / PR 到 `main`、`master` 时触发，矩阵
**Python 3.10 / 3.11 / 3.12**（`fail-fast: false`）。步骤：

1. `actions/checkout@v4` + `actions/setup-python@v5`（开 pip 缓存）
2. `pip install -e "./apps/platform[dev]"`
3. **准备演示数据**：`python scripts/demo_data/make_demo.py`、`make_coco_demo.py`、
   `make_yolo_demo.py`、`make_broken_demo.py`
4. `ruff check .` 与 `ruff format --check .`
5. `pytest`

注意 CI 跑在 `ubuntu-latest` 上：它不装 cu126 版 torch（Ubuntu runner 无 GPU），
所以 `apps/platform/logging` 里那份"运行环境概览"在 CI 上会是 CPU。

## 常见问题

<details>
<summary><b>Windows 控制台中文乱码 / 表情变问号</b></summary>

控制台默认代码页不是 UTF-8 时，`odp-*` 输出里的中文和 `│`、`•` 会变成乱码。
先切到 UTF-8 再运行：

```powershell
chcp 65001
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = 'utf-8'
```

日志**文件**本身是 `encoding="utf-8"` 写的（`common/logging_utils.get_logger`），
所以 `apps/platform/logging/**/*.log` 用编辑器打开中文正常 —— 乱码只发生在控制台。
</details>

<details>
<summary><b>GPU 没用上 / 推理训练都很慢 —— 先分 CPU 版还是 GPU 版 torch</b></summary>

```powershell
& .\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

- 版本号带 `+cu126` 且 `True` → 装的是 CUDA 构建，可用（本机就是 `2.14.1+cu126` / `True`）。
- 版本号带 `+cpu` 或 `torch.version.cuda` 是 `None` → 装成了 CPU 构建，`device: null`
  会**静默退回 CPU**。按「环境准备」第 3 步换成 cu126 索引重装。

另外 `device` 默认是 `null`（自动选卡）。要临时固定用 CLI，**不要把卡号写进 yaml**
（理由见 [ADR-003](docs/architecture/ADR-003-runtime-config-precedence.md)）。
</details>

<details>
<summary><b>`data/raw/demo/` 的图是 0 字节，拿来训练会报损坏图像</b></summary>

`scripts/demo_data/make_demo.py` 刻意用 `write_bytes(b"")` 造**空字节图**，
它的目的是演示 D3/D4 的**目录与标注链路**（覆盖率、配对、越界框、稀有类），
全程不读像素。所以 `data/raw/demo/images/*.jpg` 全是 0 字节：

```powershell
Get-ChildItem data\raw\demo\images | Select-Object -First 3 Name,Length
# c_000.jpg  0
# c_001.jpg  0
# c_002.jpg  0
```

**它不能用来训练**：ultralytics 解不了空图，会直接报损坏图像。
</details>

<details>
<summary><b>`synth` 合成数据集只能验证链路，指标没有意义</b></summary>

`scripts/demo_data/make_synth_demo.py` 用 numpy + PIL 生成 **40 张真实可解码的 JPEG**
（噪点背景 + 随机色块），配对应的 YOLO 归一化标签，3 个类别：
`red_block` / `green_block` / `blue_block`（固定种子 `SEED = 20260727`，可复现）。

它是为了**真机验证 D6 训练链路端到端可跑**（GPU / 产物 / 验章 / 归档）而存在的。
色块与噪声没有真实语义：

> **1 个 epoch 的 mAP 只能说明"路通了"，不能说明"学得好"。**

业务数据集确定后，`data/raw/synth/` 可以整个删掉，不影响任何代码路径。
</details>

<details>
<summary><b>`odp-infer --model` 给了名字却触发联网下载</b></summary>

`inference/service.py` 用 `resolve_model_path(..., search_dirs=[CHECKPOINTS_DIR, PRETRAINED_MODELS_DIR])`
解析模型，即只在 **`models/checkpoints/`** 与 **`models/pretrained/`** 里找；
`common/model_path.resolve_model_path` **不会自动补 `.pt` 后缀**。
而自训练权重归档在 `models/trained/`，**不在搜索目录里**。

所以：

- 用 `models/trained/<stem>-best.pt` → **给全路径**；
- 给 `yolo11n` 这种不带后缀的名字 → 本地找不到，会交给 ultralytics 联网下载
  （给全路径，或把权重放进 `models/pretrained/` 并写全名 `yolo11n.pt`）。
</details>

<details>
<summary><b>服务器 / 无人值守环境推理卡住</b></summary>

`--show` 会弹窗。默认 `infer.yaml` 里 `show: false`，但 CLI 一旦给了 `--show` 就会开窗。
无 GUI 环境请显式 `--no-show`。
</details>

<details>
<summary><b>美化绘制关掉之后还想改框的颜色 / 类名</b></summary>

`apps/platform/configs/runtime/infer_pipeline.yaml` 的 `visualization` 段管这些：
`enabled`、`use_label_mapping`、`label_mapping`（英文→中文）、`color_mapping`（BGR）、
`default_color`、`font_path`，以及透传给 `DrawStyle.from_image_size` 的 `style` 覆盖项。
`--no-viz` 会整体关掉美化、改走 ultralytics 原生 `plot()`。
详见 [ADR-004](docs/architecture/ADR-004-inference-output-and-viz.md)。
</details>

<details>
<summary><b>推理视频产物特别大 / 想改落盘文件名与编码器</b></summary>

落盘参数**不写死在代码里**：`apps/platform/configs/runtime/infer_pipeline.yaml` 的
`output` 段管 `filename_prefix`、`video_basename` / `video_format` / `video_fourcc`、
`video_fps` / `video_fps_fallback`、`image_format` / `image_quality`、
`anonymous_frame_template`、`overwrite_existing`、`keep_audit_json` / `audit_filename`。
默认值逐字段等于改造前 `LocalFileSink` 写死的那些值（`output.mp4` / `mp4v` / 兜底 `30.0` …），
所以不配就不会改变产物。

想换 H.264 / Motion-JPEG 只改 yaml（容器与编码器要配对）：

```yaml
output:
  video_format: "avi"
  video_fourcc: "MJPG"
```

生效值与来源会打进本次推理日志，并写进 `odp_audit.json` 的
`pipeline.output` / `pipeline.output_sources`；字段表见
[docs/manual.md](docs/manual.md) 的 `odp-infer` 一节，
决策记在 [ADR-004](docs/architecture/ADR-004-inference-output-and-viz.md)。
</details>

<details>
<summary><b>训练被拦："数据身份核验未通过"</b></summary>

`odp-train` 默认要求数据身份对账通过（`require_validated=True`）。请先跑
`odp-validate`（它会盖 `validate.stamp.json`）。确实要跳过就加 `--skip-validate`
——审计快照里会留下"跳过"的痕迹，这是有意的。
</details>

<details>
<summary><b>磁盘被 `runs/` 吃掉</b></summary>

`odp-clean` 的保留策略：在每个受管子目录里保留**最新的
`--keep-latest` 个（默认 5）**，并且**`--max-age-days` 天内（默认 30）的一律不动**；
**两个条件都满足才删**（宁可留着）。`runs/lineage/` 硬性豁免。
默认只预演，加 `--yes` 才动手。
</details>

## 数据身份与审计（一句话版）

- `odp-transform` 产出**契约指纹** `contract_fingerprint` 与 `manifest.json`，
  写进数据集 yaml 的 `odp_meta` 段（"产品合格证"）。
- 每一站在 `runs/lineage/<data_fingerprint>/` 留下一联章：
  `transform.stamp.json` / `validate.stamp.json` / `train.stamp.json`。
  **目录名就是数据指纹** —— 数据一变，指纹变，指向空目录，旧章自动失效，不需要作废机制。
- `odp-train` 开训前对账；每次训练/推理都在现场留 `odp_audit.json`。

细节见 [ADR-005](docs/architecture/ADR-005-progress-retention-eval-and-label-dir-fix.md)
与本仓库既有现场 `runs/training/20261027-115300/odp_audit.json`。

## 文档

| 文档 | 内容 |
|---|---|
| [docs/manual.md](docs/manual.md) | 使用手册：每个命令一节（用途 / 参数表 / 示例 / 产物 / 失败模式） |
| [docs/architecture/README.md](docs/architecture/README.md) | ADR 索引 |
| [docs/architecture/ADR-001-monorepo.md](docs/architecture/ADR-001-monorepo.md) | Monorepo + `apps/` 布局 |
| [docs/architecture/ADR-002-dataset-yaml-paths.md](docs/architecture/ADR-002-dataset-yaml-paths.md) | 数据集 yaml 的路径解析与可移植性 |
| [docs/architecture/ADR-003-runtime-config-precedence.md](docs/architecture/ADR-003-runtime-config-precedence.md) | 运行配置三源合并与优先级 |
| [docs/architecture/ADR-004-inference-output-and-viz.md](docs/architecture/ADR-004-inference-output-and-viz.md) | 推理输出与可视化 |
| [docs/architecture/ADR-005-progress-retention-eval-and-label-dir-fix.md](docs/architecture/ADR-005-progress-retention-eval-and-label-dir-fix.md) | 进度反馈 / `runs/` 保留 / 评估与 IoU 口径 / 标签目录缺陷修正 |