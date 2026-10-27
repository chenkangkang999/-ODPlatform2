# ADR-006：一切皆配置 —— 用配置层消灭硬编码

| 项目 | 内容 |
|---|---|
| 状态 | ✅ Accepted |
| 决策日期 | 2026-10-27 |
| 适用范围 | 全仓（`apps/platform/` 全部子系统、`scripts/`、CI） |

## 背景

平台跑通之后，评审提出一条纪律：**全局不要硬编码，遵循「一切皆配置」的原则。**

先说清"硬编码"在本仓库的准确定义 —— 不是"代码里出现数字"，而是下面三类：

| 类别 | 反例（改造前真实存在） | 危害 |
|---|---|---|
| **行为取值写死在代码里** | D4 质检的 13 个阈值只在 `common/constants.py`，15 个 check 直接 import；D3 覆盖率闸门 `0.5/0.9` 同样 | 换个数据集想收紧闸门，必须改源码；改完就跑偏了"同一份代码 + 同一份配置 = 同一结果"的可复现承诺 |
| **同一默认值存在多处** | 划分比例在 `constants.py` 有默认、在 CLI 手写一遍、在文档又写一遍 | 改一处忘一处，行为与文档不一致 |
| **机器相关信息进仓库** | 数据集 yaml 里写 `D:/ODPlatform2/...`（见 [ADR-002](ADR-002-dataset-yaml-paths.md)）；`odp-infer` 的日志落 CWD 相对的 `logs/` | clone 到别的机器就坏；CI 守卫抓得到但已经污染了历史 |

而"能配"不等于"配得好"：如果每个子系统各写一套加载/合并/校验，就会出现
"train 能配、validate 不能配""配了不生效""生效了查不到值从哪来"。

## 决定

把 D5 已经立起来的配置范式**升格为全仓唯一范式**，凡是"影响行为的取值"都必须走它：

### 1) 每个子系统一个 Settings 模型（Pydantic v2）

- 字段带 `json_schema_extra={"group","examples","tips","yaml_comment","sensitive"}`：
  字段定义、yaml 模板注释、文档共用一份真相；
- `model_config = ConfigDict(extra="forbid")`：`epchs=300` 这种拼错**在构造时**就报错，
  而不是"静默用默认值跑 5 小时"（拼写防线的价值远大于它的实现成本）；
- 校验用 validator 表达语义：`train_rate + val_rate > 1` 直接拒、`batch=True` 直接拒
  （`isinstance(True, int)` 是 Python 的经典坑）、`imgsz` 非 32 倍数**只 warning 不 raise**
  （能跑但会悄悄 pad，值得提醒不值得拦死）。

### 2) 三源合并，优先级固定 `CLI > YAML > DEFAULT`

- **DEFAULT 层就是 Settings 模型的字段默认值**，而且这些默认值**直接引用唯一出处**：
  例如 `runtime_config/transform.py` 里写
  `coverage_hard_threshold: float = C.COVERAGE_HARD_THRESHOLD`（`C = common.constants`）。
  这样"默认值只存在一处"是**代码上可验证**的，而不是靠人记住；
- YAML 放 `configs/runtime/<name>.yaml`（模板由 `odp-gen-config <name>` 生成，带中文注释）；
- CLI 只暴露最常用的那几个（`--conf` / `--epochs` / `--imgsz` / …），
  **不是所有字段都给 CLI 开关** —— 开关太多等于没开关，深水区交给 yaml。

### 3) 每个生效值都要能查到来源（可观测性）

- 启动时打印 `配置来源报告`：分组列出每个字段来自 `CLI / YAML / DEFAULT`；
- 打印 `配置覆盖情况`：`epochs : 100(DEFAULT) → 200(YAML) → 50(CLI)`；
- 敏感字段（如 `api_key`）在报告里打 `***`，且**不进**审计 JSON；
- 每次运行把生效配置写进 `runs/.../odp_audit.json`；
- Pydantic 的报错被增强成带来源链：`Input should be greater than or equal to 1 [来源: -5(YAML) ← 100(DEFAULT)]`
  —— 报错时不用猜"这个 -5 是哪来的"。

### 4) 路径只有一处真相（SSoT）

所有路径从 `common/paths.py` 出，数据集 yaml 一律写**相对仓库根**（ADR-002）；
`ultralytics` 那种"相对路径按它自己全局目录解析"的语义差异，只在
`common/dataset_yaml.write_resolved_copy()` 一个函数里处理。
CI 用**硬编码绝对路径守卫**（测试文件豁免）把这条钉死。

### 5) 配置层清单（本决策落地后的全貌）

| 子系统 | 配置模型 | yaml 模板 | 主要可调项 |
|---|---|---|---|
| 数据流水线 | `runtime_config/transform.py` | `transform.yaml` | 覆盖率硬/软闸门、稀有类阈值、划分比例、种子、策略、统计阈值 |
| 数据质检 | `runtime_config/validate.py` | `validate.yaml` | 13 个阈值（缺签比、出界比、退化框、无标注比、split 最小图数、失衡比…） |
| 训练 | `runtime_config/train.py` | `train.yaml` | 70 个字段（epochs/batch/imgsz/优化器/增强/…） |
| 评估 | `runtime_config/val.py` + `evaluation/settings.py` | `val.yaml` / `eval.yaml` | split/conf/iou/扫描阈值列表/目标精度/可靠性警告线/IoU 分析开关 |
| 推理 | `runtime_config/infer.py` + `inference/pipeline_config.py` | `infer.yaml` + `infer_pipeline.yaml` | 阈值与输入、帧源参数、美化样式、**输出文件名/格式/质量/编码器** |
| runs 保留策略 | `runtime_config/clean.py` | `clean.yaml` | 保留个数、最大保留天数、豁免清单 |
| 演示界面 | `runtime_config/demo.py` | `demo.yaml` | 服务器地址/端口、默认权重、默认阈值、是否分享 |

## 明确**不可**配置的例外（写下来是为了防止"为了配置而配置"）

配置化的边界是**行为**，不是**安全边界与协议常量**：

1. **销毁类操作的白名单/保护目录**（`common/paths.py` 的 `get_dirs_to_reset()` /
   `PROTECTED_DIRS`）：这是 [D2 的"白名单优于黑名单 + 多层防御"](ADR-001-monorepo.md) 安全设计。
   若允许用配置放开，攻击面就从"代码写错"变成"配置写错"，而配置是**用户可写的** —— 那不是配置化，那是把保险丝换成铜丝。
2. **协议与框架常量**：`cv2` 的 `CAP_PROP_*`、`FOURCC` 的语义、`torch` 的 dtype、
   `hashlib.sha256` 的选择、契约版本号 `TOOL_VERSION`（它参与指纹配方，改了就是换契约，必须走代码评审而不是改 yaml）。
3. **数学边界与容差**：`0/1` 取值范围、浮点比较容差 `1e-6`、
   IoU 分桶的 `0.5/0.6/.../1.0` 边界（这是定义，不是偏好）。
4. **幂等与安全所需的固定值**：例如 `min_interval` 的默认打点间隔、
   `total<=0` 表示"总量未知"这类哨兵语义。

判据一句话：**改它会让"同一份输入得到不同产物"** → 必须进配置；
**改它会让"安全承诺或数据契约失效"** → 必须留在代码里并由评审守住。

## 理由

1. **可复现的一半在配置，另一半在代码**：划分锁了、环境锁了（见 D0 边界声明第 2 条），
   如果阈值还散在源码里，"同样的命令跑出不同结果"就永远说不清。
2. **配置即文档**：字段的 `tips`/`examples` 直接生成 yaml 模板注释，
   用户拿到的模板本身就是说明书，不会出现"文档写的和代码写的不是一个数"。
3. **可观测性是配置化的前提**：不能回答"这个值从哪来"的配置系统，
   在出故障时等于没有配置系统 —— 所以溯源报告、覆盖链、审计快照是**同一件事的一半**。
4. **一处默认值 + 契约测试**，把"改一处忘一处"从人的纪律变成机器的断言。

## 后果

- **正面**
  - 换数据集/换机器，改 yaml 即可；同一份数据 + 同一份配置 → 同一结果；
  - `odp-gen-config` 生成的模板与代码默认值由**契约测试**保证一致（模板 ≠ 代码默认值 → 测试红）；
  - 每个字段的来源可查：来源报告 + 覆盖链 + `odp_audit.json` + 增强报错四处对齐；
  - CI 守卫把"写死盘符""忘记配置化"变成机器能拦的事。
- **负面 / 代价（如实记录）**
  - **配置面变大了**：7 套 yaml、7 个 Settings 模型，学习成本上升；
    缓解手段是"CLI 只给最常用的开关 + 模板自带注释 + `--explain` 式来源报告"。
  - **多一层间接**：读代码时要跳到配置模型才知道默认值是多少；
    缓解手段是默认值全部引用 `common/constants.py`，grep 一个常量就能看到全链路。
  - **契约测试需要维护**：新增字段忘了更新模板注释，测试会红 —— 这是有意的摩擦。
- **中性**
  - `scripts/demo_data/*` 这类**演示数据生成器**保留少量字面量（生成几张图、几个框）：
    它们是"测试夹具的生成脚本"而非平台行为，把它们配置化只会让夹具更难读。
    但**平台会读到的**一切（类别名、划分、阈值）仍走配置。

## 守卫（怎么防止回退）

| 守卫 | 位置 | 拦什么 |
|---|---|---|
| 硬编码绝对路径守卫 | `.github/workflows/ci.yml`（grep `[A-Za-z]:[\\/]`，测试文件豁免） | 盘符/绝对路径进仓库 |
| "模板 == 代码默认值"契约测试 | `apps/platform/tests/runtime_config/**` | 模板与默认值漂移 |
| `extra="forbid"` 拼写防线 | 每个 Settings 模型 | yaml 拼错字段名 |
| pytest 全量 + 阈值真生效用例 | `apps/platform/tests/**` | 改了配置不生效 / 生效了破坏默认行为 |
| 配置来源报告与审计快照 | 各子系统 service | "配了但不知道谁生效" |

## 撤销条件

- 若某个子系统的配置面被证明**无人修改**（长期只有默认值在用），
  可以把该 yaml 从模板里摘掉、只保留 Settings 模型（默认值仍只有一处），
  在其 ADR 里补注即可 —— 但**不建议**把默认值搬回散落的字面量。
- 若引入分层配置（base + 环境覆盖 + 机器覆盖），本 ADR 的"三源合并"升级为"多源合并"，
  优先级表需要重写，届时由新 ADR 取代。