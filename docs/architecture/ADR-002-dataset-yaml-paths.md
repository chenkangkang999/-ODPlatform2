# ADR-002：数据集 yaml 的路径约定 —— `path` 写【相对仓库根】，读取端统一解析

| 项目 | 内容 |
|---|---|
| 状态 | ✅ Accepted |
| 决策日期 | 2026-10-27 |
| 取代 | 本 ADR 初版（同日）曾采用"`path` 写绝对路径 + 读取端多候选探测"；当天的 CI 门禁评审后**升级为本篇的方案 A**，初版记录见「决策演化」 |

## 背景

`odp-transform` 产出的数据集 yaml（`apps/platform/configs/datasets/<name>.yaml`）是**进 git** 的：
它同时充当"产品合格证"（`odp_meta` 里有契约指纹与 manifest 引用），
`odp-validate`、`odp-train`、`odp-eval` 都要读它。

它里面有三个层级的路径，语义完全不同：

```yaml
path: data/processed/demo                     # ① 数据集根（相对仓库根）
train: images/train                           # ② 相对 ① 的 split 目录
val:   images/val
test:  images/test
odp_meta:
  manifest_path: runs/data_pipeline/20261027-115547/manifest.json   # ③ 仓库相对
```

难点在于 **三个消费方的解析规则不一样**：

| 消费方 | 它怎么解析路径 |
|---|---|
| D4 快照 `data_validation/snapshot.py` | 原先只认"绝对路径，或相对 yaml 所在目录" |
| D5 评估 `evaluation/dataset.py` | 依次尝试：yaml 里显式写的 `path` → yaml 所在目录 → yaml 的父目录；并把 `../x` 逐层剥掉前导 `..` 后再试 |
| D6 训练 `training/service.py` | 原样把 yaml 交给 ultralytics |
| ultralytics | 把相对 `path` 解析成**它自己的 `DATASETS_DIR / path`**，与本仓库约定完全不同 |

另外还有两个**真实存在**的外部写法必须兼容：

- Roboflow 导出的 `train: ../train/images` —— 它假设 `data.yaml` 被放在子目录里，
  照字面解析会跑到数据集外面去；
- Roboflow 的 `names` 是 list，本仓库生成的 `names` 是 dict（`{0: 名字, …}`）。

初版把 `path` 写成绝对路径（`D:/ODPlatform2/data/processed/demo`），
结果是把**机器信息写进了进 git 的文件**：clone 到别的盘符/目录，这几份 yaml 就指向不存在的目录，
而且仓库 CI 里"禁止硬编码绝对路径"的守卫会直接抓到它们。

## 备选方案

- **方案 A：`path` 写相对仓库根（如 `data/processed/demo`）** ← **采纳**
  可读、可移植、能进 git；代价是喂 ultralytics 之前要落一份"绝对路径运行期副本"，
  且四个消费方要统一到同一套解析规则。
- **方案 B：`path` 写绝对路径，读取端各自把能想到的写法都试一遍**（初版采用的方案）
  写入端最简单、喂 ultralytics 最省事；代价是 yaml 里混进机器信息，且与 CI 守卫冲突。
- **方案 C：把数据集 yaml 从 git 里摘掉**，每次本地生成。
  可移植性问题消失，但丢掉"合格证"的版本留痕，且新克隆的人必须先跑 `odp-transform` 才能训练。

## 决定

采用 **方案 A**，并把"路径怎么解析"收进**唯一一处公共件** `common/dataset_yaml.py`：

```python
to_repo_relative(path)                  # 写：仓库内 → 相对仓库根的 POSIX；仓库外 → 绝对路径 + warning
resolve_dataset_root(yaml_path, raw)    # 读：绝对 → 原样；相对 → 先相对 yaml 目录，再相对 ROOT_DIR
load_dataset_yaml(yaml_path)            # 读：顶层必须是 mapping
write_resolved_copy(src, dest)          # 落"运行期副本"：把 path 解析成绝对后另存一份
```

1. **写入端**（`yaml_writer.py`）：`path` 走 `to_repo_relative(dataset_root)`
   —— 数据在仓库内就写 `data/processed/<name>`；用户在仓库外自带数据集时才退回绝对路径，
   并记一条 warning 说明"该文件不可移植"（那是数据的真实位置，不是我们的选择）。
   `train/val/test` 统一写 `images/<split>`；`odp_meta.manifest_path` 写仓库相对路径
   （由 `paths.ROOT_DIR / ref` 解析，`ROOT_DIR` 靠 marker file `.odp-workspace` 向上查找）。

2. **读取端**：所有消费方都调 `resolve_dataset_root()`，不再各自实现一套：
   - D4 `snapshot._resolve_data_root()` 直接转发给它；
   - `odp-eval` 的 `build_dataset_view()` 把解析出来的数据集根插到候选目录**首位**，
     同时保留原有的"yaml 目录 → 父目录"候选与 `../` 逐层剥离，Roboflow 布局照样能解析；
   - D6 `training._run_training()` 在开训前调
     `write_resolved_copy(data_yaml, run.artifact_path("data.resolved.yaml"))`，
     **只把副本交给 ultralytics**，源 yaml 保持可移植（副本活在 `runs/` 下，不进 git）。

3. **不把 `data/raw` 直接交给框架**：`odp-eval` 用**硬链接**在
   `runs/detect_eval/<run_id>/_dataset_view/` 下搭只读视图，并写一份只用相对路径的规范化
   `data.yaml`。理由是三条真实约束：
   ① `data/raw` 是只读区，而 ultralytics 在 val 时会往数据集目录写 `labels.cache`；
   ② Roboflow 的相对路径常常是坏的；
   ③ ultralytics 强制要求 `train:` 与 `val:` 两个键都在，val-only 也必须给 `train:`。

4. **类名两种写法都认**：`class_names_from()` 同时支持 `names: [a, b]` 与 `names: {0: a, 1: b}`；
   并额外用 `reconcile_class_count()` 校验 `nc` 与 `names` 长度一致 —— 不一致会当场报错，
   因为"每类指标整体错位一格，而数字看着很正常"是最难查的一类问题。

## 理由

1. **进 git 的文件不该带机器信息**。仓库根靠 marker file 定位、契约指纹与 manifest 引用
   都不含盘符，`path` 是最后一个漏点；补上之后"整条对账链路跨机器可用"才是完整的。
2. **一处解析，四处受益**。路径规则写在 `common/dataset_yaml.py`（公共基础层，见 D0 决定 7），
   D4/D6/评估都只向下依赖它；以后加第五个消费方也不会再长出第五套解析。
3. **冗余副本只活在运行现场**。`write_resolved_copy` 把"可移植的源"与"框架要的绝对路径"
   分开：源 yaml 是对外契约，副本是一次性投喂 —— 这也是 ultralytics 的语义差异
   （它把相对 `path` 解析到自己的全局目录）在本仓库的**唯一**落点。
4. **读取端仍然宽容**。`resolve_dataset_root` 的"先 yaml 目录、后仓库根"顺序，
   加上评估侧保留下来的 `../` 剥离与多候选，使外部 Roboflow yaml 与本仓库 yaml 都能直接用；
   `resolution_notes` 还把"相对路径是怎么找到的"记进报告 —— 路径没解析对时能省掉半小时排查。
5. **CI 能守住**。`.github/workflows/ci.yml` 的"硬编码绝对路径"守卫（测试文件豁免）
   现在对本仓库源码/配置恒为 0 命中，规则从"人记得别写盘符"变成"机器拦得住"。

## 后果

- **正面**
  - `configs/datasets/*.yaml` 可以在任何机器、任何目录直接使用，不用先重跑 `odp-transform`；
  - 喂 ultralytics 的路径语义差异被收敛到 `write_resolved_copy` 一个函数；
  - CI 绝对路径守卫恒为 0 命中（验证方式见下方）；
  - `runs/lineage/<data_fingerprint>/`、`odp_meta.manifest_path`、`path` 三者都不含盘符。
- **负面 / 代价（如实记录）**
  - 每次训练多一份运行期副本 `runs/training/<run_id>/data.resolved.yaml`（KB 级，随 run 目录被 `odp-clean` 清理）；
  - **契约指纹对 yaml 文本不敏感**：改 `path` 的写法不会改变 `contract_fingerprint`
    （指纹只覆盖数据集名/策略/种子/比例/类名/样本 sha256），所以这次改造让旧 yaml 与新 yaml
    的指纹保持一致，D3/D4 的章不需要重盖 —— 这是有意的：指纹回答"数据内容是什么"，
    不回答"路径怎么写"。
  - `path` 指向**仓库外**的用户数据集时仍是绝对路径（那种数据本来就不在仓库里），
    写入端会打印 warning；这类 yaml 不应签入 git。
- **中性**
  - 读取端比写入端更宽容（保留 Roboflow 式相对解析），所以"外部 yaml 能用"这件事不依赖写入端。

## 验证方式

```bash
# 1) 写入端：path 必须是相对仓库根
python -c "import glob,yaml;[print(f, yaml.safe_load(open(f,encoding='utf-8'))['path']) for f in sorted(glob.glob('apps/platform/configs/datasets/*.yaml'))]"

# 2) 守卫：源码/配置里不得出现盘符绝对路径（测试文件豁免）
grep -rInE '(^|[^A-Za-z0-9])[A-Za-z]:[\\/]' --include='*.py' --include='*.yaml' \
     --exclude='test_*.py' --exclude-dir=.git --exclude-dir=.venv --exclude-dir=runs . || echo "✅ 0 命中"

# 3) 读取端：D4 / 评估 / 训练三条链都要能从相对 path 找到数据
odp-validate apps/platform/configs/datasets/synth.yaml
odp-eval --weights models/pretrained/yolo11n.pt --data apps/platform/configs/datasets/synth.yaml --limit 4
odp-train --data synth --epochs 1 --imgsz 64 --batch 0.2 --skip-validate
# 训练产出的 runs/training/<run_id>/data.resolved.yaml 里 path 应是绝对路径
```

## 决策演化

- 初版（同日）：采用方案 B（`path` 绝对路径 + 读取端多候选），并把"负面：yaml 带盘符"如实记入后果。
- 评审后（同日）：CI 守卫实测在 4 份 `configs/datasets/*.yaml` 上命中盘符绝对路径，
  且方案 A 的三步改造（`yaml_writer` / 读取端统一解析 / 训练前落副本）成本可控，
  于是升级为方案 A。方案 B 的"读取端宽容"被**整条保留**，因为外部 yaml 依然需要它。

## 撤销条件

- 若未来 ultralytics 支持"相对仓库根"的数据集路径语义，可删掉 `write_resolved_copy` 这一步
  （其余决策不变）。
- 若决定把 `configs/datasets/*.yaml` 移出 git（方案 C），则写入端可重新退化为绝对路径，
  本 ADR 由新决策取代。
- 若出现"数据集目录必须与 git 工作树分离（网络盘/对象存储）"的需求，
  `path` 会重新变成外部引用，需要在本 ADR 里补注新的解析来源。