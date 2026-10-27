# ADR-004：推理输出与可视化 —— 美化绘制与输出落盘参数均已落地

| 项目 | 内容 |
|---|---|
| 状态 | ✅ Accepted（第 2 条原为待办，已落地到 `infer_pipeline.yaml` 的 `output` 段） |
| 决策日期 | 2026-10-27 |
| 落地日期 | 2026-10-27（`output` 段） |

> 补注（按维护要求更新）：仓库 ADR 约定“决策变更时新增一条 ADR、旧记录只标注被取代”，
> 而本篇「撤销条件」也写着“`output` 段加成配置项后第 2 条由新 ADR 取代”。
> 本次按维护指令**就地更新**第 2 条（原文的“写死在 sink 里”改为“已配置化”），
> 并在下文逐字段标出“原写死位置”，以免丢掉历史。往后 `output` 段再扩（如新增输出后端）
> 应另开新 ADR，不再就地改写。

## 背景

推理结果的落盘与可视化原先有几处写死的值，且美化绘制曾经是个**桩**：

- `inference/sinks.py` 的 `LocalFileSink` 里写死 `"output.mp4"`、四字符编码器 `"mp4v"`、
  以及帧率兜底 `30.0`；
- `visualization/BeautifyVisualizer` 曾经是桩：`from_yolo_results()` 只打包标签
  （boxes / confidences 被丢掉），`draw()` 直接 `return image`。

第二条的后果最严重且最隐蔽：**默认（美化开启）跑出来的视频/图片没有任何检测框，
既不报错，审计里还写着 `"beautified": true`**。用户会以为模型什么都没检测到，
而统计里的检测数是正常的。

本仓库当前的实际状态是：**美化绘制已经实现**（`visualization/__init__.py` 的
`from_yolo_results()` 返回 `[{"cls", "conf", "xyxy"}, …]`，`draw()` 真的画矩形与说明文字，
`DrawStyle.from_image_size()` 按画面短边相对 720p 缩放字号与线宽），
**输出落盘参数也已从 `LocalFileSink` 提到 `infer_pipeline.yaml` 的 `output` 段**
（见「决定」第 2 条）。

## 备选方案

**关于美化绘制：**

- 方案 A：实现真正的绘制，默认走美化（**已采纳**）；
- 方案 B：删掉美化路径，统一用 ultralytics 原生 `plot()`；
- 方案 C：保留桩，但在运行开头显式告警并回落到原生绘制。

**关于输出落盘参数：**

- 方案 D：把文件名 / 编码器 / 兜底帧率提到 `infer_pipeline.yaml` 的 `output` 段（**已采纳**）；
- 方案 E：维持写死在 `LocalFileSink` 里（**已被方案 D 取代**）。

## 决定

1. **美化绘制走方案 A，并把样式做成可配置。**

   绘制约定（与 `inference/overlay.py` 的 HUD 一致：**原地修改并返回同一个数组**）：

   - 每个检测画一个矩形，颜色按类别名从 `color_mapping` 取，取不到用 `default_color`；
   - 框上方画一行说明 `类名 #轨迹号 置信度`（有哪段写哪段），类名可经 `label_mapping` 改名；
   - 说明文字先铺半透明底衬再写字 —— 否则浅色背景上白字看不见。
     实现上**先把所有底衬画在副本上、一次性按 alpha 混合**，
     而不是逐块混合（逐块混合会让重叠区域透明度叠加，颜色越压越黑）；
   - 短边小于 `DrawStyle.min_box_pixels`（默认 2px）的框不画（噪声框）；
   - 所有几何 / 字号 / 颜色都来自 `DrawStyle`，本模块不写魔法数字。

   开关与样式的**唯一配置入口**是
   `apps/platform/configs/runtime/infer_pipeline.yaml` 的 `visualization` 段：
   `enabled`、`use_label_mapping`、`label_mapping`、`color_mapping`、`default_color`、
   `font_path`、`style`（透传给 `DrawStyle.from_image_size`）。
   实际是否美化 = `CLI 的 beautify 且 pipe.viz_enabled`（`service.py` 里
   `do_beautify = beautify and pipe.viz_enabled`），审计里记的 `beautified` 就是它。

   `--no-viz` 等价于 `beautify=False` → 退回 ultralytics 原生 `plot()`；
   `--no-hud` 等价于 `show_info=False` → 不叠 HUD 信息面板。

2. **输出落盘参数走方案 D：提到 `infer_pipeline.yaml` 的 `output` 段。**

   落点（“待办”一条已落地）：

   - 解析：`inference/pipeline_config.py` 的 `OutputConfig`（Pydantic v2，`extra=forbid`）
     + `load_pipeline_config()` 读 `output` 子字典；
   - 消费：`inference/sinks.py` 的 `LocalFileSink(output_config)` 全部从 `OutputConfig` 取，
     模块内不再有写死的文件名 / 扩展名 / 编码器 / 质量；
   - 透传与可查：`inference/service.py` 把 `PipelineConfig.output` 交给 `LocalFileSink`，
     并用 `common/config_log.log_effective_config` / `log_override_chains`
     打出“每个输出字段的生效值与来源”（DEFAULT / YAML / YAML(无效,已回落默认)）；
   - 留痕：审计快照 `pipeline.output`（生效值）+ `pipeline.output_sources`（来源）一并落盘。

   字段表（默认值 = **改造前 `LocalFileSink` 的实际行为**，所以“不配”与“改前”逐字节一致）：

   | 字段 | 默认值 | 原写死位置 | 说明 |
   |---|---|---|---|
   | `filename_prefix` | `""` | （新增能力，原先没有前缀） | 产物文件名前缀，空串=不加 |
   | `video_basename` | `"output"` | `sinks.py` `"output.mp4"` | 视频文件名主干 |
   | `video_format` | `"mp4"` | `sinks.py` `"output.mp4"` | 视频容器扩展名（mp4/avi/mov/mkv） |
   | `image_format` | `"jpg"` | `sinks.py` `f"{stem}.jpg"` | 图片扩展名（jpg/jpeg/png/bmp/webp/tif/tiff） |
   | `image_quality` | `95` | `cv2.imwrite` 未传参（OpenCV 默认即 95） | JPEG 质量 1–100，仅 jpg/jpeg 生效 |
   | `anonymous_frame_template` | `"frame_{index:06d}"` | `sinks.py` `f"frame_{frame.info.frame_index:06d}"` | 帧源无文件名时的兜底命名 |
   | `video_fourcc` | `"mp4v"` | `sinks.py` `VideoWriter_fourcc(*"mp4v")` | 编码器四字符码 |
   | `video_fps` | `null`（= `auto`/`source`） | `sinks.py` `float(frame.info.fps)` | 写入帧率；`null`=跟随源帧率，数字=固定 |
   | `video_fps_fallback` | `30.0` | `sinks.py` `... else 30.0` | 源帧率缺失或为 0 时的兜底 |
   | `overwrite_existing` | `true` | `cv2.imwrite` / `VideoWriter` 的隐式覆盖 | `false`=同名跳过（日志说明跳过了谁） |
   | `keep_audit_json` | `true` | `service.py` 无条件写 | `false`=不写审计快照（日志说明是按配置跳过） |
   | `audit_filename` | `"odp_audit.json"` | `service.py` `output_dir / "odp_audit.json"` | 审计快照文件名 |

   行为不变是硬要求，因此默认值逐字段对齐上表；`MP4V` 仍是默认编码器
   （OpenCV 自带、**不依赖外部编码器**），代价仍是压缩率低。

3. **输出目录的命名规则与训练对齐。**

   `runs/<task>_infer/<name>/`，`<name>` = `config.experiment_name or "predict"`；
   重名时自增（`predict` / `predict2` / `predict3` …），由
   `_resolve_output_dir()` 实现 —— 与 ultralytics 的行为对齐
   （注释原话：「跟 ultralytics 行为对齐: 重名自增 name2/name3...」）。

4. **输出目标做成适配器（`OutputSink`），而不是把 `cv2.imwrite` 散在流程里。**

   生命周期 `open() → write() × N → close()`，实现纪律：

   - `write()` 内部 `try/except` 包住，单帧写入失败只 `logger.warning` 并跳过，**不抛**；
   - `close()` **幂等**（允许多次调用）；
   - `open()` 由 pipeline 在首批帧到达后调用（此时 `source_type` 已知）。

   仓库内置两个实现：`LocalFileSink`（CLI 默认）与 `NullSink`（`--no-save`、
   只跑统计、或把 annotated 帧经别的通道推走时用）。
   业务端可以继承 `OutputSink` 写自己的 sink（对象存储 / 推流 / Qt 信号）——
   `apps/desktop/qt_sink.py` 就是按这个接口写的。

5. **「不静默降级」是一条纪律。**
   任何一个"有配置项但没生效"或"该出图却没出图"的情况，都必须在
   日志 / 审计 / 文档三处可见。ADR-004 的上一版就是因为违反这条才需要重写；
   现在 `beautified` 记录的是真实走过的分支。

## 理由

1. 输出可视化是**使用者第一眼看到的东西**。"默认没有框"会直接摧毁对模型判断的信任，
   而对它的排查成本远高于实现绘制本身。
2. 美化牵涉视觉规范（配色、标签映射、字号自适应），把它做成 `DrawStyle` 数据类 +
   yaml 覆盖，可以让"改配色"不需要改代码。
3. `OutputSink` 抽出来之后，"写哪里"与"怎么算"解耦：
   同一套 pipeline 能同时服务 CLI（落盘）、无头服务（NullSink）、桌面端（Qt 信号）。
4. 输出落盘参数（尤其编码器）是**用户会真的想改**的东西，但当前改一处就要动代码；
   如实写清楚比假装可配置更有用。

## 后果

- **正面**
  - 默认路径（美化开启）能真的画出框与置信度，并且 `beautified: true` 名副其实；
  - 颜色 / 中文标签映射 / 字号线宽可通过 `infer_pipeline.yaml` 调，不必改代码；
  - 输出文件名（前缀 / 主干 / 扩展名）、视频编码器与容器、jpg 质量、写入帧率、
    同名覆盖策略、审计快照开关与文件名，全部可通过 `infer_pipeline.yaml` 的 `output` 段调，
    改完不必碰代码；生效值与来源既进日志也进审计快照；
  - “不配”与“改前”逐字节一致（默认值就是原先写死的那些值，已用产物 sha256 对账）；
  - 输出目录命名与训练侧对称（`runs/<task>_infer/` vs `runs/training/`），
    两者都由 `RunContext` / 自增逻辑保证不互相覆盖。
- **负面（如实记录）**
  - `font_path` 字段目前是**预留**：`DrawStyle.font` 用的是
    `cv2.FONT_HERSHEY_SIMPLEX`（其字面值就是 0，模块顶层不 `import cv2`，只在绘制时导入），
    Hershey 字体**不读外部字体文件**，所以 `label_mapping` 里的中文标签会渲染成乱码。
    手册里把这条列进了排障表。
  - `--show` 在无 GUI 环境会卡住，默认路径依赖使用者记得 `--no-show`
    （`infer.yaml` 模板里 `show: false`，但 CLI 给了 `--show` 就会开窗）。
  - 编码器与容器必须**自己配对**（如 `MJPG`+`avi`）：写错了 `cv2.VideoWriter` 会打不开，
    此时日志会打 `error` 明说视频写不出来（不静默降级），但不会替使用者自动换编码器。
  - 输出的归属目录仍是 `runs/<task>_infer/<name>/`（由 `experiment_name` 决定），
    **不在 `output` 段里** —— 目录命名规则见第 3 条，与训练侧对齐，故意不放开。
- **中性**
  - 推理侧没有 mAP，取而代之的是帧数 / 检测数 / 每类计数 / 多维帧率
    （capture / infer / render / loop / current），由 `overlay.Metrics` 采集、
    `log_infer_stats()` 打印，并写进 `odp_audit.json` 的 `stats`。

## 撤销条件

- 把 `output` 段（文件名 / `fourcc` / 兜底帧率）加成 `infer_pipeline.yaml` 的配置项、
  并由审计快照记录选定值后，本篇第 2 条由新 ADR 取代
  —— **此条件已满足**（`output` 段已存在，生效值与来源记在 `pipeline.output` /
  `pipeline.output_sources`）；第 2 条如再扩展（例如新增输出后端）建议另开新 ADR；
- 若实现真正的中文标签渲染（引入 PIL/FreeType 或点阵字库），第 4 条负面项消解，
  需在本 ADR 补注；
- 若决定永远不做美化（回到方案 B），则删除 `visualization` 模块与相关配置项，
  并在本 ADR 标注 `Superseded`。