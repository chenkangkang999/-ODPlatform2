# 渔见 FishScope：Web 与智能助手

## 项目在做什么

ODPlatform 把“数据转换 → 质量检查 → YOLO 训练 → 评估 → 推理”组织成一套可追溯的流程。
当前应用是 **33 类淡水鱼细分类检测**：既画出鱼的位置，又判断品种。

本次增加 Django + Vue 3 网页，把已有命令行能力放进可操作的页面，并补充视频推理后的自然语言解释。
页面使用顶部横向导航切换六个工作区；蓝色水下场景背景、鱼群、气泡和浅蓝色细框贯穿各页。
保留场景主视觉和浅色卡片，让鱼类主题清晰可见，操作区文字保持可读。
界面字体已放大，类别卡片使用十种装饰鱼形并去掉角落序号；置信度仍为可键盘操作的真实滑块，滑块外观改为鱼形。
首页与全页背景是AI生成的装饰素材，不作为训练样本或检测结果。

## 对应任务

| 任务 | 实现 |
| --- | --- |
| Gradio 演示 UI | 保留核心 `odp-demo`，通过网页入口及独立启动脚本访问 |
| Web 展示 | Django 同源后端 + Vue 3 / Vite 前端 |
| 项目介绍 | 鱼类检测背景、指标来源、完整工作流 |
| 数据理解 | 33 类列表、训练/验证/测试划分、文档统计与本地可用状态 |
| 模型训练 | 参数表单、启动真实训练任务、日志、取消；不生成模拟曲线 |
| 模型推理 | 图片/视频上传，本地 YOLO 检测，标注产物与结构化报告 |
| 仪表盘 | 本机任务与实验记录、文档中的独立验证集指标 |
| 智能助手 | 项目问答、对完成的检测任务解释；支持 OpenAI 兼容接口 |

## 数据与成绩的口径

来源：仓库 README 和 `apps/platform/configs/datasets/fish.yaml`。这些是同学项目的记录，**不代表网页在新机器上重新训练或复测过**。

- BengalDeltaFish，33 类，4560 张图片，6034 个标注实例。
- 冻结划分：train 3652、val 458、test 450，seed 42。
- README 独立评估记录：**验证集** mAP50 0.9073、mAP50-95 0.7442，Precision 0.8268、Recall 0.8874。
- 训练末轮记录 mAP50 0.9095，属于另一口径，不能当作独立测试集结果。
- 仓库没有提交原始数据、鱼类权重和完整运行报告。本机缺少时，网页明确显示待接入。
- 无逐类真实样本计数时，只展示类别字典，不捏造各类分布。无本机实验 CSV 时不生成训练曲线。

## 首次运行

推荐 Python 3.12 和 Node.js 20 或更高版本，在仓库根目录执行。
先按根 README 的平台环境准备步骤安装引擎，CPU/GPU 的 Torch 构建根据本机情况选择。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e "./apps/platform[ui]"
.\.venv\Scripts\python.exe -m pip install -r apps/web-backend/requirements.txt
cd apps/web-frontend
npm ci
npm run build
cd ../..
powershell -ExecutionPolicy Bypass -File scripts/start_fish_web.ps1 -SkipBuild
```

打开 `http://127.0.0.1:8510/`。该启动方式仅供本机预览，关闭终端或按 Ctrl+C 可停止服务。
已经建好环境和前端依赖后，也可直接运行 `scripts/start_fish_web.ps1`，脚本会重新构建前端。
指定已有 Python：`-Python <虚拟环境中的python.exe>`；换端口：`-Port 8511`。

开发时可分别运行 Django 和 Vue：

```powershell
.\.venv\Scripts\python.exe apps/web-backend/manage.py runserver 127.0.0.1:8510 --noreload
```

另开终端进入 `apps/web-frontend`，执行 `npm run dev`。Vite 将 `/api` 代理到 8510。

## 接入真实数据和模型

将同学训练好的 **33 类鱼 YOLO `.pt` 权重**放入 `models/trained/`，建议保留含 `fish` 的原始文件名。
推理时还会核验模型中的类别字典，防止把通用 COCO 或 smoke 测试模型误当成鱼类模型。
权重只从管理员准备的本地模型目录读取，网页不接受任意上传的模型文件。

数据按已有平台流程接入 `data/raw/fish/`，再运行 `odp-transform` 和 `odp-validate`。
处理后的数据应位于 `data/processed/fish/`，并保留平台生成的 manifest 和验证章。
训练页面复用平台训练入口，因此平台原有的数据身份检查仍生效。

上传图片或视频后，选择本地鱼类权重与参数，点击“开始推理”，等待任务完成。
视频采用逐帧或间隔帧检测，产物包括标注媒体和检测报告。视频统计的“检测框次”会在多帧重复累计，**不是独立鱼的数量**。
视频画面只在本机处理。点击智能解释时，后端向已配置的模型服务发送结构化检测摘要，不发送原始视频。
模型只能解释检测证据和局限，不能从框数推断鱼类健康、生态因果或实际种群数量。

## 配置大模型

推荐复制示例文件到本机配置，再填写这三项：

```powershell
Copy-Item apps/web-backend/.env.example apps/web-backend/.env
```

编辑 `apps/web-backend/.env` 中的 `FISH_LLM_BASE_URL`、`FISH_LLM_MODEL`、`FISH_LLM_API_KEY` 后重新启动后端。
入口会自动加载这个文件；已设置的环境变量优先。真实 `.env` 被 `.gitignore` 排除，仓库只提交空值示例。
密钥只保存在后端，不返回前端，也不写入 `.gitignore` 的正文。不要把真实密钥写入 Git。
也可在启动 Django 的终端中设置：

```powershell
$env:FISH_LLM_BASE_URL = 'https://your-provider.example/v1'
$env:FISH_LLM_MODEL = 'your-model-name'
$env:FISH_LLM_API_KEY = '<在本机填写你的密钥>'
powershell -ExecutionPolicy Bypass -File scripts/start_fish_web.ps1 -SkipBuild
```

配置齐全后，页面只显示“大模型已配置”状态；实际请求成功后才显示模型回答。
没有配置时，助手仍能解释项目、数据划分和检测报告，但会明确标记为 **本地规则说明**。
请求失败会显示失败原因或明确的本地回退状态，不把规则模板伪装成大模型回答。
这里只检查配置是否齐全，**已配置不等于已经验证服务连通**。

## Gradio 演示

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_fish_gradio.ps1
```

独立地址：`http://127.0.0.1:7860/`。使用原有的配置、绘制与权重搜索逻辑。
缺少真实权重时界面可以查看，点击识别会提示所缺文件。

## 验证

```powershell
.\.venv\Scripts\python.exe apps/web-backend/manage.py check
.\.venv\Scripts\python.exe apps/web-backend/manage.py test fishweb.tests
cd apps/web-frontend
npm run build
```

后端测试覆盖参数校验、路径约束、任务状态、缺少鱼权重、结构化统计和 LLM 回退等关键分支。
测试中的引擎夹具仅用来验证程序连接，不作为真实鱼类识别成绩。

## 文件位置与提交范围

- `apps/web-frontend/`：六个 Vue 页面、图表、样式与图片来源记录。
- `apps/web-backend/`：Django API、本地任务、YOLO 与 LLM 服务。
- `scripts/start_fish_web.ps1`、`scripts/start_fish_gradio.ps1`：本地启动。
- `docs/fish_project_evidence.json`：带来源的项目事实记录。
- 本地上传、任务产物、依赖、权重和密钥均在 Git 忽略范围内。

通过独立分支和 PR 提交，供组员审阅后合并。未修改共享平台引擎、训练配置或已有 Gradio 配置。
原有 README 内容保留，新增说明见本文件及 `FISHSCOPE_HANDOFF.md`。
