# 渔见 FishScope · Web 与 LLM 交接备注

## 本次提交范围

新增 Django + Vue 3 门户，包含项目介绍、数据理解、模型训练、模型推理、仪表盘和智能助手。
页面已按本次项目要求加入顶部导航、较大字号、蓝色水下背景、多种鱼形图标及鱼形置信度滑块。

- `apps/web-frontend/`：Vue 页面、原创图标、生成式装饰图片与来源说明。
- `apps/web-backend/fishweb/`：本地 API、后台任务、YOLO 媒体处理和证据解释。
- `apps/web-backend/manage.py`、`requirements.txt`、`.env.example`：运行入口和配置示例。
- `apps/web-backend/run_gradio.py`：使用公开配置接口定制展示标题，复用原有 Gradio。
- `apps/web-backend/README.fishscope.md`、`docs/WEB_AND_LLM.md`：安装及使用说明。
- `scripts/start_fish_web.ps1`、`scripts/start_fish_gradio.ps1`：本机启动脚本。
- `.github/workflows/fishscope-web.yml`：仅针对 Web 扩展的构建及测试。

原 `apps/platform/` 引擎源码、共享 YAML、训练数据配置及同学已有的后端占位 README 均保留。
根 README 只追加本扩展入口，`.gitignore` 追加本地配置/依赖/运行产物排除项。
基于上传前的最新主分支开发，保留同学已提交的 YOLO11s 训练参数调整。
基础版本已上传项目主分支。本轮更新按用户要求预览并直接上传，合为一个后续功能提交，不增加额外的合并提交。

## 启动与模型接入

详见 [Web 与 LLM 指南](WEB_AND_LLM.md)。先安装 Python 3.12 的平台与 Web 依赖，并在前端运行 `npm ci`、`npm run build`，再启动本机 Django 服务。
Web 训练复用原有平台 CLI，保留数据身份核验；推理前校验本地权重的 33 类标签。

用户提供的 `fish.zip` 已本地核对：4560 张图片、4560 份 YOLO 标签、6034 个有效框，类别 ID 为0–32。
该包没有类别名字典、冻结划分、平台血缘文件或 `.pt` 权重。现已复用仓库类别契约及平台转换、分层划分、质检流程，生成本机数据与独立配置 `runs/web/dataset/fish.yaml`；实际3652/458/450、15/15 PASS，身份核验成功。原共享配置未改动。
原始压缩包、数据、权重、运行产物和本地环境不在本次提交中。
页面展示的 mAP 指标来源于仓库文档，不能称为本轮重训或复测成绩。
蓝色背景和主图为 AI 装饰素材，不是训练样本或识别证据。
类别卡片新增居中的中文显示名，原训练名称及编号保持不变；移除用户指定的来源说明。顶部“监控仪表盘”显示真实本机资源与近期任务日志。

后续收到的 `20261027-124755.zip` 已完成接入：实际 best 权重与 33 类字典一致，训练数据身份与本机数据一致。单张真实 CPU 推理成功，未重训。10 轮 CSV 曲线和训练完成验证汇总保存为脱敏轻量摘要 `fish_training_run.json`；权重、训练原媒体和本地数据仍被忽略。四段用户指定的页面说明已删除，原有居中排版保留。

Web 训练适配补充：平台 CLI 的 `--batch` 参数会转成浮点数，整数批大小会在配置校验时被拒绝。Web 工作进程现在将完整共享训练配置复制为任务目录下的私有 YAML，仅覆盖整数 batch，并通过 `--config` 调用原 CLI；共享配置和数据身份门都保留。此修复未改动平台代码，也未重试任何训练任务。

本轮后端 45 项测试通过；仅安装 Web 依赖的 CI 环境中 43 项通过、2 项可选平台集成测试跳过。单张真实图片及三帧合成验证短视频识别成功，后者验证了 H.264 标注输出、浏览器播放和基于检测证据的本地规则解释。合成视频仅用于程序验收，不作为实际水下场景效果评测。

## 大模型当前状态与密钥

**尚未提供真实 API，未验证真实模型服务连通性。**
当前可用的是清楚标记为“本地规则解释”的问答模式；程序已实现 OpenAI 兼容接口，但不能把规则回答算成大模型调用。

将 `.env.example` 复制成同目录 `.env`，只在本机填写：

```dotenv
FISH_LLM_BASE_URL=
FISH_LLM_API_KEY=
FISH_LLM_MODEL=
```

后端入口自动加载 `apps/web-backend/.env`，已存在的环境变量优先。
`.env`、`.env.*` 已在 `.gitignore` 排除，只有空值的 `.env.example` 例外允许提交。
密钥不写进前端、不返回给浏览器，也不把密钥本身写到 `.gitignore` 中。
未来配置完成后，需要实际验证一次模型请求；失败会明确回退为本地规则说明。
模型服务只接收问题、项目事实和结构化检测摘要，原视频留在本机。

## 验证口径

```powershell
python apps/web-backend/manage.py check
python apps/web-backend/manage.py test fishweb.tests
cd apps/web-frontend
npm ci
npm run build
```

测试覆盖文件/路径与参数校验、任务状态、33类字典、视频编码、HTTP Range、LLM摘要与失败回退、`.env`加载及环境变量优先。
单元测试中的媒体夹具只验证程序链路；另使用本次收到的真实鱼类权重验证图片识别。单张成功不代表完整验证集精度，训练指标来源于同学提供的原始记录。

上传前本地验证：平台测试 **534 passed / 1 skipped**；Web 后端 **26 passed**；Django配置检查及Vue生产构建通过。
已核对原始平台源码与配置相对最新主分支没有差异，并检查暂存区不含真实密钥、数据、权重或运行产物。
