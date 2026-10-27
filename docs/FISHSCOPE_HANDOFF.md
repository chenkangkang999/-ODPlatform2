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
使用独立分支与 PR，不直接合并主分支，不强制推送。

## 启动与模型接入

详见 [Web 与 LLM 指南](WEB_AND_LLM.md)。先安装 Python 3.12 的平台与 Web 依赖，并在前端运行 `npm ci`、`npm run build`，再启动本机 Django 服务。
Web 训练复用原有平台 CLI，保留数据身份核验；推理前校验本地权重的 33 类标签。

用户提供的 `fish.zip` 已本地核对：4560 张图片、4560 份 YOLO 标签、6034 个有效框，类别 ID 为0–32。
该包没有类别名字典、冻结划分、平台血缘文件或 `.pt` 权重，只作为原始数据参考，未覆盖现有数据配置。
原始压缩包、数据、权重、运行产物和本地环境不在本次提交中。
页面展示的 mAP 指标来源于仓库文档，不能称为本轮重训或复测成绩。
蓝色背景和主图为 AI 装饰素材，不是训练样本或识别证据。

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
媒体链路测试使用明确的预测器替身，不代表真实鱼模型精度。当前缺少鱼权重，真实检测尚待接入。

上传前本地验证：平台测试 **534 passed / 1 skipped**；Web 后端 **26 passed**；Django配置检查及Vue生产构建通过。
已核对原始平台源码与配置相对最新主分支没有差异，并检查暂存区不含真实密钥、数据、权重或运行产物。
