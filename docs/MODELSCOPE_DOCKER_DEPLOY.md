# 渔见 FishScope · ModelScope 创空间 Docker 部署说明

> 本文件记录如何把「渔见 FishScope」完整 Django + Vue 应用部署到魔搭（ModelScope）创空间，
> 以及为云端部署所做的环境变量改造。仓库根目录已包含可直接部署的 `Dockerfile`、
> `requirements.txt`、前端构建产物 `apps/web-frontend/dist` 与已训练权重。

## 1. 架构

- 后端：Django（`apps/web-backend`），一个进程同时提供：
  - `/api/...` 接口（状态、数据集、任务、推理、智能助手）
  - 非 `/api/` 路径：同源返回前端静态文件 `apps/web-frontend/dist/index.html`
- 前端：Vue3 + Vite（`apps/web-frontend`），Docker的Node 22阶段从当前源码运行 `npm ci` 和 `npm run build`，Python运行阶段只接收新构建的dist，不依赖本机旧产物。
- 引擎：`apps/platform/src/od_platform`（训练走 `odp-train` 子进程，推理走 `fishweb/worker.py`）。
- 模型：`models/trained/train-20261027-124755-fish-yolo11n-best.pt`（YOLO11n，33 类淡水鱼）。
- 视频转码：ffmpeg（libx264，无声 H.264）。

## 2. 云端改造（通过环境变量，本地零影响）

| 位置 | 本地默认行为 | 云端放开方式 |
|---|---|---|
| `fishweb/settings.py` | `ALLOWED_HOSTS` 仅本机回环 | 环境变量 `FISH_ALLOWED_HOSTS` |
| `fishweb/views.py` 的 `LocalOnlyMiddleware` | 非 127.0.0.1 一律 403 | 环境变量 `FISH_PUBLIC=1` |

`Dockerfile` 已设置 `FISH_PUBLIC=1` 和 `FISH_ALLOWED_HOSTS=*`，所以容器里自动放开；
不设置这两个变量时，本地运行行为与改造前完全一致（仍只允许本机预览）。

## 3. 部署步骤

1. 在创空间创建空间（Programmatic Studio），SDK 选择 **docker**（不要选 Gradio/Streamlit）。
2. 云资源选「免费 CPU」；Docker 模式以仓库根目录的 `Dockerfile` 为准。
3. README.md 的 frontmatter 需要：
   ```yaml
   ---
   sdk: docker
   app_port: 7860
   ---
   ```
4. 推送代码（master 是受保护分支，默认部署分支）：
   ```bash
   git remote add studio https://www.modelscope.cn/studios/<用户名>/<空间名>.git
   git push studio master
   ```
   认证用户名固定为 `oauth2`，密码用「访问令牌」页生成的有写权限的 Token。
5. 等待构建及部署完成（实际耗时取决于依赖下载与平台资源），状态变为「运行中」后，
   页面即本地 `127.0.0.1:8510` 的完整渔见门户。

## 4. 本地运行

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows；macOS/Linux 用 source .venv/bin/activate
pip install -r requirements.txt
# 前端已构建；如改了前端源码，在 apps/web-frontend 执行 npm install && npm run build
python apps/web-backend/manage.py runserver 127.0.0.1:8510
```
浏览器打开 http://127.0.0.1:8510 。

## 5. 容器关键约定

- 服务必须监听 `0.0.0.0:7860`（见 Dockerfile 的 CMD 与 `EXPOSE 7860`）。
- 任务运行时目录 `runs/web/` 由 Dockerfile 预建。
- 根目录 `requirements.txt` 是「一键部署」用的合并依赖；各子包自身的依赖声明仍保留：
  - 平台引擎：`apps/platform/pyproject.toml`
  - Django 后端：`apps/web-backend/requirements.txt`
- 免费CPU容器先安装本机已验证的 `torch==2.14.1` / `torchvision==0.29.1` CPU轮子，再从官方PyPI安装其余依赖，避免通用依赖解析下载GPU专用大包。算法源码及本机GPU环境不受影响。

## 6. 纳入版本控制的二进制（一次性说明）

项目 `.gitignore` 原本排除 `dist/` 与 `models/trained/*.pt`。为保证仓库 clone 后
**可直接 Docker 部署、无需重新训练或构建前端**，已通过 `git add -f` 显式纳入：

- `apps/web-frontend/dist/`（前端构建产物，约 4.8 MB）
- `models/trained/train-20261027-124755-fish-yolo11n-best.pt`（权重，约 5.2 MB）

后续若重新构建前端或更新权重，这两个已跟踪文件会随正常提交更新。

## 7. 已知限制

- 免费CPU镜像适合演示；使用GPU时还需要GPU资源及匹配的GPU版PyTorch镜像。
- 数据集样本（`data/`）与训练运行产物（`runs/`）不进镜像，
  云端没有原始样本图时显示资源未接入，但仍可展示已提交的真实训练摘要和16/16数据检查记录；这些历史记录不是云端现场重训或复测。
- 智能助手优先支持运行环境Secret中的 `DEEPSEEK_API_KEY`，默认官方地址与 `deepseek-flash`；既有 `FISH_LLM_*` 兼容配置保持可用。缺失或服务失败时回退到明确标记的规则说明。

## 8. 当前空间的代理来源与新版页面

当前Docker为 `https://cx2474556379-fish.ms.show` 设置精确的 `FISH_TRUSTED_ORIGINS`，并通过 `FISH_TRUST_PROXY=1` 读取可信部署代理提供的外部Host/HTTPS头。合法的上传、训练与聊天请求可通过代理进入接口；其他创空间、相似域名和不在清单内的网站继续被拒绝。迁移空间域名时要同步修改这个明确清单，不使用任意域名通配。

此前“仅接受同源请求”来自浏览器外部Origin与容器内部Host不一致。现在来源检查使用完整的协议、主机和端口，支持配置的精确外部来源。此设置需要随新代码重新部署，刷新旧容器不会生效。

更新GitHub不会自动更新独立的创空间仓库。需要把同一版已提交源码同步到创空间的master并重新部署；新Docker会从源码构建最新页面。`scripts/sync_modelscope.py` 提供准备和发布入口，具体参数见 `--help`。魔搭可写令牌放在项目目录外本地 `.env.modelscope` 中，仅由部署工具使用；不放后端网页配置、不提交Git、不嵌入remote URL。

云端DeepSeek密钥应保存为创空间Secret，原始数据和本地任务目录不随同步上传。发布完成后核对首页16/16检查记录、新版助手文案，并实际上传短视频与提问，确认来源校验和远程解释都工作。
