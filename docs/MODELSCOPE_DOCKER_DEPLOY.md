# 渔见 FishScope · ModelScope 创空间 Docker 部署说明

> 本文件记录如何把「渔见 FishScope」完整 Django + Vue 应用部署到魔搭（ModelScope）创空间，
> 以及为云端部署所做的环境变量改造。仓库根目录已包含可直接部署的 `Dockerfile`、
> `requirements.txt`、前端构建产物 `apps/web-frontend/dist` 与已训练权重。

## 1. 架构

- 后端：Django（`apps/web-backend`），一个进程同时提供：
  - `/api/...` 接口（状态、数据集、任务、推理、智能助手）
  - 非 `/api/` 路径：同源返回前端静态文件 `apps/web-frontend/dist/index.html`
- 前端：Vue3 + Vite（`apps/web-frontend`），已 `vite build` 到 `dist/`，容器内无需 Node。
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
5. 首次构建约 15–25 分钟（安装 torch、ffmpeg 等），状态变为「运行中」后，
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

## 6. 纳入版本控制的二进制（一次性说明）

项目 `.gitignore` 原本排除 `dist/` 与 `models/trained/*.pt`。为保证仓库 clone 后
**可直接 Docker 部署、无需重新训练或构建前端**，已通过 `git add -f` 显式纳入：

- `apps/web-frontend/dist/`（前端构建产物，约 4.8 MB）
- `models/trained/train-20261027-124755-fish-yolo11n-best.pt`（权重，约 5.2 MB）

后续若重新构建前端或更新权重，这两个已跟踪文件会随正常提交更新。

## 7. 已知限制

- 免费 CPU 镜像推理较慢，适合演示；如需更快可升级 xGPU 资源。
- 数据集样本（`data/`）与训练运行产物（`runs/`）不进镜像，
  「数据理解」页面的样本图与历史训练记录在云端为空属正常。
- 智能助手（LLM）需配置 `FISH_LLM_BASE_URL` / `FISH_LLM_API_KEY` / `FISH_LLM_MODEL`，
  未配置时自动降级为规则说明。
