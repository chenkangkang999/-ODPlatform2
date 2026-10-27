# 渔见 FishScope · 淡水鱼识别 —— ModelScope 创空间 Docker 部署
# 单容器同时提供 Django 后端 API 与 Vue 前端静态页面（Django 同源 serve dist）
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    FISH_PUBLIC=1 \
    FISH_ALLOWED_HOSTS=*

# OpenCV(headless) 运行时依赖 + ffmpeg（视频推理 H.264 转码需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /home/studio/PROJECT

# 先装依赖，利用 Docker 层缓存
COPY requirements.txt .
RUN pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 再拷全部代码（平台引擎 / Django 后端 / 前端 dist / 配置 / 权重）
COPY . .

# 预建任务运行时目录（推理/训练任务的输出与状态写到这里）
RUN mkdir -p runs/web

# 创空间强制服务监听 7860
EXPOSE 7860

# 启动 Django：runserver 0.0.0.0:7860（--noreload 避免文件监视双进程）
CMD ["python", "apps/web-backend/manage.py", "runserver", "0.0.0.0:7860", "--noreload"]
