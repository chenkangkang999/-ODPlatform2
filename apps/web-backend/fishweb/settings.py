"""Loopback-only development settings. Deploy behind proper auth before sharing."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
ROOT = BASE_DIR.parents[1]
SECRET_KEY = os.environ.get("FISH_DJANGO_SECRET_KEY", "local-preview-only-fishvision")
DEBUG = False
# 默认仅允许本机回环；云端部署（如创空间容器）设置环境变量 FISH_ALLOWED_HOSTS=* 放开。
ALLOWED_HOSTS = os.environ.get(
    "FISH_ALLOWED_HOSTS", "127.0.0.1,localhost,[::1],testserver"
).split(",")
# Only enable this behind the deployment proxy, which supplies these headers.
USE_X_FORWARDED_HOST = os.environ.get("FISH_TRUST_PROXY") == "1"
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https") if USE_X_FORWARDED_HOST else None
ROOT_URLCONF = "fishweb.urls"
MIDDLEWARE = ["django.middleware.security.SecurityMiddleware", "fishweb.views.LocalOnlyMiddleware"]
INSTALLED_APPS = []
TEMPLATES = []
DATABASES = {}
USE_TZ = True
TIME_ZONE = "Asia/Shanghai"
DEFAULT_CHARSET = "utf-8"
DATA_UPLOAD_MAX_MEMORY_SIZE = 4 * 1024 * 1024
FILE_UPLOAD_MAX_MEMORY_SIZE = 2 * 1024 * 1024
DATA_UPLOAD_MAX_NUMBER_FILES = 1
SECURE_CONTENT_TYPE_NOSNIFF = True
