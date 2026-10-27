"""Loopback-only development settings. Deploy behind proper auth before sharing."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
ROOT = BASE_DIR.parents[1]
SECRET_KEY = os.environ.get("FISH_DJANGO_SECRET_KEY", "local-preview-only-fishvision")
DEBUG = False
ALLOWED_HOSTS = ["127.0.0.1", "localhost", "[::1]", "testserver"]
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
