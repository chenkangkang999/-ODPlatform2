# 一次性环境搭建脚本：创建 .venv 并安装 GPU 版 torch + 项目依赖
# 普通包走阿里源（pip 全局已配置），torch 走 PyTorch 官方 cu126 索引（GPU 版）
$ErrorActionPreference = "Stop"
$Repo = "D:\ODPlatform2"
Set-Location $Repo

if (-not (Test-Path "$Repo\.venv\Scripts\python.exe")) {
    Write-Host "[1/5] 创建虚拟环境 .venv ..."
    python -m venv "$Repo\.venv"
} else {
    Write-Host "[1/5] .venv 已存在，复用"
}

$PY = "$Repo\.venv\Scripts\python.exe"

Write-Host "[2/5] 升级构建工具（阿里源）..."
& $PY -m pip install --upgrade pip setuptools wheel -i https://mirrors.aliyun.com/pypi/simple/

Write-Host "[3/5] 安装 GPU 版 torch/torchvision（cu126）..."
& $PY -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

Write-Host "[4/5] 安装其余运行/开发依赖（阿里源）..."
& $PY -m pip install `
    colorlog psutil pyyaml pydantic ultralytics opencv-python `
    pytest pytest-cov ruff mypy pre-commit `
    -i https://mirrors.aliyun.com/pypi/simple/

Write-Host "[5/5] 校验 GPU ..."
& $PY -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
Write-Host "环境搭建完成"