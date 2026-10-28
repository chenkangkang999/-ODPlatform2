"""Prepare a committed FishScope revision; publish only with an explicit --publish.

ModelScope token values are used in child-process/session memory only. Neither
the token file nor the backend .env is copied into the deployment checkout.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import tempfile

import requests

API = "https://modelscope.cn/openapi/v1"
SOURCE_REPO = Path(__file__).resolve().parents[1]
MANIFEST = ".fishscope-source.json"
START = "<!-- fishscope-sync:start -->"
END = "<!-- fishscope-sync:end -->"


class SyncError(RuntimeError):
    """An intentionally bounded error: never include raw Git/API responses."""


def read_env_file(path):
    if path is None:
        return {}
    try:
        lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        raise SyncError("无法读取指定的本地配置文件；请确认路径。") from None
    values = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if value.startswith(("'", '"')):
            match = re.fullmatch(r"(['\"])(.*?)\1(?:\s+#.*)?", value)
            if not match:
                continue
            value = match[2]
        else:
            value = value.split(" #", 1)[0].rstrip()
        values[key.strip()] = value
    return values


def token_from(env_file):
    file_values = read_env_file(env_file)
    for key in ("MODELSCOPE_API_KEY", "MODELSCOPE_API_TOKEN", "API_TOKEN"):
        value = os.environ.get(key, "").strip() or file_values.get(key, "").strip()
        if value:
            return value
    raise SyncError("未找到可写的ModelScope Token。请在外部.env.modelscope文件填写MODELSCOPE_API_KEY，再使用--publish。")


def git_environment(token=None):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("GIT_CONFIG_", "GIT_TRACE"))
           and key not in {"GIT_CURL_VERBOSE", "GIT_ASKPASS", "SSH_ASKPASS"}}
    env.update(GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="Never")
    settings = [("credential.helper", ""), ("core.askPass", "")]
    if token:
        auth = base64.b64encode(("oauth2:" + token).encode()).decode()
        settings.append(("http.https://www.modelscope.cn/.extraHeader", "Authorization: Basic " + auth))
    env["GIT_CONFIG_COUNT"] = str(len(settings))
    for index, (key, value) in enumerate(settings):
        env[f"GIT_CONFIG_KEY_{index}"] = key
        env[f"GIT_CONFIG_VALUE_{index}"] = value
    return env


def git(repo, *args, token=None):
    try:
        result = subprocess.run(["git", *args], cwd=repo, env=git_environment(token),
                                capture_output=True, timeout=240, check=False)
    except (OSError, subprocess.TimeoutExpired):
        raise SyncError("Git操作无法完成，请检查网络和Git安装后重试。") from None
    if result.returncode:
        raise SyncError(f"Git {args[0]}未成功（退出码{result.returncode}）；请检查网络、空间权限和分支状态。")
    return result.stdout


def safe_child(root, name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise SyncError("导出文件包含不安全路径，已停止。")
    target = (root / path).resolve()
    if not target.is_relative_to(root.resolve()) or ".git" in path.parts:
        raise SyncError("导出文件超出部署目录，已停止。")
    return target


def export_head(repo, output):
    sha = git(repo, "rev-parse", "HEAD").decode().strip()
    contents = git(repo, "archive", "--format=tar", sha)
    paths = []
    with tarfile.open(fileobj=io.BytesIO(contents)) as archive:
        for item in archive:
            if item.isdir():
                continue
            if not item.isfile():
                raise SyncError("已提交源码包含特殊文件，停止自动同步以便人工核对。")
            name = Path(item.name).name
            if name == ".env" or (name.startswith(".env.") and name != ".env.example"):
                raise SyncError("已提交源码包含私有环境配置文件，已停止同步。")
            if item.name.startswith(("data/raw/", "data/processed/", "runs/web/")):
                raise SyncError("已提交源码包含本地数据或上传记录，已停止同步。")
            target = safe_child(output, item.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(item) as source:
                target.write_bytes(source.read())
            paths.append(item.name)
    index = output / "apps/web-frontend/dist/index.html"
    if not index.is_file():
        raise SyncError("已提交版本缺少前端构建产物，请先构建并提交dist。")
    for asset in re.findall(r'(?:src|href)="(/assets/[^\"]+)"', index.read_text(encoding="utf-8")):
        if not safe_child(output, "apps/web-frontend/dist" + asset).is_file():
            raise SyncError("已提交页面引用的构建文件不完整，请重新构建并提交。")
    return sha, paths


def merge_readme(existing, sha):
    match = re.match(r"\A(---\r?\n.*?\r?\n---(?:\r?\n|$))(.*)\Z", existing, re.DOTALL)
    if match:
        frontmatter, body = match.groups()
    else:
        frontmatter = "---\nsdk: docker\napp_port: 7860\n---\n"
        body = existing
    body = re.sub(re.escape(START) + r".*?" + re.escape(END) + r"\s*", "", body, flags=re.DOTALL)
    body = body.replace("左侧导航", "顶部导航").strip()
    block = (f"{START}\n# 渔见 FishScope · 当前页面说明\n\n"
             "通过顶部导航进入项目介绍、数据理解、模型训练、模型推理、仪表盘和智能助手。\n"
             "上传图片或视频，识别后点击“交给助手解释”，用日常语言理解结果中的数字。\n"
             "项目资料包含33类鱼、4560张图片的统计及真实历史训练记录；原始训练图片另行准备。\n"
             f"本次同步源码：`{sha}`。独立Gradio图片演示需单独启动。\n{END}\n")
    return frontmatter.rstrip() + "\n\n" + block + ("\n" + body + "\n" if body else "")


def overlay_source(source, studio, sha, paths):
    existing_readme = studio / "README.md"
    old_readme = existing_readme.read_text(encoding="utf-8") if existing_readme.is_file() else ""
    previous = studio / MANIFEST
    managed = []
    if previous.is_file():
        try:
            managed = json.loads(previous.read_text(encoding="utf-8")).get("files", [])
        except (OSError, ValueError, AttributeError):
            raise SyncError("空间已有同步清单无法读取，已停止以保留原文件。") from None
    keep = set(paths) | {"README.md", MANIFEST}
    # Only remove files explicitly owned by a previous run of this tool.
    for name in managed:
        if not isinstance(name, str):
            raise SyncError("空间同步清单包含无效路径，已停止。")
        if name not in keep:
            target = safe_child(studio, name)
            if target.is_file():
                target.unlink()
    for name in paths:
        if name == "README.md":
            continue
        target = safe_child(studio, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(safe_child(source, name).read_bytes())
    existing_readme.write_text(merge_readme(old_readme, sha), encoding="utf-8")
    previous.write_text(json.dumps({"source_commit": sha, "files": sorted(keep)}, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    for start in range(0, len(keep), 80):
        git(studio, "add", "--force", "--", *sorted(keep)[start:start + 80])
    git(studio, "add", "--update", "--", ".")
    return [name for name in git(studio, "diff", "--cached", "--name-only", "-z").decode().split("\0") if name]


class ModelScopeAPI:
    def __init__(self, token):
        self.session = requests.Session()
        self.session.headers.update(Authorization="Bearer " + token)

    def call(self, method, path, payload=None):
        try:
            response = self.session.request(method, API + path, json=payload, timeout=(10, 45))
        except requests.RequestException:
            raise SyncError("ModelScope接口连接失败，请检查网络后重试。") from None
        if response.status_code >= 400:
            detail = "Token无效或没有该空间的写入权限" if response.status_code in {401, 403} else "请求未成功"
            raise SyncError(f"ModelScope接口HTTP {response.status_code}：{detail}。")
        try:
            data = response.json()
        except ValueError:
            raise SyncError("ModelScope接口没有返回有效JSON。") from None
        if not isinstance(data, dict) or data.get("success") is not True:
            raise SyncError("ModelScope接口未确认操作成功；未输出原始响应以保护配置。")
        return data.get("data", {})


def variable_keys(data, kind):
    items = data.get(kind, []) if isinstance(data, dict) else []
    return {item["key"] for item in items if isinstance(item, dict) and isinstance(item.get("key"), str)}


def sync_deepseek_secret(api, studio_path, repo, secret_keys, variable_names):
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    backend_env = repo / "apps/web-backend/.env"
    if not key and backend_env.is_file():
        key = read_env_file(backend_env).get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        return False
    if any(name.startswith("FISH_LLM_") for name in secret_keys | variable_names):
        raise SyncError("空间已有FISH_LLM配置，可能优先于DeepSeek配置。请先确认其用途；工具保留现有设置并停止发布。")
    method = "PUT" if "DEEPSEEK_API_KEY" in secret_keys else "POST"
    api.call(method, studio_path + "/secrets", {"key": "DEEPSEEK_API_KEY", "value": key})
    return True


def sync_release(api, studio_path, sha, variable_names):
    method = "PUT" if "FISH_RELEASE" in variable_names else "POST"
    api.call(method, studio_path + "/variables", {"key": "FISH_RELEASE", "value": sha})


def remote_head(studio, token=None):
    rows = git(studio, "ls-remote", "origin", "refs/heads/master", token=token).decode().splitlines()
    if len(rows) != 1:
        raise SyncError("未能确认空间master分支，已停止。")
    return rows[0].split()[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, help="外部本地Token文件；默认准备模式不会读取")
    parser.add_argument("--target", default="cx2474556379/fish", help="ModelScope owner/repository")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare-only", action="store_true", help="只准备并显示安全摘要（默认）")
    modes.add_argument("--publish", action="store_true", help="明确授权推送一次提交、配置密钥并请求部署")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.target):
        raise SyncError("目标应为owner/repository，不能包含地址、凭据或其他路径。")
    token = token_from(args.env_file) if args.publish else None
    api = ModelScopeAPI(token) if token else None
    studio_path = "/studios/" + args.target
    secret_keys, variable_names = set(), set()
    if api:
        api.call("GET", "/users/me")
        secret_keys = variable_keys(api.call("GET", studio_path + "/secrets"), "secrets")
        variable_names = variable_keys(api.call("GET", studio_path + "/variables"), "variables")
    workspace = Path(tempfile.mkdtemp(prefix="fishscope-modelscope-")).resolve()
    if workspace.is_relative_to(SOURCE_REPO.resolve()):
        raise SyncError("临时工作目录必须位于源码仓库之外。")
    source, studio = workspace / "source", workspace / "studio"
    source.mkdir()
    sha, paths = export_head(SOURCE_REPO, source)
    if args.publish:
        remote = git(SOURCE_REPO, "ls-remote", "origin", "refs/heads/main").decode().split()
        if not remote or remote[0] != sha:
            raise SyncError("当前HEAD与GitHub main不同，请先确认并上传源码，避免同步旧版本。")
    git(workspace, "clone", "--depth", "1", "--single-branch", "--branch", "master",
        "https://www.modelscope.cn/studios/" + args.target + ".git", str(studio), token=token)
    original = git(studio, "rev-parse", "HEAD").decode().strip()
    changed = [name for name in overlay_source(source, studio, sha, paths) if name]
    report = {"mode": "publish" if args.publish else "prepare-only", "target": args.target,
              "source_commit": sha, "studio_base": original, "prepared_directory": str(studio),
              "changed_files": len(changed), "committed": False, "pushed": False,
              "secret_keys": sorted(secret_keys), "variable_keys": sorted(variable_names),
              "deepseek_secret_synced": False, "deploy_requested": False}
    if args.publish:
        if remote_head(studio, token) != original:
            raise SyncError("空间分支已被其他人更新，已停止；请重新准备，不会强制覆盖。")
        report["deepseek_secret_synced"] = sync_deepseek_secret(api, studio_path, SOURCE_REPO,
                                                                secret_keys, variable_names)
        if changed:
            git(studio, "-c", "user.name=FishScope Sync", "-c", "user.email=fishscope-sync@users.noreply.github.com",
                "commit", "-m", "同步渔见页面、项目资料与DeepSeek解释接入（源码" + sha[:12] + "）")
            if remote_head(studio, token) != original:
                raise SyncError("提交准备期间空间分支有更新，未推送；请重新运行以保留组员变更。")
            git(studio, "push", "origin", "HEAD:refs/heads/master", token=token)
            report.update(committed=True, pushed=True)
        current = git(studio, "rev-parse", "HEAD").decode().strip()
        if remote_head(studio, token) != current:
            raise SyncError("未能确认远端版本，未请求部署。")
        report["studio_commit"] = current
        sync_release(api, studio_path, sha, variable_names)
        report["release"] = sha
        api.call("POST", studio_path + "/deploy")
        report["deploy_requested"] = True
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    try:
        main()
    except SyncError as exc:
        print(json.dumps({"ok": False, "message": str(exc)}, ensure_ascii=False))
        raise SystemExit(1) from None
    except Exception as exc:
        # Unknown transport/encoding errors may carry request details. Keep
        # credentials out of tracebacks as well as ordinary command output.
        print(json.dumps({"ok": False, "message": "同步未完成，已隐藏原始错误内容以保护配置。",
                          "error_type": type(exc).__name__}, ensure_ascii=False))
        raise SystemExit(1) from None
