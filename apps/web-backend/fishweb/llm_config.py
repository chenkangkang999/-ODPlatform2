"""Resolve server-side model settings without mixing credentials across providers."""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from urllib.parse import urlparse

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-flash"


def _official_deepseek(base_url):
    try:
        return urlparse(base_url).hostname == "api.deepseek.com"
    except ValueError:
        return False


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    model: str
    api_key: str = field(repr=False)

    @property
    def is_deepseek(self):
        return _official_deepseek(self.base_url)

    @property
    def provider(self):
        return "deepseek" if self.is_deepseek else "compatible"

    @property
    def provider_label(self):
        return "DeepSeek" if self.is_deepseek else "兼容模型服务"

    @property
    def configured(self):
        return bool(self.api_key and self.base_url and self.model)

    @property
    def endpoint(self):
        base = self.base_url.rstrip("/")
        return base if base.endswith("/chat/completions") else base + "/chat/completions"

    def public_status(self):
        return {"configured": self.configured, "mode": "llm" if self.configured else "local",
                "provider": self.provider, "provider_label": self.provider_label,
                "model": self.model if self.configured else "本地规则解释",
                "disclosure": "仅发送项目资料、问题和检测统计给配置的大模型服务，不发送视频原文件。"}


def resolve_llm_config(environ: Mapping[str, str] | None = None):
    env = os.environ if environ is None else environ

    def value(name):
        return env.get(name, "").strip()

    # Keep a partially configured custom service separate: its missing key must
    # never be silently filled with a DeepSeek credential intended elsewhere.
    if any(value(key) for key in ("FISH_LLM_BASE_URL", "FISH_LLM_MODEL", "FISH_LLM_API_KEY")):
        base = value("FISH_LLM_BASE_URL") or DEEPSEEK_BASE_URL
        model = value("FISH_LLM_MODEL")
        if _official_deepseek(base):
            model = model or DEEPSEEK_MODEL
        return LLMConfig(base.rstrip("/"), model, value("FISH_LLM_API_KEY"))
    return LLMConfig((value("DEEPSEEK_BASE_URL") or DEEPSEEK_BASE_URL).rstrip("/"),
                     value("DEEPSEEK_MODEL") or DEEPSEEK_MODEL, value("DEEPSEEK_API_KEY"))
