"""Load only this portal's ignored local configuration, without interpolation."""
from __future__ import annotations

import os
import re
from collections.abc import MutableMapping
from pathlib import Path

DEFAULT_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
ALLOWED_KEYS = frozenset({
    "FISH_LLM_BASE_URL", "FISH_LLM_API_KEY", "FISH_LLM_MODEL",
    "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL",
    "FISH_ENGINE_PYTHON", "FISH_DJANGO_SECRET_KEY",
})


def _literal_value(value: str) -> str | None:
    value = value.strip()
    if value.startswith(("'", '"')):
        quoted = re.fullmatch(r"(['\"])(.*?)\1(?:\s+#.*)?", value)
        return quoted.group(2) if quoted else None
    return value.split(" #", 1)[0].rstrip()


def load_local_env(
    path: str | Path | None = None,
    environ: MutableMapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Read backend/.env only; an existing process environment always wins.

    This intentionally supports a small literal KEY=VALUE format. It never
    executes values, expands variables, searches parent directories or logs
    configuration contents. The return value contains loaded key names only.
    """
    target = DEFAULT_ENV_FILE if path is None else Path(path)
    destination = os.environ if environ is None else environ
    try:
        contents = target.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return ()
    loaded = []
    for line in contents.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, raw_value = line.partition("=")
        key = key.strip()
        if not separator or key not in ALLOWED_KEYS or key in destination:
            continue
        value = _literal_value(raw_value)
        if value is not None:
            destination[key] = value
            loaded.append(key)
    return tuple(loaded)
