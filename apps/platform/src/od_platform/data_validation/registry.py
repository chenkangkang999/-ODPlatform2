# apps/platform/src/od_platform/data_validation/registry.py
"""质检注册表 + 严重度 + 结果类型 + 执行上下文。

严重度四级:ERROR(必须修)> WARNING(可训但建议修)> INFO(提示)> PASS(通过);
rank() 供聚合取"全场最严"。重复注册直接 ValueError —— 静默覆盖等于埋雷。

★ 阈值来源(D4 配置化):CheckContext 带一个 ValidateSettings(configs/runtime/validate.yaml,
  由 odp-gen-config validate 生成;三源合并 CLI > YAML > DEFAULT)。
  check 侧统一用 ctx.setting("字段名", 常量) 取值 —— settings 优先, 读不到回落常量,
  于是"没传 settings 的旧构造点"照样能跑, 而默认值的唯一出处仍是 common/constants.py
  (ValidateSettings 的 Field(default=...) 直接引用那些常量)。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, TypeVar

from od_platform.data_validation.snapshot import DatasetSnapshot
from od_platform.runtime_config.validate import ValidateSettings, build_validate_config

T = TypeVar("T")


class CheckSeverity:
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    PASS = "PASS"

    _ORDER = {ERROR: 3, WARNING: 2, INFO: 1, PASS: 0}

    @classmethod
    def rank(cls, severity: str) -> int:
        return cls._ORDER[severity]


@dataclass
class CheckResult:
    name: str
    severity: str
    summary: str                                     # 一句话结论(给人看)
    details: Dict[str, Any] = field(default_factory=dict)  # 结构化证据(给程序看)

    @property
    def passed(self) -> bool:
        return self.severity in (CheckSeverity.PASS, CheckSeverity.INFO)


def _default_validate_settings() -> ValidateSettings:
    """默认 D4 配置: 不读 yaml, 全用 ValidateSettings 的默认值(= constants 的值)."""
    settings, _merger = build_validate_config()
    return settings


@dataclass
class CheckContext:
    """一次质检运行共享的上下文。

    ★ settings 给了默认值(default_factory), 所以 CheckContext(yaml_path=..., snapshot=...)
      这个旧构造点一个都不用改 —— 拿到的是"全默认"的 ValidateSettings。
    """

    yaml_path: Path
    snapshot: DatasetSnapshot
    settings: ValidateSettings = field(default_factory=_default_validate_settings)

    def setting(self, field_name: str, fallback: T) -> T:
        """读生效阈值: settings.<字段> 优先; 读不到(没传 / 为 None / 没这个字段)回落常量.

        Args:
            field_name: ValidateSettings 的字段名(如 "bbox_out_error_ratio")
            fallback:   对应的 common.constants 常量(唯一默认值出处)

        Returns:
            生效值 —— 类型与 fallback 一致。
        """
        settings = getattr(self, "settings", None)
        if settings is None:
            return fallback
        value = getattr(settings, field_name, None)
        return fallback if value is None else value


CheckFunc = Callable[[CheckContext], CheckResult]


@dataclass(frozen=True)
class CheckEntry:
    name: str
    func: CheckFunc


_REGISTRY: Dict[str, CheckEntry] = {}
_LAZY_INITIALIZED = False


def check(name: str):
    """装饰器:把函数注册为质检项。"""
    def deco(func: CheckFunc) -> CheckFunc:
        if name in _REGISTRY:
            raise ValueError(f"质检项重复注册: {name!r}")
        _REGISTRY[name] = CheckEntry(name=name, func=func)
        return func
    return deco


def _lazy_init() -> None:
    global _LAZY_INITIALIZED
    if _LAZY_INITIALIZED:
        return
    from od_platform.common.registry_utils import import_submodules
    from od_platform.data_validation import checks
    import_submodules(checks)
    _LAZY_INITIALIZED = True


def get_all_checks() -> List[CheckEntry]:
    _lazy_init()
    return [_REGISTRY[k] for k in sorted(_REGISTRY)]


def get_check(name: str) -> CheckEntry:
    _lazy_init()
    return _REGISTRY[name]          # 名字写错就该炸(KeyError)


def list_check_names() -> List[str]:
    _lazy_init()
    return sorted(_REGISTRY)