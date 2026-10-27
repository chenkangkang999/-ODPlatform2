# tests — 跨端(E2E)测试的家

这里是 **跨端 / 端到端(E2E)测试** 的所在地:一次测试要同时涉及多个端
(例如 `apps/platform/` 的引擎 + `apps/web-backend/` 的接口)。

- **端私有测试**放在各端内部,例如 `apps/platform/tests/`(路径常量见 `UNIT_TEST_DIR`)
- **跨端测试**放这里,与仓库根的 `pyproject.toml` 里 `[tool.pytest.ini_options]`
  的 `testpaths` 对应

运行方式:

```bash
pytest            # 从仓库根运行,自动读取根 pyproject 的共享配置
```