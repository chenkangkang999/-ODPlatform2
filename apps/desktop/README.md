# apps/desktop — 桌面端应用(占位)

> 🚧 **状态**: 占位,未来计划(V1.2+)

本子项目将提供 ODPlatform 的桌面端应用(本地可视化标注查看 / 训练监控)。

## 未来计划

- 复用 `apps/platform/` 的核心引擎做本地推理与数据检查
- 与 `apps/web-backend/` 共用数据结构定义
- 直接读写共享资产 `data/`、`models/`、`runs/`

## 为什么现在是空的?

按 ADR-001 的 Monorepo 布局,先把位置预约出来:
将来启动桌面端时目录已就位,platform 侧代码零改动。

参考: `docs/architecture/ADR-001-monorepo.md`