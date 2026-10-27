# models/trained — 训练产出权重目录

存放**训练产出的、长期保留的权重**(产物资产)。

- 权重文件与子目录不进 git(见根 `.gitignore`),只保留本 README
- `odp-init` 会自动创建本目录
- 属于**可再生产物**——重训能拿回来,所以 `reset_project` 会清理本目录