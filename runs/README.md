# runs — 训练运行现场目录

每次训练自动产生一个新目录(实验现场:权重快照、TensorBoard 日志、指标曲线等)。

- 目录内容不进 git(见根 `.gitignore`),只保留本 README
- `odp-init` 会自动创建本目录
- 属于**运行时产物**,`reset_project` 会清理本目录

建议目录命名:`runs/<实验名>_<日期>/`。