# models/pretrained — 预训练权重目录

存放**下载来的预训练权重**(输入资产),例如 `yolo11n.pt` 等。

- 权重文件(`*.pt`)不进 git(见根 `.gitignore`),只保留本 README
- `odp-init` 会自动创建本目录
- `reset_project` **绝不清理**本目录——重新下载耗时且体积大