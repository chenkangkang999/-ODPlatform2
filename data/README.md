# data — 数据集目录

本目录按"原始(只读)与派生(可再生)分家"的原则划分为两个子目录:

| 子目录 | 内容 | 进 git |
|---|---|---|
| `data/raw/` | **原始数据**:只读,绝不就地修改 | ❌ 内容 gitignored |
| `data/processed/` | **派生数据集**:由数据流水线产出,含冻结后的 train/val/test | ❌ 内容 gitignored |

`data/raw/` 下按数据集名组织,预期结构:

```
data/raw/<数据集名>/
├── images/
└── annotations/
```

> 📌 运行 `odp-init` 会自动创建 `data/raw/` 与 `data/processed/` 空目录。
> 本 README 是目录占位说明,随代码进 git。