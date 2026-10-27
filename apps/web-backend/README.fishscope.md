# FishScope · Django 后端

完整安装、模型接入和 LLM 配置见 [Web 与 LLM 指南](../../docs/WEB_AND_LLM.md)。

```powershell
python -m pip install -r apps/web-backend/requirements.txt
python apps/web-backend/manage.py check
python apps/web-backend/manage.py test fishweb.tests
python apps/web-backend/manage.py runserver 127.0.0.1:8510 --noreload
```

前端先在 `apps/web-frontend/` 构建，Django 从 `dist/` 同源提供页面。
YOLO/Gradio 依赖使用平台原有安装方式：`pip install -e "apps/platform[ui]"`。
默认当前 Python 运行引擎，也可用 `FISH_ENGINE_PYTHON` 指向另一个已装好平台依赖的 Python。
未安装引擎时仍能访问介绍、数据文档和规则助手。

| 端点 | 作用 |
| --- | --- |
| `GET /api/status/` | 模型候选、引擎、LLM 配置状态 |
| `GET /api/dataset/` | 带来源的数据字典和本机数据状态 |
| `GET /api/experiments/` | 本机真实报告及文档基线 |
| `GET/POST /api/jobs/` | 列出任务 / 创建训练任务 |
| `GET /api/jobs/<id>/` | 任务进度、日志、结果 |
| `POST /api/jobs/<id>/cancel/` | 取消任务和进程树 |
| `POST /api/inference/` | multipart `file, model, conf, iou, imgsz, device, stride` |
| `GET /api/jobs/<id>/files/<name>` | 标注媒体、缩略图、JSON；支持 Range |
| `POST /api/chat/` | JSON `message, job_id?, history?`；返回 `answer, mode, disclosure` |

`catalog.py` 读取配置与真实产物，`jobs.py` 管理后台任务，`worker.py` 隔离执行，
`assistant.py` 只把服务器保存的结构化证据交给配置的模型服务。
训练复用 `od_platform.cli.train_model`，保留数据身份核验。

这是仅供本机的开发预览服务。缺少鱼权重时不伪造结果，未配置 LLM 时标记为规则说明。
`models/trained/` 的权重只作候选登记，推理前核验标签与 fish 配置的 33 类完全一致。

测试中的预测器是明确的测试替身；OpenCV 读写、H.264 转码及 Range 可独立验证，测试结果不代表鱼类模型精度。
