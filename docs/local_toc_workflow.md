# 本地模型章节层级与目录审查

新扫描文献先完成文字提取、OCR、版面与规则目录，再由本地模型判断候选是否为章节标题，并给出父子关系和简短依据。程序按来源 ID 绑定原文证据，计算层级并校验树结构。模型不改写原始文字，也不覆盖 `iiif_ranges`。

## 自动处理与预算

`scripts/process_document.py` 已接入此阶段；复用已入库文献时也会检查缓存。`--skip-models` 跳过全部模型阶段，`LOCAL_TOC_ENABLED=false` 单独关闭目录模型。没有本地权重时记录 `unavailable` 并继续保留规则目录。**此阶段始终不调用云端，即使其他阶段允许云端回退。**

外部配置仍放在 `~/.config/archaeology-digitization/.env`，进程环境变量优先。建议明确指定目录模型：

```dotenv
LOCAL_TOC_ENABLED=true
LOCAL_TOC_MODEL_PATH=D:/Documents/Github/PKU-LM-Humanities-Data-Lab/data/processed/models/Qwen3-4B-Q4_K_M.gguf
MAX_LOCAL_TOC_NODES=48
LOCAL_TOC_BATCH_SIZE=6
MAX_LOCAL_TOC_BATCHES=8
LOCAL_TOC_BATCH_TIMEOUT=150
LOCAL_MODEL_THREADS=4
```

未指定 `LOCAL_TOC_MODEL_PATH` 时，先查找 `LOCAL_MODEL_PATH` 同目录的 `Qwen3-4B-Q4_K_M.gguf`，再尝试 `LOCAL_MODEL_PATH` 本身。手动命令的 `--model` 优先于配置。候选上限与批次上限共同限制处理量；默认最多 48 个候选，不能视为整本文献审查。单批超时单位为秒。

## 对现有文献执行一轮

在仓库根目录运行，`--manifest` 使用数据库中的文献 ID，页序从 1 开始，表示 PDF 扫描页序：

```powershell
.\.venv\Scripts\python.exe scripts/review_toc_structure.py `
  --manifest 38ef9921-7ae9-4b56-ab6a-1e887a68b1c3 `
  --start-scan 59 --end-scan 125 `
  --max-nodes 24 --batch-size 4 --max-batches 6 `
  --report reports/local_toc_review_pilot.json
```

命令先通过 SQLite backup API 建立备份，再保存独立运行记录与建议。相同来源、模型及参数复用有效缓存。文字、版面、页码映射或规则层级变化会使旧结果失效，需要重新运行。

候选抽样优先既有高层级标题，再按页序分散补充短文本候选；这是有预算的功能试跑，不是随机人工测试集。运行记录分别报告全书候选数、指定范围内候选数、选中数、处理数与未覆盖数。

## 页面查看与数据保护

打开本地文献工作台，点击“本地模型目录审查”查看模型、覆盖范围、建议、原因与证据。目录来源可以切换：

| 来源 | 当前行为 |
| --- | --- |
| 当前目录（`auto`） | 待审模型结果不替换规则基线 |
| 本地模型建议（`local_model`） | 显式预览模型建议，带“模型待审”标识 |
| 规则目录（`rules`） | 查看规则基线 |

切换来源只改变预览，不代表批准。当前界面未提供整批批准按钮。人工修订的节点及其祖先受到保护；模型不能通过删除祖先间接改变人工层级。

校验包括来源 ID、节点覆盖、父节点存在与页序、环、树深度、交叉子树和人工结构保护。超时或非法输出记录失败；不会把失败结果替换为默认目录。**结构和来源校验通过，只说明输出可追溯且结构合法，不能证明章节语义判断正确。**

## 输出与接口

文献运行目录保留：

| 文件 | 内容 |
| --- | --- |
| `toc_model_stage.json` | 本地运行状态、覆盖量、错误与云端调用数 |
| `toc_model_cache/` | 分批结构化结论缓存 |
| `toc_structure.json` | 默认目录及审查元数据 |
| `toc_model_preview.json` | 模型待审目录预览与建议 |
| `toc_model_manifest.json` | 模型预览的 IIIF Manifest |
| `manifest.json` | 默认展示策略下的 IIIF Manifest |
| `toc_local_worker.log` | 本地工作进程诊断日志 |

新增数据库表为 `toc_structure_runs`、`toc_node_proposals`，与原规则目录分离。程序只保存结构化结论和简短依据。

`GET /api/toc_structure?manifest=<ID>&source=auto|rules|local_model` 返回目录和运行元数据；原 `GET /api/toc` 仍返回数组，并支持相同来源参数。接口里的目录条目会经过页面展示筛选，因此数量可能少于原始规则记录数。

本轮真实运行结果与已发现的误判见 [部署与实测记录](../reports/local_toc_deployment_2026-11-06.md)。当前数据没有人工 Gold，不报告 Precision、Recall 或 F1。
