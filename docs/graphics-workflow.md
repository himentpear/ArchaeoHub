# 图表处理与验收

本阶段使用独立 Python 3.11 CPU 工作进程，默认 4 线程。版面检测先于图表文字识别；原始 OCR、已确认脚注、Work 和 Canvas 不重建。输出默认进入克隆的暂存库，只有独立标注验收通过后才能发布。当前仍为候选功能，不能把模型候选当作验收真值。

## 安装与模型

在仓库根目录执行（PowerShell）：

```powershell
py -3.11 -m venv data/processed/models/graphics-runtime
data/processed/models/graphics-runtime/Scripts/python.exe -m pip install -r requirements-graphics.txt
data/processed/models/graphics-runtime/Scripts/python.exe scripts/bootstrap_graphics_models.py --with-vl
data/processed/models/graphics-runtime/Scripts/python.exe scripts/bootstrap_graphics_models.py --verify
```

阅读服务继续使用原来的 `.venv`；在该环境更新 `requirements.txt`。模型缓存和环境均在被 Git 忽略的 `data/processed/models/`。Windows 不要在工作进程运行时替换 OpenCV DLL。

默认模型：PP-DocLayout_plus-L、PP-LCNet_x1_0_doc_ori、SLANet_plus；复杂表格追加 SLANeXt 和 RT-DETR 单元格检测，文字使用 RapidOCR。PP-DocLayout-S 只用于抽样对照。PaddleOCR-VL-1.6-0.9B 仅作限时的小裁片第二候选，不自动覆盖结果。复杂模型分歧进入审校。

`weights-lock.json` 记录下载文件的本地 SHA256；这是本地复现锁，不是上游签名校验。缓存还绑定原图哈希、配置和代码版本。更新权重后重新生成锁。报告中的 `graphics_models_manifest.json` 不包含本机绝对路径。

## 新文献与已有文献

新文献默认先检测版面，再做来源 OCR 和结构化处理：

```powershell
.venv/Scripts/python.exe scripts/process_document.py PATH_TO_PDF --max-pages 5 --skip-models --graphics-output data/processed/benchmarks/new-graphics
```

`--skip-models` 跳过原有语言模型富化；`--skip-graphics` 才会显式跳过新版图表阶段。图表阶段失败会报错，不静默退回不带结构的结果。已有文献走增量处理，不调用 `--rebuild-index`。

观台重点页抽样：

```powershell
.venv/Scripts/python.exe scripts/process_graphics.py --manifest-id 3f6e2ea2-3baa-4462-91d1-f0051bd91d5d --pages 535-536,538-540 --output data/processed/benchmarks/graphics-focus --skip-vl
.venv/Scripts/python.exe scripts/serve_library.py --library data/processed/benchmarks/graphics-focus --source-library data/processed/library --port 8002
```

访问 `http://127.0.0.1:8002/`。`--source-library` 明确允许克隆暂存库读取原库中的页面图片，不放宽其他文件路径限制。页面图表按钮提供整图／结构表／原页定位，以及 JSON、HTML、Excel 导出。单元格修改、归组、拆分、跨片段单元格连接均需填写校对人、原图依据并检查版本号，支持撤销最近一次未被后续修改的操作。

只处理一个图表时指定它的物理页；同页多个对象一并重新检测，避免局部更新漏掉其他对象。失败后重跑同一命令，完整页面缓存会复用。配置改变会使缓存失效。`--skip-worker` 仅导入版本和原图均匹配的已有结果；源库发生修订后必须换一个暂存目录。

数据表恢复结构；统计图只保留整图与坐标轴、图例文字，不反推数据；考古图件保留整体与标注。△、○没有图例时仅记录原符号；圆形候选不能覆盖 OCR 识别出的数字 0；空白不写成 0。细小文字以 PDF 区域 300dpi 重渲染，最长边上限 4096，并保留逆变换。

续表必须同时通过图表号、列结构、边界、表头与明确“续”标记检查。结构冲突记录为 `unresolved_continuation`，不能按标题强并。几何拼接单独要求有辨识度的重叠和近刚性配准；重复网格与表头不算重叠证据。靠近页面边缘标记 `possible_truncation`；它不证明源文件确实缺失。没有证据时保留多个片段，不补绘内容。

## API 和数据

- `GET /api/graphics?manifest=ID`：图表候选列表。
- `GET /api/graphics?id=ID`：片段、单元格、来源锚点和审校历史。
- `GET /api/graphic_asset?fragment=ID` 或 `cell=ID` 或 `object=ID`：原裁片或配准图。
- `GET /api/graphic_export?id=ID&format=json|html|xlsx`：导出。Excel 的 Sources 表保留原始定位，结构重叠另记 StructureIssues，避免丢弃来源。
- `POST /api/graphic_review`：`object_id, expected_revision, actor, note, action`；支持 `correct_cell/classify/approve/reject/group/split/join_cells`。撤销使用 `undo=true, revision_id`。

数据库新增 `graphic_runs/graphic_objects/graphic_fragments/graphic_cells`，不替换来源文字。未发布候选可以在暂存阅览器审校；默认正文 RAG 只排除已发布图表区域。新文献在入库前已把检测到的图表文字划入表格上下文。旧文献的已确认脚注不被删除。

API 复核默认关闭，启用需传 `--allow-api-review`，配置文件仍放在用户配置目录之外部 `.env`。模型固定 `deepseek-flash`，最多 20 次请求且遵循总 token 预算，仅发送已识别单元格文字。输出必须引用已有原文；不发送整本报告，不假定视觉输入能力。`--skip-vl` 用于单独测量主流程，正式困难样本复核可省略此参数。

## 独立标注与发布

```powershell
.venv/Scripts/python.exe scripts/sample_graphics.py --run-dir data/processed/library/c92f5f329d8be223-40cf08553706 --output data/processed/benchmarks/graphics-60-annotations.json
```

该命令生成 60 页清单和按续表分组隔离的 40/20 划分，不生成真值。535—540 已用于调试，固定属于训练组。同一续表的额外联系应先核验 `group_id`，不能跨训练和验收组。旋转增强放在另外的文件统计。

可启动原图独立标注工具，拖框和编辑对象 JSON；不预填模型结果，保存有版本冲突检查和上一版备份：

```powershell
.venv/Scripts/python.exe scripts/annotate_graphics.py --manifest-id 3f6e2ea2-3baa-4462-91d1-f0051bd91d5d --annotations data/processed/benchmarks/graphics-60-annotations.json --port 8003
```

标注者只看原图，逐页填写 `reviewer`，完成后才设置 `reviewed=true`。空 `objects` 只有在确认整页没有对象后才是负例。所有边界使用数据库 Canvas 坐标（不是原图片像素）；Canvas 尺寸可以从 `/api/page` 取得。每个对象例如：

```json
{"id":"p545-table1","category":"table","bbox":[100,120,1400,1800],"orientation":0,"html":"<table><tr><td>A</td></tr></table>","cell_checks":[{"row":0,"column":0,"kind":"identifier","text":"A"}]}
```

`bbox` 为 `[x0,y0,x1,y1]`，方向为顺时针纠正角度。`cell_checks.kind` 分别使用 `identifier/unit/symbol`；行列从 0 开始。续表和反例放在顶层 `relations`：`{"a":"对象ID","b":"对象ID","kind":"continuation或negative","split":"test"}`。验收关系必须两端都在独立验收组。为物理拼接建立单独真实扫描正负例报告，填写 `stitch_suite.verified_report`；合成图报告不能替代真实扫描验收。

```powershell
.venv/Scripts/python.exe scripts/evaluate_graphics.py --library STAGE --run-id RUN_ID --annotations ANNOTATIONS.json --output EVALUATION.json
.venv/Scripts/python.exe scripts/process_graphics.py --library data/processed/library --output STAGE --publish RUN_ID --evaluation EVALUATION.json
```

评估检查 IoU≥0.5 的检测精确率≥90%、召回率≥95%，结构 TEDS≥0.90、编号／单位／符号分项准确率、零跨页误并和归组召回≥90%、方向及来源覆盖。缺失标注直接失败。发布时重新计算门槛、检查源库版本并备份；裁片复制到生产图表目录，校对历史随对象复制。当前使用结构 TEDS（不计正文字符编辑距离），字符准确率由分项检查单独报告。

撤回某次图表发布：

```powershell
.venv/Scripts/python.exe scripts/process_graphics.py --library data/processed/library --unpublish RUN_ID
```

该操作先备份再停用图表覆盖层，旧图表版本会重新生效，OCR 和审校记录保留。服务进程需重启以刷新检索缓存。

## 当前证据与限制

60 页版面速度对照见 `reports/graphics_layout_comparison.json`：plus-L 中位约 4.12 秒，S 约 1.83 秒；并发 CPU 负载下测得，没有独立准确率结论。合成拼接测试 5 个正例、20 个反例通过，但不能代表历史扫描表现。局部 VLM 小裁片已成功返回候选，整表在 CPU 上曾超过 120 秒期限。

第535页分为附表一和附表二开头，第536页方向纠正为 0°，只有附表二与第536页归组。修正扫描倾斜和细竖线过滤后，538—540 的附表四均恢复为29列并逻辑归组，540页附表五保持独立。重点页的来源覆盖率为100%、坐标核验无异常，见 `reports/graphics_focus_checks.json`；这些是机械来源与归组检查，不代表符号文字或 TEDS 已通过独立验收。

60页完整主流程基线已处理完成，约25分13秒、峰值约2.67GiB，见 `reports/graphics_60_runtime.json`。它运行于网格修正之前，且受并发 CPU 任务影响；不能代表最终四线程版本的精度或稳定速度。最终单进程四线程重点5页耗时约2分40秒，峰值约2.02GiB，见 `reports/graphics_focus_runtime.json`。缓存复用实测5页全部命中，见 `reports/graphics_cache_runtime.json`。

793页整册增量图表阶段的时间情景见 `reports/graphics_book_time_scenarios.json`：假设40页表格约67—82分钟，80页约86—100分钟，160页约124—136分钟。表格页数是假设，尚未测得整册分布；估算混合了不同版本的基线耗时，应预留余量，不包含原始 OCR、独立标注和 VL/API 复核。默认最多12个局部 VLM 复核，每个上限120秒，额外时间最多约24分钟。

60 页独立标注、真实扫描拼接验收和生产发布尚未完成，`reports/graphics_acceptance_status.json` 中精确率、召回率与 TEDS 均为未测量，发布被门槛阻止。运行耗时与峰值内存由每次暂存输出的版本化缓存目录中 `runtime.json` 记录（`graphics-job.json` 给出缓存路径），旧运行裁片不会被新配置覆盖。整册估时必须按实际图表类别比例加权，不能简单外推附表密集样本。
