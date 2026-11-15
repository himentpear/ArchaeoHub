# 图表处理与验收

图表使用独立 Python 3.11 CPU 工作进程，默认 4 线程。版面检测先于图表文字识别。正式部署按全量来源哈希、裁片、坐标和数据完整性检查增量发布候选结果，结构与语义疑点进入审校队列。原页、单元格与修订版本提供回溯依据。

## 安装与模型

在仓库根目录执行（PowerShell）：

```powershell
py -3.11 -m venv data/processed/models/graphics-runtime
data/processed/models/graphics-runtime/Scripts/python.exe -m pip install -r requirements-graphics.txt
data/processed/models/graphics-runtime/Scripts/python.exe scripts/bootstrap_graphics_models.py --with-vl
data/processed/models/graphics-runtime/Scripts/python.exe scripts/bootstrap_graphics_models.py --verify
```

阅读服务继续使用原来的 `.venv`；在该环境更新 `requirements.txt`。模型缓存和环境均在被 Git 忽略的 `data/processed/models/`。Windows 不要在工作进程运行时替换 OpenCV DLL。

版面、方向和结构模型使用 PP-DocLayoutV3、PP-LCNet_x1_0_doc_ori、SLANet_plus。网格候选追加 PP-DocLayout_plus-L 交叉判断。复杂表格追加 SLANeXt 和 RT-DETR 单元格检测。文字和表格 HTML 通过独立 OCR API 调用 `PaddleOCRVL(pipeline_version="v1.6")`，保存原生 JSON、Markdown 和图片。PP-DocLayout-S 用于抽样对照。模型分歧进入审校。

`weights-lock.json` 记录下载文件的本地 SHA256；这是本地复现锁，不是上游签名校验。缓存还绑定原图哈希、配置和代码版本。更新权重后重新生成锁。报告中的 `graphics_models_manifest.json` 不包含本机绝对路径。

## 新文献与已有文献

新文献默认先检测版面，再做来源 OCR 和结构化处理：

```powershell
.venv/Scripts/python.exe scripts/process_document.py PATH_TO_PDF --skip-models
```

新报告默认按文献和来源修订创建独立图表暂存目录，并启动持续部署进程。完成页通过来源校验后写入正式库，异常重识别具有重试次数记录。`--stage-graphics-only` 可保留暂存输出，`--skip-models` 控制原有语言模型富化阶段，`--skip-graphics` 控制图表阶段。

## 全量任务与正式部署

已有全量表格任务可独立接入持续入库进程。

```powershell
.venv/Scripts/python.exe scripts/deploy_table_batch.py --staging data/processed/benchmarks/tables-reprocessed-20261115 --library data/processed/library --watch
.venv/Scripts/python.exe scripts/review_table_jobs.py --library data/processed/library --watch
```

正式阅览器的“表格任务”入口展示页数、记录数、失败状态与自动语义复核进度，每15秒更新。相关接口为 `GET /api/table_jobs`、`GET /api/table_inventory?job=ID` 和 `GET /api/table_semantic_result?id=ID`。

任务状态保存在正式库的 `table-jobs/`。每个来源配置对应稳定的入库批次，追加页面时保留已有单元格及校订。任务去重依靠文件锁，完成页缓存支持断点续跑，识别进程恢复次数写入任务状态。正式库首次写入前生成数据库快照，裁片按哈希复制并核验。

各表的 JSON、CSV、HTML、Excel 与正文核对结果位于 `exports/tables/批次ID/图表ID/`。自动规则核对检查结构问题、表头范围、单位、未读出格及明确器号的正文关联，带来源的疑点写入审校任务。

语义复核使用 `LOCAL_LLM_*` 配置的 OpenAI 兼容端点。工作进程检查所有触发复核条件的表格，覆盖对应表头、脚注、数量字段与疑点单元格，按80格分批传输。每项建议通过原文引用和 JSON Schema 校验，并绑定表格修订版本。结果保存在 `table-semantic-results/`，处理状态写入任务页面。

批次回滚可执行以下命令。

```powershell
.venv/Scripts/python.exe scripts/deploy_table_batch.py --staging data/processed/benchmarks/tables-reprocessed-20261115 --library data/processed/library --rollback
```

回滚先暂停该批次的持续入库，再备份并停用对应图表发布版本。版本化数据和裁片支持回溯。

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
- `GET /api/graphics?id=ID` 返回片段、单元格、来源锚点、审校历史和 `table` 数据记录。
- `GET /api/graphic_asset?fragment=ID` 或 `cell=ID` 或 `object=ID`：原裁片或配准图。
- `GET /api/graphic_export?id=ID&format=json|records|csv|html|xlsx` 提供导出。`records` 返回逻辑数据表。CSV 保留来源页和单元格编号。Excel 包含 Table、Records、Columns、Sources、StructureIssues。
- `POST /api/graphic_review` 使用 `object_id, expected_revision, actor, note, action`，支持 `correct_cell/classify/approve/reject/group/split/join_cells/set_header_rows`。`set_header_rows` 同时提供片段 `target_id` 和整数 `header_rows`，0 表示无表头。撤销使用 `undo=true, revision_id`。
- `POST /api/table_semantic_review` 使用 `object_id, expected_revision`，按需调用 `LOCAL_LLM_*` 配置的 OpenAI 兼容端点，返回带来源引用的待审建议。

## 表格版式与数据抽取

表格按测得的物理列宽、多级表头、横向合并和纵向分组展示。`<br>` 和单元格内的换行保留。逻辑记录沿明确的 rowspan 继承分组值，并记录 `inherited` 和原单元格编号。每个字段保留有效文字、原始 OCR、解析类型、单位候选和来源锚点。人工改字或调整表头后，字段路径和数据导出即时重新派生，操作纳入版本检查和撤销。

新报告通过 `prepare_pdf` 自动进入该流程。PP-DocLayoutV3 提供主版面判断，网格补充区域由另一版面模型交叉判断，并结合图表标题选择处理分支。模型共同指向图件的方格区域归入图件。分类有分歧的区域会自动调用 PaddleOCR-VL 的原生版面解析流程确认类型。仍有分歧时保存裁片和分类证据，并生成页面审校任务。后续扫描可直接执行常规文献入库命令，表格页由流程自动选择。

超过 160 个物理单元格的大表，优先在完整行边界分段识别，分段边界避开纵向合并单元格。结果恢复全局行号和原图坐标，保留各段的原生 OCR 输出。识别行数与网格不一致时记入问题列表。表内 `tfoot` 和跨列注释单独导出为 `notes`，展示保留原位置。

稀疏符号矩阵按原图网格逐格读取。PaddleOCR-VL 识别表头和有文字的单元格，清晰孤立的三角符号按轮廓记录为带来源定位的观察候选，空白保留独立状态。该分支保留 `source-observations.json` 和每个文字裁片的 OCR 结果。单元格有墨迹却未读出文字时记为 `unreadable`。行列数一致而合并关系有分歧的普通表格，保留已测得的单元格边界，并将由网格轴线推定的边界标记为 `grid_axes`。

斜线表头的左上角区域可结合相邻单元格、四边原图线条和对角线恢复为一个表头区域，边界标记为 `grid_axes`。短续表也参与稀疏符号矩阵判断，表头缺角时结合相邻层级确定表头范围。

表头识别发生列数分歧时，只有一整行的单元格数量和跨度序列一致，才按阅读顺序对应到物理单元格，其余表头从原图单元格裁片重新识别。对应方式保存在单元格的 `method` 中。

OCR API 另提供 `task="spotting"`，返回原生文字定位的行级坐标。调用时可通过 `max_tokens` 设置本次请求的生成上限。第 538 页左侧 356×1400 像素裁片的本机试验返回 17 个文字定位，耗时 442.58 秒，罗马类型号出现漏读。该接口作为可选能力保留。常规批量流程采用网格定位、单元格文字识别和符号观察。

解析支持编号、十进制数、范围、尺寸和名称数量列表。空白、破折号与图形符号分别保存。地层编号中的圈码保留在编号中，数量列中的圈码另给待核查的数量解释。多级表头的单位归属、单元格缺失或重叠、面积超过布方尺寸乘积等情况写入问题列表。可疑的面积与深度表头混入情况，会按已匹配的物理单元格重新调用 PaddleOCR-VL，并保存整表候选和裁片候选。

额外文本 LLM 建议保留为按需语义复核。确定性表头和几何规则负责常规恢复，PaddleOCR-VL 负责图像转录。文本 LLM 可辅助判断复杂表头角色、单位归属、数量圈码和脚注意义。接口采用受约束 JSON，要求引用真实 `cell_id` 和对应原文，建议交由审校操作应用。每次最多选取 80 个来源单元格，优先选表头和有问题的单元格，并补充前部数据行，响应报告实际抽样范围。默认全量重处理使用 OCR 和规则流程，语义复核由阅览器按钮单独触发。

全量覆盖本机当前存在的已登记 PDF，可执行下面的命令。

```powershell
.venv/Scripts/python.exe scripts/reprocess_tables.py --output data/processed/benchmarks/tables-reprocessed-20261115
```

该流程逐页检测全部可用文献，随后识别检测出的数据表并按来源和结构证据处理续表。`--discover-only` 可单独完成检测。`--resume` 会校验已完成文献的全部原页图像和版面模型权重后复用检测结果。运行状态记录在 `progress.json`，逐页检测缓存位于 `discovery/`，最终清单为 `table-inventory.json`，各表导出位于 `exports/`，分类分歧汇总在 `classification-review.json`。中断后重复执行相同命令，已完成的检测页和 OCR 请求可复用。扫描范围及候选数量由运行报告给出，独立人工标注可用于进一步测量漏检和字符准确率。

数据库新增 `graphic_runs/graphic_objects/graphic_fragments/graphic_cells`，不替换来源文字。未发布候选可以在暂存阅览器审校；默认正文 RAG 只排除已发布图表区域。新文献在入库前已把检测到的图表文字划入表格上下文。旧文献的已确认脚注不被删除。

API 复核默认关闭，启用需传 `--allow-api-review`，配置文件仍放在用户配置目录之外部 `.env`。模型固定 `deepseek-flash`，最多 20 次请求且遵循总 token 预算，仅发送已识别单元格文字。输出必须引用已有原文；不发送整本报告，不假定视觉输入能力。`--skip-vl` 用于单独测量主流程，正式困难样本复核可省略此参数。

## 独立质量评估

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
