# 考古文献自动处理与本地部署

## 安装

在仓库根目录运行，推荐 Python 3.11—3.13：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -r requirements-models.txt --only-binary=llama-cpp-python
```

基础流程使用 PyMuPDF 提取可用的原生文字，扫描页和图表文字通过独立 API 调用 PaddleOCR-VL v1.6。本机使用 CPU。语言模型补充抽取属于可选阶段，其结果需要原文证据校验。PyMuPDF 软件分发需确认相应许可要求。

## PaddleOCR-VL v1.6

OCR运行于独立的 Python 3.11 环境，底层使用 `PaddleOCRVL(pipeline_version="v1.6")`，同时保存原生 JSON 与 Markdown 导出。

```powershell
py -3.11 -m venv data/processed/models/graphics-runtime
data/processed/models/graphics-runtime/Scripts/python.exe -m pip install -r requirements-graphics.txt
data/processed/models/graphics-runtime/Scripts/python.exe scripts/serve_ocr.py --port 18082 --device cpu
```

服务提供 `POST /v1/chat/completions`，支持单张 base64 `image_url`，返回包含识别块、原图坐标、Markdown及模型来源的 JSON。当前模型名为 `PaddleOCR-VL-1.6-0.9B`，模型信息与运行状态可从本机查看。[模型列表](http://127.0.0.1:18082/v1/models)

默认 API 地址为 `http://127.0.0.1:18082/v1`，调用超时1800秒，生成上限8192 token，单个识别区域像素上限1003520。本机采用单批次串行推理。可通过以下配置项调整。

```dotenv
OCR_BASE_URL=http://127.0.0.1:18082/v1
OCR_MODEL=PaddleOCR-VL-1.6-0.9B
OCR_TIMEOUT=1800
OCR_MAX_TOKENS=8192
OCR_MAX_PIXELS=1003520
OCR_API_KEY=
```

扫描页识别结果写入对应运行目录的 `ocr-vl/<页图名>/`。图表流程先定位并校正区域，再用 v1.6 识别整表和图中文字，结果写入区域目录的 `paddleocr-vl/`。`result.json` 保留模型信息与原生结果，`document.md` 及配套图片可直接查看。独立服务的原生导出与重试缓存位于 `data/processed/models/ocr-vl-cache/`。

v1.6提供块级坐标。整表结构与已有网格一致时使用单元格定位，存在结构歧义时使用整表范围并记录待核查状态。识别器的置信度缺省会明确标记。修改模型或生成参数会生成新的缓存标识，历史识别可通过原有运行记录回溯。

已有文献需要重新处理才能生成 v1.6 结果。表格可用 `process_graphics.py` 在暂存库运行，新文献通过 `process_document.py` 进入完整流程。`process_document_cuda.py` 调用GPU模式的同一OCR服务，设备由服务端配置。

2026年11月15日实机核验通过125项测试。《观台磁州窑址》物理页535的最新一轮结果包含2张表、103个单元格，首轮CPU处理926.52秒，缓存重试33.12秒。表头文字的跨格归属与数值仍需人工核查，样例以待审候选保存。暂存阅览已指向该模型样例库。[打开暂存阅览](http://127.0.0.1:18002)

样例的 JSON、Markdown 和运行统计位于 `data/processed/benchmarks/paddleocr-vl-20261115/`。完整模型输出可直接查看。[第一张表的Markdown](../data/processed/benchmarks/paddleocr-vl-20261115/graphics-cache/bc17cc30c1327f0555b7/00535-00/paddleocr-vl/document.md)

## 外部配置

默认读取 `~/.config/archaeology-digitization/.env`，位于 Git 仓库之外。当前机器的路径是 `C:/Users/jisub/.config/archaeology-digitization/.env`。也可用 `process_document.py --settings <外部路径>` 指定其他文件。不要把密钥作为命令行参数传入。

```dotenv
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-flash
ALLOW_CLOUD_FALLBACK=false
MAX_CLOUD_CALLS_PER_DOCUMENT=20
MAX_CLOUD_TOKENS_PER_DOCUMENT=50000
LOCAL_MODEL_PATH=D:/path/to/model.gguf
LOCAL_MODEL_THREADS=4
LOCAL_LLM_BASE_URL=http://127.0.0.1:18081/v1
LOCAL_LLM_MODEL=Qwen3-4B-Q4_K_M.gguf
LOCAL_LLM_RESPONSE_FORMAT=json_schema
LOCAL_LLM_TIMEOUT=150
MAX_LOCAL_BLOCKS_PER_DOCUMENT=30
```

第三方兼容服务需填写其实际地址及模型名。填好凭据后，将 `ALLOW_CLOUD_FALLBACK` 设为 `true`。配置中允许的环境变量可覆盖文件值。代码、运行元数据和用量报告不保存密钥。

## 单份文献

```powershell
.venv\Scripts\python.exe scripts/process_document.py "D:\path\report.pdf"
```

输入不需要书名、遗址名或预先标注的正文起始页。默认输出到 `data/processed/library/`：

- `library.db`：独立文献库，不覆盖之前的 `data/processed/humanities_lab.db`。
- `<源哈希-配置哈希>/job.json`：状态、已完成页数和错误类型。
- `pages/*.json`：逐页提取缓存、引擎、耗时及原始文字框。
- `images/*.jpg`：原页渲染；`figures/*.jpg`：图版候选。
- `manifest.json`、`bundles/*.json`、`alto/*.xml`：IIIF、页面数据和 ALTO 4 输出。
- `tables/*.json`：表格候选、单元格文字/坐标及待审标识。
- `quality.json`：质量报告与审校任务。
- `model_stage.json`、`model_cache/`：可选语言模型阶段的结果、缓存和有界 API 用量。

相同文件、相同提取配置使用同一运行目录。失败后重新运行原命令会校验并复用逐页缓存。数据库写入在最终事务内完成，避免留下半卷索引。已经入库的文献再次运行会复用已有记录并重新导出，不清库、不重复导入。

`--max-pages N` 仅用于抽样，其配置哈希与全卷不同，因此会成为独立测试记录。`--stop-after N` 可模拟提取阶段中断；去掉该参数再次运行即可续跑。代码和配置规则改变后，正式重建应先做数据库备份，在新库验证结果，不要把已人工校对的数据直接覆盖。

## 全量实机测试

完整考古报告 PDF 不随 Git 仓库分发。运行真实文献集成测试前，将本地原件放入 `docs/考古报告/`；对应文件名与哈希可在探查报告中核对。数据库、页图缓存和模型权重也不会随代码推送。

```powershell
.venv\Scripts\python.exe scripts/benchmark_corpus.py --output data/processed/library
.venv\Scripts\python.exe scripts/verify_library.py
.venv\Scripts\python.exe -m pytest tests -q
```

批处理将目录中全部 PDF 送入同一条通用流程，使用文件名作为临时显示名，不使用报告专属页码偏移或正文模板。`corpus_results.json` 记录逐份状态和运行时间。`verify_library.py` 检查所有已入库文献的页数、原文件哈希、外键、文字框范围、图像与导出文件，但**不把这些结构检查当作 OCR 或实体语义准确率**。

## 本地语言模型

本地语言模型通过独立 HTTP 服务运行，实体、属性和目录阶段使用 OpenAI 兼容的 `POST /v1/chat/completions`。在一个终端启动推理服务，再在另一个终端启动阅览室或处理任务。

```powershell
.venv\Scripts\python.exe scripts/serve_model.py --model data/processed/models/Qwen3-4B-Q4_K_M.gguf --port 18081
.venv\Scripts\python.exe scripts/serve_library.py --library data/processed/library --port 18000
```

本机部署地址为 [阅览室](http://127.0.0.1:18000) 和 [模型列表](http://127.0.0.1:18081/v1/models)。`serve_model.py` 默认使用4线程、8192上下文及本机回环地址，可通过参数设置模型、端口、线程和上下文。

`LOCAL_LLM_RESPONSE_FORMAT` 可设为 `json_schema`、`llama_json` 或 `json_object`。默认的 `json_schema` 使用 OpenAI 格式，由推理服务入口转换为底层 Schema 约束。应用端继续校验结构与逐字证据。兼容服务可配置 `LLM_BASE_URL`、`LLM_MODEL`、`LLM_API_KEY` 和 `LLM_RESPONSE_FORMAT`，已有 DeepSeek 配置也可读取。模型服务只接受已部署的模型名，换用其他权重后应同步更新模型别名和配置。

```powershell
.venv\Scripts\python.exe scripts/download_local_model.py
# 需要更高质量时对比 4B 的 4-bit 量化版本
.venv\Scripts\python.exe scripts/download_local_model.py --repo Qwen/Qwen3-4B-GGUF --file Qwen3-4B-Q4_K_M.gguf
.venv\Scripts\python.exe scripts/enrich_models.py --library data/processed/library
```

下载使用官方模型仓库固定 revision，完成后校验 LFS SHA-256，模型文件不纳入 Git。0.6B 是速度基线，实际抽样已发现实体漏提与类型判断错误，不能把“能输出 JSON”视为质量通过。4B 或 API 仍需要相同来源约束及样本评测。

模型阶段只处理有领域线索的有限数量文本块，配置控制每份文献的上限。其余候选在 `remaining_candidates` 中明确记录；规则结果依然保留。模型实体区分候选与已审结果，保留类型、逐字原文位置及来源锚点。重跑模型阶段使用缓存，避免重复调用。

## 器物属性增量补填

已有编号索引可继续补充器类、型式、材质、明确年代、纹饰和尺寸：

```powershell
.venv\Scripts\python.exe scripts/enrich_artifacts.py --dry-run
.venv\Scripts\python.exe scripts/enrich_artifacts.py
```

命令先备份数据库，随后按标本绑定属性，生成有出处的字段候选。无冲突值以待审状态填入器物字段；冲突和既有人工填写不会被自动覆盖。可选模型阶段通过 `--models --max-model-units 5` 显式开启。

操作与审校方式详见 [器物属性工作流](artifact_attribute_workflow.md)。

## 阅览与审校

```powershell
.venv\Scripts\python.exe scripts/serve_library.py --library data/processed/library
```

打开 [本地阅览室](http://127.0.0.1:8000)。可切换文献、跳页、检索中文和器物编号、点击文字行高亮原页，查看审校任务及导出文件。

审校保存要求审校人、依据和当前页面版本；过期版本返回冲突，避免覆盖其他人的修改。原始 `raw_ocr` 保留，校对文本另存；编号/尺寸等受保护内容的修改要求详细说明。文字、全文检索、注释、索引与导出随审校更新，并记录 Revision。

服务默认仅监听本机。当前提供原尺寸图像及像素区域裁切，IIIF Image API 宣告 level0，不宣称完整 level2 能力。默认地址为 `http://127.0.0.1:8000/iiif`；使用其他端口时，生成阶段也需通过 `--base-url` 设置一致地址。

## 编号核查与比对标注

检索结果以线图作为主图，正文、图注、表格、脚注和标题带来源标签。历史段落中的编号型图注还会结合括号内器号识别。标签仍依赖既有版面与图注证据，可通过原页核验。

“编号问题”分页展示同号记录中的器类、出土单位、层位或属性候选冲突，可按当前检索词和卷次筛选。在器物的“属性、编号与证据”或问题清单中打开“校订编号”，填写校订号、同号消歧标记、审校人和依据。相同消歧标记用于跨卷关联。

校订保存于器物元数据，界面和 `artifacts.json` 同时提供原书编号与校订编号。搜索支持两种编号，保存同步刷新跨卷缓存并导出器物数据。修订带版本检查与历史记录，可撤销最近一次修改。当前编辑粒度为一个器物档案，同档案内的多件器物仍需后续拆分工具。

自由比对画布每张卡片提供“箭头”“撤销箭头”和“删除箭头”。启用箭头后拖动画线，拖动端点调整位置。坐标绑定图像，随缩放、镜像和卡片尺寸变化同步。比对卡片及箭头保存在当前浏览器的本地存储中。

在检索主图的大图窗口中点击“单器物精确切分”，在原页上拖框，选择器号和视图后保存。已有裁片可调整坐标或撤回，保存采用版本与原图哈希检查。确认的精确切片优先展示于检索主图，器物导出记录包含 `image_regions`。

结构化表格详情中的“与正文核对”按明确器号绑定行，使用有出处的正文尺寸与表格同名尺寸比较。单位换算复用属性工作流，编号歧义和缺少单位时保留待核查状态。接口为 `GET /api/table_alignment?id=<图表ID>`。

## Agent试验

PAI Agent SDK 在独立环境中运行，使用本机模型 API 与检索工具完成器号核查。

```powershell
.venv\Scripts\python.exe -m venv data/processed/models/agent-runtime
data/processed/models/agent-runtime/Scripts/python.exe -m pip install -r requirements-agent.txt
data/processed/models/agent-runtime/Scripts/python.exe scripts/check_artifact_agent.py --query "T20D2∶314"
```

器物卡片提供“Agent核查（试验）”。任务先调用检索工具，再通过 Schema 约束选择差异类型和来源对，界面使用已检索原文生成核查条目。任务限3次模型请求、2次工具调用和240秒总时限，结果及来源存入文献库的 `agent_runs/`。关闭窗口后可通过同一器物档案重新连接正在运行的任务。

Agent任务绑定当前器物档案及报告体系，同号消歧后的记录各自检索。运行期间原文或编号版本变化会提示重跑。条目为待审候选，实机试跑仍出现来源配对与差异类型误报，需核验原文。

自动视觉 Grounding、轮廓分割及 OCR 换模准确率仍需后续模型与独立标注评测。本轮部署与实机验收见 [2026-11-15 部署记录](../reports/phased_deployment_2026-11-15.md)。

## 备份恢复

```powershell
.venv\Scripts\python.exe scripts/backup_library.py "D:\backup\library-2026-10-30.db"
```

脚本通过 SQLite backup API 生成一致快照并验证完整性，拒绝覆盖已有备份。数据库、运行目录中的原页/图版和外部配置应分别备份；密钥按用户自己的凭据备份方式保存。恢复时先停服务，将备份复制到新的库目录，并恢复对应运行文件，再以 `--library` 指向该目录启动验证。

## 当前边界

新扫描文献已加入独立的本地模型目录审查阶段，默认有界处理候选，结果以待审预览保存；缺少权重时保留规则目录，目录阶段不使用云端回退。配置、手动审查命令和目录来源切换见 [本地目录工作流](local_toc_workflow.md)，真实模型试跑见 [2026-11-06 部署记录](../reports/local_toc_deployment_2026-11-06.md)。

图版连接区域与扫描表格网格是候选结果，存在拆分/合并错误，需要抽查；未可靠定位的图版有明确待审标识。无框表格、罕见字、模糊标本号和复杂跨页关系不能以自动成功掩盖。API 不可用、预算耗尽或结果未通过证据校验时，保留本地结果并进入审校，不伪造补全。
