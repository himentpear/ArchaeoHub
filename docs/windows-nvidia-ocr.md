# Windows NVIDIA OCR 单页验证

从仓库根目录用 PowerShell 执行。测试机为 Windows、Python 3.13、RTX 4060 Laptop（8188 MiB），驱动 552.46。默认入口仍使用仓库原有的 CPU 依赖。GPU 入口从项目虚拟环境内单独的 gpu 目录加载 CUDA 运行库，无需替换默认环境。

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pip install --timeout 180 --target .venv/gpu --no-deps onnxruntime-gpu==1.22.0 nvidia-cuda-runtime-cu12==12.4.127 nvidia-cublas-cu12==12.4.5.8 nvidia-cudnn-cu12==9.1.1.17 nvidia-cufft-cu12==11.2.1.3
```

准备一份由真实扫描页组成的单页 PDF（下例为 sample.pdf）。两次传入相同文件。输出目录须分别为空；已有缓存或原生文字页可能不触发 OCR。

```powershell
.\.venv\Scripts\python scripts/process_document.py sample.pdf --output data/processed/cpu-test --skip-models
.\.venv\Scripts\python scripts/process_document_cuda.py sample.pdf --output data/processed/gpu-test --skip-models
```

检查每个输出目录内的 pages/00001.json（在任务编号子目录内）：比较逐行文字和 elapsed_seconds。GPU 验证依据是本次执行的 ONNX 节点记录，不是仅检测到显卡。详细 profile 和最近一次成功摘要 last-run.json 位于 data/processed/gpu-evidence。显存为整卡采样值，可能包括其他程序。

| 《观台磁州窑址》PDF 第302页，180 dpi | CPU | CUDA |
|---|---:|---:|
| 页面提取耗时（含初始化） | 6.024 秒 | 3.846 秒 |
| 输出行数 | 35 | 35 |
| 检测 / 方向分类 / 识别的 CUDA 节点执行次数 | — | 190 / 1074 / 1086 |

两次逐行文本相同。识别阶段另有 12 次 CPU 节点执行；采样整卡显存峰值为 1779 MiB。本次跳过可选语言模型阶段；单页数据不代表准确率、整卷吞吐量或全平台验收。
