"""Windows CUDA entry point; accepts the arguments of process_document.py."""
import collections, json, os, runpy, sys, time, subprocess, threading
from pathlib import Path
root=Path(__file__).resolve().parents[1]
gpu=root/'.venv/gpu'
sys.path.insert(0,str(gpu))
sys.path.insert(0,str(root))
dirs=[p for p in (gpu/'nvidia').rglob('bin') if p.is_dir()]
handles=[os.add_dll_directory(str(p)) for p in dirs]
os.environ['PATH']=os.pathsep.join(map(str,dirs))+os.pathsep+os.environ['PATH']
import onnxruntime as ort
if not Path(ort.__file__).resolve().is_relative_to(gpu): raise RuntimeError('GPU runtime not installed')

out=root/'data/processed/gpu-evidence';out.mkdir(parents=True,exist_ok=True)
sessions=[]
class Profiled(ort.InferenceSession):
    def __init__(self,path_or_bytes,sess_options=None,**kwargs):
        sess_options=sess_options or ort.SessionOptions()
        sess_options.enable_profiling=True
        sess_options.profile_file_prefix=str(out/Path(path_or_bytes).stem)
        super().__init__(path_or_bytes,sess_options,**kwargs)
        if self.get_providers()[0]!='CUDAExecutionProvider':raise RuntimeError('CUDA fallback')
        sessions.append(self)
ort.InferenceSession=Profiled
from rapidocr import RapidOCR
from src.ingest import page_pipeline as pp
class CudaOcr(pp.LocalOcr):
    def __init__(self,threads=4):
        self.engine=RapidOCR(params={'EngineConfig.onnxruntime.use_cuda':True,'EngineConfig.onnxruntime.cuda_ep_cfg.cudnn_conv_algo_search':'HEURISTIC','EngineConfig.onnxruntime.intra_op_num_threads':threads,'EngineConfig.onnxruntime.inter_op_num_threads':1})
pp.LocalOcr=CudaOcr
stop=threading.Event();memory=[]
def monitor():
    while not stop.is_set():
        try: memory.append(int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()[0]))
        except Exception: pass
        stop.wait(.25)
thread=threading.Thread(target=monitor,daemon=True);thread.start()
started=time.perf_counter()
try: runpy.run_path(str(root/'scripts/process_document.py'),run_name='__main__')
finally: stop.set();thread.join()
counts=[]
for s in sessions:
    p=Path(s.end_profiling())
    c=collections.Counter(e['args']['provider'] for e in json.loads(p.read_text()) if e.get('args',{}).get('provider'))
    if not c['CUDAExecutionProvider']: raise RuntimeError('No CUDA kernel ran')
    counts.append(dict(c))
report=dict(runtime=ort.__version__,seconds=time.perf_counter()-started,device_memory_start_MiB=memory[0] if memory else None,device_memory_peak_MiB=max(memory,default=None),sessions=counts)
print(report)
if not counts: raise SystemExit('No OCR session ran; use a scanned page and fresh output directory.')
(out/'last-run.json').write_text(json.dumps(report,indent=2))
