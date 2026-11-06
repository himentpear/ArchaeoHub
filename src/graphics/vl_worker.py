"""Isolated small local VLM second opinion, with a hard parent-process deadline."""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
os.environ.setdefault('PADDLE_PDX_CACHE_HOME',str(ROOT/'data/processed/models/graphics-weights'))
os.environ.setdefault('PADDLE_PDX_MODEL_SOURCE','bos');os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK','True')
from src.graphics.worker import digest,save_json,result_json

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--image',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    from paddleocr import PaddleOCRVL
    model=PaddleOCRVL(pipeline_version='v1.6',device='cpu',cpu_threads=4,enable_mkldnn=False,
        use_layout_detection=False,use_doc_orientation_classify=False,use_doc_unwarping=False,
        use_chart_recognition=False,use_seal_recognition=False)
    prediction=list(model.predict(str(a.image),prompt_label='table',use_layout_detection=False,max_new_tokens=256,max_pixels=262144))[0]
    save_json(a.output,{'status':'candidate','provider':'PaddleOCR-VL-1.6-0.9B','source_hash':digest(a.image),
        'result':result_json(prediction),'requires_review':True,'generation_limits':{'max_new_tokens':256,'max_pixels':262144}})
