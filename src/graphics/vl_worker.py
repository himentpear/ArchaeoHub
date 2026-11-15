"""Compatibility worker using the independent PaddleOCR-VL v1.6 service."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.graphics.worker import digest,save_json
from src.ocr_api import PaddleOcrClient

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--image',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    client=PaddleOcrClient()
    prediction=client.recognize(a.image,task='text',output_dir=a.output.with_suffix(''))
    save_json(a.output,{'status':'candidate','provider':'PaddleOCR-VL-1.6-0.9B','source_hash':digest(a.image),
        'result':prediction,'requires_review':True,'generation_limits':client.identity})
