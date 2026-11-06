"""Evaluate a staging run; missing labels are a failed gate, not claimed accuracy."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.graphics.evaluation import evaluate
from src.ingest.page_pipeline import atomic_json
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--library',type=Path,required=True);p.add_argument('--run-id',required=True)
    p.add_argument('--annotations',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=evaluate(a.library,a.run_id,a.annotations);atomic_json(a.output,result);print(json.dumps(result,ensure_ascii=False,indent=2))
    raise SystemExit(0 if result['passed'] else 1)
