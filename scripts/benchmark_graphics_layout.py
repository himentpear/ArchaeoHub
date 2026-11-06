"""Compare CPU layout models on the same source pages; no accuracy without labels."""
import argparse,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.graphics.pipeline import prepare_request,DEFAULT_RUNTIME,ROOT
from src.db.session import create_db_engine
from src.ingest.page_pipeline import atomic_json

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--library',type=Path,default=Path('data/processed/library'));p.add_argument('--manifest-id',required=True)
    p.add_argument('--annotations',type=Path,required=True);p.add_argument('--output',type=Path,default=Path('data/processed/benchmarks/graphics-layout'));a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);data=json.loads(a.annotations.read_text(encoding='utf8'));pages=[r['page'] for r in data['pages']]
    engine=create_db_engine('sqlite:///'+(a.library.resolve()/'library.db').as_posix());reports=[]
    try:
        for name in ['PP-DocLayout_plus-L','PP-DocLayout-S']:
            directory=a.output/name;directory.mkdir(exist_ok=True)
            request=prepare_request(engine,a.manifest_id,pages,{'layout_model':name,'threads':4,'layout_only':True})
            atomic_json(directory/'request.json',request)
            with (directory/'worker.log').open('w',encoding='utf8') as log:
                subprocess.run([str(DEFAULT_RUNTIME),str(ROOT/'src/graphics/worker.py'),'--request',str(directory/'request.json'),'--output',str(directory/'output')],
                    stdout=log,stderr=subprocess.STDOUT,check=True)
            results=[json.loads((directory/'output'/f'{n:05d}.json').read_text(encoding='utf8')) for n in pages]
            times=sorted(r['elapsed_seconds'] for r in results)
            reports.append({'model':name,'pages':len(results),'median_seconds':times[len(times)//2],
                'p95_seconds':times[int((len(times)-1)*.95)],'sum_seconds':sum(times),'regions':sum(len(r['regions']) for r in results),
                'accuracy':'requires independent source annotations','runtime':json.loads((directory/'output/runtime.json').read_text(encoding='utf8'))})
        atomic_json(ROOT/'reports/graphics_layout_comparison.json',reports)
        print(json.dumps([{k:v for k,v in r.items() if k!='runtime'} for r in reports],ensure_ascii=False,indent=2))
    finally:engine.dispose()
