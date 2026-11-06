"""Download official weights separately from bounded per-region inference."""
import argparse,hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('PADDLE_PDX_CACHE_HOME',str(ROOT/'data/processed/models/graphics-weights'))
os.environ.setdefault('PADDLE_PDX_MODEL_SOURCE','bos');os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK','True')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--with-vl',action='store_true');p.add_argument('--output',type=Path,default=ROOT/'data/processed/models/graphics-weights/weights-lock.json')
    p.add_argument('--verify',action='store_true',help='Verify existing local files against the saved hash lock without downloading')
    a=p.parse_args()
    if a.verify:
        lock=json.loads(a.output.read_text(encoding='utf8'));mismatches=[]
        for name,entry in lock.items():
            for relative,expected in entry['files_sha256'].items():
                file=Path(entry['directory'])/relative
                if not file.is_file():mismatches.append(f'{name}/{relative}: missing');continue
                sha=hashlib.sha256()
                with file.open('rb') as stream:
                    for chunk in iter(lambda:stream.read(4*1024*1024),b''):sha.update(chunk)
                if sha.hexdigest()!=expected:mismatches.append(f'{name}/{relative}: hash mismatch')
        print(json.dumps({'models':len(lock),'verified':not mismatches,'errors':mismatches}))
        raise SystemExit(1 if mismatches else 0)
    from paddlex.inference.utils.official_models import official_models
    names=['PP-DocLayout_plus-L','PP-DocLayout-S','PP-LCNet_x1_0_doc_ori','PP-LCNet_x1_0_table_cls','SLANet_plus',
        'SLANeXt_wired','SLANeXt_wireless','RT-DETR-L_wired_table_cell_det','RT-DETR-L_wireless_table_cell_det']
    if a.with_vl:names.append('PaddleOCR-VL-1.6-0.9B')
    result={}
    for name in names:
        path=Path(official_models[name]);files={}
        for file in path.rglob('*'):
            if file.is_file():
                sha=hashlib.sha256()
                with file.open('rb') as f:
                    for chunk in iter(lambda:f.read(4*1024*1024),b''):sha.update(chunk)
                files[str(file.relative_to(path))]=sha.hexdigest()
        result[name]={'directory':str(path),'files_sha256':files}
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
        print(json.dumps({'model':name,'files':len(files),'status':'downloaded_and_hashed'}),flush=True)
