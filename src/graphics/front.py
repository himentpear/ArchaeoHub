"""Layout-first entry point for new PDFs; legacy OCR is retained as source evidence."""
from pathlib import Path
import json,subprocess
from src.ingest.page_pipeline import file_hash,atomic_json
from src.graphics.pipeline import DEFAULT_RUNTIME,ROOT


def prepare_pdf(pdf,output,*,max_pages=None,dpi=180,runtime=DEFAULT_RUNTIME):
    import pymupdf
    pdf=Path(pdf).resolve();sha=file_hash(pdf)
    root=Path(output).resolve()/'graphics-front'/f'{sha[:16]}-{dpi}-{max_pages or "all"}'
    images=root/'images';images.mkdir(parents=True,exist_ok=True)
    request={'config':{'layout_model':'PP-DocLayout_plus-L','threads':4,'complex_review':True},'pages':[]}
    from src.graphics.pipeline import implementation_hash,model_fingerprint
    request['config']['implementation_hash']=implementation_hash()
    request['config']['weights_lock_sha256']=model_fingerprint()
    with pymupdf.open(pdf) as document:
        count=min(len(document),max_pages) if max_pages else len(document)
        for i in range(count):
            path=images/f'{i+1:05d}.jpg'
            if not path.exists():document[i].get_pixmap(matrix=pymupdf.Matrix(dpi/72,dpi/72),alpha=False).save(path,jpg_quality=88)
            request['pages'].append({'page':i+1,'image_path':str(path),'source_pdf':str(pdf),'pdf_page_index':i,'lines':[]})
    atomic_json(root/'request.json',request);cache=root/'results'
    with (root/'worker.log').open('w',encoding='utf8') as log:
        result=subprocess.run([str(runtime),str(ROOT/'src/graphics/worker.py'),'--request',str(root/'request.json'),'--output',str(cache)],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
    if result.returncode:raise RuntimeError(f'Layout-first graphic stage failed; inspect {root / "worker.log"}')
    return cache
