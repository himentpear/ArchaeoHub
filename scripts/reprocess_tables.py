"""Discover and reprocess tables in every available source PDF, with resumable caches."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqlalchemy import select
from src.db.models import Canvas, Manifest
from src.db.session import get_session, init_db
from src.graphics.pipeline import DEFAULT_RUNTIME, DEFAULT_LAYOUT_MODEL, model_fingerprint, prepare_request, run_graphics
from src.ingest.page_pipeline import atomic_json, job_lock, file_hash

CLASSIFICATION_VERSION=3


def available_documents(library):
    engine = init_db('sqlite:///' + (library / 'library.db').as_posix())
    try:
        with get_session(engine) as session:
            return [
                {'id': manifest.id, 'pdf': manifest.source_pdf_path,
                 'pages': list(session.scalars(select(Canvas.scan_index).where(
                     Canvas.manifest_id == manifest.id).order_by(Canvas.scan_index)))}
                for manifest in session.scalars(select(Manifest).order_by(Manifest.id))
                if manifest.source_pdf_path and Path(manifest.source_pdf_path).is_file()
            ]
    finally:
        engine.dispose()


def discover(library, document, output, runtime, settings):
    config = {'layout_model': DEFAULT_LAYOUT_MODEL, 'threads': 4, 'layout_only': True,
              'discovery_version': 2, 'classification_version':CLASSIFICATION_VERSION,'weights_lock_sha256': model_fingerprint()}
    older_layouts=[]
    for request_file in sorted((output/'discovery'/document['id']).glob('*/request.json')):
        previous=json.loads(request_file.read_text(encoding='utf8'))
        if previous.get('config',{}).get('layout_model')=='PP-DocLayout_plus-L':
            config['secondary_layout_cache']=str(request_file.parent/'pages')
        if (previous.get('config',{}).get('layout_model')==DEFAULT_LAYOUT_MODEL
                and previous['config'].get('classification_version',0)<CLASSIFICATION_VERSION):
            older_layouts.append(request_file)
    for request_file in reversed(older_layouts):
        found={'cache':str(request_file.parent/'pages'),'scanned_pages':len(document['pages'])}
        if reusable_discovery(library,document,found,require_current_classification=False):
            config['layout_cache']=found['cache'];config['secondary_layout_cache']=found['cache']
            break
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    directory = output / 'discovery' / document['id'] / key
    directory.mkdir(parents=True, exist_ok=True)
    engine = init_db('sqlite:///' + (library / 'library.db').as_posix())
    try:
        request = prepare_request(engine, document['id'], document['pages'], config)
    finally:
        engine.dispose()
    request_path = directory / 'request.json'
    atomic_json(request_path, request)
    command = [str(runtime), str(ROOT / 'src/graphics/worker.py'), '--request', str(request_path),
               '--output', str(directory / 'pages')]
    if settings:
        command.extend(['--settings', str(settings)])
    with (directory / 'worker.log').open('w', encoding='utf-8') as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    tables = []
    count = 0
    excluded = []
    from src.graphics.grouping import data_table_regions
    lines = {page['page']: page['lines'] for page in request['pages']}
    for page in document['pages']:
        value = json.loads((directory / 'pages' / f'{page:05d}.json').read_text(encoding='utf-8'))
        if 'table_assessment' in value:
            regions=[region for region in value['regions'] if region['category']=='table']
            discarded=value.get('excluded_table_grids',[])
        else:
            regions, discarded = data_table_regions(value['regions'],value['layout'],lines[page],value['image_size'][1])
        excluded.extend({'page':page, **region} for region in discarded)
        if regions:
            tables.append(page)
            count += len(regions)
    atomic_json(directory / 'excluded-grids.json',excluded)
    return {'manifest_id': document['id'], 'pdf': document['pdf'], 'scanned_pages': len(document['pages']),
            'table_pages': tables, 'table_regions': count, 'excluded_illustration_grids':len(excluded),
            'exclusions':str(directory / 'excluded-grids.json'),'cache': str(directory / 'pages')}


def export_tables(output, results):
    from src.graphics.models import GraphicObject
    from src.graphics.service import object_detail, render_csv, render_html, render_xlsx
    engine = init_db('sqlite:///' + (output / 'library.db').as_posix())
    inventory = []
    try:
        with get_session(engine) as session:
            for result in results:
                objects = session.scalars(select(GraphicObject).where(
                    GraphicObject.run_id == result['run_id'], GraphicObject.category == 'table',
                    GraphicObject.status != 'superseded'))
                for obj in objects:
                    detail = object_detail(session, obj.id)
                    directory = output / 'exports' / obj.manifest_id / obj.id
                    directory.mkdir(parents=True, exist_ok=True)
                    atomic_json(directory / 'table.json', detail)
                    (directory / 'table.html').write_bytes(render_html(detail))
                    (directory / 'records.csv').write_bytes(render_csv(detail))
                    (directory / 'table.xlsx').write_bytes(render_xlsx(detail))
                    inventory.append({'id': obj.id, 'manifest_id': obj.manifest_id, 'label': obj.label,
                                      'pages': [part['scan_index'] for part in detail['fragments']],
                                      'records': len(detail['table']['records']),
                                      'issues': detail['table']['issues'], 'directory': str(directory)})
    finally:
        engine.dispose()
    atomic_json(output / 'table-inventory.json', inventory)
    return inventory


def reusable_discovery(library, document, found,*,require_current_classification=True):
    cache=Path(found['cache'])
    runtime=cache/'runtime.json'
    if found['scanned_pages']!=len(document['pages']) or not runtime.is_file():
        return False
    lock=json.loads((ROOT/'data/processed/models/graphics-weights/weights-lock.json').read_text(encoding='utf8'))
    weights=json.loads(runtime.read_text(encoding='utf8')).get('weights_sha256',{})
    for name in [DEFAULT_LAYOUT_MODEL,'PP-DocLayout_plus-L']:
        for relative,expected in lock[name]['files_sha256'].items():
            if weights.get(str(Path(name)/relative))!=expected:
                return False
    engine=init_db('sqlite:///'+(library/'library.db').as_posix())
    try:
        with get_session(engine) as session:
            canvases=list(session.scalars(select(Canvas).where(Canvas.manifest_id==document['id'])))
            for canvas in canvases:
                page=cache/f'{canvas.scan_index:05d}.json'
                if not page.is_file():return False
                value=json.loads(page.read_text(encoding='utf8'))
                if (value['signature']['source_hash']!=file_hash(Path(canvas.image_storage_path))
                        or value['signature']['config']['layout_model']!=DEFAULT_LAYOUT_MODEL
                        or require_current_classification and value['signature']['config'].get('classification_version')!=CLASSIFICATION_VERSION):
                    return False
    finally:
        engine.dispose()
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, default=ROOT / 'data/processed/library')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/processed/benchmarks/tables-reprocessed-20261115')
    parser.add_argument('--runtime', type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument('--settings', type=Path)
    parser.add_argument('--discover-only', action='store_true')
    parser.add_argument('--resume', action='store_true',help='Reuse completed discoveries after checking source images and layout weights')
    args = parser.parse_args()
    args.library, args.output = args.library.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    with job_lock(args.output / 'tables-batch.lock'):
        documents = available_documents(args.library)
        if not documents:
            raise ValueError('No registered source PDFs are available')
        started = time.time()
        previous={}
        if args.resume and (args.output/'progress.json').is_file():
            previous=json.loads((args.output/'progress.json').read_text(encoding='utf8'))
        discovered={document['manifest_id']:document for document in previous.get('documents',[])}
        state = {'status': 'discovering', 'started_at': started, 'documents': [], 'results': []}
        atomic_json(args.output / 'progress.json', state)
        for document in documents:
            state['current_manifest'] = document['id']
            atomic_json(args.output / 'progress.json', state)
            found=discovered.get(document['id'])
            if found is None or not reusable_discovery(args.library,document,found):
                found = discover(args.library, document, args.output, args.runtime, args.settings)
            state['documents'].append(found)
            atomic_json(args.output / 'progress.json', state)
            print(json.dumps(found, ensure_ascii=False), flush=True)
        state['status'] = 'discovered'
        atomic_json(args.output / 'progress.json', state)
        if args.discover_only:
            return
        for document in state['documents']:
            if not document['table_pages']:
                continue
            state['status'] = 'recognizing'
            state['current_manifest'] = document['manifest_id']
            atomic_json(args.output / 'progress.json', state)
            result = run_graphics(args.library, document['manifest_id'], args.output, document['table_pages'],
                                  runtime=args.runtime, settings_path=args.settings, with_vl=False,
                                  tables_only=True, layout_cache=Path(document['cache']))
            state['results'].append(result)
            atomic_json(args.output / 'progress.json', state)
        inventory = export_tables(args.output, state['results'])
        classification_review=[]
        for result in state['results']:
            for page in result['pages']:
                value=json.loads((Path(result['cache'])/f'{page:05d}.json').read_text(encoding='utf8'))
                classification_review.extend({'manifest_id':result['manifest_id'],'page':page,**region}
                    for region in value.get('excluded_table_grids',[]) if region.get('reason')=='native_ocr_classification_pending')
        atomic_json(args.output/'classification-review.json',classification_review)
        state.update(status='complete', finished_at=time.time(), tables=len(inventory),
                     records=sum(table['records'] for table in inventory),classification_pending=len(classification_review))
        atomic_json(args.output / 'progress.json', state)
        print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
