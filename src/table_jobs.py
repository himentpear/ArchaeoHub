"""Deploy completed table pages with durable progress and source validation."""
import hashlib
import json
import math
from pathlib import Path
import time

from sqlalchemy import select
from PIL import Image

from src.db.session import get_session, init_db
from src.db.models import ReviewTask
from src.graphics.models import GraphicObject, GraphicRun
from src.graphics.pipeline import import_pages, source_guard
from src.graphics.service import object_detail, render_csv, render_html, render_xlsx
from src.ingest.page_pipeline import atomic_json, file_hash, job_lock
from src.io_utils import atomic_write
from src.table_alignment import align_table


def list_table_jobs(library, manifest_id=None):
    jobs=[]
    for path in (Path(library)/'table-jobs').glob('*.json'):
        if path.name.endswith(('-inventory.json','-semantic.json')):continue
        job=json.loads(path.read_text(encoding='utf8'))
        semantic=path.with_name(path.stem+'-semantic.json')
        if semantic.is_file():job['semantic_review']=json.loads(semantic.read_text(encoding='utf8'))
        if manifest_id and manifest_id not in {item['manifest_id'] for item in job.get('documents',[])}:continue
        jobs.append(job)
    return sorted(jobs,key=lambda job:job.get('updated_at',0),reverse=True)


def rollback_table_batch(staging, library):
    from src.graphics.pipeline import unpublish
    staging,library=Path(staging).resolve(),Path(library).resolve()
    job_id=hashlib.sha256(str(staging).encode()).hexdigest()[:20]
    path=library/'table-jobs'/f'{job_id}.json'
    with job_lock(path.with_suffix('.lock')):
        state=json.loads(path.read_text(encoding='utf8'))
        state.update(status='paused',updated_at=time.time())
        atomic_json(path,state)
        state['rollbacks']=[unpublish(library,run_id) for run_id in state.get('runs',{}).values()]
        atomic_json(path,state)
        return state


def validate_completed_page(path, request):
    data=json.loads(path.read_text(encoding='utf8'))
    expected=file_hash(Path(request['image_path']))
    if data['signature']['source_hash']!=expected:raise ValueError(f"Page {request['page']} image changed")
    width,height=data['image_size']
    with Image.open(request['image_path']) as image:
        if image.size!=(width,height):raise ValueError('Source image dimensions differ')
    for region in data['regions']:
        if region['source_hash']!=expected:raise ValueError('Region source hash mismatch')
        if not Path(region['crop_path']).is_file():raise ValueError('Missing region crop')
        for cell in region.get('cells',[]):
            if not Path(cell['crop_path']).is_file():raise ValueError('Missing cell crop')
            if any(type(cell[key]) is not int or cell[key]<1 for key in ('row_span','column_span')):
                raise ValueError('Invalid cell span')
            if cell['row']<0 or cell['column']<0:raise ValueError('Invalid cell position')
            for x,y in cell['polygon_source']:
                if not math.isfinite(x) or not math.isfinite(y) or not -1<=x<=width+1 or not -1<=y<=height+1:
                    raise ValueError('Cell anchor outside source image')
    return data


def store_table_check(engine, detail, alignment):
    issues=list(detail['table']['issues'])
    if detail['evidence'].get('unresolved_continuation'):
        issues.append({'code':'unresolved_continuation','message':'续表关联证据存在分歧','cell_ids':[]})
    snapshot={'revision':detail['revision'],'issues':issues,'body_conflicts':alignment['conflicts'],
        'ambiguous_rows':alignment['ambiguous_rows'],'identifier_rows':alignment['identifier_rows']}
    with get_session(engine) as session:
        current=session.get(GraphicObject,detail['id'])
        if current is None or current.revision_no!=detail['revision']:return
        task=session.scalar(select(ReviewTask).where(ReviewTask.target_type=='graphic_object',
            ReviewTask.target_id==detail['id'],ReviewTask.risk_type=='table_automatic_checks').order_by(ReviewTask.created_at.desc()))
        if task and task.candidates_snapshot==[snapshot]:return
        unresolved=bool(issues or alignment['conflicts'] or alignment['ambiguous_rows'])
        if task is None or task.status!='pending':
            if not unresolved:return
            task=ReviewTask(manifest_id=detail['manifest_id'],canvas_id=detail['fragments'][0]['canvas_id'],
                target_type='graphic_object',target_id=detail['id'],risk_type='table_automatic_checks',priority_score=.7)
            session.add(task)
        task.candidates_snapshot=[snapshot]
        task.description=f"表格自动核对，修订 {detail['revision']}，结构问题 {len(issues)} 项，正文冲突 {alignment['conflicts']} 项，编号歧义 {alignment['ambiguous_rows']} 行。"
        if not unresolved:
            task.status='approved';task.reviewer='workflow';task.resolution_note='当前修订的结构与关联字段规则检查通过'


def export_deployed_tables(engine, library, run_ids, job_id):
    inventory=[];checks=[]
    with get_session(engine) as session:
        objects=session.scalars(select(GraphicObject).where(GraphicObject.run_id.in_(run_ids),
            GraphicObject.category=='table',GraphicObject.status!='superseded').order_by(GraphicObject.id))
        for obj in objects:
            detail=object_detail(session,obj.id)
            directory=Path(library)/'exports'/'tables'/obj.run_id/obj.id
            atomic_json(directory/'table.json',detail)
            atomic_write(directory/'table.html',render_html(detail))
            atomic_write(directory/'records.csv',render_csv(detail))
            atomic_write(directory/'table.xlsx',render_xlsx(detail))
            alignment=align_table(session,obj.id,limit=500)
            while alignment['remaining_rows']:
                more=align_table(session,obj.id,limit=500,offset=alignment['processed_rows'])
                alignment['rows'].extend(more['rows'])
                alignment['processed_rows']+=more['processed_rows']
                alignment['remaining_rows']=more['remaining_rows']
                alignment['conflicts']+=more['conflicts']
                alignment['ambiguous_rows']+=more['ambiguous_rows']
            atomic_json(directory/'body-alignment.json',alignment)
            checks.append((detail,alignment))
            inventory.append({'id':obj.id,'manifest_id':obj.manifest_id,'label':obj.label,'status':obj.status,
                'revision':obj.revision_no,'pages':[part['scan_index'] for part in detail['fragments']],
                'records':len(detail['table']['records']),'issues':detail['table']['issue_counts'],
                'body_conflicts':alignment['conflicts'],'ambiguous_rows':alignment['ambiguous_rows'],
                'semantic_review_recommended':detail['table']['semantic_review']['recommended'],
                'directory':str(directory.resolve())})
    for detail,alignment in checks:store_table_check(engine,detail,alignment)
    atomic_json(Path(library)/'table-jobs'/f'{job_id}-inventory.json',inventory)
    return inventory


def sync_table_batch(staging, library):
    staging,library=Path(staging).resolve(),Path(library).resolve()
    if staging==library:raise ValueError('Choose distinct OCR staging and deployment libraries')
    job_id=hashlib.sha256(str(staging).encode()).hexdigest()[:20]
    directory=library/'table-jobs'
    state_path=directory/f'{job_id}.json'
    with job_lock(directory/'publication.lock'),job_lock(directory/f'{job_id}.lock'):
        state=json.loads(state_path.read_text(encoding='utf8')) if state_path.is_file() else {
            'id':job_id,'created_at':time.time(),'runs':{},'status':'discovering','deployed_pages':0}
        if state['status']=='paused':return state
        progress=json.loads((staging/'progress.json').read_text(encoding='utf8'))
        state.update(updated_at=time.time(),source_directory=str(staging),documents=progress.get('documents',[]),
            scanned_pages=sum(item['scanned_pages'] for item in progress.get('documents',[])),
            expected_pages=sum(len(item['table_pages']) for item in progress.get('documents',[])),
            expected_regions=sum(item['table_regions'] for item in progress.get('documents',[])))
        request_path=staging/'request.json'
        if progress['status']=='discovering' or not request_path.is_file():
            state['status']='discovering';atomic_json(state_path,state);return state
        request=json.loads(request_path.read_text(encoding='utf8'))
        config=request['config'];manifest_id=config['manifest_id']
        key=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:20]
        cache=staging/'graphics-cache'/key
        pages=[item['page'] for item in request['pages'] if (cache/f"{item['page']:05d}.json").is_file()]
        state.update(current_manifest=manifest_id,current_page=max(pages,default=None),cache=key)
        if progress.get('workflow')=='graphics':state['recognition_retries']=progress.get('retry_count',0)
        engine=init_db('sqlite:///'+(library/'library.db').as_posix())
        try:
            guard=source_guard(staging/'library.db',manifest_id)
            if source_guard(library/'library.db',manifest_id)!=guard:raise ValueError('Deployment source revision changed')
            fingerprint=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
            with get_session(engine) as session:
                run=session.scalar(select(GraphicRun).where(GraphicRun.manifest_id==manifest_id,GraphicRun.fingerprint==fingerprint))
                imported=set(run.config.get('pages',[])) if run else set()
            inputs={item['page']:item for item in request['pages']}
            validate_pages=set(pages) if state.get('validation_version')!=2 else set(pages)-imported
            for page in sorted(validate_pages):
                data=validate_completed_page(cache/f'{page:05d}.json',inputs[page])
                if data['signature'].get('config')!=config:raise ValueError('Completed page configuration mismatch')
            state['validation_version']=2
            if pages:
                if not state.get('backup'):
                    from src.pipeline import _backup_database
                    state['backup']=str(_backup_database(library))
                    atomic_json(state_path,state)
                run_id=import_pages(engine,manifest_id,cache,pages,config,guard,incremental=True,
                    published=True,asset_root=library/'graphics')
                state['runs'][manifest_id]=run_id
            run_ids=list(state['runs'].values())
            with get_session(engine) as session:
                runs=list(session.scalars(select(GraphicRun).where(GraphicRun.id.in_(run_ids))))
                state['deployed_pages']=sum(run.metrics.get('pages',0) for run in runs)
                state['deployed_regions']=sum(run.metrics.get('regions',0) for run in runs)
                revisions=list(session.execute(select(GraphicObject.id,GraphicObject.revision_no,GraphicObject.status).where(
                    GraphicObject.run_id.in_(run_ids)).order_by(GraphicObject.id)))
            revision_key=hashlib.sha256(json.dumps({'schema_version':2,'objects':[tuple(row) for row in revisions]}).encode()).hexdigest()
            if state.get('export_revision')!=revision_key:
                inventory=export_deployed_tables(engine,library,run_ids,job_id)
                state.update(export_revision=revision_key,tables=len(inventory),records=sum(item['records'] for item in inventory),
                    inventory=f'/api/table_inventory?job={job_id}')
            state['status']='recognizing'
            if progress['status']=='complete' and state['deployed_pages']==state['expected_pages']:
                state['status']='complete'
                state.setdefault('finished_at',time.time())
            failure=cache/'failure.json'
            state.pop('failure',None)
            if failure.is_file() and state['status']!='complete' and (progress.get('workflow')!='graphics' or progress['status']=='failed'):
                failed=json.loads(failure.read_text(encoding='utf8'))
                if failed.get('page') not in pages:state.update(status='recognition_failed',failure=failed)
            state.pop('deployment_error',None)
        except Exception as error:
            state.update(status='deployment_failed',deployment_error=str(error))
            atomic_json(state_path,state)
            raise
        finally:engine.dispose()
        atomic_json(state_path,state)
        return state
