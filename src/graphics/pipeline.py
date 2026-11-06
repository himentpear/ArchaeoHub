"""Incremental staging and gated promotion; never rebuild a reviewed document."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import time
from sqlalchemy import select,delete,update
from src.db.models import Canvas,Manifest,AltoLine,ReviewTask
from src.db.session import init_db,get_session
from src.graphics.models import GraphicRun,GraphicObject,GraphicFragment,GraphicCell
from src.graphics.geometry import transform_points,box_polygon
from src.graphics.grouping import continuation_evidence,stitch_pair
from src.ingest.page_pipeline import atomic_json,file_hash,job_lock

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_RUNTIME=ROOT/'data/processed/models/graphics-runtime/Scripts/python.exe'


def source_guard(db_path,manifest_id):
    connection=sqlite3.connect(db_path.resolve().as_uri()+'?mode=ro',uri=True)
    h=hashlib.sha256()
    queries=[('select id,source_pdf_sha256 from manifests where id=?',),
        ('select id,scan_index,revision_no,image_storage_path from canvases where manifest_id=? order by id',),
        ('select l.id,l.raw_ocr,l.content_text,l.approved_text from alto_lines l join canvases c on c.id=l.canvas_id where c.manifest_id=? order by l.id',),
        ('select r.id,r.status,r.footnote_id,r.char_start,r.char_end from footnote_references r join canvases c on c.id=r.canvas_id where c.manifest_id=? order by r.id',)]
    for (query,) in queries:
        for row in connection.execute(query,(manifest_id,)):h.update(json.dumps(tuple(row),ensure_ascii=False).encode())
    connection.close();return h.hexdigest()


def clone_database(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():return
    a=sqlite3.connect(source.resolve().as_uri()+'?mode=ro',uri=True);b=sqlite3.connect(target)
    try:a.backup(b)
    finally:b.close();a.close()


def implementation_hash():
    h=hashlib.sha256()
    for path in sorted((ROOT/'src/graphics').glob('*.py')):h.update(path.read_bytes().replace(b'\r\n',b'\n'))
    return h.hexdigest()


def model_fingerprint():
    """Bind page caches to downloaded weights, not only model names."""
    lock=ROOT/'data/processed/models/graphics-weights/weights-lock.json'
    return file_hash(lock) if lock.is_file() else 'unlocked'


def prepare_request(engine,manifest_id,pages,config):
    with get_session(engine) as session:
        manifest=session.get(Manifest,manifest_id)
        if manifest is None:raise LookupError('Manifest not found')
        selected=list(session.scalars(select(Canvas).where(Canvas.manifest_id==manifest_id,Canvas.scan_index.in_(pages)).order_by(Canvas.scan_index)))
        if len(selected)!=len(set(pages)):raise ValueError('Requested physical pages do not all exist')
        items=[]
        from PIL import Image
        for canvas in selected:
            with Image.open(canvas.image_storage_path) as image:w,h=image.size
            sx=w/canvas.width;sy=h/canvas.height
            lines=[{'text':l.content_text,'bbox':[l.x*sx,l.y*sy,(l.x+l.w)*sx,(l.y+l.h)*sy]} for l in session.scalars(
                select(AltoLine).where(AltoLine.canvas_id==canvas.id))]
            items.append({'page':canvas.scan_index,'image_path':canvas.image_storage_path,'source_pdf':manifest.source_pdf_path,
                'pdf_page_index':canvas.pdf_page_index,'lines':lines})
        return {'config':config,'pages':items}


def import_pages(engine,manifest_id,cache,pages,config,guard):
    fingerprint=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    with get_session(engine) as session:
        manifest=session.get(Manifest,manifest_id)
        existing=session.scalar(select(GraphicRun).where(GraphicRun.manifest_id==manifest_id,GraphicRun.fingerprint==fingerprint))
        if existing:
            if existing.status=='complete':return existing.id
            raise ValueError('Incomplete run exists; use another output directory to preserve it')
        run=GraphicRun(manifest_id=manifest_id,fingerprint=fingerprint,config={**config,'source_guard':guard,'pages':pages,'created_at':time.time()})
        session.add(run);session.flush();descriptors=[]
        for n in pages:
            data=json.loads((cache/f'{n:05d}.json').read_text(encoding='utf8'))
            if config.get('implementation_hash') and data.get('signature',{}).get('config')!=config:
                raise ValueError('Worker cache configuration/model version mismatch; rerun the worker')
            canvas=session.scalar(select(Canvas).where(Canvas.manifest_id==manifest_id,Canvas.scan_index==n))
            if data['signature']['source_hash']!=file_hash(Path(canvas.image_storage_path)):raise ValueError('Page image changed')
            sx=canvas.width/data['image_size'][0];sy=canvas.height/data['image_size'][1]
            scale=[[sx,0,0],[0,sy,0],[0,0,1]]
            for r in data['regions']:
                obj=GraphicObject(run_id=run.id,manifest_id=manifest_id,category=r['category'],label=r.get('label',''),title=r.get('title',''),
                    evidence={'provider':config['layout_model'],'source_region':r['key'],'source_page':n,'model_structure':r.get('structure',{})})
                session.add(obj);session.flush()
                polygon=transform_points(box_polygon(r['bbox']),scale)
                fragment=GraphicFragment(object_id=obj.id,canvas_id=canvas.id,source_key=r['key'],bbox=[r['bbox'][0]*sx,r['bbox'][1]*sy,
                    (r['bbox'][2]-r['bbox'][0])*sx,(r['bbox'][3]-r['bbox'][1])*sy],polygon=polygon,rotation=r['rotation'],
                    transform={'crop_to_source':r['crop_to_source'],'source_to_canvas':scale},source_hash=r['source_hash'],crop_path=r['crop_path'],confidence=r['score'],
                    evidence={'header_rows':r.get('structure',{}).get('header_rows',0),'signature':r.get('structure',{}).get('signature',{}),
                        'continuation':r.get('continuation',False),'appendix':r.get('appendix',False),'method':r['method'],
                        'completeness':r.get('completeness',{})})
                session.add(fragment);session.flush()
                for c in r.get('cells',[]):
                    poly=transform_points(c['polygon_source'],scale)
                    session.add(GraphicCell(fragment_id=fragment.id,row_index=c['row'],column_index=c['column'],row_span=c['row_span'],column_span=c['column_span'],
                        raw_text=c.get('text',''),value_kind=c.get('value_kind','unreadable'),symbols=c.get('symbols',[]),header_path=c.get('header_path',[]),
                        anchors=[{'canvas_id':canvas.id,'scan_index':n,'polygon':poly,'source_hash':r['source_hash']}],crop_path=c.get('crop_path',''),
                        confidence=c.get('confidence',0),evidence={'method':c.get('method'),'line_results':c.get('line_results',[])}))
                descriptors.append({'object_id':obj.id,'fragment_id':fragment.id,'manifest_id':manifest_id,'namespace':manifest.work.metadata_json.get('source_namespace') or manifest.work_id,
                    'page':n,'label':obj.label,'category':obj.category,'signature':fragment.evidence['signature'],'appendix':r.get('appendix',False),
                    'section_id':'appendix_tables' if r.get('appendix') else None,'continuation':r.get('continuation',False)})
                session.add(ReviewTask(manifest_id=manifest_id,canvas_id=canvas.id,target_type='graphic_object',target_id=obj.id,
                    risk_type='graphic_structure_candidate',priority_score=.65,description='图表结构候选；请核对分类、单元格、符号和边界。'))
        session.flush()
        # Only explicit, structurally supported continuations are grouped. Raw
        # fragment rows remain unchanged, so header folding and undo are lossless.
        for i,b in enumerate(descriptors):
            conflicts=[]
            for a in reversed(descriptors[:i]):
                evidence=continuation_evidence(a,b)
                if not evidence['accepted']:
                    if a['label'] and a['label']==b['label']:conflicts.append({'candidate_fragment':a['fragment_id'],**evidence})
                    continue
                source=session.get(GraphicObject,b['object_id']);target=session.get(GraphicObject,a['object_id'])
                while target.status=='superseded':target=session.get(GraphicObject,target.evidence['merged_into'])
                part=session.get(GraphicFragment,b['fragment_id']);part.object_id=target.id;part.part_index=b['page']
                part.evidence={**part.evidence,'fold_repeated_header':True}
                target.structure={**target.structure,'continuations':target.structure.get('continuations',[])+[{'fragment_id':part.id,**evidence}]}
                source.status='superseded';source.evidence={**source.evidence,'merged_into':target.id}
                b['object_id']=target.id
                break
            else:
                if b.get('continuation'):
                    obj=session.get(GraphicObject,b['object_id'])
                    obj.evidence={**obj.evidence,'unresolved_continuation':True,'grouping_candidates':conflicts,
                        'missing_content':'unknown; source fragments retained without synthesis'}
        # Physical assembly is a separate decision from logical continuation.
        # Repeated table headers cannot trigger this path.
        import cv2,numpy as np
        for i,b in enumerate(descriptors):
            for a in reversed(descriptors[:i]):
                if b['page']-a['page']!=1 or a['category']!=b['category']:continue
                if a['label'] and b['label'] and a['label']!=b['label']:continue
                target=session.get(GraphicObject,a['object_id']);source=session.get(GraphicObject,b['object_id'])
                if target.id==source.id or target.status=='superseded' or source.status=='superseded':continue
                first=session.get(GraphicFragment,a['fragment_id']);second=session.get(GraphicFragment,b['fragment_id'])
                first_path=target.structure.get('composite_path',first.crop_path)
                im1=cv2.imdecode(np.fromfile(first_path,np.uint8),cv2.IMREAD_COLOR)
                im2=cv2.imdecode(np.fromfile(second.crop_path,np.uint8),cv2.IMREAD_COLOR)
                image,evidence=stitch_pair(im1,im2,b['category'])
                if not evidence['accepted']:continue
                composite=cache/f'composite-{target.id}.png';cv2.imencode('.png',image)[1].tofile(composite)
                previous=target.structure.get('fragment_to_composite',{first.id:np.eye(3).tolist()})
                transforms={fid:(np.asarray(evidence['first_to_composite'])@np.asarray(matrix)).tolist() for fid,matrix in previous.items()}
                transforms[second.id]=evidence['second_to_composite']
                target.structure={**target.structure,'composite_path':str(composite.resolve()),'stitch':evidence,
                    'stitch_steps':target.structure.get('stitch_steps',[])+[evidence],'fragment_to_composite':transforms,
                    'table_structure_pending':b['category']=='table'}
                second.object_id=target.id;second.part_index=b['page'];source.status='superseded'
                source.evidence={**source.evidence,'merged_into':target.id};b['object_id']=target.id;break
        run.status='complete';run.metrics={'pages':len(pages),'regions':len(descriptors)};session.flush();return run.id


def run_graphics(library,manifest_id,output,pages,*,runtime=DEFAULT_RUNTIME,layout_model='PP-DocLayout_plus-L',complex_review=True,skip_worker=False,
                 with_vl=True,allow_api_review=False,settings_path=None,max_vl_regions=12):
    library,output=Path(library).resolve(),Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    if library==output:raise ValueError('A separate staging library is required until benchmark gates pass')
    with job_lock(output/'graphics.lock'):
        source=library/'library.db';guard=source_guard(source,manifest_id)
        clone_database(source,output/'library.db')
        if source_guard(output/'library.db',manifest_id)!=guard:
            raise ValueError('Staging source snapshot is stale; choose a new output directory before processing')
        engine=init_db('sqlite:///'+(output/'library.db').as_posix())
        try:
            config={'layout_model':layout_model,'threads':4,'complex_review':complex_review,'implementation_hash':implementation_hash(),
                'weights_lock_sha256':model_fingerprint(),
                'selected_pages':sorted(set(pages)),'with_vl':with_vl,'allow_api_review':allow_api_review,'max_vl_regions':max_vl_regions}
            request=prepare_request(engine,manifest_id,pages,config)
            atomic_json(output/'request.json',request)
            # Historical revisions must never point at overwritten crop files.
            cache_key=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:20]
            cache=output/'graphics-cache'/cache_key;cache.mkdir(parents=True,exist_ok=True)
            if not skip_worker:
                with (output/'worker.log').open('w',encoding='utf8') as log:
                    result=subprocess.run([str(runtime),str(ROOT/'src/graphics/worker.py'),'--request',str(output/'request.json'),'--output',str(cache)],
                        cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                if result.returncode:raise RuntimeError('Graphic worker failed; inspect staging worker.log (no production rows changed)')
            if source_guard(source,manifest_id)!=guard:raise ValueError('Source was edited during processing; keep staging and reconcile before import')
            refinements=[];router=None
            if allow_api_review:
                from src.settings import load_settings
                from src.model_router import ModelRouter
                settings=load_settings(settings_path)
                settings['DEEPSEEK_MODEL']='deepseek-flash'
                router=ModelRouter.from_settings(cache/'api-review',settings)
                router.max_calls=min(20,router.max_calls)
            for n in pages:
                data=json.loads((cache/f'{n:05d}.json').read_text(encoding='utf8'))
                for region in data['regions']:
                    if region['category']!='table' or not region.get('structure',{}).get('needs_review'):continue
                    if len(refinements)>=max_vl_regions:break
                    from src.graphics.refinement import local_vl,grounded_structure_review
                    entry={'page':n,'region':region['key']}
                    if with_vl:
                        hard=sorted([c for c in region.get('cells',[]) if c.get('crop_path') and c['row']<2],key=lambda c:c.get('confidence',0))
                        crop=hard[0]['crop_path'] if hard else region['crop_path']
                        entry['local_vl_scope']='header_cell' if hard else 'table_region'
                        entry['local_vl']=local_vl(runtime,crop,cache/f'vl-{n:05d}-{region["key"]}.json')
                    if router and entry.get('local_vl',{}).get('status')!='candidate':
                        try:entry['api']=grounded_structure_review(router,region.get('cells',[]))
                        except (ValueError,RuntimeError):entry['api']={'status':'unavailable_or_unverified'}
                    refinements.append(entry)
                    region.setdefault('structure',{})['secondary_review']=entry
                atomic_json(cache/f'{n:05d}.json',data)
            atomic_json(output/'refinements.json',refinements)
            run_id=import_pages(engine,manifest_id,cache,pages,config,guard)
            result={'run_id':run_id,'manifest_id':manifest_id,'library':str(output),'cache':str(cache),'status':'staged','published':False,'pages':pages}
            atomic_json(output/'graphics-job.json',result);return result
        finally:engine.dispose()


def promote(staging,production,run_id,evaluation_path):
    evaluation=json.loads(Path(evaluation_path).read_text(encoding='utf8'))
    if evaluation.get('run_id')!=run_id or not evaluation.get('passed') or not evaluation.get('annotations_complete'):
        raise ValueError('Complete, passing evaluation for this exact run is required')
    # Recompute rather than trust an edited passed flag in a report.
    from src.graphics.evaluation import evaluate
    verified=evaluate(Path(staging),run_id,Path(evaluation['annotation_path']))
    if not verified['passed']:raise ValueError('Recomputed acceptance gates failed')
    a=init_db('sqlite:///'+(Path(staging).resolve()/'library.db').as_posix())
    try:
        with get_session(a) as s:
            run=s.get(GraphicRun,run_id)
            if source_guard(Path(production)/'library.db',run.manifest_id)!=run.config['source_guard']:raise ValueError('Production source revision changed')
            from src.pipeline import _backup_database
            backup=_backup_database(Path(production).resolve())
            b=init_db('sqlite:///'+(Path(production).resolve()/'library.db').as_posix())
            try:
                with get_session(b) as target:
                    if target.get(GraphicRun,run_id):raise ValueError('Run already imported')
                    # Published partial runs coexist; page routing selects the
                    # newest published run covering that page, preserving others.
                    from sqlalchemy import inspect
                    for cls,stmt in [(GraphicRun,select(GraphicRun).where(GraphicRun.id==run_id)),
                        (GraphicObject,select(GraphicObject).where(GraphicObject.run_id==run_id)),
                        (GraphicFragment,select(GraphicFragment).join(GraphicObject).where(GraphicObject.run_id==run_id)),
                        (GraphicCell,select(GraphicCell).join(GraphicFragment).join(GraphicObject).where(GraphicObject.run_id==run_id))]:
                        for row in s.scalars(stmt):
                            values={c.key:getattr(row,c.key) for c in inspect(cls).columns}
                            if cls is GraphicRun:values['published']=True;values['metrics']=verified
                            if cls in {GraphicCell,GraphicFragment} and values.get('crop_path'):
                                values['crop_path']=publish_asset(values['crop_path'],Path(production)/'graphics'/run_id)
                            if cls is GraphicObject and values['structure'].get('composite_path'):
                                values['structure']={**values['structure'],'composite_path':publish_asset(values['structure']['composite_path'],Path(production)/'graphics'/run_id)}
                            target.add(cls(**values))
                        target.flush()
                    # Keep review provenance with the promoted records.
                    from src.db.models import Revision
                    ids=select(GraphicObject.id).where(GraphicObject.run_id==run_id)
                    for cls,stmt in [(Revision,select(Revision).where(Revision.entity_type=='graphic_object',Revision.entity_id.in_(ids))),
                        (ReviewTask,select(ReviewTask).where(ReviewTask.target_type=='graphic_object',ReviewTask.target_id.in_(ids)))]:
                        for row in s.scalars(stmt):target.add(cls(**{c.key:getattr(row,c.key) for c in inspect(cls).columns}))
            finally:b.dispose()
            return {'published':run_id,'backup':str(backup)}
    finally:a.dispose()


def publish_asset(source,directory):
    import shutil
    source=Path(source);directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True)
    sha=file_hash(source);destination=directory/(sha+source.suffix)
    if not destination.exists():shutil.copy2(source,destination)
    if file_hash(destination)!=sha:raise ValueError('Published asset checksum mismatch')
    return str(destination)


def unpublish(production,run_id):
    """Deactivate a graphic overlay; retain assets and all source/review records."""
    from src.pipeline import _backup_database
    production=Path(production).resolve();backup=_backup_database(production)
    engine=init_db('sqlite:///'+(production/'library.db').as_posix())
    try:
        with get_session(engine) as s:
            run=s.get(GraphicRun,run_id)
            if run is None:raise LookupError('Graphic run not found')
            run.published=False
        return {'unpublished':run_id,'backup':str(backup),'source_records_deleted':False}
    finally:engine.dispose()
