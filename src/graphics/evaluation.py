"""Acceptance is computed from independently reviewed annotations, never predictions."""
from __future__ import annotations
import json
from pathlib import Path
from sqlalchemy import select
from src.db.session import init_db,get_session
from src.db.models import Canvas
from src.graphics.models import GraphicRun,GraphicObject,GraphicFragment
from src.graphics.geometry import iou
from src.graphics.service import object_detail,render_html


def structure_teds(predicted,reference):
    from apted import APTED,Config
    from lxml import html
    def tree(value):
        root=html.fromstring(value)
        tables=root.xpath('//table') if root.tag!='table' else [root]
        if not tables:raise ValueError('Table HTML required for TEDS')
        def node(el):
            children=[node(c) for c in el if isinstance(c.tag,str) and c.tag in {'thead','tbody','tfoot','tr','td','th'}]
            if el.tag in {'thead','tbody','tfoot'}:return ('group',1,1,children)
            return (el.tag,int(el.get('rowspan',1)),int(el.get('colspan',1)),children)
        t=node(tables[0])
        def flatten(n):
            children=[]
            for c in n[3]:
                f=flatten(c)
                children.extend(f[3] if f[0]=='group' else [f])
            return (*n[:3],children)
        return flatten(t)
    class Structure(Config):
        def rename(self,a,b):return int(a[:3]!=b[:3])
        def children(self,n):return n[3]
    a,b=tree(predicted),tree(reference)
    count=lambda n:1+sum(count(c) for c in n[3])
    return max(0.,1-APTED(a,b,Structure()).compute_edit_distance()/max(count(a),count(b)))


def evaluate(library,run_id,annotation_path):
    annotations=json.loads(Path(annotation_path).read_text(encoding='utf8'))
    engine=init_db('sqlite:///'+(Path(library).resolve()/'library.db').as_posix())
    try:
        with get_session(engine) as session:
            run=session.get(GraphicRun,run_id)
            if run is None:raise LookupError('Run not found')
            pages=annotations.get('pages',[])
            complete=len(pages)==60 and len({p['page'] for p in pages})==60 and sum(p.get('split')=='train' for p in pages)==40 and sum(p.get('split')=='test' for p in pages)==20
            complete=complete and set(run.config.get('pages',[]))=={p['page'] for p in pages}
            complete=complete and all(p.get('reviewed') and p.get('reviewer') and p.get('image_sha256') for p in pages)
            complete=complete and not any(p.get('augmentation',False) for p in pages)
            object_splits={}
            for p in pages:
                for obj in p.get('objects',[]):
                    if obj['id'] in object_splits:complete=False
                    object_splits[obj['id']]=p['split']
            groups={}
            for p in pages:groups.setdefault(p.get('group_id',str(p['page'])),set()).add(p.get('split'))
            if any(len(splits)>1 for splits in groups.values()):complete=False
            from src.ingest.page_pipeline import file_hash
            image_hashes={}
            def image_hash(canvas):
                if canvas.id not in image_hashes:image_hashes[canvas.id]=file_hash(Path(canvas.image_storage_path))
                return image_hashes[canvas.id]
            known={};predicted={};tp=fp=fn=0;teds=[];source_errors=[];orientation=[];checks={};scored_test_pages=0
            fragments=list(session.scalars(select(GraphicFragment).join(GraphicObject).where(GraphicObject.run_id==run_id,GraphicObject.status!='superseded')))
            by_page={}
            for f in fragments:
                canvas=session.get(Canvas,f.canvas_id);by_page.setdefault(canvas.scan_index,[]).append(f)
            for page in pages:
                canvas=session.scalar(select(Canvas).where(Canvas.manifest_id==run.manifest_id,Canvas.scan_index==page['page']))
                if canvas is None or image_hash(canvas)!=page.get('image_sha256'):complete=False
                selected=by_page.get(page['page'],[])
                if selected and any(f.source_hash!=page.get('image_sha256') for f in selected):complete=False
                # Unreviewed empty lists are unknown labels, never negatives.
                if not page.get('reviewed') or not page.get('reviewer'):continue
                if page.get('split')=='test':scored_test_pages+=1
                ground=page.get('objects',[]);pairs=[]
                for i,g in enumerate(ground):
                    for j,f in enumerate(selected):
                        x,y,w,h=f.bbox;score=iou(g['bbox'],[x,y,x+w,y+h])
                        obj=session.get(GraphicObject,f.object_id)
                        if score>=.5 and g['category']==obj.category:pairs.append((score,i,j))
                used_g=set();used_p=set()
                for score,i,j in sorted(pairs,reverse=True):
                    if i in used_g or j in used_p:continue
                    used_g.add(i);used_p.add(j);g,f=ground[i],selected[j]
                    known[g['id']]=f.object_id;predicted[g['id']]=f.id
                    if page.get('split')!='test':continue
                    tp+=1
                    if 'orientation' in g:orientation.append(f.rotation==g['orientation'])
                    if g.get('html'):
                        detail=object_detail(session,f.object_id)
                        # Evaluate a physical fragment, not the whole continued table.
                        detail['cells']=[{**c,'row':c['source_row'],'repeated_header':False} for c in detail['cells'] if c['fragment_id']==f.id]
                        detail['structure']={**detail['structure'],'table_structure_pending':False}
                        teds.append(structure_teds(render_html(detail).decode('utf8'),g['html']))
                    if g.get('cell_checks'):
                        detail=object_detail(session,f.object_id)
                        physical={(c['source_row'],c['column']):c for c in detail['cells'] if c['fragment_id']==f.id}
                        for check in g['cell_checks']:
                            actual=physical.get((check['row'],check['column']),{}).get('text','')
                            normalize=lambda s:''.join(s.split())
                            role=check['kind']
                            bucket=checks.setdefault(role,{'correct':0,'total':0})
                            bucket['total']+=1;bucket['correct']+=int(normalize(actual)==normalize(check['text']))
                if page.get('split')=='test':fp+=len(selected)-len(used_p);fn+=len(ground)-len(used_g)
            positive=negative=correct=false_merges=0
            for relation in annotations.get('relations',[]):
                if relation.get('split')!='test':continue
                a,b=relation['a'],relation['b'];merged=a in known and b in known and known[a]==known[b]
                if object_splits.get(a)!='test' or object_splits.get(b)!='test':
                    complete=False;continue
                if relation['kind']=='continuation':positive+=1;correct+=int(merged)
                elif relation['kind']=='negative':negative+=1;false_merges+=int(merged)
            # Each fragment/cell has an explicit source anchor in the right report.
            anchors=0;cell_count=0
            for object_id in set(f.object_id for f in fragments):
                detail=object_detail(session,object_id)
                if detail['structure'].get('composite_path'):
                    mapping=detail['structure'].get('fragment_to_composite',{})
                    if not Path(detail['structure']['composite_path']).is_file() or set(mapping)!={p['id'] for p in detail['fragments']}:
                        source_errors.append(object_id)
                for part in detail['fragments']:
                    canvas=session.get(Canvas,part['canvas_id'])
                    if (not part['polygon'] or not part['transform'] or not Path(asset_file(session,part['id'])).is_file()
                        or canvas.manifest_id!=run.manifest_id or part['source_hash']!=image_hash(canvas)):
                        source_errors.append(part['id'])
                for cell in detail['cells']:
                    cell_count+=1
                    valid=bool(cell['anchors']) and all(a.get('polygon') and a.get('source_hash') for a in cell['anchors'])
                    for anchor in cell['anchors']:
                        canvas=session.get(Canvas,anchor['canvas_id'])
                        if (canvas is None or canvas.manifest_id!=run.manifest_id
                            or anchor.get('source_hash')!=image_hash(canvas)
                            or any(x<-.5 or y<-.5 or x>canvas.width+.5 or y>canvas.height+.5 for x,y in anchor.get('polygon',[]))):valid=False
                    if valid:anchors+=1
                    else:source_errors.append(cell['id'])
            precision=tp/max(1,tp+fp);recall=tp/max(1,tp+fn);mean_teds=sum(teds)/max(1,len(teds))
            gates={'annotations':bool(complete),'detection_precision':precision>=.9,'detection_recall':recall>=.95,
                'table_structure':len(teds)>=5 and mean_teds>=.9,
                'cell_metrics_reported':all(checks.get(kind,{}).get('total',0)>0 for kind in ['identifier','unit','symbol']),
                'source_coverage':not source_errors and cell_count>0,
                'logical_grouping':positive>=10 and negative>=20 and false_merges==0 and correct/max(1,positive)>=.9,
                'stitching':bool(annotations.get('stitch_suite',{}).get('verified_report')),
                'direction':len(orientation)>=10 and sum(orientation)/len(orientation)>=.98}
            # A path to a report alone is not accepted as proof of physical stitching.
            if gates['stitching']:
                report=Path(annotations['stitch_suite']['verified_report'])
                if not report.is_file():gates['stitching']=False
                else:
                    s=json.loads(report.read_text(encoding='utf8'))
                    gates['stitching']=(s.get('dataset_type')=='real_scan' and bool(s.get('reviewer'))
                        and s.get('positive_cases',0)>=5 and s.get('negative_cases',0)>=20 and s.get('false_merges',1)==0 and s.get('recall',0)>=.8)
            return {'run_id':run_id,'annotation_path':str(Path(annotation_path).resolve()),'annotations_complete':bool(complete),
                'passed':all(gates.values()),'gates':gates,'scored_test_pages':scored_test_pages,
                'metrics_status':'unannotated' if not scored_test_pages else 'complete' if scored_test_pages==20 else 'partial',
                'detection':{'tp':tp,'fp':fp,'fn':fn,'precision':precision if scored_test_pages else None,'recall':recall if scored_test_pages else None},
                'TEDS_structure':mean_teds if teds else None,'tables_scored':len(teds),'cell_source_coverage':anchors/max(1,cell_count),
                'grouping':{'positive_cases':positive,'negative_cases':negative,'recall':correct/max(1,positive),'false_merges':false_merges},
                'cell_accuracy':{kind:{**v,'accuracy':v['correct']/v['total']} for kind,v in checks.items()},
                'source_errors':source_errors}
    finally:engine.dispose()


def asset_file(session,fragment_id):
    return session.get(GraphicFragment,fragment_id).crop_path
