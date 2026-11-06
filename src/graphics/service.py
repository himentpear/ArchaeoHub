"""Graphic queries, exports and CAS reviews; source OCR is never overwritten."""
from __future__ import annotations
from html import escape
import io
import json
from pathlib import Path
from sqlalchemy import select,update
from src.db.models import Canvas,Manifest,Revision
from src.graphics.models import GraphicRun,GraphicObject,GraphicFragment,GraphicCell


def object_detail(session,object_id):
    obj=session.get(GraphicObject,object_id)
    if obj is None:raise LookupError('Graphic object not found')
    run=session.get(GraphicRun,obj.run_id)
    fragments=list(session.scalars(select(GraphicFragment).where(GraphicFragment.object_id==obj.id).order_by(GraphicFragment.part_index,GraphicFragment.id)))
    cells=[];parts=[];offset=0
    for part in fragments:
        canvas=session.get(Canvas,part.canvas_id)
        physical=list(session.scalars(select(GraphicCell).where(GraphicCell.fragment_id==part.id).order_by(GraphicCell.row_index,GraphicCell.column_index)))
        skip=part.evidence.get('header_rows',0) if parts and part.evidence.get('fold_repeated_header') else 0
        for cell in physical:
            hidden=cell.row_index<skip
            cells.append({'id':cell.id,'fragment_id':part.id,'row':cell.row_index-skip+offset,'source_row':cell.row_index,
                'column':cell.column_index,'row_span':cell.row_span,'column_span':cell.column_span,
                'text':cell.corrected_text if cell.corrected_text is not None else cell.raw_text,'raw_text':cell.raw_text,
                'value_kind':cell.value_kind,'symbols':cell.symbols,'header_path':cell.header_path,
                'anchors':cell.anchors,'confidence':cell.confidence,'repeated_header':hidden,
                'crop_url':f'/api/graphic_asset?cell={cell.id}'})
        count=max((c.row_index+c.row_span for c in physical),default=0)
        offset+=max(0,count-skip)
        parts.append({'id':part.id,'part_index':part.part_index,'canvas_id':part.canvas_id,'scan_index':canvas.scan_index,'printed_page':canvas.printed_page_label,
            'bbox':part.bbox,'polygon':part.polygon,'rotation':part.rotation,'transform':part.transform,
            'source_hash':part.source_hash,'evidence':part.evidence,'confidence':part.confidence,
            'image_url':f'/api/graphic_asset?fragment={part.id}',
            'source_image_url':f'/iiif/image/{obj.manifest_id}/{canvas.scan_index}/full/max/0/default.jpg'})
    # Explicitly reviewed cell joins preserve every physical source cell.
    by_id={c['id']:c for c in cells}
    for joined in obj.structure.get('cell_joins',[]):
        chosen=[by_id[i] for i in joined if i in by_id]
        if len(chosen)<2:continue
        chosen[0]['text']='\n'.join(c['text'] for c in chosen)
        chosen[0]['anchors']=[a for c in chosen for a in c['anchors']]
        chosen[0]['source_cells']=[c['id'] for c in chosen]
        for c in chosen[1:]:c['joined_into']=chosen[0]['id']
    revisions=[{'id':r.id,'revision':r.revision_no,'actor':r.actor,'reason':r.reason} for r in session.scalars(select(Revision).where(
        Revision.entity_type=='graphic_object',Revision.entity_id==obj.id).order_by(Revision.revision_no.desc()).limit(10))]
    return {'id':obj.id,'run_id':obj.run_id,'manifest_id':obj.manifest_id,'category':obj.category,'label':obj.label,'title':obj.title,
        'status':obj.status,'revision':obj.revision_no,'published':run.published,'evidence':obj.evidence,'structure':obj.structure,
        'fragments':parts,'cells':cells,'revisions':revisions,'composite_url':f'/api/graphic_asset?object={obj.id}' if obj.structure.get('composite_path') else None}


def page_graphics(session,canvas_id,include_candidates=True):
    from src.graphics.context import latest_page_run
    newest=latest_page_run(session,canvas_id,not include_candidates)
    stmt=select(GraphicObject).join(GraphicFragment,GraphicFragment.object_id==GraphicObject.id).join(GraphicRun,GraphicRun.id==GraphicObject.run_id).where(
        GraphicFragment.canvas_id==canvas_id,GraphicObject.status!='superseded')
    if not include_candidates:stmt=stmt.where(GraphicRun.published.is_(True))
    if newest:stmt=stmt.where(GraphicRun.id==newest)
    objects=list(session.scalars(stmt).unique())
    if not objects:return []
    newest=max((session.get(GraphicRun,o.run_id) for o in objects),key=lambda r:r.config.get('created_at',0)).id
    return [{'id':o.id,'category':o.category,'label':o.label,'title':o.title,'status':o.status,'revision':o.revision_no,
        'published':session.get(GraphicRun,o.run_id).published,'url':f'/api/graphics?id={o.id}'} for o in objects if o.run_id==newest]


def render_html(detail):
    if detail.get('structure',{}).get('table_structure_pending'):
        output=[b'<!doctype html><meta charset="utf-8"><p>Physical registration candidate: table structures remain separate pending review.</p>']
        for fragment in detail['fragments']:
            part={**detail,'title':detail['title']+' / page '+str(fragment['scan_index']),
                'structure':{},'cells':[{**c,'row':c['source_row'],'repeated_header':False} for c in detail['cells'] if c['fragment_id']==fragment['id']]}
            output.append(render_html(part))
        return b''.join(output)
    cells=[c for c in detail['cells'] if not c['repeated_header'] and not c.get('joined_into')]
    rows={}
    for c in cells:rows.setdefault(c['row'],[]).append(c)
    result=['<!doctype html><meta charset="utf-8"><title>'+escape(detail['title'] or detail['label'])+'</title>',
        '<style>table{border-collapse:collapse}td{border:1px solid #777;padding:5px;white-space:pre-wrap}caption{padding:1em}</style>',
        '<table><caption>'+escape(detail['label']+' '+detail['title'])+'</caption>']
    for row in sorted(rows):
        result.append('<tr>')
        for c in sorted(rows[row],key=lambda c:c['column']):
            result.append(f'<td rowspan="{c["row_span"]}" colspan="{c["column_span"]}" data-cell-id="{c["id"]}">'+escape(c['text'])+'</td>')
        result.append('</tr>')
    result.append('</table>');return ''.join(result).encode('utf8')


def render_xlsx(detail):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment
    workbook=Workbook();sheet=workbook.active;sheet.title='Table';sources=workbook.create_sheet('Sources')
    issues=workbook.create_sheet('StructureIssues');issues.append(['cell','issue'])
    if detail.get('structure',{}).get('table_structure_pending'):
        issues.append(['ALL','Physical registration candidate only; Table stacks fragment candidates. Sources retains physical row and column.'])
    occupied=set()
    sources.append(['cell','page','source_region','text','value_kind','review_status','row','column','row_span','column_span'])
    for cell in detail['cells']:
        for anchor in cell['anchors']:
            sources.append([cell['id'],anchor.get('scan_index'),json.dumps(anchor.get('polygon')),cell['text'],cell['value_kind'],detail['status'],
                cell['source_row'],cell['column'],cell['row_span'],cell['column_span']])
        if cell['repeated_header'] or cell.get('joined_into'):continue
        r,c=cell['row']+1,cell['column']+1
        slots={(rr,cc) for rr in range(r,r+cell['row_span']) for cc in range(c,c+cell['column_span'])}
        if occupied & slots:
            issues.append([cell['id'],'Overlapping candidate span; full text and source retained in Sources'])
            continue
        occupied.update(slots)
        target=sheet.cell(r,c);target.value=cell['text'];target.data_type='s';target.alignment=Alignment(wrap_text=True,vertical='top')
        if cell['row_span']>1 or cell['column_span']>1:
            sheet.merge_cells(start_row=r,start_column=c,end_row=r+cell['row_span']-1,end_column=c+cell['column_span']-1)
    for row in sources:
        for cell in row:
            if isinstance(cell.value,str):cell.data_type='s'
    output=io.BytesIO();workbook.save(output);return output.getvalue()


def asset_path(session,*,fragment=None,cell=None,object=None):
    if sum(bool(v) for v in [fragment,cell,object])!=1:raise ValueError('One asset identifier required')
    if fragment:
        item=session.get(GraphicFragment,fragment);value=item.crop_path if item else None
    elif cell:
        item=session.get(GraphicCell,cell);value=item.crop_path if item else None
    else:
        item=session.get(GraphicObject,object);value=item.structure.get('composite_path') if item else None
    if not value or not Path(value).is_file():raise LookupError('Graphic asset not found')
    return Path(value)


def review_graphic(session,object_id,*,expected_revision,actor,note,action,cell_id=None,text=None,target_id=None,cell_ids=None,category=None):
    if not actor.strip() or len(note.strip())<6:raise ValueError('Reviewer and source evidence required')
    obj=session.get(GraphicObject,object_id)
    if obj is None:raise LookupError('Graphic object not found')
    before=object_detail(session,obj.id)
    claimed=session.execute(update(GraphicObject).where(GraphicObject.id==obj.id,GraphicObject.revision_no==expected_revision)
        .values(revision_no=expected_revision+1).execution_options(synchronize_session=False))
    if claimed.rowcount!=1:raise ValueError('Graphic revision conflict')
    if action=='correct_cell':
        cell=session.get(GraphicCell,cell_id)
        fragment=session.get(GraphicFragment,cell.fragment_id) if cell else None
        if fragment is None or fragment.object_id!=obj.id or text is None or len(text)>20000:raise ValueError('Cell must belong to object')
        cell.corrected_text=text;cell.value_kind='text' if text else 'blank'
    elif action=='classify':
        if category not in {'table','chart','figure'}:raise ValueError('Invalid graphic category')
        obj.category=category
    elif action=='approve':obj.status='reviewed'
    elif action=='reject':obj.status='rejected'
    elif action=='group':
        target=session.get(GraphicObject,target_id)
        if target is None or target.manifest_id!=obj.manifest_id or target.run_id!=obj.run_id or target.id==obj.id:
            raise ValueError('Cannot group across reports or processing runs')
        before['related']=[object_detail(session,target.id)]
        # Grouping locks both objects; source version is included in audit.
        lock=session.execute(update(GraphicObject).where(GraphicObject.id==target.id,GraphicObject.revision_no==target.revision_no)
            .values(revision_no=target.revision_no+1).execution_options(synchronize_session=False))
        if lock.rowcount!=1:raise ValueError('Target graphic revision conflict')
        fragments=list(session.scalars(select(GraphicFragment).where(GraphicFragment.object_id.in_([obj.id,target.id]))))
        fragments.sort(key=lambda f:(session.get(Canvas,f.canvas_id).scan_index,f.bbox[1],f.bbox[0]))
        obj.evidence={**obj.evidence,'review_group':{'source_objects':[obj.id,target.id]}}
        for i,f in enumerate(fragments):f.object_id=obj.id;f.part_index=i
        target.status='superseded';target.revision_no+=1
        target.evidence={**target.evidence,'merged_into':obj.id}
    elif action=='split':
        part=session.get(GraphicFragment,target_id)
        if part is None or part.object_id!=obj.id:raise ValueError('Fragment must belong to object')
        child=GraphicObject(run_id=obj.run_id,manifest_id=obj.manifest_id,category=obj.category,label=obj.label,title=obj.title,
            evidence={'split_from':obj.id,'review_note':note})
        session.add(child);session.flush();part.object_id=child.id;part.part_index=0
        obj.structure={**obj.structure,'split_child':child.id}
    elif action=='join_cells':
        chosen=[session.get(GraphicCell,i) for i in (cell_ids or [])]
        if len(chosen)<2 or any(c is None or session.get(GraphicFragment,c.fragment_id).object_id!=obj.id for c in chosen):
            raise ValueError('At least two source cells belonging to object required')
        if len({c.column_index for c in chosen})!=1 or len({c.fragment_id for c in chosen})!=len(chosen):
            raise ValueError('Only cells in the same column from distinct fragments may be joined')
        obj.structure={**obj.structure,'cell_joins':obj.structure.get('cell_joins',[])+[[c.id for c in chosen]]}
    else:raise ValueError('Unknown graphic review action')
    obj.revision_no=expected_revision+1;session.flush()
    after=object_detail(session,obj.id)
    if action=='group':after['related']=[object_detail(session,target.id)]
    revision=Revision(entity_type='graphic_object',entity_id=obj.id,revision_no=obj.revision_no,
        actor=actor,reason=note,before_state=before,after_state=after)
    session.add(revision);session.flush()
    return {'id':obj.id,'revision':obj.revision_no,'status':obj.status,'revision_id':revision.id}


def undo_review(session,object_id,revision_id,*,expected_revision,actor,note):
    revision=session.get(Revision,revision_id);obj=session.get(GraphicObject,object_id)
    if revision is None or revision.entity_id!=object_id or revision.entity_type!='graphic_object':raise ValueError('Wrong revision')
    if obj is None or obj.revision_no!=expected_revision:raise ValueError('Graphic revision conflict')
    if revision.revision_no!=expected_revision:raise ValueError('Only the latest unchanged review can be undone')
    if not actor.strip() or len(note.strip())<6:raise ValueError('Reviewer and reason required')
    before=object_detail(session,obj.id);snapshot=revision.before_state
    for related in snapshot.get('related',[]):
        owner=session.get(GraphicObject,related['id'])
        after=next((r for r in revision.after_state.get('related',[]) if r['id']==owner.id),None)
        if after is None or owner.revision_no!=after['revision']:raise ValueError('Related object changed after this review')
        for part in related['fragments']:
            session.get(GraphicFragment,part['id']).object_id=owner.id
        owner.status=related['status'];owner.evidence=related['evidence'];owner.structure=related['structure'];owner.revision_no+=1
        for part in related['fragments']:
            session.get(GraphicFragment,part['id']).part_index=part.get('part_index',related['fragments'].index(part))
    child_id=obj.structure.get('split_child')
    if child_id and child_id!=snapshot['structure'].get('split_child'):
        child=session.get(GraphicObject,child_id)
        if child.revision_no!=1:raise ValueError('Split object was subsequently reviewed')
        child.status='superseded'
    claimed=session.execute(update(GraphicObject).where(GraphicObject.id==obj.id,GraphicObject.revision_no==expected_revision)
        .values(revision_no=expected_revision+1).execution_options(synchronize_session=False))
    if claimed.rowcount!=1:raise ValueError('Graphic revision conflict')
    for index,part in enumerate(snapshot['fragments']):
        fragment=session.get(GraphicFragment,part['id'])
        if fragment is None:raise ValueError('Source fragment no longer exists')
        if fragment.object_id!=obj.id:
            owner=session.get(GraphicObject,fragment.object_id)
            if owner and owner.status=='reviewed':raise ValueError('Fragment was subsequently reviewed')
        fragment.object_id=obj.id;fragment.part_index=part.get('part_index',index)
    for c in snapshot['cells']:
        cell=session.get(GraphicCell,c['id'])
        if cell:
            cell.corrected_text=c['text'] if c['text']!=cell.raw_text else None
            cell.value_kind=c['value_kind']
    obj.category=snapshot['category'];obj.status=snapshot['status'];obj.structure=snapshot['structure'];obj.evidence=snapshot['evidence']
    obj.revision_no=expected_revision+1;session.flush()
    session.add(Revision(entity_type='graphic_object',entity_id=obj.id,revision_no=obj.revision_no,actor=actor,reason=note,
        before_state=before,after_state=object_detail(session,obj.id)))
    return {'id':obj.id,'revision':obj.revision_no}
