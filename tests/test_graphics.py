"""Source-preserving graphic processing, adverse merge cases and publication gates."""
import io,json
import hashlib
from pathlib import Path
import cv2
import numpy as np
import pymupdf
import pytest
from sqlalchemy import select
from src.db.models import Canvas,Manifest
from src.db.session import init_db,get_session
from src.pipeline import run_document
from src.graphics.geometry import rotation_matrix,transform_points,box_polygon
from src.graphics.vision import recover_grid_cells,classify_ink,parse_structure,grid_regions,combine_regions
from src.graphics.grouping import continuation_evidence,stitch_pair
from src.graphics.models import GraphicObject,GraphicFragment,GraphicCell,GraphicRun
from src.graphics.pipeline import source_guard,clone_database,import_pages,promote
from src.graphics.service import object_detail,render_html,render_xlsx,review_graphic,undo_review,page_graphics
from src.graphics.evaluation import structure_teds,evaluate
from src.graphics.refinement import grounded_structure_review


def grid(width=400,height=300,rows=4,cols=4):
    im=np.full((height,width,3),255,np.uint8)
    for x in np.linspace(10,width-10,cols+1).astype(int):cv2.line(im,(x,10),(x,height-10),(0,0,0),2)
    for y in np.linspace(10,height-10,rows+1).astype(int):cv2.line(im,(10,y),(width-10,y),(0,0,0),2)
    return im


@pytest.mark.parametrize('angle',[0,90,180,270])
def test_transform_roundtrip(angle):
    points=box_polygon([10,15,80,50]);m=rotation_matrix(100,70,angle)
    assert np.allclose(transform_points(transform_points(points,m),np.linalg.inv(m)),points)


def test_grid_cells_symbols_and_two_tables():
    assert len(recover_grid_cells(grid()))==16
    triangle=np.full((60,60,3),255,np.uint8)
    cv2.polylines(triangle,[np.array([[30,12],[12,46],[48,46]])],True,(0,0,0),2)
    assert classify_ink(triangle)[1][0]['text']=='△'
    assert classify_ink(np.full((40,40),255,np.uint8))==('blank',[])
    page=np.full((800,500,3),255,np.uint8);page[50:350,50:450]=grid();page[450:750,50:450]=grid()
    supplemental=grid_regions(page)
    regions=combine_regions([{'bbox':[45,40,460,760],'label':'table','score':.9,'method':'neural_layout'}],supplemental)
    assert len(regions)==2


def test_broken_sheared_rules_keep_column_indices():
    image=grid(width=700,height=900,rows=30,cols=8)
    # Short gaps in each vertical rule defeat a long-component filter.
    for y in range(25,880,29):image[y:y+2,:]=255
    image=cv2.warpAffine(image,np.float32([[1,.008,0],[0,1,0]]),(720,900),borderValue=(255,255,255))
    cells=recover_grid_cells(image)
    assert len(cells)>100
    assert max(c['column']+c['column_span'] for c in cells)==8


def test_signature_uses_rules_instead_of_text_box_edges():
    from src.graphics.grouping import table_signature
    image=grid(width=700,height=180,rows=3,cols=3)
    cells=[{'row':r,'column':c,'row_span':1,'column_span':1,'bbox':[c*200+40,r*50+20,c*200+140,r*50+40],'text':'Header'}
        for r in range(3) for c in range(3)]
    sig=table_signature(cells,700,image,1)
    assert sig['edge_method']=='supported_ruled_boundaries'
    assert np.allclose(sig['edges'],[.01,.34,.66],atol=.015)


def test_html_spans_and_structure_teds():
    html='<table><tr><td rowspan="2">A</td><td>B</td></tr><tr><td>C</td></tr></table>'
    cells=parse_structure(html)
    assert [(c['row'],c['column'],c['row_span']) for c in cells]==[(0,0,2),(0,1,1),(1,1,1)]
    assert structure_teds(html,html)==1
    assert structure_teds(html,html.replace('rowspan="2"','rowspan="1"'))<1


def descriptor(label='附表四',page=1,source='a'):
    return {'manifest_id':source,'namespace':source,'category':'table','label':label,'page':page,'appendix':True,
        'section_id':'appendix','continuation':page>1,'signature':{'columns':3,'edges':[0,.3,.6],'headers':['探方','地层','型式']}}


def test_continuation_rejects_cross_source_and_same_headers_wrong_number():
    a=descriptor();b=descriptor(page=2)
    assert continuation_evidence(a,b)['accepted']
    assert not continuation_evidence(a,descriptor('附表五',2))['accepted']
    assert not continuation_evidence(a,descriptor(page=2,source='b'))['accepted']
    b['signature']['columns']=4
    assert not continuation_evidence(a,b)['accepted']


def test_physical_stitch_has_positive_cases_and_rejects_repeated_grid():
    rng=np.random.default_rng(1234)
    base=cv2.GaussianBlur(rng.integers(0,256,(400,900,3),dtype=np.uint8),(3,3),0)
    composite,evidence=stitch_pair(base[:,:600],base[:,300:])
    assert evidence['accepted'] and composite.shape[1]>=890
    _,table_evidence=stitch_pair(base[:,:600],base[:,300:],category='table')
    assert table_evidence['accepted']
    _,negative=stitch_pair(grid(),grid(),category='table')
    assert not negative['accepted']


@pytest.fixture
def staged(tmp_path):
    pdf=tmp_path/'source.pdf'
    with pymupdf.open() as document:
        for i in range(2):
            page=document.new_page(width=300,height=400);page.insert_text((30,40),f'Table {i+1} source record')
        document.save(pdf)
    production=tmp_path/'production';result=run_document(pdf,production,skip_models=True)
    mid=result['stats']['manifest_id'];guard=source_guard(production/'library.db',mid)
    stage=tmp_path/'stage';clone_database(production/'library.db',stage/'library.db')
    engine=init_db('sqlite:///'+(stage/'library.db').as_posix());cache=stage/'graphics-cache';cache.mkdir()
    with get_session(engine) as s:
        canvases=list(s.scalars(select(Canvas).where(Canvas.manifest_id==mid).order_by(Canvas.scan_index)))
        for n,canvas in enumerate(canvases,1):
            from PIL import Image
            from src.ingest.page_pipeline import file_hash
            with Image.open(canvas.image_storage_path) as im:size=list(im.size)
            crop=cache/f'crop-{n}.png';cv2.imencode('.png',grid())[1].tofile(crop)
            cells=[]
            for row in range(2):
                for col in range(3):
                    cells.append({'row':row,'column':col,'row_span':1,'column_span':1,'text':f'R{row}C{col}',
                        'bbox':[col*50,row*30,col*50+50,row*30+30],'polygon_source':box_polygon([col*50,row*30,col*50+50,row*30+30]),
                        'value_kind':'text','symbols':[],'header_path':['heading'],'crop_path':str(crop),'confidence':.99})
            region={'category':'table','label':'附表四','title':'Test table','key':'000','bbox':[10,10,300,200],'rotation':0,
                'crop_to_source':np.eye(3).tolist(),'source_hash':file_hash(Path(canvas.image_storage_path)),'crop_path':str(crop),
                'score':.99,'method':'fixture','cells':cells,'continuation':n>1,'appendix':True,
                'structure':{'header_rows':1,'signature':descriptor()['signature']}}
            (cache/f'{n:05d}.json').write_text(json.dumps({'signature':{'source_hash':region['source_hash']},'image_size':size,'regions':[region]}),encoding='utf8')
    run=import_pages(engine,mid,cache,[1,2],{'layout_model':'test'},guard)
    yield engine,production,stage,mid,run,guard
    engine.dispose()


def test_staging_grouping_exports_and_source_guard(staged):
    engine,production,stage,mid,run,guard=staged
    assert source_guard(production/'library.db',mid)==guard
    with get_session(engine) as s:
        obj=s.scalar(select(GraphicObject).where(GraphicObject.run_id==run,GraphicObject.status!='superseded'))
        detail=object_detail(s,obj.id)
        assert len(detail['fragments'])==2 and len(detail['cells'])==12
        assert len([c for c in detail['cells'] if c['repeated_header']])==3
        assert all(c['anchors'] for c in detail['cells'])
        assert b'<table>' in render_html(detail)
        from openpyxl import load_workbook
        workbook=load_workbook(io.BytesIO(render_xlsx(detail)))
        assert workbook['Table'].max_row==3 and workbook['Sources'].max_row==13
        assert len(page_graphics(s,detail['fragments'][0]['canvas_id']))==1
    annotation=stage/'annotations.json';annotation.write_text('{"pages":[]}',encoding='utf8')
    result=evaluate(stage,run,annotation)
    assert not result['passed'] and not result['annotations_complete']
    assert result['metrics_status']=='unannotated' and result['detection']['precision'] is None
    path=stage/'evaluation.json';path.write_text(json.dumps({**result,'passed':True}),encoding='utf8')
    with pytest.raises(ValueError):promote(stage,production,run,path)


def test_header_review_and_cell_edits_refresh_logical_records_and_undo(staged):
    engine,_,_,_,run,_=staged
    with get_session(engine) as session:
        obj=session.scalar(select(GraphicObject).where(GraphicObject.run_id==run,GraphicObject.status!='superseded'))
        before=object_detail(session,obj.id)
        cell=before['cells'][0]
        result=review_graphic(session,obj.id,expected_revision=1,actor='tester',note='核对原书第一列的表头名称',
                              action='correct_cell',cell_id=cell['id'],text='新表头')
        updated=object_detail(session,obj.id)
        assert updated['table']['columns'][0]['header_path']==['新表头']
        assert updated['cells'][3]['header_path']==['新表头']
        header_result=review_graphic(session,obj.id,expected_revision=2,actor='tester',note='原书该片段从数据开始没有表头',
                                     action='set_header_rows',target_id=before['fragments'][0]['id'],header_rows=0)
        assert object_detail(session,obj.id)['table']['header_rows']==0
        undo_review(session,obj.id,header_result['revision_id'],expected_revision=3,actor='tester',note='恢复原书片段的表头设置')
        restored=object_detail(session,obj.id)
        assert restored['table']['header_rows']==1
        assert restored['table']['columns'][0]['header_path']==['新表头']


def test_incremental_publication_preserves_edits_and_is_idempotent(staged):
    _,production,stage,mid,_,guard=staged
    engine=init_db('sqlite:///'+(production/'library.db').as_posix())
    try:
        options={'incremental':True,'published':True,'asset_root':production/'graphics'}
        run_id=import_pages(engine,mid,stage/'graphics-cache',[1],{'layout_model':'test'},guard,**options)
        with get_session(engine) as session:
            obj=session.scalar(select(GraphicObject).where(GraphicObject.run_id==run_id))
            original_id=obj.id
            detail=object_detail(session,obj.id)
            cell_id=detail['cells'][-1]['id']
            review_graphic(session,obj.id,expected_revision=1,actor='tester',note='核对原书并保留单元格校订',
                action='correct_cell',cell_id=cell_id,text='corrected')
        assert import_pages(engine,mid,stage/'graphics-cache',[1,2],{'layout_model':'test'},guard,**options)==run_id
        with get_session(engine) as session:
            detail=object_detail(session,original_id)
            assert len(detail['fragments'])==2 and len(detail['cells'])==12
            assert next(cell for cell in detail['cells'] if cell['id']==cell_id)['text']=='corrected'
            revision=detail['revision']
            run=session.get(GraphicRun,run_id)
            assert run.published and run.config['pages']==[1,2] and run.metrics['regions']==2
            assert all(Path(part.crop_path).is_relative_to(production/'graphics') for part in session.scalars(
                select(GraphicFragment).join(GraphicObject).where(GraphicObject.run_id==run_id)))
        import_pages(engine,mid,stage/'graphics-cache',[1,2],{'layout_model':'test'},guard,**options)
        with get_session(engine) as session:
            assert object_detail(session,original_id)['revision']==revision
    finally:engine.dispose()


def test_running_table_batch_deploys_with_progress_and_rollback(staged):
    from src.graphics.pipeline import prepare_request
    from src.table_jobs import list_table_jobs, rollback_table_batch, sync_table_batch
    from src.ingest.page_pipeline import atomic_json
    stage_engine,production,stage,mid,_,_=staged
    config={'layout_model':'test','manifest_id':mid}
    request=prepare_request(stage_engine,mid,[1,2],config)
    key=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:20]
    cache=stage/'graphics-cache'/key
    cache.mkdir()
    for page in (1,2):
        data=json.loads((stage/'graphics-cache'/f'{page:05d}.json').read_text(encoding='utf8'))
        data['signature']['config']=config
        atomic_json(cache/f'{page:05d}.json',data)
    atomic_json(stage/'request.json',request)
    atomic_json(stage/'progress.json',{'status':'complete','documents':[{'manifest_id':mid,'scanned_pages':2,
        'table_pages':[1,2],'table_regions':2}]})
    result=sync_table_batch(stage,production)
    assert result['status']=='complete' and result['deployed_pages']==2
    assert result['tables']==1 and result['records']==2
    assert list_table_jobs(production)[0]['id']==result['id']
    assert sync_table_batch(stage,production)['runs']==result['runs']
    rollback_table_batch(stage,production)
    assert sync_table_batch(stage,production)['status']=='paused'
    engine=init_db('sqlite:///'+(production/'library.db').as_posix())
    try:
        with get_session(engine) as session:
            assert not session.get(GraphicRun,result['runs'][mid]).published
    finally:engine.dispose()


def test_graphics_entrypoint_registers_deployment_and_retries_worker(staged,tmp_path,monkeypatch):
    from types import SimpleNamespace
    from src.graphics.pipeline import run_graphics
    _,production,stage,mid,_,guard=staged
    calls=[];children=[]
    def worker(command,**options):
        calls.append(command)
        if len(calls)==1:return SimpleNamespace(returncode=1)
        request=json.loads(Path(command[command.index('--request')+1]).read_text(encoding='utf8'))
        destination=Path(command[command.index('--output')+1])
        for page in request['pages']:
            data=json.loads((stage/'graphics-cache'/f"{page['page']:05d}.json").read_text(encoding='utf8'))
            data['signature']['config']=request['config']
            (destination/f"{page['page']:05d}.json").write_text(json.dumps(data),encoding='utf8')
        return SimpleNamespace(returncode=0)
    def deploy(command,**options):
        children.append(command)
        return SimpleNamespace(pid=123)
    monkeypatch.setattr('src.graphics.pipeline.subprocess.run',worker)
    monkeypatch.setattr('src.graphics.pipeline.subprocess.Popen',deploy)
    output=tmp_path/'new-graphics'
    result=run_graphics(production,mid,output,[1,2],with_vl=False,deploy_to=production)
    progress=json.loads((output/'progress.json').read_text(encoding='utf8'))
    request=json.loads((output/'request.json').read_text(encoding='utf8'))
    assert len(calls)==2 and len(children)==1
    assert progress['status']=='complete' and progress['retry_count']==1
    assert request['config']['source_revision']==guard
    assert result['deployment']['status']=='queued' and '--watch' in children[0]


def test_cache_config_rejected_and_published_asset_is_independent(staged,tmp_path):
    from src.graphics.pipeline import publish_asset,unpublish
    engine,production,stage,mid,run,guard=staged
    with pytest.raises(ValueError,match='cache configuration'):
        import_pages(engine,mid,stage/'graphics-cache',[1,2],{'layout_model':'test','implementation_hash':'new-version'},guard)
    with pytest.raises(ValueError,match='source revision'):
        import_pages(engine,mid,stage/'graphics-cache',[1,2],{'layout_model':'test'},'stale')
    source=tmp_path/'crop.png';source.write_bytes(b'original source crop')
    copied=Path(publish_asset(source,tmp_path/'published'));source.write_bytes(b'new local candidate')
    assert copied.read_bytes()==b'original source crop'
    with get_session(engine) as s:s.get(GraphicRun,run).published=True
    result=unpublish(stage,run)
    assert Path(result['backup']).is_file()
    with get_session(engine) as s:assert not s.get(GraphicRun,run).published
    assert source_guard(production/'library.db',mid)==guard


def test_promotion_copies_assets_and_audit_with_stubbed_acceptance(staged,monkeypatch):
    """Test publication mechanics in temp DBs; this is not real acceptance."""
    from src.db.models import Revision,ReviewTask
    engine,production,stage,mid,run,guard=staged
    with get_session(engine) as s:
        obj=s.scalar(select(GraphicObject).where(GraphicObject.run_id==run,GraphicObject.status!='superseded'))
        detail=object_detail(s,obj.id)
        review_graphic(s,obj.id,expected_revision=1,actor='tester',note='原图单元格来源核对测试',action='correct_cell',cell_id=detail['cells'][0]['id'],text='△')
        oid=obj.id
    report=stage/'test-only-acceptance.json'
    report.write_text(json.dumps({'run_id':run,'passed':True,'annotations_complete':True,'annotation_path':'test-only-stub'}),encoding='utf8')
    monkeypatch.setattr('src.graphics.evaluation.evaluate',lambda *a:{'passed':True,'annotations_complete':True})
    result=promote(stage,production,run,report)
    assert Path(result['backup']).is_file()
    target=init_db('sqlite:///'+(production/'library.db').as_posix())
    try:
        with get_session(target) as s:
            assert s.get(GraphicRun,run).published
            assert s.scalar(select(Revision).where(Revision.entity_id==oid)) is not None
            assert s.scalar(select(ReviewTask).where(ReviewTask.target_id==oid)) is not None
            fragment=s.scalar(select(GraphicFragment).where(GraphicFragment.object_id==oid))
            assert Path(fragment.crop_path).is_relative_to(production/'graphics')
            assert Path(fragment.crop_path).is_file()
    finally:target.dispose()
    assert source_guard(production/'library.db',mid)==guard


def test_review_cas_correct_cell_split_undo_and_cross_report_guard(staged):
    engine,production,stage,mid,run,guard=staged
    with get_session(engine) as s:
        obj=s.scalar(select(GraphicObject).where(GraphicObject.run_id==run,GraphicObject.status!='superseded'))
        detail=object_detail(s,obj.id);cell=detail['cells'][0]
        r=review_graphic(s,obj.id,expected_revision=1,actor='tester',note='核对原图更正表格文字',action='correct_cell',cell_id=cell['id'],text='=literal')
        assert s.get(GraphicCell,cell['id']).raw_text=='R0C0'
        assert s.get(GraphicCell,cell['id']).corrected_text=='=literal'
        undo_review(s,obj.id,r['revision_id'],expected_revision=2,actor='tester',note='核对后撤销刚才修订')
        assert s.get(GraphicCell,cell['id']).corrected_text is None
        split=review_graphic(s,obj.id,expected_revision=3,actor='tester',note='核对原图拆分跨页片段',action='split',target_id=detail['fragments'][1]['id'])
        assert len(object_detail(s,obj.id)['fragments'])==1
        undo_review(s,obj.id,split['revision_id'],expected_revision=4,actor='tester',note='恢复拆分前的跨页分组')
        assert len(object_detail(s,obj.id)['fragments'])==2
    with pytest.raises(ValueError,match='revision conflict'):
        with get_session(engine) as s:review_graphic(s,obj.id,expected_revision=1,actor='tester',note='过期版本不能覆盖修改',action='approve')


def test_api_structure_review_cannot_invent_values():
    class Router:
        model='deepseek-flash'
        def _cloud(self,*a,**k):return {'choices':[{'message':{'content':json.dumps({'issues':[{'cell_id':'0','kind':'unreadable','quote':'999'}]})}}]}
    with pytest.raises(ValueError,match='Ungrounded'):
        grounded_structure_review(Router(),[{'row':0,'column':0,'row_span':1,'column_span':1,'text':'123'}])


def test_circle_shape_cannot_overwrite_numeric_zero(tmp_path):
    from src.graphics.worker import cell_content
    class Models:
        def recognize(self,image,**kwargs):return {'blocks':[{'label':'table','text':'<table><tr><td>0</td></tr></table>'}]}
    image=np.full((60,60,3),255,np.uint8);cv2.circle(image,(30,30),16,(0,0,0),2)
    result=cell_content(Models(),image,[{'bbox':[0,0,60,60],'row':0,'column':0}],tmp_path)
    assert result[0]['text']=='0' and result[0]['value_kind']=='text'
    assert result[0]['symbols'][0]['status']=='candidate'


def test_graphic_http_details_exports_review_and_sources(staged):
    import threading,httpx
    from src.db.models import PlateFigure
    from PIL import Image
    from src.server import make_server
    engine,production,stage,mid,run,guard=staged
    with get_session(engine) as s:
        obj=s.scalar(select(GraphicObject).where(GraphicObject.run_id==run,GraphicObject.status!='superseded'))
        oid=obj.id
        canvas=s.scalar(select(Canvas).where(Canvas.manifest_id==mid,Canvas.scan_index==1))
        figure=PlateFigure(manifest_id=mid,canvas_id=canvas.id,figure_number='图1',
                           crop_x=20,crop_y=20,crop_w=150,crop_h=100)
        s.add(figure);s.flush();figure_id=figure.id
        sample_cell=s.scalar(select(GraphicCell).join(GraphicFragment).where(GraphicFragment.object_id==oid))
        sample_cell.evidence={**sample_cell.evidence,'ocr_engine':'PaddleOCR-VL-1.6-0.9B',
                              'confidence_available':False,'geometry_level':'table'}
        sample_cell_id=sample_cell.id
    server=make_server(stage,port=0,source_library=production);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        with httpx.Client(base_url=f'http://127.0.0.1:{server.server_port}') as client:
            detail=client.get('/api/graphics',params={'id':oid}).json()
            assert len(detail['fragments'])==2 and not detail['published']
            sample=next(cell for cell in detail['cells'] if cell['id']==sample_cell_id)
            assert not sample['confidence_available'] and sample['geometry_level']=='table'
            assert client.get(detail['fragments'][0]['image_url']).headers['content-type']=='image/png'
            assert client.get(detail['fragments'][0]['source_image_url']).status_code==200
            for crop,expected_size in [('',(150,100)),('20,20,60,40',(60,40))]:
                response=client.get('/api/figure_image',params={'id':figure_id,'xywh':crop})
                assert response.status_code==200
                assert Image.open(io.BytesIO(response.content)).size==expected_size
            for fmt in ['json','html','xlsx']:
                response=client.get('/api/graphic_export',params={'id':oid,'format':fmt})
                assert response.status_code==200 and len(response.content)>100
            cell=detail['cells'][0]
            body={'object_id':oid,'expected_revision':1,'actor':'tester','note':'对照原图核对编号修改','action':'correct_cell','cell_id':cell['id'],'text':'③'}
            assert client.post('/api/graphic_review',json=body).status_code==200
            assert client.post('/api/graphic_review',json=body).status_code==409
            assert client.get('/api/graphics',params={'id':'missing'}).status_code==404
    finally:server.shutdown();thread.join();server.server_close();server.engine.dispose()
    assert source_guard(production/'library.db',mid)==guard


def test_manual_group_undo_and_cross_run_rejection(staged):
    engine,_,_,mid,run,_=staged
    with get_session(engine) as s:
        obj=s.scalar(select(GraphicObject).where(GraphicObject.run_id==run,GraphicObject.status!='superseded'))
        part=object_detail(s,obj.id)['fragments'][1]
        review_graphic(s,obj.id,expected_revision=1,actor='tester',note='原图核对分开两个片段',action='split',target_id=part['id'])
        child=s.get(GraphicFragment,part['id']).object_id
        result=review_graphic(s,obj.id,expected_revision=2,actor='tester',note='确认相同报告的续表片段',action='group',target_id=child)
        assert len(object_detail(s,obj.id)['fragments'])==2
        undo_review(s,obj.id,result['revision_id'],expected_revision=3,actor='tester',note='撤销归组并恢复原片段')
        assert len(object_detail(s,obj.id)['fragments'])==1
        assert s.get(GraphicObject,child).status=='candidate'
        another=GraphicRun(manifest_id=mid,fingerprint='other',status='complete');s.add(another);s.flush()
        foreign=GraphicObject(manifest_id=mid,run_id=another.id,category='table');s.add(foreign);s.flush();foreign_id=foreign.id;oid=obj.id
    with pytest.raises(ValueError,match='across reports'):
        with get_session(engine) as s:
            review_graphic(s,oid,expected_revision=4,actor='tester',note='不同运行不能自动归组',action='group',target_id=foreign_id)


def test_empty_new_published_page_replaces_old_overlay(staged):
    from src.graphics.context import published_regions
    engine,_,_,mid,run,_=staged
    with get_session(engine) as s:
        old=s.get(GraphicRun,run);old.published=True;old.config={**old.config,'created_at':1}
        canvas=s.scalar(select(Canvas).where(Canvas.manifest_id==mid,Canvas.scan_index==1))
        assert published_regions(s,canvas.id)
        new=GraphicRun(manifest_id=mid,fingerprint='empty',status='complete',published=True,config={'pages':[1],'created_at':2})
        s.add(new);s.flush()
        assert published_regions(s,canvas.id)==[]
        assert page_graphics(s,canvas.id,False)==[]
        new.published=False;s.flush()
        assert published_regions(s,canvas.id)


def test_layout_first_native_text_routing_and_hash_guard(tmp_path):
    from src.ingest.page_pipeline import PageExtractor,file_hash
    from src.db.models import AltoBlock,AltoLine,RagChunk
    pdf=tmp_path/'native.pdf'
    with pymupdf.open() as doc:
        p=doc.new_page(width=300,height=400)
        p.insert_text((30,80),'T3 code 3 42');p.insert_text((30,250),'Ordinary body paragraph.')
        doc.save(pdf)
    extracted=tmp_path/'probe'
    with pymupdf.open(pdf) as doc:
        page=PageExtractor(extracted).extract(doc,0,file_hash(pdf))
    front=tmp_path/'front';front.mkdir()
    from PIL import Image
    with Image.open(page.image_path) as im:w,h=im.size
    record={'signature':{'source_hash':file_hash(Path(page.image_path))},'image_size':[w,h],
        'regions':[{'category':'table','key':'000','bbox':[0,0,w,h*.4]}]}
    (front/'00001.json').write_text(json.dumps(record),encoding='utf8')
    out=tmp_path/'library';result=run_document(pdf,out,skip_models=True,graphics_front_dir=front)
    engine=init_db('sqlite:///'+(out/'library.db').as_posix())
    try:
        with get_session(engine) as s:
            blocks=list(s.scalars(select(AltoBlock)))
            assert any(b.block_type=='table' and 'T3' in b.content_text for b in blocks)
            assert all('T3' not in c.content_text for c in s.scalars(select(RagChunk)))
    finally:engine.dispose()
    record['signature']['source_hash']='different-source';(front/'00001.json').write_text(json.dumps(record),encoding='utf8')
    with pytest.raises(ValueError,match='different rendered source'):
        PageExtractor(extracted,graphics_front_dir=front).attach_regions(page)


def test_table_circled_counts_excluded_while_footer_note_survives(monkeypatch):
    from src.ingest.digital_pdf_ingestor import ExtractedLine,ExtractedPageData
    from src.ingest.footnotes import detect
    monkeypatch.setattr('src.ingest.footnotes.separator_above',lambda *a:True)
    lines=[ExtractedLine('正文与其他标本相似①。',60,180,450,16,12,.99),
        ExtractedLine('③ 崇宁通宝①，数量②。',60,650,450,12,8,.99),
        ExtractedLine('① 作者：《考古报告》，1980年。',60,760,450,12,8,.99)]
    page=ExtractedPageData(1,0,'1',600,900,0,lines)
    plan=detect(page,excluded_indices={1})
    assert len(plan.notes)==1 and plan.notes[0].indices==[2]
    assert len(plan.references)==1 and plan.references[0].note_index==0


def test_independent_annotation_source_only_cas_and_bounds(staged):
    import threading,httpx
    from src.graphics.annotation import make_annotation_server
    from src.ingest.page_pipeline import file_hash
    engine,production,stage,mid,_,guard=staged
    with get_session(engine) as s:
        canvas=s.scalar(select(Canvas).where(Canvas.manifest_id==mid,Canvas.scan_index==1))
        page={'page':1,'image_sha256':file_hash(Path(canvas.image_storage_path)),'reviewed':False,'reviewer':'','objects':[],'split':'train','group_id':'one'}
    annotation=stage/'independent.json';annotation.write_text(json.dumps({'pages':[page],'relations':[]}),encoding='utf8')
    server=make_annotation_server(production,mid,annotation,port=0)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        with httpx.Client(base_url=f'http://127.0.0.1:{server.server_port}') as client:
            initial=client.get('/api/annotations').json()
            assert initial['data']['pages'][0]['objects']==[]
            assert client.get('/source?page=1').status_code==200
            body={'page':1,'version':initial['version'],'fields':{'reviewer':'tester','reviewed':True,'objects':[
                {'id':'p1table','category':'table','bbox':[0,0,20,20],'orientation':0}]}}
            assert client.post('/api/save',json=body).status_code==200
            assert client.post('/api/save',json=body).status_code==409
            body['version']=client.get('/api/annotations').json()['version'];body['fields']['objects'][0]['bbox']=[-1,0,20,20]
            assert client.post('/api/save',json=body).status_code==409
            assert annotation.with_suffix('.json.bak').is_file()
    finally:server.shutdown();thread.join();server.server_close();server.engine.dispose()
    assert source_guard(production/'library.db',mid)==guard
