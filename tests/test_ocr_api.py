"""OCR model replacement, API isolation, source geometry and cache regressions."""
import io
import json
from pathlib import Path
import threading

import httpx
from PIL import Image
import pymupdf
import pytest

from src.ingest.digital_pdf_ingestor import ExtractedLine
from src.ingest.page_pipeline import LocalOcr, PageExtractor, file_hash
from src.ocr_api import OCR_MODEL, PaddleOcrClient
from src.ocr_server import PaddleOcrService, make_ocr_server


def test_pipeline_service_exports_and_reuses_model_result(tmp_path):
    saved=[]
    class Result:
        json={'res':{'parsing_res_list':[{'block_label':'text','block_content':'原文测试','block_bbox':[0,0,10,8]}]}}
        markdown={'markdown_texts':'原文测试'}
        def save_to_json(self,save_path):saved.append(('json',save_path))
        def save_to_markdown(self,save_path):
            saved.append(('markdown',save_path))
            asset=Path(save_path)/'imgs/figure.png';asset.parent.mkdir(parents=True,exist_ok=True)
            Image.new('RGB',(10,8),'white').save(asset)
    class Pipeline:
        calls=[]
        def predict(self,path,**options):
            self.calls.append(options)
            return [Result()]
    pipeline=Pipeline()
    service=PaddleOcrService(pipeline,tmp_path/'server',{'model':OCR_MODEL,'pipeline_version':'v1.6'})
    server=make_ocr_server(service,port=0)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    output=io.BytesIO();Image.new('RGB',(20,16),'white').save(output,'PNG')
    client=PaddleOcrClient({'OCR_BASE_URL':f'http://127.0.0.1:{server.server_port}/v1'})
    try:
        for _ in range(2):
            result=client.recognize(output.getvalue(),task='text',output_dir=tmp_path/'exports')
            assert result['blocks'][0]['text']=='原文测试'
        assert len(pipeline.calls)==1 and pipeline.calls[0]['prompt_label']=='ocr'
        assert [kind for kind,_ in saved]==['json','markdown']
        assert (tmp_path/'exports/document.md').read_text(encoding='utf-8')=='原文测试'
        assert (tmp_path/'exports/imgs/figure.png').is_file()
        assert json.loads((tmp_path/'exports/result.json').read_text(encoding='utf-8'))['image_size']==[20,16]
        with httpx.Client(base_url=f'http://127.0.0.1:{server.server_port}',trust_env=False) as transport:
            assert transport.get('/v1/models').json()['data'][0]['id']==OCR_MODEL
            assert transport.post('/v1/chat/completions',json={'model':OCR_MODEL,'messages':[]}).status_code==400
            assert transport.post('/v1/chat/completions',json={'model':'unknown'}).status_code==404
    finally:
        server.shutdown();thread.join();server.server_close()


def test_block_geometry_is_preserved_without_inventing_line_boxes(tmp_path,monkeypatch):
    result={'blocks':[{'label':'text','text':'第一行\n第二行','bbox':[20,40,120,100]},
                      {'label':'table','text':'<table><tr><td>T1∶1</td><td>5</td></tr></table>','bbox':[20,110,120,150]}]}
    monkeypatch.setattr(PaddleOcrClient,'recognize',lambda *args,**kwargs:result)
    ocr=LocalOcr(settings={})
    lines=ocr.recognize(tmp_path/'image.png',2)
    assert len(lines)==2 and lines[0].text=='第一行\n第二行'
    assert (lines[0].x_pt,lines[0].y_top_pt,lines[0].w_pt,lines[0].h_pt)==(10,20,50,30)
    assert lines[0].confidence is None
    assert lines[1].text=='T1∶1\t5' and ocr.regions[0]['bbox_pt']==[10,55,60,75]


def test_spotting_exports_native_text_coordinates_and_per_call_limits(tmp_path):
    class Result:
        json={'res':{'parsing_res_list':[], 'spotting_res':{'rec_texts':['T1'],
            'rec_polys':[[[2,1],[12,1],[12,9],[2,9]]]}}}
        markdown={'markdown_texts':'T1'}
        def save_to_json(self,save_path):pass
        def save_to_markdown(self,save_path):pass
    class Pipeline:
        options=None
        def predict(self,path,**options):self.options=options;return [Result()]
    pipeline=Pipeline();service=PaddleOcrService(pipeline,tmp_path/'server',{'model':OCR_MODEL,'pipeline_version':'v1.6'})
    server=make_ocr_server(service,port=0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    data=io.BytesIO();Image.new('RGB',(20,16),'white').save(data,'PNG')
    client=PaddleOcrClient({'OCR_BASE_URL':f'http://127.0.0.1:{server.server_port}/v1'})
    try:
        result=client.recognize(data.getvalue(),task='spotting',max_tokens=512)
        assert result['geometry_level']=='line'
        assert result['blocks']==[{'label':'textline','text':'T1','bbox':[2.,1.,12.,9.]}]
        assert pipeline.options['prompt_label']=='spotting' and pipeline.options['max_new_tokens']==512
        assert client.max_tokens==8192
        with pytest.raises(ValueError,match='token limit'):client.recognize(data.getvalue(),task='spotting',max_tokens=0)
    finally:server.shutdown();thread.join();server.server_close()


def test_ocr_api_rejects_unbounded_geometry_and_hides_credentials():
    client=PaddleOcrClient({'OCR_API_KEY':'test-secret'})
    assert 'test-secret' not in json.dumps(client.identity)
    result={'model':OCR_MODEL,'image_size':[20,16],'markdown':'text',
            'blocks':[{'label':'text','text':'text','bbox':[0,0,21,10]}]}
    with pytest.raises(ValueError,match='outside'):
        client.validate(result)


def test_model_settings_invalidate_scanned_page_cache(tmp_path,monkeypatch):
    output=io.BytesIO();Image.new('RGB',(100,160),'black').save(output,'PNG')
    source=tmp_path/'scan.pdf'
    with pymupdf.open() as document:
        page=document.new_page(width=100,height=160)
        page.insert_image(page.rect,stream=output.getvalue());document.save(source)
    calls=[]
    def recognize(self,*args):
        calls.append(True);self.regions=[]
        return [ExtractedLine('测试',10,10,20,10,8)]
    monkeypatch.setattr(LocalOcr,'recognize',recognize)
    with pymupdf.open(source) as document:
        extractor=PageExtractor(tmp_path/'run',settings={})
        assert extractor.extract(document,0,file_hash(source)).method=='PaddleOCR-VL-v1.6'
        extractor.extract(document,0,file_hash(source))
        assert len(calls)==1
        changed=PageExtractor(tmp_path/'run',settings={'OCR_MAX_PIXELS':'262144'})
        changed.extract(document,0,file_hash(source))
        assert len(calls)==2


def test_table_mismatch_keeps_whole_region_anchors(tmp_path):
    import numpy as np
    from src.graphics.worker import cell_content
    class Models:
        def recognize(self,*args,**kwargs):
            return {'blocks':[{'label':'table','text':'<table><tr><td>甲</td><td>乙</td></tr></table>'}]}
    image=np.full((60,100,3),255,dtype=np.uint8)
    cells=cell_content(Models(),image,[{'row':0,'column':0,'bbox':[0,0,50,60]}],tmp_path)
    assert [cell['text'] for cell in cells]==['甲','乙']
    assert all(cell['geometry_level']=='table' and cell['bbox']==[0,0,100,60] for cell in cells)
