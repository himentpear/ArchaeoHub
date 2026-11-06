"""Python 3.11 CPU worker. Neural layout precedes every region OCR operation."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
import re
from pathlib import Path
import sys
import time
import threading

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ.setdefault("PADDLE_PDX_CACHE_HOME",str(ROOT/"data/processed/models/graphics-weights"))
os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE","bos")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK","True")
os.environ.setdefault("OMP_NUM_THREADS","4")
os.environ.setdefault("OPENBLAS_NUM_THREADS","4")
os.environ.setdefault("MKL_NUM_THREADS","4")
os.environ.setdefault("FLAGS_use_mkldnn","0")
import cv2
import numpy as np
from src.graphics.geometry import rotation_matrix, transform_points, box_polygon
from src.graphics.vision import (grid_regions,combine_regions,recover_grid_cells,parse_structure,classify_ink,header_rows)
from src.graphics.grouping import caption_for,table_signature
from src.io_utils import atomic_write


def save_json(path,value):
    atomic_write(Path(path),(json.dumps(value,ensure_ascii=False,indent=2)+"\n").encode("utf8"))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def result_json(result):
    value=result.json
    if isinstance(value,str):value=json.loads(value)
    return value.get("res",value)


class Models:
    def __init__(self,config):
        self.config=config;self.loaded={};self.timings={}
        cv2.setNumThreads(config.get('threads',4))
    def get(self,name):
        if name not in self.loaded:
            from paddleocr import LayoutDetection,DocImgOrientationClassification,TableStructureRecognition,TableCellsDetection,TableClassification
            cls=(LayoutDetection if "DocLayout" in name else DocImgOrientationClassification if "doc_ori" in name
                else TableCellsDetection if "cell_det" in name else TableClassification if "table_cls" in name
                else TableStructureRecognition)
            started=time.perf_counter()
            self.loaded[name]=cls(model_name=name,device="cpu",cpu_threads=self.config.get('threads',4),
                enable_mkldnn=False,enable_hpi=False)
            self.timings[name]=time.perf_counter()-started
        return self.loaded[name]
    def predict(self,name,image,**kwargs):
        return result_json(self.get(name).predict(image,**kwargs)[0])
    def ocr(self):
        if 'rapidocr' not in self.loaded:
            from rapidocr import RapidOCR
            self.loaded['rapidocr']=RapidOCR(params={'EngineConfig.onnxruntime.intra_op_num_threads':4,
                'EngineConfig.onnxruntime.inter_op_num_threads':1,'Rec.rec_batch_num':12})
        return self.loaded['rapidocr']


def orientation(models,image):
    value=models.predict('PP-LCNet_x1_0_doc_ori',image)
    labels=value.get('label_names',[]);scores=value.get('scores',[])
    return (int(labels[0]) if labels else 0,float(scores[0]) if scores else 0.,value)


def rotated(image,degrees):
    return np.ascontiguousarray(np.rot90(image,-degrees//90)) if degrees else image


def choose_table_orientation(models,crop,prior):
    scores=[]
    for angle in [0,90,180,270]:
        candidate=rotated(crop,angle)
        strip=candidate[:min(candidate.shape[0],max(160,int(candidate.shape[0]*.18))) ]
        factor=min(1.,1400/max(strip.shape[:2]));strip=cv2.resize(strip,None,fx=factor,fy=factor)
        result=models.ocr()(strip,use_det=True,use_cls=False,use_rec=True)
        texts=list(result.txts or []);conf=list(result.scores or [])
        score=0.;unit_score=0.
        for text,p in zip(texts,conf):
            if p<.65:continue
            chinese=len(re.findall(r'[\u3400-\u9fff]',text))
            units=len(re.findall(r'[THYMF]\d+',text))
            terms=len(re.findall(r'探方|地层|型式|编号|尺寸|面积|统计|名称|数量|深度|遗迹|相邻',text))
            score+=float(p)*(min(chinese,12)*.3+terms*3)
            unit_score+=float(p)*units*2
        score+=min(12.,unit_score)
        scores.append({'degrees':angle,'score':round(score,4),'texts':texts})
    scores.sort(key=lambda x:x['score'],reverse=True)
    clear=scores[0]['score']>=scores[1]['score']*1.15+1
    return (scores[0]['degrees'] if clear else prior),scores,not clear


def region_pixels(request,image,box,category):
    x0,y0,x1,y1=box
    crop=image[y0:y1,x0:x1]
    transform=np.array([[1.,0,x0],[0,1.,y0],[0,0,1.]])
    if category!='table' or not request.get('source_pdf'):return crop,transform,{'source':'original_page_pixels'}
    import pymupdf
    with pymupdf.open(request['source_pdf']) as doc:
        page=doc[request['pdf_page_index']]
        sx=page.rect.width/image.shape[1];sy=page.rect.height/image.shape[0]
        rect=pymupdf.Rect(x0*sx,y0*sy,x1*sx,y1*sy)
        zoom=min(300/72,4096/max(rect.width,rect.height))
        pix=page.get_pixmap(matrix=pymupdf.Matrix(zoom,zoom),clip=rect,alpha=False)
        array=np.frombuffer(pix.samples,dtype=np.uint8).reshape(pix.height,pix.width,pix.n)
        crop=cv2.cvtColor(array,cv2.COLOR_RGB2BGR)
        # Pixmap origins are integer raster coordinates; include the raster
        # offset rather than assuming the requested clip starts on a pixel.
        transform=np.array([[1/(zoom*sx),0,pix.x/(zoom*sx)],[0,1/(zoom*sy),pix.y/(zoom*sy)],[0,0,1.]])
    return crop,transform,{'source':'pdf_region','render_dpi':zoom*72,'max_side':4096}


def deskew(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    lines=cv2.HoughLinesP(cv2.Canny(gray,60,160),1,np.pi/1800,threshold=100,minLineLength=int(image.shape[1]*.4),maxLineGap=20)
    angles=[]
    for x0,y0,x1,y1 in np.asarray(lines).reshape(-1,4) if lines is not None else []:
        angle=np.degrees(np.arctan2(y1-y0,x1-x0))
        if abs(angle)<=3:angles.append(angle)
    angle=float(np.median(angles)) if angles else 0.
    matrix=np.eye(3)
    if abs(angle)>=.08:
        h,w=image.shape[:2];matrix[:2,:]=cv2.getRotationMatrix2D((w/2,h/2),angle,1)
        image=cv2.warpAffine(image,matrix[:2],(w,h),borderValue=(255,255,255))
    return image,matrix,angle


def structural_parse(models,image,complex_review=True):
    raw=models.predict('SLANet_plus',image)
    neural=parse_structure(raw.get('structure',[]),raw.get('bbox'))
    neural=[c for c in neural if c.get('bbox') and c['bbox'][2]>c['bbox'][0] and c['bbox'][3]>c['bbox'][1]]
    grid=recover_grid_cells(image)
    review={"SLANet_plus":raw,"grid_cells":len(grid),"neural_cells":len(neural)}
    # Grid boxes offer strong source anchors on wired tables, but disagreements
    # remain visible; neither output is mislabeled as verified ground truth.
    cells=grid if len(grid)>=max(4,len(neural)*.8) else neural
    disagreement=abs(len(grid)-len(neural))/max(1,len(grid),len(neural))
    if complex_review and (disagreement>.15 or len(cells)>160 or any(c.get('row_span',1)>2 for c in cells)):
        cls=models.predict('PP-LCNet_x1_0_table_cls',image)
        wireless=len(grid)<4 and 'wireless' in str(cls.get('label_names',[])).lower()
        suffix='wireless' if wireless else 'wired'
        nxt=models.predict('SLANeXt_'+suffix,image)
        detection=models.predict('RT-DETR-L_'+suffix+'_table_cell_det',image)
        review.update(table_classification=cls,complex_structure=nxt,cell_detection=detection)
        if not cells:
            boxes=[r['coordinate'] for r in detection.get('boxes',[])]
            candidate=parse_structure(nxt.get('structure',[]),boxes)
            cells=[c for c in candidate if c.get('bbox')]
    review['needs_review']=bool(disagreement>.15 or not cells)
    return cells,review


def text_line_crops(crop):
    """Cell boundaries are already known: project ink rows before neural recognition.

    This avoids running a full-page text detector separately for thousands of
    tiny cells. Ambiguous nonempty regions still use the neural detector.
    """
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    margin=max(2,min(gray.shape)//14)
    if min(gray.shape)<=margin*2+3:return []
    inside=gray[margin:-margin,margin:-margin]
    mask=(inside<170).astype(np.uint8)
    profile=np.count_nonzero(mask,axis=1)
    active=np.where(profile>=max(1,int(inside.shape[1]*.002)))[0]
    groups=[]
    for y in active:
        if groups and y-groups[-1][-1]<=3:groups[-1].append(int(y))
        else:groups.append([int(y)])
    result=[]
    for group in groups:
        a=max(0,group[0]-2);b=min(inside.shape[0],group[-1]+3)
        if b-a<4:continue
        columns=np.where(np.count_nonzero(mask[a:b],axis=0)>0)[0]
        if not len(columns):continue
        x0=max(0,int(columns[0])-3);x1=min(inside.shape[1],int(columns[-1])+4)
        result.append((crop[a+margin:b+margin,x0+margin:x1+margin],[x0+margin,a+margin,x1+margin,b+margin]))
    return result


def cell_content(models,image,cells,crop_dir):
    """Detect within each cell; recognize detected line crops in one local batch."""
    from rapidocr.ch_ppocr_rec.typings import TextRecInput
    engine=models.ocr();recognizer=engine._load_rec_model()
    tasks=[];owners=[]
    for index,cell in enumerate(cells):
        x0,y0,x1,y1=[int(round(v)) for v in cell['bbox']]
        x0=max(0,x0);y0=max(0,y0);x1=min(image.shape[1],x1);y1=min(image.shape[0],y1)
        cell['bbox']=[x0,y0,x1,y1]
        if x1<=x0 or y1<=y0:
            cell.update(text='',value_kind='unreadable',symbols=[],confidence=0);continue
        crop=image[y0:y1,x0:x1]
        path=crop_dir/f'cell-{index:05d}.png';cv2.imencode('.png',crop)[1].tofile(path)
        kind,symbols=classify_ink(crop)
        cell.update(text='',value_kind=kind,symbols=symbols,confidence=0.,crop_path=str(path.resolve()),line_results=[])
        if kind=='blank':cell['confidence']=1.;continue
        if symbols and len(symbols)==1 and symbols[0]['text']=='△' and (x1-x0)<(y1-y0)*2 and (y1-y0)<(x1-x0)*2:
            cell.update(text=symbols[0]['text'],value_kind='symbol',confidence=symbols[0]['confidence']);continue
        # Detection inside the cell prevents crossing rows or columns. Keeping
        # this separate from recognition permits batch inference of tiny crops.
        projected=text_line_crops(crop)
        if projected:
            for pixel,box in projected:tasks.append(pixel);owners.append((index,box))
            cell['line_segmentation']='cell_ink_projection'
            continue
        out=engine(crop,use_rec=False,use_cls=False,use_det=True)
        boxes=getattr(out,'boxes',None)
        if boxes is None or not len(boxes):
            tasks.append(crop);owners.append((index,[0,0,x1-x0,y1-y0]));continue
        for box in boxes:
            p=np.asarray(box);a,b=np.maximum(0,np.floor(p.min(axis=0)).astype(int));c,d=np.ceil(p.max(axis=0)).astype(int)
            c=min(c,crop.shape[1]);d=min(d,crop.shape[0])
            if c>a and d>b:tasks.append(crop[b:d,a:c]);owners.append((index,[int(a),int(b),int(c),int(d)]))
    if tasks:
        output=recognizer(TextRecInput(img=tasks))
        for (owner,box),text,score in zip(owners,output.txts,output.scores):
            cells[owner]['line_results'].append({'text':text,'confidence':float(score),'bbox':box})
    for cell in cells:
        lines=cell.get('line_results',[])
        if not lines:continue
        # True vertical headers: top-to-bottom within columns, columns right-to-left.
        vertical=(cell['bbox'][3]-cell['bbox'][1])>1.8*(cell['bbox'][2]-cell['bbox'][0]) and all(len(l['text'])<=2 for l in lines)
        lines.sort(key=lambda l:(-round(l['bbox'][0]/15),l['bbox'][1]) if vertical else (round(l['bbox'][1]/8),l['bbox'][0]))
        cell['text']=('' if vertical else '\n').join(l['text'] for l in lines)
        cell['confidence']=sum(l['confidence'] for l in lines)/len(lines)
        cell['value_kind']='text' if cell['text'].strip() else 'unreadable'
        cell['reading_direction']='vertical' if vertical else 'horizontal'
        for symbol in cell.get('symbols',[]):symbol['status']='candidate' if symbol['text']!=cell['text'] else 'supported_by_ocr'
    return cells


def process_page(models,request,out_dir):
    started=time.perf_counter();source=Path(request['image_path']);sha=digest(source)
    signature={'source_hash':sha,'config':models.config,'version':'graphics-1'}
    cache=out_dir/f"{request['page']:05d}.json"
    if cache.exists():
        value=json.loads(cache.read_text(encoding='utf8'))
        if value.get('signature')==signature:
            value['cache_hit']=True;return value
    image=cv2.imdecode(np.fromfile(source,dtype=np.uint8),cv2.IMREAD_COLOR)
    # Always run layout before region text recognition, including native PDFs.
    layout=models.predict(models.config.get('layout_model','PP-DocLayout_plus-L'),image,threshold=.3)
    regions=combine_regions([{'bbox':r['coordinate'],'label':r['label'],'score':float(r['score']),'method':'neural_layout'}
        for r in layout.get('boxes',[])],grid_regions(image))
    if models.config.get('layout_only'):
        value={'signature':signature,'page':request['page'],'image_path':str(source.resolve()),'image_size':[image.shape[1],image.shape[0]],
            'layout':layout,'regions':regions,'mode':'layout_only','elapsed_seconds':time.perf_counter()-started,'cache_hit':False}
        save_json(cache,value);return value
    page_angle,page_confidence,page_orientation=orientation(models,image)
    if not request.get('lines'):
        # New documents have no legacy OCR yet. Read only detected title regions
        # AFTER layout, so caption discovery never depends on whole-page OCR.
        request={**request,'lines':[]}
        for box in layout.get('boxes',[]):
            if not any(v in box['label'].lower() for v in ['title','caption']):continue
            x0,y0,x1,y1=[int(v) for v in box['coordinate']]
            title=models.ocr()(image[max(0,y0):y1,max(0,x0):x1],use_det=True,use_rec=True,use_cls=False)
            if title.txts:request['lines'].append({'text':' '.join(title.txts),'bbox':[x0,y0,x1,y1]})
    results=[]
    for index,region in enumerate(regions):
        x0,y0,x1,y1=[int(round(v)) for v in region['bbox']]
        x0=max(0,x0);y0=max(0,y0);x1=min(image.shape[1],x1);y1=min(image.shape[0],y1)
        if x1<=x0 or y1<=y0:continue
        crop,unrotated_to_source,render_info=region_pixels(request,image,[x0,y0,x1,y1],region['category'])
        caption=caption_for([x0,y0,x1,y1],request.get('lines',[]),image.shape[0])
        directory=out_dir/f"{request['page']:05d}-{index:02d}";directory.mkdir(exist_ok=True)
        region_angle,confidence,ori=orientation(models,crop)
        angle=(-region_angle)%360
        if confidence<.85:angle=(-page_angle)%360
        rotation_candidates=[];orientation_ambiguous=False
        if region['category']=='table':
            angle,rotation_candidates,orientation_ambiguous=choose_table_orientation(models,crop,angle)
        upright=rotated(crop,angle)
        upright,deskew_matrix,skew_angle=deskew(upright) if region['category']=='table' else (upright,np.eye(3),0.)
        to_page=unrotated_to_source@np.linalg.inv(rotation_matrix(crop.shape[1],crop.shape[0],angle))@np.linalg.inv(deskew_matrix)
        crop_path=directory/'region.png';cv2.imencode('.png',upright)[1].tofile(crop_path)
        cells=[];structure={}
        if region['category']=='table':
            cells,structure=structural_parse(models,upright,models.config.get('complex_review',True))
            if not cells:
                attempts=[]
                for candidate in [0,90,180,270]:
                    candidate_image,candidate_matrix,candidate_skew=deskew(rotated(crop,candidate))
                    cc,rr=structural_parse(models,candidate_image,False)
                    attempts.append((len(cc)*float(rr['SLANet_plus'].get('structure_score',0)),candidate,cc,rr,candidate_image,candidate_matrix,candidate_skew))
                attempts.sort(key=lambda a:a[0],reverse=True)
                selected=next((a for a in attempts if a[1]==angle and a[2]),attempts[0])
                _,angle,cells,structure,upright,deskew_matrix,skew_angle=selected
                structure['rotation_candidates']=[{'degrees':a[1],'score':a[0]} for a in attempts]
                structure['orientation_ambiguous']=len(attempts)>1 and attempts[0][0]<attempts[1][0]*1.1
                to_page=unrotated_to_source@np.linalg.inv(rotation_matrix(crop.shape[1],crop.shape[0],angle))@np.linalg.inv(deskew_matrix)
                cv2.imencode('.png',upright)[1].tofile(crop_path)
            cells=cell_content(models,upright,cells,directory)
            for cell in cells:cell['polygon_source']=transform_points(box_polygon(cell['bbox']),to_page)
            hr=header_rows(cells)
            for cell in cells:
                cell['header_path']=[c['text'] for c in cells if c['row']<hr and c.get('text')
                    and c['column']<=cell['column']<c['column']+c['column_span']]
            structure.update(header_rows=hr,signature=table_signature(cells,upright.shape[1],upright,hr),
                rotation_text_candidates=rotation_candidates,orientation_ambiguous=orientation_ambiguous)
        else:
            text=models.ocr()(upright,use_det=True,use_cls=True,use_rec=True)
            structure['labels']=[{'text':t,'confidence':float(s),'polygon_source':transform_points(b,to_page)}
                for t,s,b in zip(text.txts or [],text.scores or [],text.boxes if text.boxes is not None else [])]
        margin=max(3,min(image.shape[:2])*.005)
        touches=[side for side,yes in [('left',x0<=margin),('top',y0<=margin),
            ('right',x1>=image.shape[1]-margin),('bottom',y1>=image.shape[0]-margin)] if yes]
        completeness={'status':'possible_truncation' if touches else 'unknown',
            'page_edges':touches,'requires_review':bool(touches),'synthesized_pixels':False}
        results.append({**region,**caption,'bbox':[x0,y0,x1,y1],'rotation':angle,'crop_to_source':to_page.tolist(),
            'crop_path':str(crop_path.resolve()),'cells':cells,'structure':structure,'orientation':ori,
            'source_hash':sha,'key':f'{index:03d}','render':render_info,'deskew_degrees':skew_angle,'completeness':completeness})
    value={'signature':signature,'page':request['page'],'image_path':str(source.resolve()),'image_size':[image.shape[1],image.shape[0]],
        'layout':layout,'page_orientation':page_orientation,'regions':results,'elapsed_seconds':time.perf_counter()-started,
        'cache_hit':False}
    save_json(cache,value);return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();request=json.loads(args.request.read_text(encoding='utf-8-sig'));args.output.mkdir(parents=True,exist_ok=True)
    config=request.get('config',{});models=Models(config);timings=[]
    import psutil
    process=psutil.Process();peak=[process.memory_info().rss];done=threading.Event()
    def sample_memory():
        while not done.wait(.25):peak[0]=max(peak[0],process.memory_info().rss)
    monitor=threading.Thread(target=sample_memory,daemon=True);monitor.start()
    cold=time.perf_counter()
    for item in request['pages']:
        wall=time.perf_counter()
        try:value=process_page(models,item,args.output)
        except Exception as error:
            save_json(args.output/'failure.json',{'page':item['page'],'error_type':type(error).__name__,'retry':'rerun the same command; completed page caches are retained'})
            raise
        timings.append({'page':item['page'],'seconds':value['elapsed_seconds'],'wall_seconds':time.perf_counter()-wall,'cache_hit':value['cache_hit']})
        print(json.dumps({'stage':'graphics','page':item['page'],'regions':len(value['regions']),'seconds':value['elapsed_seconds'],'cache_hit':value['cache_hit']}),flush=True)
    weights=Path(os.environ['PADDLE_PDX_CACHE_HOME'])/'official_models'
    files={str(p.relative_to(weights)):digest(p) for p in weights.rglob('*') if p.is_file() and p.suffix in {'.json','.pdiparams','.yml','.yaml'}}
    done.set();monitor.join(timeout=1)
    save_json(args.output/'runtime.json',{'packages':{p:importlib.metadata.version(p) for p in ['paddleocr','paddlepaddle','paddlex','rapidocr','onnxruntime']},
        'weights_sha256':files,'model_load_seconds':models.timings,'page_timings':timings,'rss_bytes':process.memory_info().rss,
        'peak_rss_bytes':peak[0],'total_wall_seconds':time.perf_counter()-cold,
        'first_page_wall_seconds':timings[0]['wall_seconds'] if timings else None})


if __name__=='__main__':main()
