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
from src.graphics.grouping import caption_for,table_signature,data_table_regions,box_coverage,boxes_coverage
from src.graphics.tables import table_row_bands
from src.io_utils import atomic_write
from src.ocr_api import PaddleOcrClient, OCR_MODEL,save_ocr_result
from src.settings import load_settings


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
    def __init__(self,config,settings=None):
        self.config=config;self.loaded={};self.timings={}
        self.ocr_client=PaddleOcrClient(settings)
        if config.get('ocr') and config['ocr']!=self.ocr_client.identity:
            raise ValueError('OCR configuration changed; prepare the graphic request again')
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
    def recognize(self,image,*,task='ocr',output_dir=None):
        data=cv2.imencode('.png',image)[1].tobytes()
        return self.ocr_client.recognize(data,task=task,output_dir=output_dir)


def orientation(models,image):
    value=models.predict('PP-LCNet_x1_0_doc_ori',image)
    labels=value.get('label_names',[]);scores=value.get('scores',[])
    return (int(labels[0]) if labels else 0,float(scores[0]) if scores else 0.,value)


def rotated(image,degrees):
    return np.ascontiguousarray(np.rot90(image,-degrees//90)) if degrees else image


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


def assess_table_regions(models,image,layout,regions,lines,source_hash,page):
    kept,excluded=data_table_regions(regions,layout,lines,image.shape[0])
    candidates=[region for region in regions if region['category']=='table']
    if not candidates:return kept,excluded,{}
    primary=models.config.get('layout_model','PP-DocLayoutV3')
    secondary=models.config.get('secondary_layout_model') or ('PP-DocLayout_plus-L' if primary=='PP-DocLayoutV3' else 'PP-DocLayoutV3')
    second=None
    cache=models.config.get('secondary_layout_cache')
    if cache and (Path(cache)/f'{page:05d}.json').is_file():
        value=json.loads((Path(cache)/f'{page:05d}.json').read_text(encoding='utf8'))
        signature=value.get('signature',{})
        if signature.get('source_hash')==source_hash:
            if signature.get('config',{}).get('layout_model')==secondary:second=value['layout']
            elif value.get('table_assessment',{}).get('model')==secondary:second=value['table_assessment']['layout']
    if second is None:second=models.predict(secondary,image,threshold=.3)
    table_boxes=[box['coordinate'] for box in second.get('boxes',[]) if box['label'].lower()=='table']
    image_boxes=[box['coordinate'] for box in second.get('boxes',[]) if box['label'].lower() in {'image','figure','chart'}]
    excluded_boxes={tuple(region['bbox']):region for region in excluded}
    accepted=[]
    rejected=[]
    for region in candidates:
        caption=caption_for(region['bbox'],lines,image.shape[0])
        table_support=max((box_coverage(region['bbox'],box) for box in table_boxes),default=0)
        image_support=boxes_coverage(region['bbox'],image_boxes)
        first_rejected=excluded_boxes.get(tuple(region['bbox']))
        evidence={'primary_model':primary,'secondary_model':secondary,'secondary_table_coverage':round(table_support,4),
                  'secondary_image_coverage':round(image_support,4)}
        if '表' in caption['label']:
            accepted.append({**region,'classification':{**evidence,'decision':'table_caption'}})
        elif (first_rejected or caption['label'].startswith('图')) and image_support>=.8 and table_support<.5:
            rejected.append({**(first_rejected or region),'reason':'grid_in_illustration','caption':caption,
                             'classification':{**evidence,'decision':'illustration_model_agreement'}})
        elif table_support>=.6 and not first_rejected:
            accepted.append({**region,'classification':{**evidence,'decision':'table_model_agreement'}})
        else:
            accepted.append({**region,'requires_type_check':True,'classification':{**evidence,'decision':'native_ocr_type_check'}})
    return accepted,rejected,{'model':secondary,'layout':second}


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


def symbol_matrix_cells(models,image,cells,crop_dir):
    """Read sparse ruled matrices at measured source positions."""
    if len(cells)<64 or not all(cell.get('method')=='grid' for cell in cells):return None
    columns=max(cell['column']+cell['column_span'] for cell in cells)
    if columns<8:return None
    count=header_rows(cells)
    observations={}
    for index,cell in enumerate(cells):
        if cell['row']<count:continue
        x0,y0,x1,y1=[int(round(value)) for value in cell['bbox']]
        observations[index]=classify_ink(image[y0:y1,x0:x1])
    triangles=sum(any(symbol['text']=='△' for symbol in symbols) for _,symbols in observations.values())
    supported=sum(kind=='blank' or any(symbol['text']=='△' for symbol in symbols) for kind,symbols in observations.values())
    if triangles<3 or supported/max(1,len(observations))<.9:return None
    headers=[cell for cell in cells if cell['row']<count]
    directory=crop_dir/'matrix-headers';directory.mkdir(parents=True,exist_ok=True)
    bottom=min(image.shape[0],max(int(cell['bbox'][3]) for cell in headers)+4)
    decoded=cell_content(models,image[:bottom],headers,directory)
    key=lambda cell:(cell['row'],cell['column'],cell['row_span'],cell['column_span'])
    recognized={key(cell):cell for cell in decoded if cell['geometry_level'] in {'cell','grid_axes'}}
    for row in sorted({cell['row'] for cell in headers}):
        physical=sorted((cell for cell in headers if cell['row']==row),key=lambda cell:cell['column'])
        observed=sorted((cell for cell in decoded if cell['row']==row),key=lambda cell:cell['column'])
        # A diagonal corner decoded one column too narrow shifts every label.
        # Match a row only when its full ordered span sequence is preserved.
        spans=lambda values:[(cell['row_span'],cell['column_span']) for cell in values]
        if spans(physical)==spans(observed):
            for source,candidate in zip(physical,observed):
                recognized.setdefault(key(source),{**candidate,'header_alignment':'physical_row_order'})
    result=[]
    for index,cell in enumerate(cells):
        x0,y0,x1,y1=[int(round(value)) for value in cell['bbox']]
        crop=image[y0:y1,x0:x1]
        path=crop_dir/f'cell-{index:05d}.png';cv2.imencode('.png',crop)[1].tofile(path)
        kind,symbols=observations.get(index,('unreadable',[]))
        candidate=recognized.get(key(cell)) if cell['row']<count else None
        ocr_result=None;ocr_candidates=[];engine=None
        if candidate is not None:
            text=candidate['text'];engine=OCR_MODEL
            method='matrix_header_row_alignment' if candidate.get('header_alignment') else 'matrix_header_ocr'
            ocr_result=candidate.get('ocr_result');kind='text' if text else kind
        elif cell['row']>=count and kind=='blank':
            text='';method='grid_blank_observation'
        elif cell['row']>=count and any(symbol['text']=='△' for symbol in symbols):
            text='△';kind='symbol';method='source_symbol_observation'
        else:
            local=models.recognize(crop,task='text',output_dir=crop_dir/f'cell-ocr-{index:05d}')
            text='\n'.join(block['text'].strip() for block in local['blocks'] if block['text'].strip())
            ocr_candidates=[{'scope':'cell_crop','text':text}]
            if re.fullmatch(r'\s*\$?\s*\\triangle\s*\$?\s*',text):text='△'
            kind='text' if text else kind;method='matrix_cell_ocr';engine=OCR_MODEL
            ocr_result=str(crop_dir/f'cell-ocr-{index:05d}'/'result.json')
        result.append({**cell,'text':text,'value_kind':kind,'symbols':[{**symbol,'status':'candidate'} for symbol in symbols],
                       'confidence':0.,'confidence_available':False,'crop_path':str(path.resolve()),'line_results':[],
                       'method':method,'ocr_engine':engine,'geometry_level':cell.get('geometry_level','cell'),'requires_review':True,
                       'ocr_result':ocr_result,'ocr_candidates':ocr_candidates,'source_matrix':True})
    save_json(crop_dir/'source-observations.json',{'header_rows':count,'columns':columns,'triangles':triangles,
        'cells':[{'row':cell['row'],'column':cell['column'],'method':cell['method'],'text':cell['text']} for cell in result]})
    return result


def cell_content(models,image,cells,crop_dir,recognized_result=None):
    """Read a complete table with VL, retaining source-supported cell geometry."""
    matrix=symbol_matrix_cells(models,image,cells,crop_dir)
    if matrix is not None:return matrix
    bands=table_row_bands(cells)
    if len(bands)>1:
        combined=[];row_offset=0
        for index,(start,stop) in enumerate(bands):
            selected=[cell for cell in cells if start<=cell['row']<stop]
            y0=max(0,min(int(cell['bbox'][1]) for cell in selected)-4)
            y1=min(image.shape[0],max(int(cell['bbox'][3]) for cell in selected)+4)
            local=[{**cell,'row':cell['row']-start,'bbox':[cell['bbox'][0],cell['bbox'][1]-y0,cell['bbox'][2],cell['bbox'][3]-y0]} for cell in selected]
            directory=crop_dir/f'band-{index:03d}';directory.mkdir(parents=True,exist_ok=True)
            decoded=cell_content(models,image[y0:y1],local,directory)
            count=max((cell['row']+cell['row_span'] for cell in decoded),default=0)
            for cell in decoded:
                cell['row']+=row_offset
                cell['bbox'][1]+=y0;cell['bbox'][3]+=y0
                cell['source_band']={'start_row':start,'end_row':stop,'bbox':[0,y0,image.shape[1],y1],
                                     'row_count_matches':count==stop-start}
            combined.extend(decoded);row_offset+=count
        return combined
    result=recognized_result or models.recognize(image,task='table',output_dir=crop_dir/'paddleocr-vl')
    if recognized_result:
        save_ocr_result(result,crop_dir/'paddleocr-vl')
    tables=[block['text'] for block in result['blocks'] if block['label']=='table']
    if len(tables)!=1 or '</table>' not in tables[0].lower():
        raise ValueError('PaddleOCR-VL returned an incomplete table; increase OCR_MAX_TOKENS and retry')
    recognized=parse_structure(tables[0])
    if not recognized:
        raise ValueError('PaddleOCR-VL returned no table cells')
    def key(cell):
        return (cell['row'],cell['column'],cell.get('row_span',1),cell.get('column_span',1))
    geometry={key(cell):cell['bbox'] for cell in cells}
    geometry_levels={key(cell):cell.get('geometry_level','cell') for cell in cells}
    aligned=set(geometry)=={key(cell) for cell in recognized}
    extent=lambda values:(max((cell['row']+cell.get('row_span',1) for cell in values),default=0),
                          max((cell['column']+cell.get('column_span',1) for cell in values),default=0))
    compatible=bool(cells) and extent(cells)==extent(recognized)
    axes=[{},{}]
    if compatible and not aligned:
        for cell in cells:
            for axis,start,span in [(0,cell['column'],cell.get('column_span',1)),(1,cell['row'],cell.get('row_span',1))]:
                axes[axis].setdefault(start,[]).append(cell['bbox'][axis])
                axes[axis].setdefault(start+span,[]).append(cell['bbox'][axis+2])
        axes=[{index:float(np.median(values)) for index,values in axis.items()} for axis in axes]
    for index,cell in enumerate(recognized):
        level='table';box=[0,0,image.shape[1],image.shape[0]]
        if (aligned or compatible) and key(cell) in geometry:
            box=geometry[key(cell)];level=geometry_levels[key(cell)]
        elif compatible:
            left,right=cell['column'],cell['column']+cell['column_span']
            top,bottom=cell['row'],cell['row']+cell['row_span']
            if left in axes[0] and right in axes[0] and top in axes[1] and bottom in axes[1]:
                box=[axes[0][left],axes[1][top],axes[0][right],axes[1][bottom]];level='grid_axes'
        cell['bbox']=box
        x0,y0,x1,y1=[int(round(v)) for v in cell['bbox']]
        x0=max(0,x0);y0=max(0,y0);x1=min(image.shape[1],x1);y1=min(image.shape[0],y1)
        cell['bbox']=[x0,y0,x1,y1]
        if x1<=x0 or y1<=y0:
            raise ValueError('Invalid table cell geometry')
        crop=image[y0:y1,x0:x1]
        path=crop_dir/f'cell-{index:05d}.png';cv2.imencode('.png',crop)[1].tofile(path)
        kind,symbols=classify_ink(crop)
        cell.update(value_kind='text' if cell['text'].strip() else kind,symbols=symbols,
                    confidence=0.,confidence_available=False,crop_path=str(path.resolve()),line_results=[],
                    method=OCR_MODEL,ocr_engine=OCR_MODEL,geometry_level=level,requires_review=True,
                    ocr_result=str(crop_dir/'paddleocr-vl/result.json'))
        for symbol in cell.get('symbols',[]):symbol['status']='candidate' if symbol['text']!=cell['text'] else 'supported_by_ocr'
    # A header phrase describing depth can leak into its adjacent area column
    # during whole-table decoding. Re-read the measured source cells together.
    count=header_rows(recognized)
    headers=[cell for cell in recognized if cell['row']<count]
    suspects=[cell for cell in headers if '面积' in cell['text'] and re.search(r'最深|深度|高度|厚度',cell['text'])]
    if aligned and suspects:
        selected={id(cell) for cell in suspects}
        selected.update(id(cell) for cell in headers if re.search(r'深度|高度|厚度',cell['text']))
        for index,cell in enumerate(recognized):
            if id(cell) not in selected:continue
            x0,y0,x1,y1=cell['bbox']
            candidate=models.recognize(image[y0:y1,x0:x1],task='text',output_dir=crop_dir/f'header-check-{index:05d}')
            text='\n'.join(block['text'].strip() for block in candidate['blocks'] if block['text'].strip())
            cell['ocr_candidates']=[{'scope':'whole_table','text':cell['text']},
                                    {'scope':'cell_crop','text':text,'result':str(crop_dir/f'header-check-{index:05d}'/'result.json')}]
            if text:
                cell['text']=text
                cell['method']=OCR_MODEL+'_cell_recheck'
    return recognized


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
    layout_cache=models.config.get('layout_cache')
    discovered=None
    if layout_cache:
        discovered=json.loads((Path(layout_cache)/f"{request['page']:05d}.json").read_text(encoding='utf8'))
        expected=discovered.get('signature',{})
        if expected.get('source_hash')!=sha or expected.get('config',{}).get('layout_model')!=models.config.get('layout_model'):
            raise ValueError('Layout discovery cache differs from source image or model')
    layout=discovered['layout'] if discovered else models.predict(models.config.get('layout_model','PP-DocLayoutV3'),image,threshold=.3)
    regions=combine_regions([{'bbox':r['coordinate'],'label':r['label'],'score':float(r['score']),'method':'neural_layout'}
        for r in layout.get('boxes',[])],grid_regions(image))
    if models.config.get('layout_only'):
        tables,excluded,assessment=assess_table_regions(models,image,layout,regions,request.get('lines',[]),sha,request['page'])
        regions=[region for region in regions if region['category']!='table']+tables
        value={'signature':signature,'page':request['page'],'image_path':str(source.resolve()),'image_size':[image.shape[1],image.shape[0]],
            'layout':layout,'regions':regions,'mode':'layout_only','elapsed_seconds':time.perf_counter()-started,'cache_hit':False,
            'excluded_table_grids':excluded,'table_assessment':assessment}
        save_json(cache,value);return value
    page_angle,page_confidence,page_orientation=orientation(models,image)
    if not request.get('lines'):
        # New documents have no legacy OCR yet. Read only detected title regions
        # AFTER layout, so caption discovery never depends on whole-page OCR.
        request={**request,'lines':[]}
        for box in layout.get('boxes',[]):
            if not any(v in box['label'].lower() for v in ['title','caption']):continue
            x0,y0,x1,y1=[int(v) for v in box['coordinate']]
            title=models.recognize(image[max(0,y0):y1,max(0,x0):x1],task='text')
            text=' '.join(block['text'] for block in title['blocks'])
            if text:request['lines'].append({'text':text,'bbox':[x0,y0,x1,y1]})
    if discovered and discovered.get('table_assessment'):
        tables=[region for region in discovered['regions'] if region['category']=='table']
        excluded=discovered.get('excluded_table_grids',[])
    else:
        tables,excluded,_=assess_table_regions(models,image,layout,regions,request.get('lines',[]),sha,request['page'])
    regions=tables if models.config.get('tables_only') else sorted(
        [region for region in regions if region['category']!='table']+tables,key=lambda region:(region['bbox'][1],region['bbox'][0]))
    results=[]
    for index,region in enumerate(regions):
        x0,y0,x1,y1=[int(round(v)) for v in region['bbox']]
        x0=max(0,x0);y0=max(0,y0);x1=min(image.shape[1],x1);y1=min(image.shape[0],y1)
        if x1<=x0 or y1<=y0:continue
        crop,unrotated_to_source,render_info=region_pixels(request,image,[x0,y0,x1,y1],region['category'])
        caption=caption_for([x0,y0,x1,y1],request.get('lines',[]),image.shape[0])
        directory=out_dir/f"{request['page']:05d}-{index:02d}";directory.mkdir(exist_ok=True)
        caption_evidence={}
        if region['category']=='table' and caption.get('caption_bbox'):
            cx0,cy0,cx1,cy1=caption['caption_bbox']
            cx0=max(0,int(cx0)-3);cy0=max(0,int(cy0)-3)
            cx1=min(image.shape[1],int(cx1)+4);cy1=min(image.shape[0],int(cy1)+4)
            candidate=models.recognize(rotated(image[cy0:cy1,cx0:cx1],(-page_angle)%360),task='text',output_dir=directory/'caption-ocr')
            text='\n'.join(block['text'] for block in candidate['blocks'])
            revised=caption_for([x0,y0,x1,y1],[{'text':text,'bbox':caption['caption_bbox']}],image.shape[0])
            caption_evidence={'previous':caption,'text':text,'ocr_result':str(directory/'caption-ocr/result.json'),
                              'source_bbox':[cx0,cy0,cx1,cy1],'requires_review':not bool(revised['label'])}
            if revised['label']:caption=revised
        region_angle,confidence,ori=orientation(models,crop)
        angle=(-region_angle)%360
        if confidence<.85:angle=(-page_angle)%360
        rotation_candidates=[];orientation_ambiguous=confidence<.85
        upright=rotated(crop,angle)
        upright,deskew_matrix,skew_angle=deskew(upright) if region['category']=='table' else (upright,np.eye(3),0.)
        to_page=unrotated_to_source@np.linalg.inv(rotation_matrix(crop.shape[1],crop.shape[0],angle))@np.linalg.inv(deskew_matrix)
        crop_path=directory/'region.png';cv2.imencode('.png',upright)[1].tofile(crop_path)
        cells=[];structure={}
        if region['category']=='table':
            recognized_result=None
            if region.get('requires_type_check'):
                checked=models.recognize(upright,task='ocr',output_dir=directory/'type-check')
                blocks=[block for block in checked['blocks'] if block['label']=='table'
                        and box_coverage([0,0,upright.shape[1],upright.shape[0]],block['bbox'])>=.5]
                if len(blocks)!=1:
                    excluded.append({**region,'reason':'native_ocr_classification_pending','type_check_result':str(directory/'type-check/result.json'),
                                     'native_labels':[block['label'] for block in checked['blocks']]})
                    continue
                recognized_result={**checked,'blocks':blocks}
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
            cells=cell_content(models,upright,cells,directory,recognized_result)
            for cell in cells:cell['polygon_source']=transform_points(box_polygon(cell['bbox']),to_page)
            hr=header_rows(cells)
            for cell in cells:
                cell['header_path']=[c['text'] for c in cells if c['row']<hr and c.get('text')
                    and c['column']<=cell['column']<c['column']+c['column_span']]
            structure.update(header_rows=hr,signature=table_signature(cells,upright.shape[1],upright,hr),
                rotation_text_candidates=rotation_candidates,orientation_ambiguous=orientation_ambiguous,
                ocr_engine=OCR_MODEL,ocr_result=cells[0].get('ocr_result') if cells else None,
                ocr_results=sorted({cell['ocr_result'] for cell in cells if cell.get('ocr_result')}),
                classification=region.get('classification',{'decision':'primary_layout_table'}),
                cell_geometry=next(iter({cell['geometry_level'] for cell in cells})) if len({cell['geometry_level'] for cell in cells})==1 else 'mixed',
                symbol_matrix=any(cell.get('source_matrix') for cell in cells),needs_review=True)
        else:
            text=models.recognize(upright,task='text',output_dir=directory/'paddleocr-vl')
            structure['labels']=[{'text':block['text'],'confidence':0.,'confidence_available':False,
                'polygon_source':transform_points(box_polygon(block['bbox']),to_page)} for block in text['blocks']]
            structure['ocr_engine']=OCR_MODEL
        if caption_evidence:structure['caption_ocr']=caption_evidence
        margin=max(3,min(image.shape[:2])*.005)
        touches=[side for side,yes in [('left',x0<=margin),('top',y0<=margin),
            ('right',x1>=image.shape[1]-margin),('bottom',y1>=image.shape[0]-margin)] if yes]
        completeness={'status':'possible_truncation' if touches else 'unknown',
            'page_edges':touches,'requires_review':bool(touches),'synthesized_pixels':False}
        results.append({**region,**caption,'bbox':[x0,y0,x1,y1],'rotation':angle,'crop_to_source':to_page.tolist(),
            'crop_path':str(crop_path.resolve()),'cells':cells,'structure':structure,'orientation':ori,
            'source_hash':sha,'key':f'{index:03d}','render':render_info,'crop_size':[upright.shape[1],upright.shape[0]],
            'deskew_degrees':skew_angle,'completeness':completeness})
    value={'signature':signature,'page':request['page'],'image_path':str(source.resolve()),'image_size':[image.shape[1],image.shape[0]],
        'layout':layout,'page_orientation':page_orientation,'regions':results,'elapsed_seconds':time.perf_counter()-started,
        'cache_hit':False,'excluded_table_grids':excluded}
    save_json(cache,value);return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--settings',type=Path)
    args=parser.parse_args();request=json.loads(args.request.read_text(encoding='utf-8-sig'));args.output.mkdir(parents=True,exist_ok=True)
    config=request.get('config',{});models=Models(config,load_settings(args.settings));timings=[]
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
    save_json(args.output/'runtime.json',{'packages':{p:importlib.metadata.version(p) for p in ['paddleocr','paddlepaddle','paddlex','httpx']},
        'ocr':models.ocr_client.identity,
        'weights_sha256':files,'model_load_seconds':models.timings,'page_timings':timings,'rss_bytes':process.memory_info().rss,
        'peak_rss_bytes':peak[0],'total_wall_seconds':time.perf_counter()-cold,
        'first_page_wall_seconds':timings[0]['wall_seconds'] if timings else None})


if __name__=='__main__':main()
