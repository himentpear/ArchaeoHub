"""Independent source-only annotation service; predictions are never prefilled."""
from http.server import BaseHTTPRequestHandler,HTTPServer
import json,shutil
from pathlib import Path
from urllib.parse import urlparse,parse_qs
from sqlalchemy import select
from src.db.models import Canvas
from src.db.session import init_db,get_session
from src.ingest.page_pipeline import atomic_json,file_hash


def make_annotation_server(library,manifest_id,annotations,port=8003):
    path=Path(annotations).resolve();engine=init_db('sqlite:///'+(Path(library).resolve()/'library.db').as_posix())
    def load():
        data=json.loads(path.read_text(encoding='utf8'));sizes={}
        with get_session(engine) as session:
            for item in data['pages']:
                canvas=session.scalar(select(Canvas).where(Canvas.manifest_id==manifest_id,Canvas.scan_index==item['page']))
                if canvas is None or file_hash(Path(canvas.image_storage_path))!=item['image_sha256']:raise ValueError('Annotation source mismatch')
                sizes[str(item['page'])]=[canvas.width,canvas.height]
        return data,sizes
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send(self,value,status=200,mime='application/json; charset=utf-8'):
            raw=value if isinstance(value,bytes) else json.dumps(value,ensure_ascii=False).encode('utf8')
            self.send_response(status);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(raw)))
            self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(raw)
        def do_GET(self):
            try:
                route=urlparse(self.path)
                if route.path=='/':return self.send(Path(__file__).with_name('annotation.html').read_bytes(),mime='text/html; charset=utf-8')
                data,sizes=load()
                if route.path=='/api/annotations':return self.send({'data':data,'sizes':sizes,'version':file_hash(path)})
                if route.path=='/source':
                    number=int(parse_qs(route.query)['page'][0]);item=next(p for p in data['pages'] if p['page']==number)
                    with get_session(engine) as s:
                        canvas=s.scalar(select(Canvas).where(Canvas.manifest_id==manifest_id,Canvas.scan_index==number))
                        return self.send(Path(canvas.image_storage_path).read_bytes(),mime='image/jpeg')
                self.send({'error':'Not found'},404)
            except (ValueError,KeyError,StopIteration) as error:self.send({'error':str(error)},400)
        def do_POST(self):
            if self.path!='/api/save':return self.send({'error':'Not found'},404)
            origin=self.headers.get('Origin')
            if origin and origin!=f'http://127.0.0.1:{self.server.server_port}':return self.send({'error':'Origin rejected'},403)
            try:
                length=int(self.headers.get('Content-Length',0))
                if not 0<length<=2000000:raise ValueError('Invalid body size')
                value=json.loads(self.rfile.read(length));data,sizes=load()
                if value['version']!=file_hash(path):raise ValueError('Annotations changed; reload before saving')
                item=next(p for p in data['pages'] if p['page']==value['page']);fields=value['fields']
                if set(fields)!={'reviewer','reviewed','objects'}:raise ValueError('Only annotation fields may change')
                if fields['reviewed'] and not str(fields['reviewer']).strip():raise ValueError('Reviewer required')
                if not isinstance(fields['objects'],list):raise ValueError('Object list required')
                width,height=sizes[str(item['page'])];ids=set()
                for obj in fields['objects']:
                    if not obj.get('id') or obj['id'] in ids:raise ValueError('Unique object IDs required')
                    ids.add(obj['id'])
                    if obj.get('category') not in {'table','chart','figure'}:raise ValueError('Invalid category')
                    x0,y0,x1,y1=obj['bbox']
                    if not 0<=x0<x1<=width or not 0<=y0<y1<=height:raise ValueError('Box outside Canvas')
                    if obj.get('orientation',0) not in {0,90,180,270}:raise ValueError('Invalid orientation')
                item.update(fields)
                all_ids=[obj['id'] for p in data['pages'] for obj in p.get('objects',[])]
                if len(all_ids)!=len(set(all_ids)):raise ValueError('Object IDs must be unique across pages')
                relations=value.get('relations',data.get('relations',[]))
                if not isinstance(relations,list):raise ValueError('Relations must be an array')
                data['relations']=relations
                shutil.copy2(path,path.with_suffix(path.suffix+'.bak'));atomic_json(path,data)
                self.send({'version':file_hash(path),'saved':item['page']})
            except (ValueError,KeyError,TypeError,StopIteration) as error:self.send({'error':str(error)},409)
    server=HTTPServer(('127.0.0.1',port),Handler);server.engine=engine
    return server
