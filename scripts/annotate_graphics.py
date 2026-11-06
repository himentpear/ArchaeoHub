"""Serve independent source-only graphical annotation on loopback."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.graphics.annotation import make_annotation_server
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--library',type=Path,default=Path('data/processed/library'))
    p.add_argument('--manifest-id',required=True);p.add_argument('--annotations',type=Path,required=True);p.add_argument('--port',type=int,default=8003);a=p.parse_args()
    server=make_annotation_server(a.library,a.manifest_id,a.annotations,a.port)
    print(f'Independent annotations: http://127.0.0.1:{server.server_port}',flush=True)
    try:server.serve_forever()
    finally:server.server_close();server.engine.dispose()
