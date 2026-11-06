"""Process any archaeological PDF without report-specific metadata or page offsets."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.pipeline import run_document


def main():
    if hasattr(sys.stdout,"reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf",type=Path)
    parser.add_argument("--output",type=Path,default=Path("data/processed/library"))
    parser.add_argument("--max-pages",type=int)
    parser.add_argument("--dpi",type=int,default=180)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--base-url",default="http://127.0.0.1:8000/iiif")
    parser.add_argument("--stop-after",type=int,help="Interrupt after N extracted pages to verify recovery")
    parser.add_argument("--settings",dest="settings_path",type=Path,help="External .env file; defaults to user profile configuration")
    parser.add_argument("--rebuild-index",action="store_true",help="Back up and atomically regenerate an unreviewed stale index")
    parser.add_argument("--skip-models",action="store_true",help="Run extraction and indexing only; language model stage can be resumed separately")
    parser.add_argument('--skip-graphics',action='store_true',help='Explicit legacy text-only path; no graphic structure is inferred')
    parser.add_argument('--graphics-runtime',type=Path,help='Isolated Python 3.11 worker executable')
    parser.add_argument('--graphics-output',type=Path,help='Separate staging library for graph objects')
    args=parser.parse_args()
    options=vars(args).copy();skip=options.pop('skip_graphics');runtime=options.pop('graphics_runtime');graphics_output=options.pop('graphics_output')
    if not skip:
        import sqlite3
        from src.graphics.front import prepare_pdf
        from src.graphics.pipeline import DEFAULT_RUNTIME,run_graphics
        from src.ingest.page_pipeline import file_hash
        runtime=runtime or DEFAULT_RUNTIME
        if not runtime.exists():parser.error('Graphic runtime missing; install requirements-graphics.txt in Python 3.11 or explicitly use --skip-graphics')
        existing=None;database=args.output/'library.db'
        if database.exists():
            connection=sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True)
            existing=connection.execute('select id,total_pages from manifests where source_pdf_sha256=?',(file_hash(args.pdf),)).fetchone();connection.close()
        if existing:
            result=run_graphics(args.output,existing[0],graphics_output or args.output/'graphics-stage',
                list(range(1,min(existing[1],args.max_pages or existing[1])+1)),runtime=runtime)
            print(json.dumps(result,ensure_ascii=False,indent=2));return
        options['graphics_front_dir']=prepare_pdf(args.pdf,args.output,max_pages=args.max_pages,dpi=args.dpi,runtime=runtime)
    result=run_document(**options)
    if not skip and result['status']=='completed':
        result['graphics']=run_graphics(args.output,result['stats']['manifest_id'],graphics_output or args.output/'graphics-stage',
            list(range(1,result['stats']['pages_ingested']+1)),runtime=runtime)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
