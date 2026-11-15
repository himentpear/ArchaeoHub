"""Incrementally deploy a running full-report table batch to the document library."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import hashlib

from sqlalchemy.exc import SQLAlchemyError

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.ingest.page_pipeline import atomic_json, job_lock
from src.table_jobs import rollback_table_batch, sync_table_batch


def recovery_command(staging, library):
    progress=json.loads((staging/'progress.json').read_text(encoding='utf8'))
    if progress.get('workflow')!='graphics':
        return [sys.executable,'-u',str(ROOT/'scripts/reprocess_tables.py'),'--library',str(library),'--output',str(staging),'--resume']
    request=json.loads((staging/'request.json').read_text(encoding='utf8'))
    config=request['config']
    command=[sys.executable,'-u',str(ROOT/'scripts/process_graphics.py'),'--library',str(library),
        '--output',str(staging),'--manifest-id',config['manifest_id'],'--pages',','.join(map(str,config['selected_pages'])),
        '--layout-model',config['layout_model'],'--deploy']
    for option,key in [('--runtime','runtime'),('--settings','settings_path')]:
        if progress.get(key):command.extend([option,progress[key]])
    if config.get('layout_cache'):command.extend(['--layout-cache',config['layout_cache']])
    if not config.get('with_vl'):command.append('--skip-vl')
    if not config.get('complex_review'):command.append('--no-complex-review')
    if config.get('tables_only'):command.append('--tables-only')
    if config.get('allow_api_review'):command.append('--allow-api-review')
    return command


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--staging',type=Path,required=True)
    parser.add_argument('--library',type=Path,default=ROOT/'data/processed/library')
    parser.add_argument('--watch',action='store_true')
    parser.add_argument('--interval',type=float,default=30)
    parser.add_argument('--retry-limit',type=int,default=2)
    parser.add_argument('--rollback',action='store_true')
    args=parser.parse_args()
    if args.interval<1 or args.retry_limit<0:parser.error('Invalid interval or retry limit')
    args.staging=args.staging.resolve();args.library=args.library.resolve()
    if args.rollback:
        print(json.dumps(rollback_table_batch(args.staging,args.library),ensure_ascii=False),flush=True)
        return
    import cv2
    cv2.setNumThreads(1)
    retry=None
    job_id=hashlib.sha256(str(args.staging.resolve()).encode()).hexdigest()[:20]
    with job_lock(args.library/'table-jobs'/f'{job_id}-worker.lock'):
        while True:
            try:
                state=sync_table_batch(args.staging,args.library)
            except (OSError,ValueError,KeyError,RuntimeError,SQLAlchemyError) as error:
                print(json.dumps({'status':'deployment_failed','error':str(error)}),flush=True)
                if not args.watch:raise
                time.sleep(args.interval)
                continue
            print(json.dumps({key:state.get(key) for key in ('id','status','deployed_pages','expected_pages','tables','records')},ensure_ascii=False),flush=True)
            if not args.watch or state['status'] in {'complete','paused'}:return
            attempts=state.get('restart_attempts',0)
            if state['status'] in {'recognition_failed','recognizing'} and attempts<args.retry_limit and (retry is None or retry.poll() is not None):
                try:
                    with job_lock(args.staging/'graphics.lock'):pass
                    with job_lock(args.staging/'tables-batch.lock'):pass
                except OSError:
                    time.sleep(args.interval)
                    continue
                attempts+=1
                state['restart_attempts']=attempts
                atomic_json(args.library/'table-jobs'/f'{job_id}.json',state)
                log=args.library/'table-jobs'/f"{state['id']}-retry-{attempts}.log"
                env={**os.environ,'OCR_TIMEOUT':'14400'}
                with log.open('ab') as stream:
                    retry=subprocess.Popen(recovery_command(args.staging,args.library),cwd=ROOT,env=env,
                        stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            time.sleep(args.interval)


if __name__=='__main__':main()
