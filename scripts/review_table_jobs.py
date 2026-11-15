"""Process semantic table review queues through the configured local model API."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sqlalchemy import select
from src.db.models import ReviewTask
from src.db.session import get_session, init_db
from src.graphics.models import GraphicObject
from src.graphics.refinement import review_table_semantics_all
from src.graphics.service import object_detail
from src.ingest.page_pipeline import atomic_json, job_lock
from src.model_router import ModelRouter
from src.settings import load_settings
from src.table_jobs import list_table_jobs


def review_jobs(library):
    engine=init_db('sqlite:///'+(library/'library.db').as_posix())
    endpoint=ModelRouter.from_settings(library/'table-semantic-cache',load_settings()).local_endpoint
    try:
        for job in list_table_jobs(library):
            if job['status']=='paused':continue
            state_path=library/'table-jobs'/f"{job['id']}-semantic.json"
            with get_session(engine) as session:
                objects=list(session.scalars(select(GraphicObject).where(GraphicObject.run_id.in_(list(job.get('runs',{}).values())),
                    GraphicObject.category=='table',GraphicObject.status.not_in(['superseded','rejected']))))
                targets=[object_detail(session,obj.id) for obj in objects]
            targets=[detail for detail in targets if detail['table']['semantic_review']['recommended']]
            summary={'total':len(targets),'completed':0,'failed':0,'updated_at':time.time(),
                'status':'running' if endpoint else 'unconfigured'}
            atomic_json(state_path,summary)
            for detail in targets:
                path=library/'table-semantic-results'/detail['id']/f"{detail['revision']}.json"
                previous=json.loads(path.read_text(encoding='utf8')) if path.is_file() else {}
                if previous.get('status')=='complete':summary['completed']+=1
                elif previous.get('attempts',0)>=2:summary['failed']+=1
                elif endpoint is not None:
                    attempt=previous.get('attempts',0)+1
                    try:
                        result=review_table_semantics_all(detail,endpoint,library/'table-semantic-cache')
                        with get_session(engine) as session:
                            current=session.get(GraphicObject,detail['id'])
                            if current.revision_no!=detail['revision']:continue
                            if result['issues']:
                                task=session.scalar(select(ReviewTask).where(ReviewTask.target_id==detail['id'],
                                    ReviewTask.risk_type=='table_semantic_review',ReviewTask.status=='pending'))
                                if task is None:
                                    task=ReviewTask(manifest_id=detail['manifest_id'],canvas_id=detail['fragments'][0]['canvas_id'],
                                        target_type='graphic_object',target_id=detail['id'],risk_type='table_semantic_review',priority_score=.6)
                                    session.add(task)
                                task.description=f"表格语义复核，修订 {detail['revision']}，{len(result['issues'])} 项带原文引用的建议。"
                                task.candidates_snapshot=[result]
                        atomic_json(path,{'status':'complete','attempts':attempt,'completed_at':time.time(),'result':result})
                        summary['completed']+=1
                    except Exception as error:
                        atomic_json(path,{'status':'failed','attempts':attempt,'error_type':type(error).__name__,
                            'error':str(error)[:400],'revision':detail['revision']})
                        if attempt>=2:summary['failed']+=1
                summary['updated_at']=time.time()
                atomic_json(state_path,summary)
            summary['pending']=len(targets)-summary['completed']-summary['failed']
            if endpoint:
                summary['status']='running' if summary['pending'] else 'complete' if job['status']=='complete' else 'waiting_for_pages'
            summary['updated_at']=time.time()
            atomic_json(state_path,summary)
    finally:engine.dispose()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',type=Path,default=ROOT/'data/processed/library')
    parser.add_argument('--watch',action='store_true')
    parser.add_argument('--interval',type=float,default=30)
    args=parser.parse_args()
    if args.interval<1:parser.error('Invalid review interval')
    args.library=args.library.resolve()
    with job_lock(args.library/'table-jobs'/'semantic-worker.lock'):
        while True:
            try:review_jobs(args.library)
            except Exception as error:
                print(json.dumps({'status':'semantic_worker_failed','error_type':type(error).__name__}),flush=True)
                if not args.watch:raise
            if not args.watch:return
            time.sleep(args.interval)


if __name__=='__main__':main()
