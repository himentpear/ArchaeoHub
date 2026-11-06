"""Summarize measured runtime without treating predictions as accuracy labels."""
import argparse,json,statistics,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.ingest.page_pipeline import atomic_json


def summarize(stage):
    stage=Path(stage).resolve();job=json.loads((stage/'graphics-job.json').read_text(encoding='utf8'))
    cache=Path(job.get('cache',stage/'graphics-cache'));runtime=json.loads((cache/'runtime.json').read_text(encoding='utf8'))
    groups={'no_detected_graphic':[],'table_page':[],'figure_or_chart_page':[]};regions=review=0
    for timing in runtime['page_timings']:
        page=json.loads((cache/f'{timing["page"]:05d}.json').read_text(encoding='utf8'))
        key='table_page' if any(r['category']=='table' for r in page['regions']) else 'figure_or_chart_page' if page['regions'] else 'no_detected_graphic'
        groups[key].append(timing.get('wall_seconds',timing['seconds']));regions+=len(page['regions'])
        review+=sum(bool(r.get('structure',{}).get('needs_review')) for r in page['regions'])
    def stats(values):
        return {'pages':len(values),'mean_seconds':statistics.mean(values),'median_seconds':statistics.median(values),
            'p95_seconds':sorted(values)[min(len(values)-1,int(len(values)*.95))]} if values else {'pages':0}
    return {'run_id':job['run_id'],'published':False,'accuracy':'not measured; independent annotations required',
        'implementation_hash':json.loads((stage/'request.json').read_text(encoding='utf8'))['config'].get('implementation_hash'),
        'wall_seconds':runtime.get('total_wall_seconds'),'peak_rss_mib':runtime.get('peak_rss_bytes',0)/2**20,
        'first_page_wall_seconds':runtime.get('first_page_wall_seconds'),'model_load_seconds':runtime['model_load_seconds'],
        'cache_hits':sum(p['cache_hit'] for p in runtime['page_timings']),'page_groups':{k:stats(v) for k,v in groups.items()},
        'detected_regions':regions,'local_structure_review_candidates':review,'review_fraction':review/max(1,regions),
        'sampling_note':'Purposeful appendix-heavy sample; not a random estimate of the whole book. CPU contention may affect times.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=summarize(a.stage);atomic_json(a.output,result);print(json.dumps(result,ensure_ascii=False,indent=2))
