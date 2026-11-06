"""Stage incremental graphic processing, or publish a verified processing run."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.graphics.pipeline import run_graphics,promote,unpublish,DEFAULT_RUNTIME


def page_list(value):
    result=set()
    for part in value.split(','):
        bounds=[int(v) for v in part.split('-')]
        result.update(range(bounds[0],bounds[-1]+1))
    if not result or min(result)<1:raise argparse.ArgumentTypeError('Physical pages start at 1')
    return sorted(result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',type=Path,default=Path('data/processed/library'))
    parser.add_argument('--manifest-id')
    parser.add_argument('--output',type=Path,default=Path('data/processed/benchmarks/graphics-stage'))
    parser.add_argument('--pages',type=page_list)
    parser.add_argument('--runtime',type=Path,default=DEFAULT_RUNTIME)
    parser.add_argument('--layout-model',default='PP-DocLayout_plus-L',choices=['PP-DocLayout_plus-L','PP-DocLayout-S'])
    parser.add_argument('--skip-worker',action='store_true',help='Import existing source-hash-checked worker cache')
    parser.add_argument('--no-complex-review',action='store_true')
    parser.add_argument('--skip-vl',action='store_true',help='Skip optional local VLM second opinions during model benchmarking')
    parser.add_argument('--allow-api-review',action='store_true');parser.add_argument('--settings',type=Path)
    parser.add_argument('--publish',metavar='RUN_ID');parser.add_argument('--evaluation',type=Path)
    parser.add_argument('--unpublish',metavar='RUN_ID',help='Back up and deactivate a published graphic overlay without deleting source rows')
    args=parser.parse_args()
    if args.publish and args.unpublish:parser.error('Choose publish or unpublish')
    if args.unpublish:result=unpublish(args.library,args.unpublish)
    elif args.publish:
        if not args.evaluation:parser.error('--evaluation required to publish')
        result=promote(args.output,args.library,args.publish,args.evaluation)
    else:
        if not args.manifest_id or not args.pages:parser.error('--manifest-id and --pages required')
        result=run_graphics(args.library,args.manifest_id,args.output,args.pages,runtime=args.runtime,layout_model=args.layout_model,
            complex_review=not args.no_complex_review,skip_worker=args.skip_worker,with_vl=not args.skip_vl,
            allow_api_review=args.allow_api_review,settings_path=args.settings)
    print(json.dumps(result,ensure_ascii=False,indent=2))
