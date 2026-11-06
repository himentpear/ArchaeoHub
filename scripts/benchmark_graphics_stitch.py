"""Synthetic overlap positives and hard negatives, reported separately from scans."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np
from src.graphics.grouping import stitch_pair
from src.ingest.page_pipeline import atomic_json


def benchmark():
    rng=np.random.default_rng(20261031);correct=0;false_merges=0;cases=[]
    for i in range(5):
        base=cv2.GaussianBlur(rng.integers(0,256,(400,900,3),dtype=np.uint8),(3,3),0)
        _,e=stitch_pair(base[:,:600],base[:,300:]);correct+=int(e['accepted']);cases.append({'id':f'overlap-{i}','positive':True,'result':e})
    for i in range(20):
        if i<10:
            first=rng.integers(0,256,(300,400,3),dtype=np.uint8);second=rng.integers(0,256,(300,400,3),dtype=np.uint8);category='figure'
        else:
            first=np.full((300,400,3),255,np.uint8)
            for x in range(10,400,40):cv2.line(first,(x,10),(x,290),(0,0,0),2)
            for y in range(10,300,30):cv2.line(first,(10,y),(390,y),(0,0,0),2)
            second=first.copy();category='table'
        _,e=stitch_pair(first,second,category);false_merges+=int(e['accepted']);cases.append({'id':f'negative-{i}','positive':False,'result':e})
    return {'dataset':'synthetic, not historical scan accuracy','seed':20261031,'positive_cases':5,'negative_cases':20,
        'recall':correct/5,'false_merges':false_merges,'passed':correct>=4 and not false_merges,'cases':cases}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('reports/graphics_stitch_benchmark.json'));a=p.parse_args()
    result=benchmark();atomic_json(a.output,result);print(json.dumps({k:v for k,v in result.items() if k!='cases'}));raise SystemExit(0 if result['passed'] else 1)
