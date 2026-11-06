"""Prepare 60 source pages with group-disjoint 40/20 splits, without inventing labels."""
import argparse,json,random,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.graphics.grouping import LABEL
from src.ingest.page_pipeline import atomic_json


def sample(run_dir):
    numbers=list(range(535,575))+[1,17,22,32,35,40,57,80,100,130,160,199,250,300,331,400,470,597,650,730]
    parent={n:n for n in numbers};labels={};records=[]
    def find(n):
        while parent[n]!=n:parent[n]=parent[parent[n]];n=parent[n]
        return n
    for n in numbers:
        d=json.loads((run_dir/'pages'/f'{n:05d}.json').read_text(encoding='utf8'))
        p=d['page']
        if n>=535 and n<=574:
            for l in p['lines']:
                m=LABEL.match(l['text'].strip())
                if m and '附' in m[2]:
                    label=re.sub(r'\s+','',m[2]+m[3])
                    if label in labels:parent[find(n)]=find(labels[label])
                    else:labels[label]=n
        records.append({'page':n,'image_path':p['image_path'],'image_sha256':d['image_sha256'],
            'reviewed':False,'reviewer':'','objects':[],'augmentation':False})
    groups={}
    for n in numbers:groups.setdefault(find(n),[]).append(n)
    components=list(groups.values());random.Random(20261031).shuffle(components)
    states={(0,0):[]}
    for group in components:
        if any(n in {535,536,538,539,540} for n in group):continue  # already used during model debugging
        for (count,tables),chosen in list(states.items()):
            nc=count+len(group);nt=tables+sum(535<=n<=574 for n in group)
            if nc<=20:states.setdefault((nc,nt),chosen+group)
    candidates=[key for key in states if key[0]==20 and 10<=key[1]<=16]
    if not candidates:raise ValueError('Cannot create a balanced, group-disjoint 20-page holdout')
    test=set(states[min(candidates,key=lambda k:abs(k[1]-12))])
    for record in records:
        record['split']='test' if record['page'] in test else 'train'
        record['group_id']=f'component-{find(record["page"])}'
    return {'version':1,'pages':records,'relations':[],'stitch_suite':{},
        'annotation_policy':'Annotate all visible objects independently from original images. Empty unreviewed pages are NOT negative labels.',
        'required_counts':{'pages':60,'train':40,'test':20,'positive_relations':10,'negative_relations':20}}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    data=sample(a.run_dir);atomic_json(a.output,data)
    print(json.dumps({'pages':len(data['pages']),'train':40,'test':20,'test_pages':[p['page'] for p in data['pages'] if p['split']=='test'],'annotations_complete':False}))
