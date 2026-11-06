"""Bounded second opinions; a model's suggestion never silently edits a cell."""
import json
from pathlib import Path
import subprocess
import os,signal
from src.ingest.page_pipeline import atomic_json,file_hash


def local_vl(runtime,crop,output,*,timeout=120):
    script=Path(__file__).with_name('vl_worker.py');output=Path(output)
    key=file_hash(Path(crop));version=file_hash(script)
    if output.exists():
        old=json.loads(output.read_text(encoding='utf8'))
        if old.get('source_hash')==key and old.get('worker_version')==version and old.get('status')=='candidate':return old
    log=output.with_suffix('.log')
    with log.open('w',encoding='utf8') as stream:
        try:
            process=subprocess.Popen([str(runtime),str(script),'--image',str(crop),'--output',str(output)],
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=os.name!='nt')
            try:code=process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Windows venv launchers can spawn a second Python executable.
                # Kill the owned tree so a timed-out VLM cannot keep using CPU.
                if os.name=='nt':subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                else:
                    try:os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                if process.poll() is None:process.kill()
                process.wait();raise
            if code:value={'status':'unavailable','error_type':'WorkerFailure','source_hash':key}
            else:
                value=json.loads(output.read_text(encoding='utf8'));value['worker_version']=version
                atomic_json(output,value);return value
        except subprocess.TimeoutExpired:value={'status':'timeout','source_hash':key,'worker_version':version}
    atomic_json(output,value);return value


def grounded_structure_review(router,cells):
    if len(cells)>500:cells=cells[:500]
    observed=[{'id':str(i),'row':c['row'],'column':c['column'],'row_span':c['row_span'],'column_span':c['column_span'],
        'text':c.get('text','')} for i,c in enumerate(cells)]
    messages=[{'role':'system','content':'核对表格结构，仅返回JSON {"issues":[{"cell_id":"编号","kind":"span_ambiguity|header_ambiguity|unreadable","quote":"该单元格已有原文"}]}。只能引用输入单元格原文，不补数值，不猜测图像，不修改结构。'},
        {'role':'user','content':json.dumps(observed,ensure_ascii=False)}]
    response=router._cloud(messages,structured=True,max_tokens=1200)
    value=json.loads(response['choices'][0]['message']['content']);by_id={c['id']:c for c in observed}
    if set(value)!={'issues'} or not isinstance(value['issues'],list):raise ValueError('Invalid structure review')
    for issue in value['issues']:
        if set(issue)!={'cell_id','kind','quote'} or issue['cell_id'] not in by_id:raise ValueError('Unknown source cell')
        if issue['kind'] not in {'span_ambiguity','header_ambiguity','unreadable'}:raise ValueError('Invalid issue kind')
        if not issue['quote'] or issue['quote'] not in by_id[issue['cell_id']]['text']:raise ValueError('Ungrounded structure suggestion')
    return {**value,'model':router.model,'requires_review':True,'images_sent':False}
