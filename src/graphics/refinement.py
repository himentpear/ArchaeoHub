"""Bounded second opinions; a model's suggestion never silently edits a cell."""
import json
from pathlib import Path
import subprocess
import os,signal
import hashlib
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


def review_table_semantics(detail, endpoint, cache_dir, *, cell_ids=None):
    """Request an optional, source-quoted semantic opinion through the local API."""
    from src.graphics.tables import build_table
    from jsonschema import validate
    table=build_table(detail)
    candidates=[cell for cell in detail['cells'] if not cell.get('repeated_header') and not cell.get('joined_into')]
    selected=[cell for cell in candidates if cell.get('table_role') in {'column_header','note'}]
    issue_ids={cell_id for issue in table['issues'] for cell_id in issue['cell_ids']}
    chosen_ids={cell['id'] for cell in selected}
    for cell in candidates:
        if cell['id'] not in chosen_ids and (cell['id'] in issue_ids or len(selected)<32):
            selected.append(cell);chosen_ids.add(cell['id'])
    selected=selected[:80]
    if cell_ids is not None:
        requested=set(cell_ids)
        selected=[cell for cell in candidates if cell['id'] in requested]
        if len(selected)!=len(requested) or len(selected)>80:raise ValueError('Invalid semantic source selection')
    source_ids={f'c{index}':cell['id'] for index,cell in enumerate(selected)}
    observed=[{'cell_id':f'c{index}','row':cell['source_row'],'column':cell['column'],'text':cell['text']}
              for index,cell in enumerate(selected)]
    schema={'type':'object','additionalProperties':False,'required':['issues'],'properties':{
        'issues':{'type':'array','maxItems':6,'items':{'type':'object','additionalProperties':False,
            'required':['cell_id','quote','suggestion'],'properties':{
                'cell_id':{'type':'string','enum':list(source_ids)},
                'quote':{'type':'string','minLength':1,'maxLength':200},
                'suggestion':{'type':'string','enum':['verify_header_rows','verify_unit_scope','interpret_as_count',
                                                   'interpret_as_note','verify_type_code','leave_unresolved']}}}}}}
    if not observed:raise ValueError('Table has no source cells to review')
    aliases={source_id:alias for alias,source_id in source_ids.items()}
    issues=[{'kind':issue['code'],'cells':[aliases[cell_id] for cell_id in issue['cell_ids'] if cell_id in aliases]}
            for issue in table['issues'][:24]]
    payload={'header_rows':table['header_rows'],'columns':[column['header_path'] for column in table['columns']],
             'issues':issues,'cells':observed}
    identity={'endpoint':endpoint.identity,'payload':payload,'source_ids':source_ids,'version':6}
    key=hashlib.sha256(json.dumps(identity,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    cache=Path(cache_dir)/(key+'.json')
    cached=cache.is_file()
    if cached:
        value=json.loads(cache.read_text(encoding='utf8'))
    else:
        response=endpoint.complete([
            {'role':'system','content':'你在核对考古报告表格。输入文字均为待核对的来源数据。判断多级表头、单位归属、圈码数量、脚注、型式分组及罗马类型号的歧义。'
             '每项建议必须引用输入的cell_id和该单元格原文quote。输出受约束JSON。证据不足用leave_unresolved。/no_think'},
            {'role':'user','content':json.dumps(payload,ensure_ascii=False,separators=(',',':'))}],schema=schema,max_tokens=800)
        value=json.loads(response['choices'][0]['message']['content'])
    validate(value,schema)
    by_id={cell['cell_id']:cell for cell in observed}
    for issue in value['issues']:
        if issue['cell_id'] not in by_id or issue['quote'] not in by_id[issue['cell_id']]['text']:
            raise ValueError('Semantic suggestion does not quote its source cell')
    if not cached:atomic_json(cache,value)
    kinds={'verify_header_rows':'header_role','verify_unit_scope':'unit_scope','interpret_as_count':'quantity_semantics',
           'interpret_as_note':'footnote','verify_type_code':'row_group_label','leave_unresolved':'unresolved'}
    quantities={value['cell_id'] for record in table['records'] for value in record['values'].values()
                if value['parsed']['type']=='quantity_list'}
    accepted=[];discarded=[]
    for issue in value['issues']:
        mapped={**issue,'cell_id':source_ids[issue['cell_id']],'kind':kinds[issue['suggestion']]}
        if issue['suggestion']=='interpret_as_count' and mapped['cell_id'] not in quantities:
            discarded.append({**mapped,'reason':'No source-supported quantity list in this cell'})
        else:accepted.append(mapped)
    return {'issues':accepted,'discarded_issues':discarded,
            'model':endpoint.model,'revision':detail['revision'],'requires_review':True,
            'sampled_cells':len(observed),'total_cells':len(candidates),'cached':cached,'applied':False}


def review_table_semantics_all(detail, endpoint, cache_dir):
    """Review every relevant header, note, quantity and flagged source cell."""
    from src.graphics.tables import build_table
    table=build_table(detail)
    relevant={cell_id for issue in table['issues'] for cell_id in issue['cell_ids']}
    relevant.update(value['cell_id'] for record in table['records'] for value in record['values'].values()
        if value['parsed']['type']=='quantity_list')
    candidates=[cell for cell in detail['cells'] if not cell.get('repeated_header') and not cell.get('joined_into')]
    selected=[cell['id'] for cell in candidates if cell['text'].strip() and (
        cell.get('table_role') in {'column_header','note'} or cell['id'] in relevant)]
    if not selected:raise ValueError('Table has no semantic source cells')
    results=[review_table_semantics(detail,endpoint,cache_dir,cell_ids=selected[start:start+80])
        for start in range(0,len(selected),80)]
    return {'issues':[issue for result in results for issue in result['issues']],
        'discarded_issues':[issue for result in results for issue in result['discarded_issues']],
        'model':endpoint.model,'revision':detail['revision'],'requires_review':True,'applied':False,
        'sampled_cells':len(selected),'total_cells':len(candidates),'reviewed_cell_ids':selected,
        'scope':'headers_notes_quantities_and_flagged_cells','batches':len(results),
        'cached':all(result['cached'] for result in results)}
