"""Regressions for book table layout, data provenance and optional semantic review."""
import csv
import io
import json
from pathlib import Path

import pytest
from src.graphics.tables import build_table, infer_header_rows, parsed_value, table_row_bands
from src.graphics.vision import parse_structure
from src.graphics.service import render_csv, render_html, render_xlsx
from src.graphics.refinement import review_table_semantics
from src.graphics.grouping import data_table_regions,table_label_key,continuation_evidence,boxes_coverage,caption_for


def table_detail(html):
    cells = parse_structure(html)
    for index, cell in enumerate(cells):
        cell.update(id=f'cell-{index}', fragment_id='part', source_row=cell['row'], raw_text=cell['text'],
                    anchors=[{'scan_index':535,'polygon':[[0,0],[1,0],[1,1],[0,1]]}],
                    repeated_header=False, geometry_level='cell',value_kind='text')
    return {'id':'table', 'revision':1, 'category':'table', 'title':'探方表', 'label':'附表一', 'status':'candidate',
            'structure':{}, 'cells':cells,
            'fragments':[{'id':'part','scan_index':535,'table_header_rows':infer_header_rows(cells)}]}


def test_multilevel_headers_ignore_feature_identifiers_in_data_columns():
    detail = table_detail('<table><tr><td rowspan="2" colspan="2">探方编号</td><td colspan="2">发掘面积（平方米）</td><td rowspan="2">方内遗迹</td></tr>'
                          '<tr><td>布方尺寸</td><td>实际面积</td></tr>'
                          '<tr><td rowspan="2">A区</td><td>T1</td><td>5×6</td><td>30</td><td>Y1、H1、H2</td></tr>'
                          '<tr><td>T2</td><td>5×5</td><td></td><td>H3、H4</td></tr></table>')
    table = build_table(detail)
    assert table['header_rows'] == 2
    assert len(table['records']) == 2
    assert table['columns'][2]['header_path'] == ['发掘面积（平方米）','布方尺寸']
    assert all('T1' not in column['header_path'] for column in table['columns'])
    inherited = table['records'][1]['values']['c0']
    assert inherited['text'] == 'A区' and inherited['inherited']
    assert inherited['cell_id'] == table['records'][0]['values']['c0']['cell_id']
    assert table['records'][1]['values']['c3']['parsed'] == {'type':'blank','value':None}
    assert table['records'][0]['values']['c2']['parsed']['value'] == ['5','6']
    assert table['columns'][2]['unit'] is None
    assert table['issue_counts'] == {'unit_scope':1}


def test_html_preserves_breaks_and_row_header_scope():
    detail = table_detail('<table><thead><tr><th>地层单位</th><th>深度（米）<br/>（方内最深处）</th></tr></thead>'
                          '<tbody><tr><th scope="row">T1②</th><td>0</td></tr></tbody></table>')
    assert detail['cells'][1]['text'] == '深度（米）\n（方内最深处）'
    assert infer_header_rows(detail['cells']) == 1
    rendered = render_html(detail).decode()
    assert '<colgroup>' in rendered and '<thead>' in rendered and '<tbody>' in rendered
    assert 'scope="row"' in rendered and 'data-cell-id="cell-2"' in rendered
    assert '（方内最深处）' in rendered


def test_excavated_area_smaller_than_layout_is_supported():
    detail=table_detail('<table><tr><td>编号</td><td>布方尺寸</td><td>实际发掘面积（平方米）</td></tr>'
                        '<tr><td>T1</td><td>5×5</td><td>30</td></tr>'
                        '<tr><td>T2</td><td>10×11</td><td>86</td></tr></table>')
    table=build_table(detail)
    assert table['issue_counts']=={'area_dimension_difference':1}
    assert table['issues'][0]['row']==1
    assert table['records'][0]['values']['c1']['text']=='5×5'


def test_quantity_symbols_identifiers_and_numeric_precision_stay_distinct():
    column = {'role':'value','header_path':['铜钱名称及数量'],'unit':None}
    value = parsed_value('崇宁通宝①，政和通宝②', column)
    assert value['type'] == 'quantity_list'
    assert [item['quantity'] for item in value['value']] == [1,2]
    assert value['interpretation_status'] == 'candidate'
    assert parsed_value('T1②', column)['type'] == 'identifier'
    assert parsed_value('○', column) == {'type':'symbol','value':'○'}
    assert parsed_value('—', column) == {'type':'symbol','value':'—'}
    assert parsed_value('', column)['value'] is None
    assert parsed_value('0', column)['value'] == '0'
    assert parsed_value('0.1234567890123456789', column)['value'] == '0.1234567890123456789'


def test_column_proportions_follow_measured_book_cells():
    detail = table_detail('<table><tr><td>编号</td><td>描述</td></tr><tr><td>T1</td><td>内容</td></tr></table>')
    for cell in detail['cells']:
        cell['bbox'] = [0,0,100,30] if cell['column']==0 else [100,0,500,30]
    table = build_table(detail)
    assert [column['width'] for column in table['columns']] == [.2,.8]
    assert table['column_width_method'] == 'measured_cell_boundaries'


def test_record_exports_keep_source_ids_and_literal_strings():
    detail = table_detail('<table><tr><td>编号</td><td>内容</td></tr><tr><td>T1</td><td>=literal</td></tr></table>')
    data = list(csv.reader(io.StringIO(render_csv(detail).decode('utf-8-sig'))))
    assert data[1][:2] == ['T1','=literal']
    assert data[1][-2] == '535'
    assert json.loads(data[1][-1]) == [['cell-2'],['cell-3']]
    from openpyxl import load_workbook
    workbook = load_workbook(io.BytesIO(render_xlsx(detail)))
    assert {'Table','Records','Columns','Sources','StructureIssues'} == set(workbook.sheetnames)
    assert workbook['Records']['B2'].value == '=literal' and workbook['Records']['B2'].data_type == 's'
    assert workbook['Table']['B2'].value == '=literal' and workbook['Table']['B2'].data_type == 's'


def test_no_header_and_missing_cells_are_representable():
    detail = table_detail('<table><tr><td>T1</td><td>0</td></tr><tr><td>T2</td></tr></table>')
    table = build_table(detail)
    assert table['header_rows'] == 0 and len(table['records']) == 2
    assert table['records'][1]['values']['c1']['parsed']['type'] == 'missing'
    assert table['issue_counts'] == {'missing_cell':1}
    detail['cells'][1]['text']='';detail['cells'][1]['value_kind']='unreadable'
    unresolved=build_table(detail)
    assert unresolved['records'][0]['values']['c1']['parsed']['type']=='unreadable'
    assert unresolved['issue_counts']['unreadable_cell']==1


def test_legacy_header_metadata_survives_data_only_fragments():
    detail=table_detail('<table><tr><td>T1①∶1</td><td>60</td></tr></table>')
    detail['cells'][1]['header_path']=['高（毫米）']
    table=build_table(detail)
    assert table['header_rows']==0
    assert table['columns'][1]['header_path']==['高（毫米）']
    assert table['columns'][1]['unit']=='毫米'


def test_illustration_grids_are_classified_before_table_decoding():
    region={'category':'table','method':'grid_supplement','bbox':[10,20,110,100]}
    layout={'boxes':[{'label':'image','coordinate':[0,0,120,120]}]}
    kept,excluded=data_table_regions([region],layout,[],200)
    assert kept==[] and excluded[0]['reason']=='grid_in_illustration'
    caption=[{'text':'附表一 数据统计','bbox':[10,10,110,18]}]
    assert data_table_regions([region],layout,caption,200)[0]==[region]
    neural={**region,'method':'neural_layout'}
    assert data_table_regions([neural],layout,[],200)[0]==[neural]


def test_large_table_bands_keep_merged_cells_whole():
    cells=[{'row':row,'column':column,'row_span':1,'column_span':1} for row in range(30) for column in range(10)]
    cells=[cell for cell in cells if cell['column']!=0 or cell['row'] not in {8,9,10,11}]
    cells.append({'row':8,'column':0,'row_span':4,'column_span':1})
    bands=table_row_bands(cells,limit=100)
    assert bands[0]==(0,8)
    assert bands[-1][1]==30
    assert all(not cell['row']<stop<cell['row']+cell['row_span'] for cell in cells for _,stop in bands)


def test_table_notes_remain_in_layout_and_out_of_data_records():
    detail=table_detail('<table><tr><td>地层单位</td><td>数量</td></tr><tr><td>T1</td><td>2</td></tr>'
                        '<tr><td colspan="2">注：圈码表示钱币数量。</td></tr></table>')
    table=build_table(detail)
    assert len(table['records'])==1
    assert table['notes'][0]['text']=='注：圈码表示钱币数量。'
    assert '<tfoot>' in render_html(detail).decode()


def test_matrix_table_identifiers_can_be_column_headers():
    detail=table_detail('<table><tr><td>地层</td><td>T1</td><td>T2</td></tr>'
                        '<tr><td>第一组</td><td>5</td><td>7</td></tr></table>')
    table=build_table(detail)
    assert table['header_rows']==1
    assert [column['label'] for column in table['columns']]==['地层','T1','T2']
    assert len(table['records'])==1


def test_large_table_recognition_restores_global_rows_and_source_boxes(tmp_path):
    import numpy as np
    from src.graphics.worker import cell_content
    class Models:
        calls=0
        def recognize(self,image,*,task,output_dir):
            rows=16 if self.calls==0 else 14
            self.calls+=1
            html='<table>'+''.join('<tr>'+'<td>value</td>'*10+'</tr>' for _ in range(rows))+'</table>'
            return {'blocks':[{'label':'table','text':html}]}
    image=np.full((300,100,3),255,np.uint8)
    cells=[{'row':row,'column':column,'row_span':1,'column_span':1,'bbox':[column*10,row*10,(column+1)*10,(row+1)*10]}
           for row in range(30) for column in range(10)]
    models=Models();decoded=cell_content(models,image,cells,tmp_path)
    assert models.calls==2 and len(decoded)==300
    assert decoded[-1]['row']==29 and decoded[-1]['bbox']==[90,290,100,300]
    assert all(cell['geometry_level']=='cell' and cell['source_band']['row_count_matches'] for cell in decoded)


def test_continuations_match_equivalent_printed_number_styles():
    assert table_label_key('附表一二')==table_label_key('附表十二')==table_label_key('附表12')
    assert table_label_key('附表一〇')==table_label_key('附表十')
    a={'manifest_id':'source','namespace':'source','category':'table','label':'附表一二','page':1,
       'signature':{'columns':2,'edges':[0,.5],'headers':['地层','数量']}}
    b={**a,'label':'附表十二','page':2,'continuation':True}
    assert continuation_evidence(a,b)['accepted']


def test_below_figure_caption_and_multiple_image_boxes_support_classification():
    region={'category':'table','method':'grid_supplement','bbox':[0,0,100,100]}
    layout={'boxes':[{'label':'image','coordinate':[0,0,50,100]}, {'label':'image','coordinate':[50,0,100,100]}]}
    assert boxes_coverage(region['bbox'],[box['coordinate'] for box in layout['boxes']])==1
    assert boxes_coverage(region['bbox'],[[0,0,50,100],[0,0,50,100]])==.5
    lines=[{'text':'图四三 铜钱拓片','bbox':[0,103,100,110]}]
    assert caption_for(region['bbox'],lines,200)['label']=='图四三'
    kept,excluded=data_table_regions([region],layout,lines,200)
    assert kept==[] and excluded[0]['figure_coverage']==1


def test_late_roman_ocr_error_and_kiln_suffix_do_not_extend_headers():
    detail=table_detail('<table><tr><td rowspan="2" colspan="2">探方地层型式</td><td>T1</td><td>T2</td></tr>'
                        '<tr><td>2</td><td>3</td></tr>'
                        '<tr><td>I</td><td>1</td><td>△</td><td></td></tr>'
                        '<tr><td>X1</td><td>2</td><td></td><td>△</td></tr></table>')
    assert infer_header_rows(detail['cells'])==2
    coins=table_detail('<table><tr><td>地层单位</td><td>名称及数量</td></tr>'
                       '<tr><td>Y1烟</td><td>开元通宝①</td></tr><tr><td>T2③</td><td>通宝②</td></tr></table>')
    assert infer_header_rows(coins['cells'])==1


def test_missing_diagonal_corner_does_not_turn_subheaders_into_records():
    cells=[{'row':0,'column':2,'row_span':1,'column_span':2},
           {'row':1,'column':2,'row_span':1,'column_span':1},
           {'row':1,'column':3,'row_span':1,'column_span':1},
           {'row':2,'column':0,'row_span':3,'column_span':1}]
    assert infer_header_rows(cells)==2


def test_diagonal_corner_requires_source_boundaries():
    import cv2
    import numpy as np
    from src.graphics.vision import diagonal_header_cell
    binary=np.zeros((200,300),np.uint8)
    cv2.rectangle(binary,(10,10),(110,90),255,2)
    cv2.line(binary,(10,10),(110,90),255,2)
    cv2.line(binary,(10,10),(110,50),255,2)
    cells=[{'row':0,'column':2},{'row':1,'column':2},{'row':2,'column':0}]
    corner=diagonal_header_cell(cells,binary,[10,60,110,210,290],[10,50,90,190],0)
    assert corner['row_span']==corner['column_span']==2
    assert corner['bbox']==[10,10,110,90]
    assert corner['geometry_level']=='grid_axes' and corner['diagonal_header']
    binary[20:85,7:14]=0
    assert diagonal_header_cell(cells,binary,[10,60,110,210,290],[10,50,90,190],0) is None


@pytest.mark.parametrize('rows',[8,22])
def test_sparse_symbol_matrix_uses_physical_positions(tmp_path,rows):
    import cv2
    import numpy as np
    from src.graphics.worker import cell_content
    image=np.full((rows*40,10*40,3),255,np.uint8)
    cells=[{'row':row,'column':column,'row_span':1,'column_span':1,
            'bbox':[column*40,row*40,(column+1)*40,(row+1)*40],'method':'grid'}
           for row in range(rows) for column in range(10)]
    for row in range(1,rows):
        x=row%2*160+80;y=row*40
        cv2.polylines(image,[np.array([[x+20,y+8],[x+8,y+30],[x+32,y+30]])],True,(0,0,0),2)
    cv2.putText(image,'I',(12,105),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),1)
    class Models:
        table_calls=0
        def recognize(self,image,*,task,output_dir):
            if task=='table':
                self.table_calls+=1
                return {'blocks':[{'label':'table','text':'<table><tr>'+'<td>字段</td>'*10+'</tr></table>'}]}
            return {'blocks':[{'label':'text','text':'Ⅰ'}]}
    models=Models();decoded=cell_content(models,image,cells,tmp_path)
    assert models.table_calls==1 and len(decoded)==rows*10
    assert all(cell['source_matrix'] and cell['geometry_level']=='cell' for cell in decoded)
    for row in range(1,rows):
        column=row%2*4+2
        cell=next(cell for cell in decoded if cell['row']==row and cell['column']==column)
        assert cell['text']=='△' and cell['method']=='source_symbol_observation'
    assert next(cell for cell in decoded if cell['row']==1 and cell['column']==0)['text']==''


def test_span_disagreement_keeps_measured_cells_and_marks_inferred_boxes(tmp_path):
    import numpy as np
    from src.graphics.worker import cell_content
    cells=[{'row':row,'column':column,'row_span':1,'column_span':1,'bbox':[column*40,row*40,(column+1)*40,(row+1)*40]}
           for row in range(3) for column in range(2) if column==0 or row==0]
    cells.append({'row':1,'column':1,'row_span':2,'column_span':1,'bbox':[40,40,80,120]})
    class Models:
        def recognize(self,image,*,task,output_dir):
            return {'blocks':[{'label':'table','text':'<table><tr><td>编号</td><td>名称</td></tr>'
                              '<tr><td>T1</td><td>甲</td></tr><tr><td>T2</td><td>乙</td></tr></table>'}]}
    decoded=cell_content(Models(),np.full((120,80,3),255,np.uint8),cells,tmp_path)
    assert sum(cell['geometry_level']=='cell' for cell in decoded)==4
    inferred=[cell for cell in decoded if cell['geometry_level']=='grid_axes']
    assert [cell['bbox'] for cell in inferred]==[[40,40,80,80],[40,80,80,120]]


def test_matrix_header_alignment_uses_complete_row_order(tmp_path):
    import cv2
    import numpy as np
    from src.graphics.worker import cell_content
    image=np.full((400,320,3),255,np.uint8)
    cells=[{'row':0,'column':0,'row_span':2,'column_span':2,'bbox':[0,0,80,80],'method':'grid'}]
    cells.extend({'row':0,'column':column,'row_span':1,'column_span':2,
                  'bbox':[column*40,0,(column+2)*40,40],'method':'grid'} for column in (2,4,6))
    cells.extend({'row':row,'column':column,'row_span':1,'column_span':1,
                  'bbox':[column*40,row*40,(column+1)*40,(row+1)*40],'method':'grid'}
                 for row in range(1,10) for column in range(2 if row==1 else 0,8))
    for row in range(2,10):
        y=row*40
        cv2.polylines(image,[np.array([[100,y+8],[88,y+30],[112,y+30]])],True,(0,0,0),2)
    class Models:
        def recognize(self,image,*,task,output_dir):
            if task=='table':
                html='<table><tr><td rowspan="2">axes</td><td colspan="2">T1</td><td colspan="2">T2</td><td colspan="2">T3</td></tr><tr>'
                html+=''.join(f'<td>{number}</td>' for number in range(6))+'</tr></table>'
                return {'blocks':[{'label':'table','text':html}]}
            return {'blocks':[{'label':'text','text':'crop'}]}
    decoded=cell_content(Models(),image,cells,tmp_path)
    leaves=[cell for cell in decoded if cell['row']==1]
    assert [cell['column'] for cell in leaves]==list(range(2,8))
    assert [cell['text'] for cell in leaves]==list(map(str,range(6)))
    assert all(cell['method']=='matrix_header_row_alignment' for cell in leaves)
    assert all(cell['text']=='crop' for cell in decoded if cell['row']==0)


def test_repeated_type_group_readings_are_flagged_with_sources():
    detail=table_detail('<table><tr><td>型式</td><td>子型</td></tr>'
                        '<tr><td rowspan="2">I</td><td>1</td></tr><tr><td>2</td></tr>'
                        '<tr><td rowspan="2">Ⅰ</td><td>1</td></tr><tr><td>2</td></tr></table>')
    detail['title']='器物型式对照表'
    table=build_table(detail)
    assert table['issue_counts']=={'row_group_label_conflict':1}
    assert table['issues'][0]['cell_ids']==['cell-2','cell-5']
    assert 'row_group_label_conflict' in table['semantic_review']['reasons']
    assert detail['cells'][5]['text']=='Ⅰ'


def test_semantic_opinions_are_source_quoted_and_cached(tmp_path):
    detail = table_detail('<table><tr><td>地层单位</td><td>名称及数量</td></tr><tr><td>T1②</td><td>通宝①</td></tr></table>')
    class Endpoint:
        model = 'local-test'
        identity = {'endpoint':'http://127.0.0.1:18081/v1','model':model}
        def complete(self, messages, **options):
            return {'choices':[{'message':{'content':json.dumps({'issues':[{
                'cell_id':'c3','quote':'通宝①','suggestion':'interpret_as_count'}]})}}]}
    result = review_table_semantics(detail,Endpoint(),tmp_path)
    assert result['applied'] is False and result['requires_review'] is True
    assert result['issues'][0]['cell_id']=='cell-3'
    assert review_table_semantics(detail,Endpoint(),tmp_path)['cached'] is True
    assert detail['cells'][3]['text'] == '通宝①'
    for path in tmp_path.glob('*.json'):
        path.write_text(json.dumps({'issues':[{'cell_id':'c3',
                                             'quote':'不存在的原文','suggestion':'interpret_as_count'}]}),encoding='utf8')
    with pytest.raises(ValueError, match='does not quote'):
        review_table_semantics(detail,Endpoint(),tmp_path)


def test_automatic_semantic_review_covers_every_relevant_cell_in_batches(tmp_path):
    from src.graphics.refinement import review_table_semantics_all
    detail=table_detail('<table><tr><td>地层单位</td><td>名称及数量</td></tr>'+''.join(
        f'<tr><td>T{index}</td><td>通宝①</td></tr>' for index in range(1,102))+'</table>')
    class Endpoint:
        model='local-test'
        identity={'endpoint':'http://127.0.0.1:18081/v1','model':model}
        def __init__(self):self.observed=[]
        def complete(self,messages,**options):
            payload=json.loads(messages[1]['content'])
            self.observed.extend(cell['text'] for cell in payload['cells'])
            return {'choices':[{'message':{'content':'{"issues":[]}'}}]}
    endpoint=Endpoint()
    result=review_table_semantics_all(detail,endpoint,tmp_path)
    assert result['batches']==2
    assert result['sampled_cells']==103
    assert len(set(result['reviewed_cell_ids']))==103
    assert endpoint.observed.count('通宝①')==101
    assert review_table_semantics_all(detail,endpoint,tmp_path)['cached']
