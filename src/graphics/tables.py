"""Source-backed table headers, book layout and logical data records."""
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
import re
from statistics import median
import unicodedata


IDENTIFIER = re.compile(r'(?:(?:[A-Za-z]+\d+[A-Za-z]?|[THYMFQZGDJL]\d+(?:[HYMFZGD]\d+)+)(?:扩|烟|火)?[①-⑳㉑-㊿０-９]*[a-z]?(?:[.:：∶-]\d+[A-Za-z]?)?|[甲乙丙丁戊己庚辛壬癸]区)',re.I)
HEADER_TERMS = re.compile(r'编号|名称|尺寸|数量|面积|深度|地层|层位|器类|型式|形状|单位|时代|备注|组别|时期|探方|类型|样品|年代|釉色|装饰|化学成分|unit|type|size|area|depth|quantity|name',re.I)
NUMBER = r'[+-]?(?:\d+(?:\.\d+)?|\.\d+)'
UNITS = re.compile(r'平方(?:米|厘米|毫米)|[cm毫厘分]?米|mm²|cm²|m²|mm2|cm2|m2|mm|cm|m|千克|公斤|克|件|枚|个|%')
CIRCLED = {chr(0x2460 + index): index + 1 for index in range(20)}


def cell_text(cell):
    return cell.get('text', cell.get('raw_text', '')) or ''


def infer_header_rows(cells):
    """Infer a leading header band using explicit markup and leading identifiers."""
    if not cells:
        return 0
    explicit = [cell for cell in cells if cell.get('header_section') == 'thead']
    if explicit:
        return max(cell['row'] + cell.get('row_span', 1) for cell in explicit)
    rows = sorted({cell['row'] for cell in cells})
    marked_depth = 0
    for row_index in rows:
        row = [cell for cell in cells if cell['row'] == row_index]
        if row and all(cell.get('header') and cell.get('scope') not in {'row', 'rowgroup'} for cell in row):
            marked_depth = max(marked_depth, max(cell['row'] + cell.get('row_span', 1) for cell in row))
        else:
            break
    if marked_depth:
        return marked_depth
    first = [cell for cell in cells if cell['row'] == rows[0]]
    first_identifiers = [cell for cell in first if IDENTIFIER.fullmatch(re.sub(r'\s+', '', cell_text(cell)))]
    matrix_header = len(first_identifiers) >= 2 and len(first_identifiers) >= len(first) / 2
    matrix_header = matrix_header or bool(first_identifiers and any(
        re.search(r'层位|地层|探方|器类|组别|型式', cell_text(cell)) for cell in first if cell['column'] == 0))
    if not matrix_header and any(cell['column'] <= 1 for cell in first_identifiers):
        return rows[0]
    grouped = any(cell.get('column_span', 1) > 1 for cell in first)
    has_header_words = any(HEADER_TERMS.search(cell_text(cell)) for cell in first)
    depth = rows[0] + max(cell.get('row_span', 1) for cell in first) if grouped or matrix_header or has_header_words else rows[0] + 1
    if grouped and min(cell['column'] for cell in first)>0:
        leading_rows=[cell['row'] for cell in cells if cell['column']==0]
        if leading_rows and min(leading_rows)<=depth+1:
            depth=max(depth,min(leading_rows))
    for row_index in rows:
        if row_index >= depth:
            break
        row = [cell for cell in cells if cell['row'] == row_index]
        if any(cell.get('column_span', 1) > 1 and cell.get('row_span', 1) == 1
               and HEADER_TERMS.search(cell_text(cell)) for cell in row):
            depth = max(depth, row_index + 2)
    # Body identifiers provide a boundary near the geometric header band. A
    # late OCR reading such as XII -> X1 cannot turn the data into headers.
    for row_index in rows:
        if row_index <= rows[0] or row_index > depth:
            continue
        row = [cell for cell in cells if cell['row'] == row_index]
        if any(IDENTIFIER.fullmatch(re.sub(r'\s+', '', cell_text(cell))) for cell in row if cell['column'] <= 1):
            return row_index
        if any(cell.get('value_kind') == 'symbol' or cell_text(cell).strip() in {'△','○','〇'} for cell in row):
            return row_index
    return min(rows[-1] + 1, depth)


def header_paths(cells, count, columns):
    paths = []
    for column in range(columns):
        values = []
        for cell in sorted(cells, key=lambda item: (item['row'], item['column'])):
            if cell['row'] >= count or not cell['column'] <= column < cell['column'] + cell.get('column_span', 1):
                continue
            value = cell_text(cell).strip()
            if value and (not values or values[-1] != value):
                values.append(value)
        paths.append(values)
    return paths


def column_widths(cells, columns):
    """Use measured cell boundaries, interpolating only unobserved boundaries."""
    boundaries = defaultdict(list)
    for cell in cells:
        box = cell.get('bbox')
        if not box or cell.get('geometry_level', 'cell') != 'cell':
            continue
        boundaries[cell['column']].append(box[0])
        boundaries[cell['column'] + cell.get('column_span', 1)].append(box[2])
    edges = {index: median(values) for index, values in boundaries.items()}
    if 0 not in edges or columns not in edges or edges[columns] <= edges[0]:
        return [1 / max(1, columns)] * columns, 'equal_width_fallback'
    observed = sorted(edges)
    for left, right in zip(observed, observed[1:]):
        for index in range(left + 1, right):
            edges[index] = edges[left] + (edges[right] - edges[left]) * (index - left) / (right - left)
    widths = [edges[index + 1] - edges[index] for index in range(columns)]
    if min(widths, default=0) <= 0:
        return [1 / max(1, columns)] * columns, 'equal_width_fallback'
    total = sum(widths)
    return [round(width / total, 6) for width in widths], 'measured_cell_boundaries'


def table_row_bands(cells, limit=160):
    """Choose complete row bands without cutting measured merged cells."""
    if not cells:
        return []
    end=max(cell['row']+cell.get('row_span',1) for cell in cells)
    boundaries=[row for row in range(end+1) if not any(
        cell['row']<row<cell['row']+cell.get('row_span',1) for cell in cells)]
    bands=[];start=0
    while start<end:
        choices=[row for row in boundaries if row>start]
        fitting=[row for row in choices if sum(start<=cell['row']<row for cell in cells)<=limit]
        stop=max(fitting) if fitting else min(choices)
        bands.append((start,stop));start=stop
    return bands


def parsed_value(text, column):
    compact = re.sub(r'\s+', '', text)
    if not compact:
        return {'type': 'blank', 'value': None}
    if compact in {'—', '–', '-', '△', '○', '〇', '/', '〃', '同上'}:
        return {'type': 'symbol', 'value': compact}
    if column['role'] == 'identifier' or IDENTIFIER.fullmatch(compact):
        return {'type': 'identifier', 'value': text}
    dimensions = re.fullmatch(rf'({NUMBER})[×xX*]({NUMBER})(?:[×xX*]({NUMBER}))?', compact)
    if dimensions:
        return {'type': 'dimensions', 'value': [value for value in dimensions.groups() if value is not None],
                'unit': column['unit']}
    if re.fullmatch(NUMBER, compact):
        try:
            return {'type': 'decimal', 'value': str(Decimal(compact)), 'unit': column['unit']}
        except InvalidOperation:
            pass
    interval = re.fullmatch(rf'({NUMBER})[～~至—]({NUMBER})', compact)
    if interval:
        return {'type': 'range', 'value': list(interval.groups()), 'unit': column['unit']}
    if '数量' in ''.join(column['header_path']):
        pieces = [piece.strip() for piece in re.split(r'[，,、；;\n]+', text) if piece.strip()]
        pairs = []
        for piece in pieces:
            match = re.fullmatch(r'(.+?)([①-⑳]|\d+)(?:枚|件|个)?', piece)
            if not match:
                return {'type': 'text', 'value': text}
            name, marker = match.groups()
            pairs.append({'name': name, 'quantity': CIRCLED[marker] if marker in CIRCLED else int(marker),
                          'marker': marker, 'status': 'candidate'})
        if pairs:
            return {'type': 'quantity_list', 'value': pairs, 'interpretation_status': 'candidate'}
    return {'type': 'text', 'value': text}


def build_table(detail):
    cells = [cell for cell in detail['cells'] if not cell.get('repeated_header') and not cell.get('joined_into')]
    columns_count = max((cell['column'] + cell.get('column_span', 1) for cell in cells), default=0)
    rows_count = max((cell['row'] + cell.get('row_span', 1) for cell in cells), default=0)
    if rows_count > 10000 or columns_count > 1000 or rows_count * columns_count > 500000:
        raise ValueError('Table dimensions exceed supported source grid size')
    parts = detail.get('fragments', [])
    first_id = parts[0]['id'] if parts else None
    first_cells = [cell for cell in cells if first_id is None or cell.get('fragment_id') == first_id]
    count = parts[0].get('table_header_rows', infer_header_rows(first_cells)) if parts else infer_header_rows(first_cells)
    paths = header_paths(first_cells, count, columns_count)
    if count==0:
        for index in range(columns_count):
            stored=[cell.get('header_path',[]) for cell in first_cells if cell['column']==index and cell.get('header_path')]
            if stored and all(path==stored[0] for path in stored):
                paths[index]=list(stored[0])
    widths, width_method = column_widths(first_cells, columns_count)
    issues = []
    columns = []
    for index, path in enumerate(paths):
        label = ' / '.join(path) or f'列{index + 1}'
        matches = UNITS.findall(' '.join(path))
        unit = matches[-1] if matches else None
        if unit and '平方' in unit and path and re.search(r'尺寸|长度|宽度|高度|深度', path[-1]):
            issues.append({'code': 'unit_scope', 'message': '分组表头的面积单位与子字段尺寸含义需要核对',
                           'column': index, 'cell_ids': []})
            unit = None
        role = 'identifier' if re.search(r'编号|器号|标本号|地层单位', ' '.join(path)) else 'value'
        columns.append({'key': f'c{index}', 'index': index, 'header_path': path, 'label': label,
                        'unit': unit, 'unit_candidates': matches, 'role': role, 'width': widths[index]})
    occupancy = {}
    for cell in cells:
        for row in range(cell['row'], cell['row'] + cell.get('row_span', 1)):
            for column in range(cell['column'], cell['column'] + cell.get('column_span', 1)):
                if (row, column) in occupancy:
                    issues.append({'code': 'overlap', 'message': '单元格占位重叠', 'row': row, 'column': column,
                                   'cell_ids': [occupancy[row, column]['id'], cell['id']]})
                else:
                    occupancy[row, column] = cell
    part_by_id = {part['id']: part for part in parts}
    header_rows_set = set()
    for cell in cells:
        part = part_by_id.get(cell.get('fragment_id'), {})
        header_count = part.get('table_header_rows', count)
        source_row = cell.get('source_row', cell['row'])
        if source_row < header_count:
            header_rows_set.update(range(cell['row'], cell['row'] + min(cell.get('row_span', 1),header_count-source_row)))
            if source_row+cell.get('row_span',1)>header_count:
                issues.append({'code':'header_crossing','message':'合并单元格跨越了当前表头范围，需要核对行跨度或表头行数',
                               'cell_ids':[cell['id']]})
    for cell in cells:
        cell['header_path'] = paths[cell['column']] if cell['column'] < len(paths) else []
        cell['table_role'] = 'column_header' if cell['row'] in header_rows_set else 'data'
    notes=[];note_rows=set()
    for cell in cells:
        full_width=cell['column']==0 and cell.get('column_span',1)==columns_count
        if cell.get('header_section')=='tfoot' or full_width and re.match(r'^(?:注[：:、.．\s]|备注|说明|资料来源|Notes?\b)',cell_text(cell).strip(),re.I):
            cell['table_role']='note'
            note_rows.update(range(cell['row'],cell['row']+cell.get('row_span',1)))
            notes.append({'cell_id':cell['id'],'text':cell_text(cell),'anchors':cell.get('anchors',[]),'row':cell['row']})
    records = []
    sizes=[column for column in columns if column['header_path'] and '布方尺寸' in column['header_path'][-1]]
    areas=[column for column in columns if column['header_path'] and '面积' in column['header_path'][-1]
           and '尺寸' not in column['header_path'][-1]]
    for row in range(rows_count):
        if row in header_rows_set or row in note_rows:
            continue
        values = {}
        source_pages = set()
        for column in columns:
            cell = occupancy.get((row, column['index']))
            if cell is None:
                values[column['key']] = {'text': None, 'parsed': {'type': 'missing', 'value': None},
                                          'cell_id': None, 'anchors': [], 'inherited': False}
                issues.append({'code': 'missing_cell', 'message': '该行缺少物理单元格', 'row': row,
                               'column': column['index'], 'cell_ids': []})
                continue
            source_pages.update(anchor.get('scan_index') for anchor in cell.get('anchors', []) if anchor.get('scan_index') is not None)
            text = cell_text(cell)
            parsed=parsed_value(text,column)
            if not text.strip() and cell.get('value_kind')=='unreadable':
                parsed={'type':'unreadable','value':None,'symbols':cell.get('symbols',[])}
            values[column['key']] = {'text': text, 'raw_text': cell.get('raw_text', text),
                                      'parsed': parsed, 'cell_id': cell['id'],
                                      'source_cell_ids': cell.get('source_cells', [cell['id']]),
                                      'anchors': cell.get('anchors', []),
                                      'inherited': row != cell['row'], 'column_merged': cell.get('column_span', 1) > 1}
        records.append({'row': row, 'pages': sorted(source_pages), 'values': values})
        for size_column in sizes:
            size=values[size_column['key']]
            if size['parsed']['type']!='dimensions' or len(size['parsed']['value'])!=2:
                continue
            product=Decimal(size['parsed']['value'][0])*Decimal(size['parsed']['value'][1])
            for area_column in areas:
                area=values[area_column['key']]
                if area['parsed']['type']=='decimal' and product<Decimal(area['parsed']['value']):
                    issues.append({'code':'area_dimension_difference','message':'发掘面积超过布方尺寸乘积，需要结合原书核对',
                                   'row':row,'cell_ids':[size['cell_id'],area['cell_id']],
                                   'dimension_product':str(product),'area_text':area['text'],'status':'candidate'})
    table_geometry = [cell['id'] for cell in cells if cell.get('geometry_level') == 'table']
    unreadable=[cell['id'] for cell in cells if not cell_text(cell).strip() and cell.get('value_kind')=='unreadable']
    if unreadable:
        issues.append({'code':'unreadable_cell','message':'部分单元格有墨迹且文字尚未读出，需要核对原图', 'cell_ids':unreadable})
    if table_geometry:
        issues.append({'code': 'table_geometry', 'message': '部分单元格只有整表级定位，需要核对单元格边界',
                       'cell_ids': table_geometry})
    inferred_geometry=[cell['id'] for cell in cells if cell.get('geometry_level')=='grid_axes']
    if inferred_geometry:
        issues.append({'code':'inferred_cell_geometry','message':'部分单元格边界由同表网格轴线推定，需要核对合并关系',
                       'cell_ids':inferred_geometry})
    band_mismatch=[cell['id'] for cell in cells if cell.get('source_band') and not cell['source_band']['row_count_matches']]
    if band_mismatch:
        issues.append({'code':'band_row_mismatch','message':'分段识别的行数与物理网格有差异，需要核对行列关系',
                       'cell_ids':band_mismatch})
    if detail.get('structure', {}).get('table_structure_pending'):
        issues.append({'code': 'fragment_structure_pending', 'message': '片段的行列关系需要核对', 'cell_ids': []})
    if not records and cells:
        issues.append({'code': 'no_data_rows', 'message': '表头范围覆盖了全部行，需要核对表头行数', 'cell_ids': []})
    if '型式' in (detail.get('title','')+' '.join(' '.join(path) for path in paths)):
        groups=defaultdict(list)
        for cell in cells:
            marker=re.sub(r'\s+','',unicodedata.normalize('NFKC',cell_text(cell))).upper()
            if cell['column']==0 and cell['row'] not in header_rows_set and cell.get('row_span',1)>1 and re.fullmatch(r'[IVXLCDM1]+',marker):
                groups[cell.get('fragment_id'),marker].append(cell['id'])
        for (_,marker),ids in groups.items():
            if len(ids)>1:
                issues.append({'code':'row_group_label_conflict','message':'同页多个型式分组读出了相同标记，需要核对原书字形或分组关系',
                               'cell_ids':ids,'marker':marker})
    semantic_codes = {'unit_scope', 'no_data_rows','row_group_label_conflict'}
    semantic_reasons = sorted({issue['code'] for issue in issues if issue['code'] in semantic_codes})
    quantity_candidates = sum(value['parsed']['type'] == 'quantity_list' for record in records for value in record['values'].values())
    if quantity_candidates:
        semantic_reasons.append('quantity_marker_meaning')
    return {'schema_version': 1, 'header_rows': count, 'rows': rows_count, 'column_count': columns_count,
            'columns': columns, 'column_width_method': width_method, 'records': records, 'notes':notes,'issues': issues,
            'issue_counts': dict(Counter(issue['code'] for issue in issues)),
            'semantic_review': {'enabled': False, 'recommended': bool(semantic_reasons), 'reasons': semantic_reasons,
                                'source_cell_ids_required': True, 'status': 'pending' if semantic_reasons else 'rules_sufficient'}}
