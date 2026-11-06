"""Geometric supplements to neural layout/table models, with explicit uncertainty."""
from html.parser import HTMLParser
import re
import cv2
import numpy as np
from src.graphics.geometry import iou


def binary_grid(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    binary=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,25,12)
    h,w=binary.shape
    horizontal=cv2.morphologyEx(binary,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(max(15,w//40),1)))
    vertical=cv2.morphologyEx(binary,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(1,max(15,h//40))))
    # Text strokes must not turn into miniature cells. Keep long, thin connected
    # rules, then close scan gaps. Both sides of a thick rule share one axis.
    def long_rules(mask,axis,minimum):
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        count,labels,stats,_=cv2.connectedComponentsWithStats(mask)
        clean=np.zeros_like(mask)
        for i in range(1,count):
            _,_,rw,rh,_=stats[i]
            if (rw if axis==0 else rh)>=minimum and (rh if axis==0 else rw)<=max(15,minimum*.15):clean[labels==i]=255
        return clean
    horizontal=long_rules(horizontal,0,max(45,w*.07))
    vertical=long_rules(vertical,1,max(40,h*.06))
    grid=cv2.bitwise_or(horizontal,vertical)
    grid=cv2.morphologyEx(grid,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    return binary,horizontal,vertical,grid


def grid_regions(image):
    _,horizontal,vertical,grid=binary_grid(image)
    contours,_=cv2.findContours(grid,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    h,w=grid.shape;regions=[]
    for c in contours:
        x,y,rw,rh=cv2.boundingRect(c)
        if rw>w*.12 and rh>h*.035 and rw*rh>w*h*.004:
            v=vertical[y:y+rh,x:x+rw];hh=horizontal[y:y+rh,x:x+rw]
            intersections=cv2.countNonZero(cv2.bitwise_and(v,hh))
            if intersections>12:
                regions.append({"bbox":[x,y,x+rw,y+rh],"label":"table","score":.65,"method":"grid_supplement"})
    return regions


def combine_regions(neural,grid):
    regions=[]
    for region in neural:
        label=region["label"].lower()
        if label in {"table","chart","image","figure","figure_table"}:
            regions.append({**region,"category":"table" if label=="table" else "chart" if label=="chart" else "figure"})
    for region in grid:
        matches=[r for r in regions if r["category"]=="table" and iou(r["bbox"],region["bbox"])>.5]
        if matches:
            for match in matches:
                a,b=match['bbox'],region['bbox']
                match['bbox']=[min(a[0],b[0]),min(a[1],b[1]),max(a[2],b[2]),max(a[3],b[3])]
            continue
        regions.append({**region,"category":"table"})
    # A full-page neural box may contain two independent tables. Distinct grid
    # components win only when both are substantial, disjoint table regions.
    result=[]
    for region in regions:
        inside=[g for g in grid if g["bbox"][0]>=region["bbox"][0]-10 and g["bbox"][1]>=region["bbox"][1]-10
            and g["bbox"][2]<=region["bbox"][2]+10 and g["bbox"][3]<=region["bbox"][3]+10]
        if region["method"]!="grid_supplement" and region["category"]=="table" and len(inside)>=2:
            for g in inside:
                if not any(iou(g["bbox"],r["bbox"])>.8 for r in result):
                    result.append({**g,"category":"table","evidence":{"neural_parent":region}})
        elif not any(iou(region["bbox"],r["bbox"])>.8 and region["category"]==r["category"] for r in result):
            result.append(region)
    return sorted(result,key=lambda r:(r["bbox"][1],r["bbox"][0]))


def cluster(values,tolerance=4):
    groups=[]
    for value in sorted(values):
        if groups and value-np.mean(groups[-1])<=tolerance:groups[-1].append(value)
        else:groups.append([value])
    return [int(round(np.mean(g))) for g in groups]


def supported_vertical_rules(binary,fallback):
    """Track repeated long rules through mild scanner shear without erasing them.

    A tall connected-component requirement loses broken thin lines. Projection
    is used only after measuring shear, and only when two support thresholds
    agree. Pixels are filtered, never drawn across missing source content.
    """
    h,w=binary.shape
    raw=cv2.morphologyEx(binary,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(1,max(15,h//60))))
    lines=cv2.HoughLinesP(raw,1,np.pi/1800,threshold=80,minLineLength=max(40,int(h*.12)),maxLineGap=40)
    slopes=[]
    for x0,y0,x1,y1 in np.asarray(lines).reshape(-1,4) if lines is not None else []:
        if y1!=y0 and abs((x1-x0)/(y1-y0))<.05:slopes.append((x1-x0)/(y1-y0))
    if len(slopes)<4:return fallback,None,0.
    slope=float(np.median(slopes));matrix=np.float32([[1,-slope,slope*h/2],[0,1,0]])
    aligned=cv2.warpAffine(raw,matrix,(w,h),flags=cv2.INTER_NEAREST)
    profile=np.count_nonzero(aligned,axis=0)
    def peaks(fraction):
        groups=[]
        for x in np.where(profile>=max(20,h*fraction))[0]:
            if groups and x-groups[-1][-1]<=4:groups[-1].append(int(x))
            else:groups.append([int(x)])
        return [int(round(np.mean(g))) for g in groups]
    supported=None
    # Short continuation fragments can contain more ink than grid. Raise both
    # thresholds together until stable, rather than accepting text stems.
    for lower,upper in [(.2,.3),(.3,.4),(.4,.5),(.5,.6)]:
        low,high=peaks(lower),peaks(upper)
        if len(high)<3 or abs(len(low)-len(high))>1:continue
        if any(not any(abs(a-b)<=max(8,w*.005) for b in low) for a in high):continue
        supported=high;break
    if supported is None:return fallback,None,0.
    high=supported
    bands=np.zeros_like(raw);radius=max(5,int(w*.003))
    for x in high:bands[:,max(0,x-radius):min(w,x+radius+1)]=255
    support=cv2.warpAffine(bands,cv2.invertAffineTransform(matrix),(w,h),flags=cv2.INTER_NEAREST)
    return cv2.bitwise_and(raw,support),high,slope


def recover_grid_cells(image):
    binary,horizontal,vertical,grid=binary_grid(image)
    vertical,supported_x,slope=supported_vertical_rules(binary,vertical)
    grid=cv2.morphologyEx(cv2.bitwise_or(horizontal,vertical),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    contours,hierarchy=cv2.findContours(grid,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:return []
    h,w=grid.shape;boxes=[]
    for i,c in enumerate(contours):
        x,y,rw,rh=cv2.boundingRect(c)
        if hierarchy[0][i][3]<0 or rw<8 or rh<8 or rw*rh>h*w*.9:continue
        if cv2.contourArea(c)<rw*rh*.65:continue  # diagonal header, never invent a rectangular cell
        box=[x,y,x+rw,y+rh]
        if not any(iou(box,b)>.8 for b in boxes):boxes.append(box)
    if len(boxes)<4:return []
    def axes(mask,axis):
        count,_,stats,_=cv2.connectedComponentsWithStats(mask)
        values=[float(s[axis]+s[axis+2]/2) for s in stats[1:]]
        return cluster(values,max(7,(w if axis==0 else h)*.003))
    xs=supported_x if supported_x is not None else axes(vertical,0);ys=axes(horizontal,1)
    if len(xs)<2 or len(ys)<2:return []
    cells=[]
    for b in boxes:
        shift=slope*((b[1]+b[3])/2-h/2)
        c0,c1=[min(range(len(xs)),key=lambda j:abs(xs[j]-(v-shift))) for v in (b[0],b[2])]
        r0,r1=[min(range(len(ys)),key=lambda j:abs(ys[j]-v)) for v in (b[1],b[3])]
        if c1>c0 and r1>r0:
            cells.append({"row":r0,"column":c0,"row_span":r1-r0,"column_span":c1-c0,"bbox":b,"method":"grid"})
    unique={}
    for c in cells:
        key=(c['row'],c['column'])
        if key not in unique or c['row_span']*c['column_span']>unique[key]['row_span']*unique[key]['column_span']:unique[key]=c
    return sorted(unique.values(),key=lambda c:(c["row"],c["column"]))


class TableHTML(HTMLParser):
    def __init__(self):
        super().__init__();self.cells=[];self.row=-1;self.column=0;self.occupied=set();self.current=None
    def handle_starttag(self,tag,attrs):
        if tag=="tr":self.row+=1;self.column=0
        if tag not in {"td","th"}:return
        attrs=dict(attrs)
        while (self.row,self.column) in self.occupied:self.column+=1
        rs=max(1,min(1000,int(attrs.get("rowspan",1))));cs=max(1,min(1000,int(attrs.get("colspan",1))))
        self.current={"row":max(0,self.row),"column":self.column,"row_span":rs,"column_span":cs,"text":"","header":tag=="th"}
        for r in range(max(0,self.row),max(0,self.row)+rs):
            for c in range(self.column,self.column+cs):self.occupied.add((r,c))
        self.column+=cs
    def handle_data(self,data):
        if self.current is not None:self.current["text"]+=data
    def handle_endtag(self,tag):
        if tag in {"td","th"} and self.current is not None:
            self.cells.append(self.current);self.current=None


def parse_structure(tokens,boxes=None):
    parser=TableHTML();parser.feed("".join(tokens) if isinstance(tokens,list) else tokens)
    if boxes is not None and len(boxes)==len(parser.cells):
        for cell,box in zip(parser.cells,boxes):
            p=np.asarray(box,dtype=float).reshape(-1,2)
            cell["bbox"]=[float(p[:,0].min()),float(p[:,1].min()),float(p[:,0].max()),float(p[:,1].max())]
            cell["method"]="neural_structure"
    return parser.cells


def classify_ink(image):
    """Symbols are observations, never numerical or archaeological interpretations."""
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    margin=max(2,min(gray.shape)//12)
    if min(gray.shape)>margin*2+2:gray=gray[margin:-margin,margin:-margin]
    ink=cv2.threshold(gray,160,255,cv2.THRESH_BINARY_INV)[1]
    if cv2.countNonZero(ink)/max(1,ink.size)<.003:return "blank",[]
    contours,_=cv2.findContours(ink,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    symbols=[]
    for c in contours:
        x,y,w,h=cv2.boundingRect(c);area=cv2.contourArea(c);perim=cv2.arcLength(c,True)
        if w<5 or h<5 or area<15 or not .55<w/max(1,h)<1.6:continue
        poly=cv2.approxPolyDP(c,.045*perim,True)
        component_ink=cv2.countNonZero(ink[y:y+h,x:x+w])
        isolated=component_ink>=cv2.countNonZero(ink)*.65
        if isolated and len(poly)==3 and area>w*h*.25:symbols.append({"text":"△","confidence":.9,"method":"contour"})
        elif isolated and len(poly)>=6 and perim and 4*np.pi*area/perim**2>.7:
            symbols.append({"text":"○","confidence":.8,"method":"contour"})
    return "unreadable",symbols


def header_rows(cells):
    rows=sorted({c["row"] for c in cells})
    for r in rows:
        row=[c for c in cells if c["row"]==r]
        units=[c for c in row if re.match(r"^(?:T|H|Y|M|F)\d+",c.get("text",""))]
        if any(c.get("symbols") and c.get('value_kind')=='symbol' for c in row) or (r>0 and len(units)==1 and units[0]['column']<=1):
            return min(r,4)
    return min(1,len(rows))
