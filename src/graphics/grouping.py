"""A continuation is a logical relationship, not proof of image overlap."""
import re
import cv2
import numpy as np
from src.graphics.geometry import box_polygon, transform_points, bounds


LABEL=re.compile(r"(续\s*)?((?:附\s*)?表|图版|彩版|图)\s*([一二三四五六七八九十百零〇0-9]+(?:[.．]\d+)*)")


def caption_for(region, lines, page_height):
    x0,y0,x1,y1=region
    nearby=[l for l in lines if y0-page_height*.13 <= l["bbox"][1] <= y0+8
        and l["bbox"][0]<x1 and l["bbox"][2]>x0]
    marked=[(l,LABEL.search(l["text"])) for l in nearby if LABEL.search(l["text"])]
    if not marked:return {"label":"","title":"","continuation":False,"appendix":False}
    line,m=max(marked,key=lambda pair:pair[0]["bbox"][1])
    title_lines=[l["text"] for l in nearby if abs(l["bbox"][1]-line["bbox"][1])<max(15,page_height*.012)]
    return {"label":re.sub(r"\s+","",m[2]+m[3]),"title":" ".join(title_lines),
        "continuation":bool(m[1]),"appendix":"附" in m[2]}


def table_signature(cells,width,image=None,header_count=2):
    if not cells:return {"columns":0,"edges":[],"headers":[]}
    columns=max(c["column"]+c["column_span"] for c in cells)
    headers=[re.sub(r"\s+","",c.get("text","")) for c in cells if c["row"]<header_count and c.get("text")]
    edges=sorted({round(c["bbox"][0]/max(1,width),2) for c in cells if c.get("bbox")})
    method='cell_boxes'
    if image is not None:
        # Neural bboxes can surround text rather than cell boundaries. Prefer
        # measured rules only when their count corroborates the parsed columns.
        from src.graphics.vision import binary_grid,supported_vertical_rules
        binary,_,vertical,_=binary_grid(image)
        _,axes,_=supported_vertical_rules(binary,vertical)
        if axes is not None and len(axes)==columns+1:
            edges=[round(x/max(1,width),2) for x in axes[:-1]];method='supported_ruled_boundaries'
    return {"columns":columns,"edges":edges,"headers":headers,'edge_method':method}


def continuation_evidence(a,b):
    if a["manifest_id"]!=b["manifest_id"] or a.get("namespace")!=b.get("namespace"):
        return {"accepted":False,"reason":"different_source"}
    if a.get("category")!="table" or b.get("category")!="table":
        return {"accepted":False,"reason":"not_a_table_continuation"}
    if not a.get("label") or a["label"]!=b.get("label"):
        return {"accepted":False,"reason":"different_label"}
    adjacent=0<b["page"]-a["page"]<=1
    appendix=a.get("appendix") and b.get("appendix") and a.get("section_id")==b.get("section_id")
    if not adjacent and not appendix:return {"accepted":False,"reason":"outside_search_scope"}
    sa,sb=a["signature"],b["signature"]
    if not sa["columns"] or sa["columns"]!=sb["columns"]:
        return {"accepted":False,"reason":"column_conflict"}
    ea,eb=sa.get("edges",[]),sb.get("edges",[])
    agreement=sum(any(abs(x-y)<=.025 for y in eb) for x in ea)/max(1,len(ea))
    words=lambda s:set("".join(s.get("headers",[])))
    wa,wb=words(sa),words(sb);similarity=len(wa&wb)/max(1,len(wa|wb))
    accepted=agreement>=.85 and similarity>=.5 and b.get("continuation",False)
    return {"accepted":accepted,"reason":"explicit_continuation" if accepted else "insufficient_structure_evidence",
        "edge_agreement":round(agreement,4),"header_similarity":round(similarity,4),"image_stitched":False}


def stitch_pair(first,second,category="figure"):
    """Only near-rigid, distinctive overlap can produce a composite. No inpainting."""
    factor=min(1.,1500/max(*first.shape[:2],*second.shape[:2]))
    images=[cv2.resize(im,None,fx=factor,fy=factor) for im in [first,second]]
    detector=cv2.SIFT_create(nfeatures=3000)
    features=[]
    for im in images:
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) if im.ndim==3 else im
        mask=np.full(gray.shape,255,np.uint8)
        if category=="table":
            from src.graphics.vision import binary_grid
            grid=binary_grid(gray)[3];mask[cv2.dilate(grid,np.ones((5,5),np.uint8))>0]=0
            mask[:int(gray.shape[0]*.18),:]=0  # repeated headers do not establish overlap
        features.append(detector.detectAndCompute(gray,mask))
    (ka,da),(kb,db)=features
    if da is None or db is None:return None,{"accepted":False,"reason":"no_distinctive_features"}
    pairs=cv2.BFMatcher().knnMatch(db,da,k=2)
    good=[m for pair in pairs if len(pair)==2 for m,n in [pair] if m.distance<.68*n.distance]
    if len(good)<24:return None,{"accepted":False,"reason":"insufficient_matches"}
    src=np.float32([kb[m.queryIdx].pt for m in good]);dst=np.float32([ka[m.trainIdx].pt for m in good])
    affine,inliers=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=2.)
    if affine is None:return None,{"accepted":False,"reason":"registration_failed"}
    keep=inliers.ravel().astype(bool);count=int(keep.sum());ratio=float(keep.mean())
    angle=float(np.degrees(np.arctan2(affine[1,0],affine[0,0])));scale=float(np.hypot(affine[0,0],affine[1,0]))
    if count<24 or ratio<.7 or abs(angle)>3 or not .95<scale<1.05:
        return None,{"accepted":False,"reason":"unstable_geometry","inliers":count,"inlier_ratio":ratio}
    matrix=np.vstack([affine,[0,0,1]]);matrix[:2,2]/=factor
    wa,ha=first.shape[1],first.shape[0];wb,hb=second.shape[1],second.shape[0]
    footprint=transform_points(box_polygon([0,0,wb,hb]),matrix)
    _,overlap_poly=cv2.intersectConvexConvex(np.float32(box_polygon([0,0,wa,ha])),np.float32(footprint))
    overlap=0 if overlap_poly is None else cv2.contourArea(overlap_poly)
    fraction=overlap/min(wa*ha,wb*hb)
    if not .12<fraction<.85:return None,{"accepted":False,"reason":"missing_or_duplicate_overlap","overlap":fraction}
    # Inliers must span a 2-D area, not a row of repetitive grid/axis marks.
    span=np.ptp(src[keep],axis=0)
    if span[0]<images[1].shape[1]*.15 or span[1]<images[1].shape[0]*.15:
        return None,{"accepted":False,"reason":"degenerate_overlap"}
    box=bounds(box_polygon([0,0,wa,ha])+footprint);x0,y0=np.floor(box[:2]).astype(int);x1,y1=np.ceil(box[2:]).astype(int)
    if (x1-x0)*(y1-y0)>2*(wa*ha+wb*hb):return None,{"accepted":False,"reason":"unreasonable_canvas"}
    shift=np.array([[1,0,-x0],[0,1,-y0],[0,0,1.]])
    shape=(x1-x0,y1-y0)
    a=cv2.warpPerspective(first,shift,shape);b=cv2.warpPerspective(second,shift@matrix,shape)
    ma=cv2.warpPerspective(np.full(first.shape[:2],255,np.uint8),shift,shape)>0
    mb=cv2.warpPerspective(np.full(second.shape[:2],255,np.uint8),shift@matrix,shape)>0
    out=np.full_like(a,255);out[ma]=a[ma];out[mb&~ma]=b[mb&~ma]
    both=ma&mb;out[both]=((a[both].astype(float)+b[both])/2).astype(np.uint8)
    return out,{"accepted":True,"method":"sift_similarity","inliers":count,"inlier_ratio":ratio,
        "overlap":fraction,"first_to_composite":shift.tolist(),"second_to_composite":(shift@matrix).tolist(),
        "seam":"average_only_in_verified_overlap","missing_pixels":int((~(ma|mb)).sum())}
