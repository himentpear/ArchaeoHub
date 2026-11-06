"""Published graphic regions control semantic routing; raw OCR remains searchable."""
from sqlalchemy import select
from src.graphics.models import GraphicFragment,GraphicObject,GraphicRun
from src.graphics.geometry import coverage


def latest_page_run(session,canvas_id,published_only=True):
    # Select the newest run by its page coverage, including a valid zero-object
    # result. Otherwise old detections survive a later corrected empty page.
    runs=list(session.scalars(select(GraphicRun).where(GraphicRun.status=='complete')))
    from src.db.models import Canvas
    canvas=session.get(Canvas,canvas_id)
    runs=[r for r in runs if canvas and r.manifest_id==canvas.manifest_id
          and canvas.scan_index in r.config.get('pages',r.config.get('selected_pages',[]))
          and (r.published or not published_only)]
    return max(runs,key=lambda r:r.config.get('created_at',0)).id if runs else None


def page_regions(session,canvas_id,published_only=True):
    newest=latest_page_run(session,canvas_id,published_only)
    stmt=select(GraphicFragment,GraphicObject.category).join(GraphicObject).join(GraphicRun,GraphicRun.id==GraphicObject.run_id).where(
        GraphicFragment.canvas_id==canvas_id,GraphicObject.status.not_in(['rejected','superseded']))
    if published_only:stmt=stmt.where(GraphicRun.published.is_(True))
    if newest:stmt=stmt.where(GraphicRun.id==newest)
    rows=session.execute(stmt).all()
    if rows:
        newest=max((session.get(GraphicObject,f.object_id).run_id for f,_ in rows),key=lambda rid:session.get(GraphicRun,rid).config.get('created_at',0))
        rows=[(f,c) for f,c in rows if session.get(GraphicObject,f.object_id).run_id==newest]
    return [(f,category) for f,category in rows]


def published_regions(session,canvas_id):return page_regions(session,canvas_id,True)


def covered(box,regions):
    x,y,w,h=box;a=[x,y,x+w,y+h]
    return next((category for fragment,category in regions if coverage(a,[fragment.bbox[0],fragment.bbox[1],
        fragment.bbox[0]+fragment.bbox[2],fragment.bbox[1]+fragment.bbox[3]])>=.8),None)
