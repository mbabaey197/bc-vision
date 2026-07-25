"""License plate detector with YOLO support and an OpenCV fallback."""
from pathlib import Path
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

MODEL_PATH = Path(__file__).parent / "models" / "plate.pt"
_model = None


def load_model():
    global _model
    if _model is None and YOLO is not None and MODEL_PATH.exists():
        _model = YOLO(str(MODEL_PATH))
    return _model


def _clip_box(box, width, height):
    x1,y1,x2,y2 = box
    return max(0,x1), max(0,y1), min(width,x2), min(height,y2)


def detect_plates(frame, min_confidence=0.25, max_results=5):
    """Return [{crop, bbox, confidence, method}]."""
    if frame is None or getattr(frame, 'size', 0) == 0:
        return []
    h,w = frame.shape[:2]
    model = load_model()
    found=[]
    if model is not None:
        try:
            for result in model(frame, verbose=False):
                for b in result.boxes:
                    conf=float(b.conf[0])
                    if conf < min_confidence:
                        continue
                    x1,y1,x2,y2=map(int,b.xyxy[0].tolist())
                    x1,y1,x2,y2=_clip_box((x1,y1,x2,y2),w,h)
                    if x2>x1 and y2>y1:
                        found.append({'crop':frame[y1:y2,x1:x2].copy(),'bbox':(x1,y1,x2,y2),'confidence':conf,'method':'yolo'})
            return sorted(found,key=lambda x:x['confidence'],reverse=True)[:max_results]
        except Exception:
            found=[]

    # Offline fallback: edge/geometry based plate localization.
    scale = 960 / max(w, 1) if w > 960 else 1.0
    work = cv2.resize(frame, None, fx=scale, fy=scale) if scale != 1 else frame
    gray=cv2.cvtColor(work,cv2.COLOR_BGR2GRAY)
    gray=cv2.bilateralFilter(gray,9,55,55)
    grad=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)
    grad=np.absolute(grad)
    grad=(255*(grad-grad.min())/(grad.max()-grad.min()+1e-6)).astype('uint8')
    grad=cv2.morphologyEx(grad,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_RECT,(17,3)))
    _,th=cv2.threshold(grad,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th=cv2.morphologyEx(th,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_RECT,(11,3)),iterations=2)
    contours,_=cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    wh=work.shape[:2]
    candidates=[]
    for c in contours:
        x,y,bw,bh=cv2.boundingRect(c)
        area=bw*bh
        ratio=bw/max(bh,1)
        if area < wh[0]*wh[1]*0.0007 or ratio < 2.0 or ratio > 7.5 or bh < 12:
            continue
        roi_gray=gray[y:y+bh,x:x+bw]
        edge_density=float(cv2.countNonZero(cv2.Canny(roi_gray,80,180)))/max(area,1)
        rectangularity=float(cv2.contourArea(c))/max(area,1)
        score=min(0.82,0.25+edge_density*2.2+rectangularity*0.28)
        if edge_density < 0.05:
            continue
        ox1,oy1,ox2,oy2=map(lambda v:int(v/scale),(x,y,x+bw,y+bh))
        pad_x=max(2,int((ox2-ox1)*.05)); pad_y=max(2,int((oy2-oy1)*.12))
        ox1,oy1,ox2,oy2=_clip_box((ox1-pad_x,oy1-pad_y,ox2+pad_x,oy2+pad_y),w,h)
        candidates.append({'crop':frame[oy1:oy2,ox1:ox2].copy(),'bbox':(ox1,oy1,ox2,oy2),'confidence':score,'method':'opencv'})
    candidates.sort(key=lambda x:x['confidence'],reverse=True)
    return candidates[:max_results]


def detect_plate(frame):
    rows=detect_plates(frame,max_results=1)
    return (rows[0]['crop'],rows[0]['confidence']) if rows else (None,0.0)
