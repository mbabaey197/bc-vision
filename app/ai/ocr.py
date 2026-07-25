"""OCR adapter. Uses EasyOCR or Tesseract when available."""
import cv2
import numpy as np
from .plate_rules import normalize_plate, plausible_plate, format_iran_plate

_easy_reader=None

def _variants(image):
    if image is None or image.size==0: return []
    h,w=image.shape[:2]
    scale=max(1.0, 360/max(w,1))
    img=cv2.resize(image,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) if img.ndim==3 else img
    gray=cv2.equalizeHist(gray)
    blur=cv2.GaussianBlur(gray,(3,3),0)
    return [img, gray, cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1], cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]]

def _easyocr(image):
    global _easy_reader
    try:
        import easyocr
        if _easy_reader is None:
            _easy_reader=easyocr.Reader(['fa','en'],gpu=False,verbose=False)
        best=('',0.0)
        for v in _variants(image):
            for box,text,conf in _easy_reader.readtext(v,detail=1,paragraph=False):
                norm=normalize_plate(text)
                score=float(conf)
                if plausible_plate(norm): score=min(1.0,score+.12)
                if score>best[1]: best=(norm,score)
        return best
    except Exception:
        return '',0.0

def _tesseract(image):
    try:
        import pytesseract
        best=('',0.0)
        config='--psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for v in _variants(image):
            data=pytesseract.image_to_data(v,config=config,output_type=pytesseract.Output.DICT)
            for text,conf in zip(data.get('text',[]),data.get('conf',[])):
                norm=normalize_plate(text)
                try: score=max(0,float(conf))/100
                except Exception: score=0
                if plausible_plate(norm): score=min(1.0,score+.10)
                if score>best[1]: best=(norm,score)
        return best
    except Exception:
        return '',0.0

def read_plate(image):
    candidates=[_easyocr(image),_tesseract(image)]
    text,conf=max(candidates,key=lambda x:x[1])
    return format_iran_plate(text),float(conf)
