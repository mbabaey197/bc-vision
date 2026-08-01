"""Lightweight vehicle attributes for BC Vision.
Works offline with OpenCV and can later be replaced by a trained classifier.
"""
import cv2
import numpy as np

COLOR_LABELS = {
    'white':'سفید','black':'مشکی','gray':'خاکستری','silver':'نقره‌ای',
    'red':'قرمز','blue':'آبی','green':'سبز','yellow':'زرد','brown':'قهوه‌ای','other':'سایر'
}

def _vehicle_roi(frame, plate_bbox):
    h,w=frame.shape[:2]
    x1,y1,x2,y2=plate_bbox
    pw=max(1,x2-x1); ph=max(1,y2-y1)
    # Iranian plates are normally in the lower portion of the vehicle.
    vx1=max(0,int(x1-2.3*pw)); vx2=min(w,int(x2+2.3*pw))
    vy1=max(0,int(y1-5.2*ph)); vy2=min(h,int(y2+1.7*ph))
    if vx2-vx1 < 40 or vy2-vy1 < 40:
        return frame, (0,0,w,h)
    return frame[vy1:vy2,vx1:vx2], (vx1,vy1,vx2,vy2)

def detect_color(image):
    if image is None or image.size == 0:
        return 'نامشخص',0.0
    h,w=image.shape[:2]
    # Ignore sky, road, glass and border areas; sample likely body panels.
    roi=image[int(h*.25):int(h*.82), int(w*.10):int(w*.90)]
    if roi.size == 0: roi=image
    hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    pixels=hsv.reshape(-1,3)
    # Drop very dark shadows and overexposed highlights.
    valid=pixels[(pixels[:,2]>28) & (pixels[:,2]<248)]
    if len(valid)<50: valid=pixels
    H,S,V=np.median(valid,axis=0)
    if V < 58: key='black'
    elif S < 28 and V > 188: key='white'
    elif S < 34 and V > 125: key='silver'
    elif S < 42: key='gray'
    elif H < 10 or H >= 170: key='red'
    elif H < 24: key='brown' if V < 145 else 'yellow'
    elif H < 38: key='yellow'
    elif H < 86: key='green'
    elif H < 140: key='blue'
    else: key='other'
    confidence=min(.92,max(.35,(float(S)/255)*.45 + abs(float(V)-128)/255*.15 + .42))
    if key in ('white','black','gray','silver'): confidence=.72
    return COLOR_LABELS[key],round(confidence,3)

def estimate_type(vehicle_crop, frame_shape, plate_bbox):
    if vehicle_crop is None or vehicle_crop.size == 0:
        return 'نامشخص',0.0
    vh,vw=vehicle_crop.shape[:2]; fh,fw=frame_shape[:2]
    x1,y1,x2,y2=plate_bbox
    plate_ratio=(x2-x1)/max(y2-y1,1)
    area_ratio=(vw*vh)/max(fw*fh,1)
    aspect=vw/max(vh,1)
    # Conservative geometry-based estimate; avoids claiming exact make/model.
    if aspect < .85 and area_ratio < .12:
        return 'موتورسیکلت',.48
    if vh > vw*.95 and area_ratio > .20:
        return 'اتوبوس/کامیون',.50
    if aspect > 2.25 and area_ratio > .17:
        return 'وانت/کامیونت',.47
    if plate_ratio > 4.8 and area_ratio > .22:
        return 'خودروی سنگین',.44
    return 'سواری',.58

def analyze_vehicle(frame, plate_bbox):
    crop,vehicle_bbox=_vehicle_roi(frame,plate_bbox)
    color,color_conf=detect_color(crop)
    vehicle_type,type_conf=estimate_type(crop,frame.shape,plate_bbox)
    confidence=round((color_conf+type_conf)/2,3)
    return {'vehicle_type':vehicle_type,'vehicle_color':color,'vehicle_brand':'نامشخص',
            'vehicle_confidence':confidence,'vehicle_bbox':vehicle_bbox,'vehicle_crop':crop}
