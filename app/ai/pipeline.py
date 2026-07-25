"""BC Vision plate-recognition pipeline."""
from .detector import detect_plates
from .ocr import read_plate
from .plate_rules import plausible_plate
from .vehicle_intelligence import analyze_vehicle

def process_frame(frame, min_detection_confidence=0.25):
    results=[]
    for item in detect_plates(frame,min_confidence=min_detection_confidence):
        text,ocr_conf=read_plate(item['crop'])
        combined=float(item['confidence'])*(0.35+0.65*float(ocr_conf))
        vehicle=analyze_vehicle(frame,item['bbox'])
        results.append({
            'plate':text or 'ناخوانا', 'confidence':combined,
            'detector_confidence':float(item['confidence']), 'ocr_confidence':float(ocr_conf),
            'bbox':item['bbox'], 'crop':item['crop'], 'method':item['method'],
            'valid':plausible_plate(text),
            'vehicle_type':vehicle['vehicle_type'], 'vehicle_color':vehicle['vehicle_color'],
            'vehicle_brand':vehicle['vehicle_brand'], 'vehicle_confidence':vehicle['vehicle_confidence'],
            'vehicle_bbox':vehicle['vehicle_bbox']
        })
    return results
