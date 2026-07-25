"""Video processing and event extraction for BC Vision."""
from pathlib import Path
from datetime import datetime
import hashlib
import cv2
from .pipeline import process_frame

class VideoTester:
    def __init__(self,video_path):
        self.video_path=str(video_path); self.cap=cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened(): raise ValueError('فایل ویدئو قابل باز شدن نیست.')
    def info(self):
        return {'frames':int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),'fps':float(self.cap.get(cv2.CAP_PROP_FPS)),'width':int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),'height':int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),'duration':(float(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))/max(float(self.cap.get(cv2.CAP_PROP_FPS)),1))}
    def frames(self):
        while True:
            ok,frame=self.cap.read()
            if not ok: break
            yield frame
    def close(self): self.cap.release()

def _hash_crop(crop):
    gray=cv2.resize(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),(32,8))
    return hashlib.sha1(gray.tobytes()).hexdigest()[:12]

def process_video(video_path,plate_dir,snapshot_dir,frame_step=5,max_events=100,min_confidence=.20,duplicate_seconds=2.5,roi=None):
    tester=VideoTester(video_path); info=tester.info(); fps=max(info['fps'],1.0)
    plate_dir=Path(plate_dir); snapshot_dir=Path(snapshot_dir); plate_dir.mkdir(parents=True,exist_ok=True); snapshot_dir.mkdir(parents=True,exist_ok=True)
    events=[]; seen={}; frame_no=0
    try:
        for frame in tester.frames():
            frame_no+=1
            if frame_no % max(1,int(frame_step)) != 0: continue
            source=frame
            ox=oy=0
            if roi:
                h,w=frame.shape[:2]
                rx,ry,rw,rh=roi
                x1=max(0,min(w-1,int(w*rx/100))); y1=max(0,min(h-1,int(h*ry/100)))
                x2=max(x1+1,min(w,int(w*(rx+rw)/100))); y2=max(y1+1,min(h,int(h*(ry+rh)/100)))
                source=frame[y1:y2,x1:x2]; ox,oy=x1,y1
            for result in process_frame(source,min_confidence):
                if ox or oy:
                    x1,y1,x2,y2=result['bbox']; result['bbox']=(x1+ox,y1+oy,x2+ox,y2+oy)
                crop=result['crop']; key=(result['plate'] if result['valid'] else _hash_crop(crop))
                now_sec=frame_no/fps
                if key in seen and now_sec-seen[key] < max(0,float(duplicate_seconds)): continue
                seen[key]=now_sec
                stamp=datetime.now().strftime('%Y%m%d-%H%M%S-%f')
                plate_file=plate_dir/f'plate-{stamp}.jpg'; snap_file=snapshot_dir/f'vehicle-{stamp}.jpg'
                cv2.imwrite(str(plate_file),crop,[cv2.IMWRITE_JPEG_QUALITY,92])
                annotated=frame.copy(); x1,y1,x2,y2=result['bbox']; cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,255,0),2)
                vx1,vy1,vx2,vy2=result.get('vehicle_bbox',(x1,y1,x2,y2)); cv2.rectangle(annotated,(vx1,vy1),(vx2,vy2),(255,180,0),2)
                cv2.imwrite(str(snap_file),annotated,[cv2.IMWRITE_JPEG_QUALITY,88])
                events.append({**{k:v for k,v in result.items() if k!='crop'},'plate_path':str(plate_file),'image_path':str(snap_file),'frame':frame_no,'video_second':round(now_sec,2)})
                if len(events)>=max_events: return info,events
    finally: tester.close()
    return info,events
