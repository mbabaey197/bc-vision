from __future__ import annotations
import threading, time
from dataclasses import dataclass
from typing import Iterator

try:
    import cv2
    import numpy as np
    CV_OK = True
except Exception:
    cv2 = None
    np = None
    CV_OK = False

@dataclass
class StreamState:
    online: bool = False
    last_error: str = ""
    last_frame_at: float = 0.0

class CameraStream:
    def __init__(self, camera_id: int, url: str, name: str, width=640, fps=5, quality=70):
        self.camera_id, self.url, self.name = camera_id, url, name
        self.width, self.fps, self.quality = width, max(1, fps), quality
        self.state = StreamState()
        self.latest: bytes | None = None
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.lock = threading.Lock()

    def start(self):
        if self.thread and self.thread.is_alive(): return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, daemon=True, name=f"camera-{self.camera_id}")
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def _encode(self, frame):
        if self.width and frame.shape[1] > self.width:
            scale = self.width / frame.shape[1]
            frame = cv2.resize(frame, (self.width, int(frame.shape[0]*scale)))
        ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        return bytes(buf) if ok else None

    def _demo_frame(self):
        h, w = 360, 640
        frame = np.zeros((h,w,3), dtype=np.uint8)
        t = time.strftime('%Y-%m-%d  %H:%M:%S')
        # Moving object proves the stream is live.
        x = int((time.time()*90) % (w+160)) - 160
        cv2.rectangle(frame, (x,205), (x+160,300), (70,160,225), -1)
        cv2.circle(frame, (x+35,305), 20, (220,220,220), -1)
        cv2.circle(frame, (x+125,305), 20, (220,220,220), -1)
        cv2.putText(frame, 'Gilas Vision - DEMO CAMERA', (22,48), cv2.FONT_HERSHEY_SIMPLEX, .75, (255,255,255), 2)
        cv2.putText(frame, t, (22,85), cv2.FONT_HERSHEY_SIMPLEX, .68, (210,230,255), 2)
        cv2.putText(frame, self.name, (22,130), cv2.FONT_HERSHEY_SIMPLEX, .7, (200,255,200), 2)
        return frame

    def _run(self):
        if not CV_OK:
            self.state.last_error = 'OpenCV is not available'
            return
        delay = 1.0/self.fps
        if self.url.startswith('demo://'):
            while not self.stop_event.is_set():
                frame = self._demo_frame()
                data = self._encode(frame)
                if data:
                    with self.lock: self.latest = data
                    self.state.online = True; self.state.last_frame_at = time.time(); self.state.last_error = ''
                time.sleep(delay)
            return
        while not self.stop_event.is_set():
            cap = None
            try:
                cap = cv2.VideoCapture(self.url)
                if not cap.isOpened(): raise RuntimeError('Cannot open RTSP stream')
                while not self.stop_event.is_set():
                    ok, frame = cap.read()
                    if not ok or frame is None: raise RuntimeError('Camera stopped sending frames')
                    data = self._encode(frame)
                    if data:
                        with self.lock: self.latest = data
                        self.state.online = True; self.state.last_frame_at = time.time(); self.state.last_error = ''
                    time.sleep(delay)
            except Exception as e:
                self.state.online = False; self.state.last_error = str(e)
                time.sleep(3)
            finally:
                if cap is not None: cap.release()

    def frames(self) -> Iterator[bytes]:
        self.start()
        while not self.stop_event.is_set():
            with self.lock: frame = self.latest
            if frame:
                yield b'--frame\r\nContent-Type: image/jpeg\r\nCache-Control: no-cache\r\n\r\n' + frame + b'\r\n'
            time.sleep(1.0/self.fps)

class StreamManager:
    def __init__(self): self.streams = {}; self.lock = threading.Lock()
    def get(self, camera_id, url, name, width, fps, quality):
        key = (url,name,width,fps,quality)
        with self.lock:
            old = self.streams.get(camera_id)
            if old and getattr(old,'_key',None) != key:
                old.stop(); self.streams.pop(camera_id,None); old=None
            if not old:
                old = CameraStream(camera_id,url,name,width,fps,quality); old._key=key
                self.streams[camera_id]=old; old.start()
            return old
    def remove(self, camera_id):
        with self.lock:
            s=self.streams.pop(camera_id,None)
            if s: s.stop()
    def status(self, camera_id):
        s=self.streams.get(camera_id)
        if not s: return {'online':False,'error':'stream not started'}
        return {'online':s.state.online,'error':s.state.last_error,'last_frame_at':s.state.last_frame_at}

manager = StreamManager()
