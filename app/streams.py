from __future__ import annotations

import threading
import time
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

try:
    import av
    AV_OK = True
except Exception:
    av = None
    AV_OK = False


@dataclass
class StreamState:
    online: bool = False
    last_error: str = ""
    last_frame_at: float = 0.0


class CameraStream:
    def __init__(
        self,
        camera_id: int,
        url: str,
        name: str,
        width=640,
        fps=5,
        quality=70,
    ):
        self.camera_id, self.url, self.name = camera_id, url, name
        self.width, self.fps, self.quality = (
            width,
            max(1, fps),
            quality,
        )
        self.state = StreamState()
        self.latest: bytes | None = None
        self.latest_frame = None
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.lock = threading.Lock()

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"camera-{self.camera_id}",
        )
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        try:
            from app.ai.live_worker import stop_live_camera
            stop_live_camera(self.camera_id)
        except Exception:
            pass

    def _encode(self, frame):
        display = frame
        if self.width and frame.shape[1] > self.width:
            scale = self.width / frame.shape[1]
            display = cv2.resize(
                frame,
                (self.width, int(frame.shape[0] * scale)),
            )
        ok, buffer = cv2.imencode(
            ".jpg",
            display,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.quality],
        )
        return bytes(buffer) if ok else None

    def _publish(self, frame):
        data = self._encode(frame)
        if not data:
            return
        with self.lock:
            self.latest = data
            self.latest_frame = frame
        self.state.online = True
        self.state.last_frame_at = time.time()
        self.state.last_error = ""
        try:
            from app.ai.live_worker import submit_live_frame
            submit_live_frame(
                self.camera_id,
                self.name,
                frame,
            )
        except Exception:
            # ANPR failures are reported through its own status and must never
            # interrupt or mark a healthy camera stream as offline.
            pass

    def _demo_frame(self):
        height, width = 360, 640
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        stamp = time.strftime("%Y-%m-%d  %H:%M:%S")
        x = int((time.time() * 90) % (width + 160)) - 160
        cv2.rectangle(
            frame,
            (x, 205),
            (x + 160, 300),
            (70, 160, 225),
            -1,
        )
        cv2.circle(
            frame,
            (x + 35, 305),
            20,
            (220, 220, 220),
            -1,
        )
        cv2.circle(
            frame,
            (x + 125, 305),
            20,
            (220, 220, 220),
            -1,
        )
        cv2.putText(
            frame,
            "Gilas Vision - DEMO CAMERA",
            (22, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            stamp,
            (22, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (210, 230, 255),
            2,
        )
        cv2.putText(
            frame,
            self.name,
            (22, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 255, 200),
            2,
        )
        return frame

    def _run_pyav_video(self, source, delay):
        """Decode an uploaded video with bundled FFmpeg when OpenCV cannot."""
        if not AV_OK:
            raise RuntimeError(
                "OpenCV could not decode the video and PyAV is unavailable"
            )
        while not self.stop_event.is_set():
            published = 0
            with av.open(str(source)) as container:
                for video_frame in container.decode(video=0):
                    if self.stop_event.is_set():
                        return
                    frame = video_frame.to_ndarray(format="bgr24")
                    self._publish(frame)
                    published += 1
                    if self.stop_event.wait(delay):
                        return
            if not published:
                raise RuntimeError(
                    "FFmpeg could not decode any frame from the video"
                )

    def _run(self):
        if not CV_OK:
            self.state.last_error = "OpenCV is not available"
            return
        delay = 1.0 / self.fps
        if self.url.startswith("demo://"):
            while not self.stop_event.is_set():
                self._publish(self._demo_frame())
                time.sleep(delay)
            return

        is_video_file = self.url.startswith("video://")
        capture_source = (
            self.url[len("video://"):]
            if is_video_file
            else self.url
        )
        while not self.stop_event.is_set():
            capture = None
            published = 0
            try:
                capture = cv2.VideoCapture(
                    capture_source,
                    cv2.CAP_FFMPEG,
                )
                if not capture.isOpened():
                    capture.release()
                    capture = cv2.VideoCapture(capture_source)
                if not capture.isOpened():
                    raise RuntimeError("Cannot open camera or video stream")
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                while not self.stop_event.is_set():
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        if is_video_file:
                            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ok, frame = capture.read()
                            if ok and frame is not None:
                                self._publish(frame)
                                published += 1
                                time.sleep(delay)
                                continue
                        raise RuntimeError(
                            "Camera stopped sending frames"
                        )
                    self._publish(frame)
                    published += 1
                    time.sleep(delay)
            except Exception as exc:
                if is_video_file and published == 0 and AV_OK:
                    try:
                        self._run_pyav_video(capture_source, delay)
                        continue
                    except Exception as fallback_exc:
                        exc = fallback_exc
                self.state.online = False
                self.state.last_error = str(exc)
                self.stop_event.wait(1 if is_video_file else 3)
            finally:
                if capture is not None:
                    capture.release()

    def frames(self) -> Iterator[bytes]:
        self.start()
        while not self.stop_event.is_set():
            with self.lock:
                frame = self.latest
            if frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Cache-Control: no-cache\r\n\r\n"
                    + frame
                    + b"\r\n"
                )
            time.sleep(1.0 / self.fps)


class StreamManager:
    def __init__(self):
        self.streams = {}
        self.lock = threading.Lock()

    def get(self, camera_id, url, name, width, fps, quality):
        key = (url, name, width, fps, quality)
        with self.lock:
            old = self.streams.get(camera_id)
            if old and getattr(old, "_key", None) != key:
                old.stop()
                self.streams.pop(camera_id, None)
                old = None
            if not old:
                old = CameraStream(
                    camera_id,
                    url,
                    name,
                    width,
                    fps,
                    quality,
                )
                old._key = key
                self.streams[camera_id] = old
                old.start()
            return old

    def start_enabled_cameras(self):
        """Start every enabled camera for continuous background ANPR."""
        from app.database import connect, get_setting

        width = int(get_setting("stream_width", "640"))
        fps = int(get_setting("live_fps", "5"))
        quality = int(get_setting("jpeg_quality", "70"))
        with connect() as con:
            rows = con.execute(
                "SELECT id,name,rtsp_url FROM cameras "
                "WHERE enabled=1 AND rtsp_url<>'' "
                "ORDER BY sort_order,id"
            ).fetchall()

        started = 0
        for row in rows:
            self.get(
                int(row["id"]),
                str(row["rtsp_url"]),
                str(row["name"]),
                width,
                fps,
                quality,
            )
            started += 1
        return started

    def stop_all(self):
        with self.lock:
            streams = list(self.streams.values())
            self.streams.clear()
        for stream in streams:
            stream.stop()

    def remove(self, camera_id):
        with self.lock:
            stream = self.streams.pop(camera_id, None)
        if stream:
            stream.stop()

    def status(self, camera_id):
        stream = self.streams.get(camera_id)
        base = {
            "online": False,
            "error": "stream not started",
            "last_frame_at": 0.0,
        }
        if stream:
            base = {
                "online": stream.state.online,
                "error": stream.state.last_error,
                "last_frame_at": stream.state.last_frame_at,
            }
        try:
            from app.ai.live_worker import live_anpr_status
            base["anpr"] = live_anpr_status(camera_id)
        except Exception:
            base["anpr"] = {"active": False}
        return base


manager = StreamManager()
