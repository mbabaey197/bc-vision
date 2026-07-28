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
    paused: bool = False
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
        self.pause_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.lock = threading.Lock()
        self._overlay_rows: list[dict] = []
        self._overlay_gray = None
        self._overlay_revision = 0
        self._overlay_updated_at = 0.0
        self._overlay_max_age = 4.0

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

    def pause(self):
        if self.url.startswith("video://"):
            self.pause_event.set()
            self.state.paused = True
            return True
        return False

    def resume(self):
        if self.url.startswith("video://"):
            self.pause_event.clear()
            self.state.paused = False
            return True
        return False

    def _wait_while_paused(self):
        while (
            self.pause_event.is_set()
            and not self.stop_event.is_set()
        ):
            self.state.paused = True
            self.stop_event.wait(0.10)
        if not self.stop_event.is_set():
            self.state.paused = False

    @staticmethod
    def _gray(frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _track_overlay_rows(previous_frame, current_frame, rows):
        if (
            previous_frame is None
            or current_frame is None
            or previous_frame.shape[:2] != current_frame.shape[:2]
        ):
            return [dict(row) for row in rows]
        previous_gray = (
            previous_frame
            if previous_frame.ndim == 2
            else CameraStream._gray(previous_frame)
        )
        current_gray = (
            current_frame
            if current_frame.ndim == 2
            else CameraStream._gray(current_frame)
        )
        height, width = current_gray.shape[:2]
        tracked = []
        for source in rows:
            x1, y1, x2, y2 = (
                max(0, int(value)) for value in source["bbox"]
            )
            x2, y2 = min(width, x2), min(height, y2)
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            mask = np.zeros_like(previous_gray)
            mask[y1:y2, x1:x2] = 255
            points = cv2.goodFeaturesToTrack(
                previous_gray,
                mask=mask,
                maxCorners=36,
                qualityLevel=0.01,
                minDistance=3,
                blockSize=5,
            )
            if points is None or len(points) < 3:
                continue
            moved, status, _ = cv2.calcOpticalFlowPyrLK(
                previous_gray,
                current_gray,
                points,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    20,
                    0.03,
                ),
            )
            if moved is None or status is None:
                continue
            valid = status.reshape(-1) == 1
            if int(valid.sum()) < 3:
                continue
            delta = moved.reshape(-1, 2)[valid] - points.reshape(-1, 2)[valid]
            dx, dy = np.median(delta, axis=0)
            if not np.isfinite(dx) or not np.isfinite(dy):
                continue
            row = dict(source)
            moved_box = (
                max(0, min(width - 1, int(round(x1 + dx)))),
                max(0, min(height - 1, int(round(y1 + dy)))),
                max(1, min(width, int(round(x2 + dx)))),
                max(1, min(height, int(round(y2 + dy)))),
            )
            if (
                moved_box[2] <= moved_box[0]
                or moved_box[3] <= moved_box[1]
            ):
                continue
            row["bbox"] = moved_box
            tracked.append(row)
        return tracked

    def _live_overlays(self, frame):
        now = time.time()
        received_new_snapshot = False
        try:
            from app.ai.live_worker import live_anpr_detection_snapshot
            snapshot = live_anpr_detection_snapshot(
                self.camera_id,
                self._overlay_revision,
            )
            revision = int(snapshot.get("revision", 0))
            reference = snapshot.get("frame")
            detections = snapshot.get("detections") or []
            if revision > self._overlay_revision and reference is not None:
                if detections:
                    compensated = self._track_overlay_rows(
                        reference,
                        frame,
                        detections,
                    )
                    # A failed motion match means the detector coordinates no
                    # longer have visual support on this newer display frame.
                    # Never redraw the old coordinates as a fallback.
                    self._overlay_rows = compensated
                else:
                    self._overlay_rows = []
                self._overlay_revision = revision
                self._overlay_updated_at = now
                self._overlay_max_age = float(
                    snapshot.get("max_age", 4.0)
                )
                self._overlay_gray = self._gray(frame)
                received_new_snapshot = True
        except Exception:
            pass

        if not self._overlay_rows and self._overlay_revision == 0:
            try:
                from app.ai.live_worker import live_anpr_detections
                rows = live_anpr_detections(self.camera_id)
                if rows:
                    self._overlay_rows = [dict(row) for row in rows]
                    self._overlay_updated_at = now
                    self._overlay_gray = self._gray(frame)
                    received_new_snapshot = True
            except Exception:
                pass

        if now - self._overlay_updated_at > self._overlay_max_age:
            self._overlay_rows = []
            self._overlay_gray = self._gray(frame)
            return []
        if not received_new_snapshot and self._overlay_rows:
            current_gray = self._gray(frame)
            self._overlay_rows = self._track_overlay_rows(
                self._overlay_gray,
                current_gray,
                self._overlay_rows,
            )
            self._overlay_gray = current_gray
        return [dict(row) for row in self._overlay_rows]

    def _encode(self, frame):
        display = frame.copy()
        try:
            for result in self._live_overlays(frame):
                x1, y1, x2, y2 = result["bbox"]
                color = (36, 220, 96) if result.get("valid") else (0, 190, 255)
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 3)
                confidence = int(float(result.get("confidence", 0)) * 100)
                label = f"PLATE {confidence}%"
                top = max(24, y1)
                cv2.rectangle(
                    display,
                    (x1, top - 24),
                    (min(display.shape[1] - 1, x1 + 145), top),
                    color,
                    -1,
                )
                cv2.putText(
                    display,
                    label,
                    (x1 + 5, top - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (8, 16, 24),
                    1,
                    cv2.LINE_AA,
                )
        except Exception:
            pass
        if self.width and frame.shape[1] > self.width:
            scale = self.width / display.shape[1]
            display = cv2.resize(
                display,
                (self.width, int(display.shape[0] * scale)),
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
                    self._wait_while_paused()
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
                    self._wait_while_paused()
                    if self.stop_event.is_set():
                        break
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
            "paused": False,
            "error": "stream not started",
            "last_frame_at": 0.0,
        }
        if stream:
            base = {
                "online": stream.state.online,
                "paused": stream.state.paused,
                "error": stream.state.last_error,
                "last_frame_at": stream.state.last_frame_at,
            }
        try:
            from app.ai.live_worker import live_anpr_status
            base["anpr"] = live_anpr_status(camera_id)
        except Exception:
            base["anpr"] = {"active": False}
        return base

    def set_playback(self, camera_id, action):
        with self.lock:
            stream = self.streams.get(int(camera_id))
        if not stream or not stream.url.startswith("video://"):
            return False
        if action == "pause":
            return stream.pause()
        if action == "play":
            return stream.resume()
        return False


manager = StreamManager()
