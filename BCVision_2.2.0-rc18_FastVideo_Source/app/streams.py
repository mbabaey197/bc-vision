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
        self._overlay_max_age = 1.6
        self._last_display_publish_at = 0.0

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
    def _clip_box(box, width, height):
        x1, y1, x2, y2 = (int(round(value)) for value in box)
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(x1 + 1, min(width, x2))
        y2 = max(y1 + 1, min(height, y2))
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        return x1, y1, x2, y2

    @staticmethod
    def _template_track(previous_gray, current_gray, box):
        """Fallback tracking for plates with too few stable KLT corners."""

        height, width = current_gray.shape[:2]
        x1, y1, x2, y2 = box
        template = previous_gray[y1:y2, x1:x2]
        if (
            template.size == 0
            or template.shape[0] < 8
            or template.shape[1] < 16
            or float(template.std()) < 4.0
        ):
            return None

        box_w, box_h = x2 - x1, y2 - y1
        search_x1 = max(0, x1 - box_w * 3)
        search_y1 = max(0, y1 - box_h * 4)
        search_x2 = min(width, x2 + box_w * 3)
        search_y2 = min(height, y2 + box_h * 4)
        search = current_gray[
            int(search_y1):int(search_y2),
            int(search_x1):int(search_x2),
        ]
        best = None
        for scale in (0.84, 0.92, 1.0, 1.08, 1.16):
            target_w = max(12, int(round(template.shape[1] * scale)))
            target_h = max(6, int(round(template.shape[0] * scale)))
            if target_w > search.shape[1] or target_h > search.shape[0]:
                continue
            candidate = cv2.resize(
                template,
                (target_w, target_h),
                interpolation=(
                    cv2.INTER_AREA
                    if scale < 1.0
                    else cv2.INTER_CUBIC
                ),
            )
            response = cv2.matchTemplate(
                search,
                candidate,
                cv2.TM_CCOEFF_NORMED,
            )
            _, score, _, location = cv2.minMaxLoc(response)
            if not np.isfinite(score):
                continue
            if best is None or score > best[0]:
                best = (float(score), location, target_w, target_h)
        if best is None or best[0] < 0.38:
            return None
        score, (match_x, match_y), target_w, target_h = best
        tracked = CameraStream._clip_box(
            (
                int(search_x1) + match_x,
                int(search_y1) + match_y,
                int(search_x1) + match_x + target_w,
                int(search_y1) + match_y + target_h,
            ),
            width,
            height,
        )
        if tracked is None:
            return None
        return tracked, score

    @staticmethod
    def _track_overlay_rows(previous_frame, current_frame, rows):
        if (
            previous_frame is None
            or current_frame is None
        ):
            return []
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
        previous_height, previous_width = previous_gray.shape[:2]
        scale_x = width / max(1, previous_width)
        scale_y = height / max(1, previous_height)
        scaled_rows = []
        if (previous_height, previous_width) != (height, width):
            previous_gray = cv2.resize(
                previous_gray,
                (width, height),
                interpolation=cv2.INTER_AREA,
            )
            for source in rows:
                row = dict(source)
                x1, y1, x2, y2 = source["bbox"]
                row["bbox"] = (
                    x1 * scale_x,
                    y1 * scale_y,
                    x2 * scale_x,
                    y2 * scale_y,
                )
                scaled_rows.append(row)
        else:
            scaled_rows = [dict(row) for row in rows]

        tracked = []
        for source in scaled_rows:
            box = CameraStream._clip_box(
                source["bbox"],
                width,
                height,
            )
            if box is None:
                continue
            x1, y1, x2, y2 = box
            mask = np.zeros_like(previous_gray)
            mask[y1:y2, x1:x2] = 255
            points = cv2.goodFeaturesToTrack(
                previous_gray,
                mask=mask,
                maxCorners=64,
                qualityLevel=0.006,
                minDistance=3,
                blockSize=5,
            )
            moved_box = None
            tracking_confidence = 0.0
            if points is not None and len(points) >= 3:
                moved, status_forward, _ = cv2.calcOpticalFlowPyrLK(
                    previous_gray,
                    current_gray,
                    points,
                    None,
                    winSize=(25, 25),
                    maxLevel=4,
                    criteria=(
                        cv2.TERM_CRITERIA_EPS
                        | cv2.TERM_CRITERIA_COUNT,
                        24,
                        0.02,
                    ),
                )
                backward = status_backward = None
                if moved is not None and status_forward is not None:
                    backward, status_backward, _ = cv2.calcOpticalFlowPyrLK(
                        current_gray,
                        previous_gray,
                        moved,
                        None,
                        winSize=(25, 25),
                        maxLevel=4,
                        criteria=(
                            cv2.TERM_CRITERIA_EPS
                            | cv2.TERM_CRITERIA_COUNT,
                            24,
                            0.02,
                        ),
                    )
                if (
                    moved is not None
                    and backward is not None
                    and status_forward is not None
                    and status_backward is not None
                ):
                    original_points = points.reshape(-1, 2)
                    moved_points = moved.reshape(-1, 2)
                    backward_points = backward.reshape(-1, 2)
                    roundtrip = np.linalg.norm(
                        original_points - backward_points,
                        axis=1,
                    )
                    valid = (
                        (status_forward.reshape(-1) == 1)
                        & (status_backward.reshape(-1) == 1)
                        & np.isfinite(roundtrip)
                        & (roundtrip <= 2.2)
                    )
                    if int(valid.sum()) >= 3:
                        source_points = original_points[valid]
                        target_points = moved_points[valid]
                        transform, inliers = cv2.estimateAffinePartial2D(
                            source_points,
                            target_points,
                            method=cv2.RANSAC,
                            ransacReprojThreshold=2.5,
                            maxIters=120,
                            confidence=0.96,
                        )
                        if transform is not None:
                            affine_scale = float(
                                np.hypot(
                                    transform[0, 0],
                                    transform[0, 1],
                                )
                            )
                            if 0.68 <= affine_scale <= 1.42:
                                corners = np.array(
                                    [
                                        [x1, y1, 1.0],
                                        [x2, y1, 1.0],
                                        [x2, y2, 1.0],
                                        [x1, y2, 1.0],
                                    ],
                                    dtype=np.float32,
                                )
                                transformed = corners @ transform.T
                                moved_box = CameraStream._clip_box(
                                    (
                                        transformed[:, 0].min(),
                                        transformed[:, 1].min(),
                                        transformed[:, 0].max(),
                                        transformed[:, 1].max(),
                                    ),
                                    width,
                                    height,
                                )
                                tracking_confidence = min(
                                    1.0,
                                    float(valid.mean()),
                                )
            if moved_box is None:
                fallback = CameraStream._template_track(
                    previous_gray,
                    current_gray,
                    box,
                )
                if fallback is None:
                    continue
                moved_box, tracking_confidence = fallback
            row = dict(source)
            row["bbox"] = moved_box
            row["tracking_confidence"] = round(
                float(tracking_confidence),
                4,
            )
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
                # The rectangle represents a physical plate detection, not OCR
                # confidence. Confirmed reads stay green; experimental raw
                # guesses are amber and explicitly labelled GUESS.
                experimental = bool(result.get("experimental"))
                color = (
                    (24, 178, 255)
                    if experimental
                    else (36, 220, 96)
                )
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 3)
                confidence = int(float(result.get("confidence", 0)) * 100)
                label = (
                    f"GUESS {confidence}%"
                    if experimental
                    else f"PLATE {confidence}%"
                )
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
        # Decode/ANPR and browser JPEG cadence are deliberately separate.
        # Uploaded video can be decoded at native 25-30 FPS while browser
        # encoding stays bounded, so slow OCR never slows the video clock.
        now_mono = time.monotonic()
        display_fps = (
            max(self.fps, 12)
            if self.url.startswith("video://")
            else self.fps
        )
        display_due = (
            self.latest is None
            or now_mono - self._last_display_publish_at
            >= 1.0 / max(1, display_fps)
        )
        if display_due:
            data = self._encode(frame)
            if data:
                with self.lock:
                    self.latest = data
                    self.latest_frame = frame
                self._last_display_publish_at = now_mono

        self.state.online = True
        self.state.last_frame_at = time.time()
        self.state.last_error = ""
        try:
            from app.ai.live_worker import submit_live_frame
            # The worker owns a one-frame/latest-frame queue. Submission is
            # non-blocking and stale frames are replaced rather than queued.
            submit_live_frame(self.camera_id, self.name, frame)
        except Exception:
            pass

    def _demo_frame(self):
        height, width = 360, 640
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        stamp = time.strftime("%Y-%m-%d  %H:%M:%S").translate(
            str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
        )
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
        """Decode uploaded video using its native FPS clock."""
        if not AV_OK:
            raise RuntimeError(
                "OpenCV could not decode the video and PyAV is unavailable"
            )
        while not self.stop_event.is_set():
            published = 0
            with av.open(str(source)) as container:
                stream = container.streams.video[0]
                try:
                    source_fps = float(stream.average_rate)
                except (TypeError, ValueError, ZeroDivisionError):
                    source_fps = 0.0
                if not 1.0 <= source_fps <= 120.0:
                    source_fps = max(1.0, float(self.fps))
                frame_delay = 1.0 / source_fps
                deadline = time.monotonic()
                for video_frame in container.decode(video=0):
                    if self.stop_event.is_set():
                        return
                    self._wait_while_paused()
                    if self.stop_event.is_set():
                        return
                    frame = video_frame.to_ndarray(format="bgr24")
                    self._publish(frame)
                    published += 1
                    deadline += frame_delay
                    wait = deadline - time.monotonic()
                    if wait > 0 and self.stop_event.wait(wait):
                        return
                    if wait < -0.75:
                        deadline = time.monotonic()
            if not published:
                raise RuntimeError(
                    "FFmpeg could not decode any frame from the video"
                )

    def _run(self):
        if not CV_OK:
            self.state.last_error = "OpenCV is not available"
            return
        dashboard_delay = 1.0 / self.fps
        if self.url.startswith("demo://"):
            while not self.stop_event.is_set():
                self._publish(self._demo_frame())
                self.stop_event.wait(dashboard_delay)
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
                capture = cv2.VideoCapture(capture_source, cv2.CAP_FFMPEG)
                if not capture.isOpened():
                    capture.release()
                    capture = cv2.VideoCapture(capture_source)
                if not capture.isOpened():
                    raise RuntimeError("Cannot open camera or video stream")
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                playback_delay = dashboard_delay
                if is_video_file:
                    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
                    if 1.0 <= source_fps <= 120.0:
                        playback_delay = 1.0 / source_fps
                deadline = time.monotonic()

                while not self.stop_event.is_set():
                    self._wait_while_paused()
                    if self.stop_event.is_set():
                        break
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        if is_video_file:
                            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            deadline = time.monotonic()
                            continue
                        raise RuntimeError("Camera stopped sending frames")

                    self._publish(frame)
                    published += 1
                    if is_video_file:
                        deadline += playback_delay
                        wait = deadline - time.monotonic()
                        if wait > 0:
                            if self.stop_event.wait(wait):
                                return
                        elif wait < -0.75:
                            deadline = time.monotonic()
                    elif self.stop_event.wait(dashboard_delay):
                        return
            except Exception as exc:
                if is_video_file and published == 0 and AV_OK:
                    try:
                        self._run_pyav_video(capture_source, dashboard_delay)
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
            display_fps = (
                max(self.fps, 12)
                if self.url.startswith("video://")
                else self.fps
            )
            time.sleep(1.0 / max(1, display_fps))

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
