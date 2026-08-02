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
    decoded_frames: int = 0
    ai_submitted_frames: int = 0
    preview_frames: int = 0
    source_fps: float = 0.0


class CameraStream:
    def __init__(
        self,
        camera_id: int,
        url: str,
        name: str,
        width=640,
        fps=8,
        quality=70,
    ):
        self.camera_id, self.url, self.name = camera_id, url, name
        self.width, self.preview_fps, self.quality = (
            width,
            max(1, min(10, int(fps))),
            quality,
        )
        # ``fps`` is kept as a read-only compatibility alias for older tests
        # and integrations. It now means dashboard preview FPS only.
        self.fps = self.preview_fps
        self.state = StreamState()
        self.latest: bytes | None = None
        self.latest_frame = None
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.lock = threading.Lock()
        self._preview_ready = threading.Condition(self.lock)
        self._preview_revision = 0
        self._preview_viewers = 0
        self._next_preview_at = 0.0
        self._last_source_at = 0.0
        self._source_fps_ema = 0.0
        self._overlay_rows: list[dict] = []
        self._overlay_gray = None
        self._overlay_revision = 0
        self._overlay_updated_at = 0.0
        self._overlay_max_age = 4.0

    def configure_preview(self, width, fps, quality):
        """Update dashboard rendering without restarting camera/ANPR state."""

        with self.lock:
            self.width = max(160, int(width))
            self.preview_fps = max(1, min(10, int(fps)))
            self.fps = self.preview_fps
            self.quality = max(30, min(95, int(quality)))
            self._next_preview_at = 0.0
            self.latest = None
            self._preview_revision += 1
            self._preview_ready.notify_all()

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

    def request_stop(self):
        self.stop_event.set()

    def stop(self, wait=False, timeout=3.0):
        self.request_stop()
        deadline = time.monotonic() + max(0.0, float(timeout))
        thread = self.thread
        if (
            wait
            and thread is not None
            and thread is not threading.current_thread()
        ):
            thread.join(max(0.0, deadline - time.monotonic()))
        decoder_stopped = not (thread and thread.is_alive())
        worker_stopped = True
        try:
            from app.ai.live_worker import stop_live_camera
            worker_stopped = stop_live_camera(
                self.camera_id,
                wait=wait,
                timeout=max(0.0, deadline - time.monotonic()),
            )
        except Exception:
            worker_stopped = False
        return bool(decoder_stopped and worker_stopped)

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

    def _observe_source_frame(self, captured_at):
        self.state.decoded_frames += 1
        if self._last_source_at > 0.0:
            interval = float(captured_at) - self._last_source_at
            if 1.0 / 240.0 <= interval <= 2.0:
                instant_fps = 1.0 / interval
                if self._source_fps_ema:
                    self._source_fps_ema = (
                        self._source_fps_ema * 0.85
                        + instant_fps * 0.15
                    )
                else:
                    self._source_fps_ema = instant_fps
        self._last_source_at = float(captured_at)
        self.state.source_fps = self._source_fps_ema

    def _preview_due(self, captured_at):
        with self.lock:
            if self._preview_viewers <= 0:
                return False
            if float(captured_at) + 1e-9 < self._next_preview_at:
                return False
            self._next_preview_at = (
                float(captured_at) + 1.0 / self.preview_fps
            )
            return True

    def _publish(self, frame, captured_at=None):
        """Ingest every source frame; render only the throttled preview lane."""

        captured_at = (
            time.monotonic()
            if captured_at is None
            else float(captured_at)
        )
        self._observe_source_frame(captured_at)
        self.state.online = True
        self.state.last_frame_at = time.time()
        self.state.last_error = ""
        # Recognition receives the native frame before JPEG/overlay work.
        # Preview failure or a closed dashboard must never suppress ANPR.
        try:
            from app.ai.live_worker import submit_live_frame
            submit_live_frame(
                self.camera_id,
                self.name,
                frame,
            )
            self.state.ai_submitted_frames += 1
        except Exception:
            # ANPR failures are reported through its own status and must never
            # interrupt or mark a healthy camera stream as offline.
            pass
        if not self._preview_due(captured_at):
            return
        data = self._encode(frame)
        if not data:
            return
        with self._preview_ready:
            self.latest = data
            self.latest_frame = frame
            self.state.preview_frames += 1
            self._preview_revision += 1
            self._preview_ready.notify_all()

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

    def _wait_until(self, deadline):
        remaining = float(deadline) - time.monotonic()
        if remaining > 0.0:
            self.stop_event.wait(remaining)

    @staticmethod
    def _capture_fps(capture, default=25.0):
        try:
            value = float(capture.get(cv2.CAP_PROP_FPS))
        except Exception:
            value = 0.0
        if not np.isfinite(value) or not 1.0 <= value <= 240.0:
            return float(default)
        return value

    def _run_pyav_video(self, source):
        """Decode an uploaded video with bundled FFmpeg when OpenCV cannot."""
        if not AV_OK:
            raise RuntimeError(
                "OpenCV could not decode the video and PyAV is unavailable"
            )
        while not self.stop_event.is_set():
            published = 0
            with av.open(str(source)) as container:
                source_fps = 25.0
                try:
                    video_stream = container.streams.video[0]
                    candidate = (
                        video_stream.average_rate
                        or video_stream.guessed_rate
                    )
                    if candidate:
                        source_fps = float(candidate)
                except Exception:
                    pass
                source_fps = max(1.0, min(240.0, source_fps))
                loop_started = time.monotonic()
                for frame_index, video_frame in enumerate(
                    container.decode(video=0)
                ):
                    if self.stop_event.is_set():
                        return
                    self._wait_while_paused()
                    if self.stop_event.is_set():
                        return
                    frame = video_frame.to_ndarray(format="bgr24")
                    self._publish(frame)
                    published += 1
                    try:
                        current_second = float(video_frame.time)
                    except (AttributeError, TypeError, ValueError):
                        current_second = frame_index / source_fps
                    next_second = max(
                        (frame_index + 1) / source_fps,
                        current_second + 1.0 / source_fps,
                    )
                    self._wait_until(loop_started + next_second)
            if not published:
                raise RuntimeError(
                    "FFmpeg could not decode any frame from the video"
                )

    def _run(self):
        if not CV_OK:
            self.state.last_error = "OpenCV is not available"
            return
        if self.url.startswith("demo://"):
            source_fps = 25.0
            frame_index = 0
            started = time.monotonic()
            while not self.stop_event.is_set():
                self._publish(self._demo_frame())
                frame_index += 1
                self._wait_until(started + frame_index / source_fps)
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
                source_fps = (
                    self._capture_fps(capture)
                    if is_video_file
                    else 0.0
                )
                loop_started = time.monotonic()
                frame_index = 0
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
                                frame_index = 1
                                loop_started = time.monotonic()
                                self._wait_until(
                                    loop_started
                                    + frame_index / source_fps
                                )
                                continue
                        raise RuntimeError(
                            "Camera stopped sending frames"
                        )
                    self._publish(frame)
                    published += 1
                    if is_video_file:
                        frame_index += 1
                        self._wait_until(
                            loop_started + frame_index / source_fps
                        )
            except Exception as exc:
                if is_video_file and published == 0 and AV_OK:
                    try:
                        self._run_pyav_video(capture_source)
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
        revision = -1
        with self._preview_ready:
            self._preview_viewers += 1
            self._next_preview_at = 0.0
        try:
            while not self.stop_event.is_set():
                with self._preview_ready:
                    if self._preview_revision == revision:
                        self._preview_ready.wait(
                            timeout=max(0.25, 2.0 / self.preview_fps)
                        )
                    frame = self.latest
                    current_revision = self._preview_revision
                if frame and current_revision != revision:
                    revision = current_revision
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Cache-Control: no-cache\r\n\r\n"
                        + frame
                        + b"\r\n"
                    )
                elif current_revision != revision:
                    # A settings change can intentionally invalidate the old
                    # JPEG before the next source frame arrives. Acknowledge
                    # that revision so an offline camera cannot busy-spin.
                    revision = current_revision
        finally:
            with self._preview_ready:
                self._preview_viewers = max(
                    0,
                    self._preview_viewers - 1,
                )


class StreamManager:
    def __init__(self):
        self.streams = {}
        self.lock = threading.Lock()

    def get(self, camera_id, url, name, width, fps, quality):
        key = (url, name, width, fps, quality)
        with self.lock:
            old = self.streams.get(camera_id)
            if old and getattr(old, "_key", None) != key:
                if not old.stop(wait=True):
                    raise RuntimeError(
                        "Previous camera stream did not stop in time"
                    )
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

    def configure_preview(self, width, fps, quality):
        """Apply display-only changes without interrupting recognition."""

        with self.lock:
            streams = list(self.streams.values())
        for stream in streams:
            stream.configure_preview(width, fps, quality)
            stream._key = (
                stream.url,
                stream.name,
                stream.width,
                stream.preview_fps,
                stream.quality,
            )

    def start_enabled_cameras(self):
        """Start every enabled camera for continuous background ANPR."""
        from app.database import connect, get_setting

        width = int(get_setting("stream_width", "640"))
        legacy_fps = get_setting("live_fps", "8")
        fps = int(get_setting("dashboard_preview_fps", legacy_fps))
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
            stream.request_stop()
        for stream in streams:
            stream.stop(wait=True)

    def remove(self, camera_id, wait=False):
        with self.lock:
            stream = self.streams.pop(camera_id, None)
        if stream:
            stopped = stream.stop(wait=wait)
            if wait and not stopped:
                with self.lock:
                    self.streams.setdefault(camera_id,stream)
            return stopped if wait else True
        try:
            from app.ai.live_worker import stop_live_camera
            return stop_live_camera(camera_id,wait=wait)
        except Exception:
            return False

    def status(self, camera_id):
        stream = self.streams.get(camera_id)
        base = {
            "online": False,
            "paused": False,
            "error": "stream not started",
            "last_frame_at": 0.0,
            "decoded_frames": 0,
            "ai_submitted_frames": 0,
            "preview_frames": 0,
            "source_fps": 0.0,
            "preview_fps": 0,
            "preview_viewers": 0,
        }
        if stream:
            base = {
                "online": stream.state.online,
                "paused": stream.state.paused,
                "error": stream.state.last_error,
                "last_frame_at": stream.state.last_frame_at,
                "decoded_frames": stream.state.decoded_frames,
                "ai_submitted_frames": (
                    stream.state.ai_submitted_frames
                ),
                "preview_frames": stream.state.preview_frames,
                "source_fps": round(stream.state.source_fps, 2),
                "preview_fps": stream.preview_fps,
                "preview_viewers": stream._preview_viewers,
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
