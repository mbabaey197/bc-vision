# RC30 performance baseline contract

This document defines the repeatable performance evidence required before any
CPU/camera-capacity optimization is merged after `2.2.0-rc30`.

## Pinned production baseline

- Production base: `v2.2.0-rc30`
- Commit: `21bdc99ed1afdca7423117b79c785992b04f0768`
- Runtime ABI: `2`
- Fast-update base: `2.2.0-rc30`
- Measurement branch must not change `VERSION`, `RUNTIME_ABI`,
  `FAST_UPDATE_BASE_VERSION`, `RUNTIME_CONTRACT.json`, runtime dependency locks,
  database schema, detector/OCR selection, or production application code.

The current production detector selector supports YOLO11n, the fixed YOLOv8n
seed, and a hash-verified custom YOLOX manifest. Hezar v2 is the required
primary OCR. The pinned Platrix CRNN is a fixed degraded-path fallback after a
Hezar rejection. Performance work must not silently change this composition.

## Why this baseline exists

Two code-level hypotheses are visible in RC30 and must be measured before they
are treated as fixes:

1. native RTSP/camera reads are followed by the dashboard `live_fps` delay;
2. preview resize/overlay/JPEG work can run even when no browser viewer exists.

Draft PR #70 changes both behaviors, but its head is based on the pre-RC30
lineage and is not merge-safe evidence by itself. The benchmark workflow pins
both the exact RC30 SHA and the exact PR #70 head so the isolated transport
hypothesis can be measured without merging that PR.

## Required 1 / 3 / 6 camera metrics

`tools/benchmark_runtime_capacity.py` emits JSON for the same camera counts and
records:

- process CPU seconds;
- core-equivalent CPU percent (100% = one logical core fully used);
- host-normalized CPU percent;
- decoded frames and aggregate/per-camera decode FPS;
- decode call p50/p95 wall latency;
- frames submitted toward ANPR;
- preview JPEG count, JPEG FPS, and p50/p95 encode latency;
- published FPS;
- expected source-frame opportunities and estimated source-frame-drop rate;
- event count/evaluability state.

The synthetic transport test intentionally stubs ANPR after submission so that
scheduling and JPEG cost are isolated. Its `event_count` is therefore present
but fail-closed as `not_evaluable` rather than being misrepresented as an ANPR
accuracy result.

## Transport scenario

Default scenario:

- camera counts: 1, 3, 6;
- source: 1920x1080 at 25 FPS;
- source decoder: paced local MJPEG/AVI through the real OpenCV `VideoCapture`
  interface while `CameraStream` sees an RTSP URL;
- dashboard: 5 FPS, width 640, JPEG quality 70;
- 30 decoded/published frames per camera;
- no-viewer RC30 vs no-viewer PR #70 comparison;
- an additional PR #70 viewer-on run proves preview JPEG still exists when a
  viewer is present.

This is deliberately a **transport/JPEG isolation**, not a production H.264 or
H.265 capacity claim. The same benchmark must later be repeated on the target
Windows host with real RTSP streams and the actual camera codec/decode path.

`estimated_source_frame_drop_rate` is calculated from missed 25-FPS source
opportunities during the measured wall interval. It is not packet loss and is
not a substitute for camera/NIC decoder telemetry.

## Real-model inference scenario

The inference mode uses the pinned real ONNX models and one shared engine key.
For 1, 3 and 6 concurrent workers it reports detector, Hezar v2, and optional
Platrix fallback throughput/latency plus process CPU. Synthetic pixels are used
only to make load repeatable.

This mode deliberately does **not** claim passage accuracy or event parity. It
measures model execution capacity only.

## Passage-level accuracy and event parity

The production accuracy gate remains `app/ai/pass_benchmark.py`; it is not
replaced by this performance harness. A production claim requires independently
labelled passages and must include at least:

- passage-level exact plate accuracy;
- misses and wrong reads;
- duplicate events;
- false accepts/negative passages;
- unreadable outcomes;
- day/night;
- distance/readable-zone coverage;
- angle;
- blur, glare and general image quality;
- multiple vehicles and fast vehicles;
- camera/session provenance and immutable evidence digests.

The repository contract currently requires enough independent evidence for its
99% gate (including Wilson confidence intervals). The existing `01.mp4`
reference has only three trusted known-positive labels and is not exhaustive;
it cannot prove precision, duplicate rate, false-positive rate or 99% exact
passage accuracy. The video itself is not committed to the repository.

## Decision rules

A transport optimization is only a **supported hypothesis** when the same-host
comparison shows lower CPU/JPEG work without reducing native ingest or ANPR
submission cadence. It is not merge-approved until real RTSP and end-to-end
ANPR passage evidence also preserve event behavior.

For the viewer-aware hypothesis specifically, a no-viewer candidate is expected
to perform zero preview JPEG encodes; viewer-on must continue to produce JPEG.
For the ingest hypothesis, decoded FPS must no longer be constrained by the
5-FPS dashboard setting.

If the optimization is later reimplemented on current `main`, all normal source
regression, Windows regression, real AI-runtime, install/update and runtime
contract gates remain mandatory before merge/release.

Because this measurement branch changes only `tools/`, `tests/`, `docs/` and a
measurement workflow, Runtime ABI 2 and the RC30 runtime contract remain
unchanged. A later proven `app/streams.py`-only product optimization should also
remain Fast-Update eligible unless another hashed runtime-contract input or ABI
boundary changes.
