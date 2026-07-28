# BC Vision ANPR accuracy, idle CPU and playback — RC9

Version: `2.2.0-rc9`

## Root causes confirmed

- RC8 emitted an unreadable capture on the first physical plate observation.
- The dedicated character model blocked the bundled generic OCR even when its
  own eight-character sequence was incomplete.
- A queued frame was dispatched immediately after inference, bypassing the
  adaptive interval and keeping CPU inference continuously busy.
- Empty detector results did not publish a new overlay revision.
- Failed optical-flow compensation redrew stale detector coordinates.
- Uploaded-video live cards had no server-side pause/resume control.

## Implemented

- Added a provisional `در حال بررسی` event state with the same in-place
  upgrade and non-duplicate media behavior.
- Delayed final `ناخوانا` classification until repeated failed observations.
- Added quality-gated EasyOCR/Tesseract fallback only after a physical plate
  crop is localized and the dedicated reader fails.
- Added exponential no-plate inference gaps: 0.4, 0.8, 1.6 and 3.2 seconds.
- Reduced the default Torch/OpenCV inference budget from six to four threads
  while retaining the `BCVISION_CPU_THREADS=1..8` override.
- Removed immediate queued-frame dispatch after inference.
- Published empty overlay revisions and removed stale-coordinate fallback.
- Added uploaded-video Play/Pause controls, API and stream state reporting.

## Engine review

- PaddleOCR PP-OCRv5 offers a lightweight Arabic recognition model, but it is
  a generic script model rather than an Iranian plate model. It requires
  ground-truth comparison before adoption.
- Ultralytics supports OpenVINO export and INT8/FP16 deployment. OpenVINO is
  the next preferred CPU benchmark because it can preserve the trained
  detector while replacing the PyTorch execution backend.
- ONNX Runtime provides explicit thread controls, graph optimization and CPU
  quantization. It remains a viable cross-vendor fallback.
- No new engine was silently introduced in RC9; model and packaging changes
  require real accuracy and Windows upgrade evidence.

## Verification

- Focused pipeline, worker, stream, video and web tests: `50 passed`.
- Full source regression after implementation: `98 passed, 1 skipped`.
- Dedicated tests cover delayed unreadable classification, generic OCR
  recovery, empty-result backoff, overlay clearing, failed-flow handling,
  uploaded-video pause/resume and the authenticated playback API.

## Compatibility

- No SQLite schema change.
- Existing database, users, settings, license, events, images, uploaded
  videos and AI models remain compatible with the one-click updater.
