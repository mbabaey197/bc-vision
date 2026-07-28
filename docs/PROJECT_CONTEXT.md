# BC Vision — Project Context

## Repository

- GitHub repository: `mahdibabaey197/bc-vision`
- Default branch: `main`
- Product: BC Vision
- Current application version in source: `2.2.0-rc11`
- Windows persistent data root: `C:\ProgramData\BCVision\data`

## Release contract

Every production release must deliver all three outputs together:

1. Complete source archive
2. Windows installer
3. One-click updater that preserves the previous database, settings, license data, snapshots, videos and AI models

Application data and AI models must remain outside the replaceable application directory.

## Current ANPR implementation

The Iranian plate-recognition subsystem now includes:

- verified Iranian YOLO detector model support
- hardened OpenCV detector fallback
- low-light, high-exposure, blur and contrast preprocessing
- perspective correction for angled plates
- Persian, Arabic and English digit normalization
- spatial reconstruction of split OCR tokens
- position-aware OCR confusion repair
- EasyOCR Persian/English recognition
- image quality scoring
- multi-frame tracking and weighted consensus
- isolated bad-read rejection
- canonical duplicate suppression
- vehicle direction estimation
- ROI processing for uploaded videos and live cameras
- asynchronous nonblocking live-camera recognition
- automatic background startup for every enabled camera
- safe shutdown and reconnect behavior
- backward-compatible SQLite migrations and canonical `plate_norm`
- persistent, SHA-256-verified AI model management
- AI-aware PyInstaller build configuration
- serialized shared-model inference for multi-camera correctness
- conditional mild-motion deblurring with dedicated-AI reread
- whole-plate Iranian CRNN+CTC OCR through ONNX Runtime
- per-camera ONNX sessions with a hard two-thread ceiling
- A/B evidence fusion between whole-plate and character-level readers

## ANPR model execution contract

The verified `best.pt` asset is a combined Iranian plate/character model, not
a single-class plate-only detector. Full-frame inference must select class
`30` (plate), followed by two independent crop readers: the character branch
of `best.pt`, and a segmentation-free Iranian CRNN+CTC model executed through
ONNX Runtime. Their alternatives are retained for multi-frame voting.
EasyOCR/Tesseract are reserved for the final crop-level fallback. Vehicle
attributes are calculated only after stable plate consensus, immediately
before event persistence.

The CPU-oriented defaults are 640px for plate localization, 416px for
character recognition, no test-time augmentation, and at most four plate
candidates per processed frame. A successful zero-result model inference
triggers the geometry fallback for difficult small, oblique or overexposed
plates. The live worker rate-limits this second pass adaptively from measured
processing latency so it does not run continuously on a slow CPU.

Each camera inference is hard-capped to two native compute threads across
OpenMP, BLAS, OpenCV and Torch. Deployments may reduce the per-camera budget
to one with `BCVISION_CPU_THREADS=1`; larger values are clamped to two.
Concurrent camera capacity is calculated from logical CPU count while
reserving two logical processors for decoding, the dashboard and Windows.
Each of the default three camera keys keeps a distinct bounded ONNX Runtime
session even when a small host permits only one simultaneous inference.
Every session uses `intra_op_num_threads <= 2`,
`inter_op_num_threads = 1`, sequential graph execution and disabled worker
spinning.

Ultralytics predictor state is not shared safely across simultaneous
`predict()` calls. The detector therefore owns one model instance and lock per
bounded runtime slot. A camera always maps to the same slot; cameras in
different slots can infer concurrently, while additional cameras queue safely
and continue replacing stale frames.

Mild-blur recovery is conditional and defaults on. The original crop is read
first; only a soft or uncertain crop is restored and read once more by the
dedicated character model. A recovered digit is accepted only with agreement
or a decisive confidence improvement. Conflicting plausible reads become
unreadable and are left to multi-frame consensus. Operators can disable this
pass with `BCVISION_BLUR_RECOVERY=0`.

## Important commits

- Iranian plate rules: `16f0a5354125ff90c9d45b253852e5dc56b239dc`
- Robust ANPR pipeline merge: `06320a424837cce9044e52531b530fe302b10be4`
- Continuous background camera ANPR merge: `d04a2bc64e5ecf546754f0d069cb971483329333`

## Validated runtime

The target Windows Python 3.13 environment successfully imported and executed:

- Torch `2.13.0+cpu`
- TorchVision `0.28.0+cpu`
- OpenCV `5.0.0`
- EasyOCR `1.7.2`
- Ultralytics `8.4.106`

Persian EasyOCR reader creation and inference were verified with the persistent Arabic recognition and CRAFT detector models.
ONNX Runtime `1.28.0` and the verified CRNN model are RC11 Windows acceptance
dependencies; packaged inference remains a release gate until the candidate
workflow completes.

## Automated validation

The current regression suite covers:

- Iranian plate normalization and validation
- OCR token reconstruction and positional repair
- difficult exposure preprocessing
- clear, dark, rotated and blurred plate localization
- candidate NMS
- multi-frame consensus and bad-read outvoting
- duplicate cooldown
- image quality scoring
- video event generation
- live worker nonblocking behavior
- background camera auto-start and clean shutdown
- backward-compatible database migration
- model hash verification
- shared-model concurrent-call serialization
- conservative blur-recovery gating and result selection

The latest full-suite and system optimization result is recorded in
`agent-results/latest/PROJECT_OPTIMIZATION_PHASE_1.md`: `64 passed, 1 skipped`,
plus a successful headless runtime smoke and a 1,000-write SQLite concurrency
and integrity check.

## Security hardening status

Phase 1 security hardening is included in the current optimization branch:

- uploaded video names are never used as filesystem paths
- video uploads are streamed with a 2 GiB limit and partial files are removed
- stored watchlist fields are HTML-escaped in event reports
- media access uses resolved path containment instead of string prefixes
- snapshot, plate, video and backup folders must be distinct children of the
  configured storage root
- retention cleanup refuses any folder outside the configured storage root
- camera, system, license, watchlist and video operations enforce server-side
  role permissions
- administrators and system managers can manage technical configuration
- operators can manage watchlists and process videos
- guards retain read-only monitoring and event access

The phase 1 regression tests cover traversal filenames, upload-size cleanup,
stored XSS, media-prefix bypass, unsafe retention paths and role boundaries.

SQLite now uses WAL mode, foreign-key enforcement, a bounded busy timeout and
NORMAL synchronous mode. Database backups use SQLite's online backup API,
verify integrity and publish atomically. Storage-root changes prepare and
update the destination database before atomically switching bootstrap
configuration, while leaving the old database available for rollback.

## Windows launcher and first-run dashboard

The launcher is designed to run as a PyInstaller windowed application. Uvicorn
now owns the main process, so no Tkinter keep-alive window is required. The
browser opens only after the BC Vision health endpoint verifies the service
identity. A different application listening on port 8000 is rejected instead
of being mistaken for BC Vision.

Version `2.2.0-rc3` hardens every Windows launch path against service-console
windows. Packaged runs hide any accidentally attached console as a runtime
safeguard, source runs hand off to `pythonw.exe`, and PowerShell/WMIC hardware
identity probes use hidden child-process flags. Windows CI parses the built
PE header and rejects any `BCVision.exe` that is not
`IMAGE_SUBSYSTEM_WINDOWS_GUI`.

Version `2.2.0-rc5` makes uploaded-video ANPR observable and independent of
OpenCV codec coverage. CCTV exports fall back to bundled FFmpeg/PyAV decoding,
camera cards show processed frames, detections, events, model readiness and
errors, and the consensus window adapts to slower CPU processing. Verified
detector and EasyOCR models are seeded inside the Windows package, then loaded
and exercised offline from the installed executable both before and after the
one-click update.

New databases no longer receive an enabled synthetic demo camera. A one-time,
exact-match migration removes only the historical built-in sample camera from
existing databases while preserving real cameras, user-created demo cameras,
events, settings and license data. The empty dashboard contains instructions
and an add-camera action without a placeholder image or animation.

## Operational truth

No software can recover a plate that contains no usable pixels, is completely hidden, is outside the frame, or is destroyed by severe motion blur/overexposure. BC Vision uses multiple software recovery layers, but camera placement, optical focus, shutter speed, resolution, night illumination and plate pixel height still determine the upper accuracy limit.

## Remaining acceptance work

Before declaring a production release, perform these validations with real customer-like data:

- real RTSP camera during day and night
- fast-moving vehicle and motion blur
- angled entry/exit lanes
- dirty, damaged and partially occluded plates
- multiple simultaneous cameras
- long-duration CPU and memory stability
- Windows portable build, installer and one-click updater
- upgrade from the previous installed version while preserving the database and models

The current performance/accuracy work remains on the draft ANPR pull request
until direct comparison ground truth is available. Windows packaging for
`2.2.0-rc5` includes an isolated executable self-test plus installer/updater
data-preservation verification. A production release has not yet been
declared.

## Trial operator-feedback loop

Version `2.2.0-rc6` adds an operator-confirmed correction loop without
silently weakening plate validation. The dashboard renders canonical Iranian
plates with Persian digits and a physical-plate layout, and allows an
authorized operator to submit the correct full plate next to each recent
event. The original observation, corrected canonical value, event, operator,
vehicle snapshot and plate crop are retained in `anpr_feedback`. The displayed
event is corrected immediately and an exact repeated OCR observation can reuse
the confirmed mapping locally. Broader OCR/model changes require a reviewed
offline training and validation run; feedback does not mutate model weights
inside the live process.

Live streams draw a green box around every physical plate candidate. OCR
confidence and review state are carried by result metadata instead of changing
the box colour. Dashboard video cards are capped at a smaller width, and the
old system-status card beside recent plate events was removed; technical
status remains available from Settings.

## RC7 live reliability

Version `2.2.0-rc7` preserves live overlays when the dashboard stream is
resized, tries the geometry detector after a zero-result YOLO pass, refreshes
the recent-event report within about one second of a committed event, and
adaptively spaces expensive inference on slower CPUs while retaining the
newest selected frame. Plate consensus and unreadable-result safeguards remain
unchanged.

## RC8 multi-frame capture and overlay tracking

Version `2.2.0-rc8` uses every received display frame for lightweight optical
tracking and clear-frame selection while retaining RC7's adaptive limit on
expensive detector/OCR inference. A new detector result is motion-compensated
from its source frame to the current live frame, then the plate box advances
with optical flow on every displayed frame. When tracking evidence is lost,
the box is removed instead of being left behind at a stale position.

Every detector observation can replace the pending inference frame when it is
clearer, even when a camera's historical `frame_step` is greater than one.
The expensive shared detector remains serialized and rate-limited, so this
improves multi-frame evidence without restoring continuous high CPU usage.

A physical plate detection now creates one event with a cropped plate image
and a clear cropped vehicle image even when OCR is unreadable. The event keeps
an empty canonical `plate_norm` and the visible value `ناخوانا`; it is never
guessed. If later observations produce valid per-character consensus, the
same database row and image paths are upgraded rather than inserting a
duplicate event. The dashboard recent-events table shows the vehicle image
and plate crop directly beside the recognized plate value.

## RC9 accuracy, idle CPU and uploaded-video playback

Version `2.2.0-rc9` no longer labels the first failed OCR observation as
`ناخوانا`. A physical plate capture is initially stored as
`در حال بررسی`, retaining its vehicle snapshot and plate crop. It becomes
`ناخوانا` only after at least three failed observations across at least
0.8 seconds. Tracks containing partial plausible reads receive two additional
observations before an unreadable result is allowed. Valid multi-frame
consensus upgrades the same event without creating a duplicate.

When the dedicated Iranian character detector cannot assemble a valid
eight-character sequence, a sufficiently large, credible and usable plate
crop receives one fallback read through the bundled generic OCR path.
Generic OCR is not invoked when no physical plate was localized.

An empty detector result now publishes an explicit empty overlay revision,
and failed optical-flow compensation clears the overlay instead of redrawing
detector coordinates from an older frame. This removes both stale-box paths
identified in RC8.

The live worker no longer dispatches a pending frame immediately after an
expensive inference. It enforces idle time after every transaction and grows
the no-plate gap from 0.4 to 0.8, 1.6 and at most 3.2 seconds. A physical plate
detection resets the no-plate streak. The dashboard reports when a camera is
in this low-consumption state.

Uploaded-video camera cards now contain Play and Pause controls. Pausing
stops decoding and ANPR submission while preserving the last displayed frame;
resuming continues from the current video position.

OpenVINO/ONNX and PP-OCRv5 Arabic were reviewed as future candidates. The
runtime was not replaced in RC9 because doing so without an Iranian-plate
ground-truth set would make accuracy and installer compatibility unverified.
OpenVINO export is the preferred next CPU benchmark, while PP-OCRv5 should be
evaluated only as a crop-level fallback against the current dedicated Iranian
character model.

## RC10 per-camera CPU and multi-hypothesis OCR

Version `2.2.0-rc10` hard-caps every camera at two native compute threads.
The live scheduler separately limits simultaneous cameras from logical CPU
count and reserves two logical processors for Windows, decoding and the web
panel. On an eight-logical-processor host the default is three simultaneous
camera slots. Each slot owns a separate Ultralytics model instance and
inference lock, removing RC9's process-wide model bottleneck without sharing
unsafe predictor state.

The character decoder now groups overlapping class predictions instead of
discarding every lower-confidence alternative through global NMS. It keeps up
to five valid Iranian-plate hypotheses and the temporal tracker aggregates
their probability independently at all eight positions. A character that is
the consistent second model choice across frames can therefore defeat
different one-frame errors, while equal alternatives still fail the ambiguity
margin for automatic confirmation. After the review window expires, the
strongest fully observed alternative is retained as an explicitly reviewable
best-effort read rather than being discarded.

Incomplete dedicated-model reads are also retained. Five-to-seven detected
characters are aligned to the physical Iranian plate positions using crop
geometry. Different incomplete frames may supply the missing positions, but
every final position still requires at least three credible observations plus
the existing ratio and margin checks. No missing character is invented.

EasyOCR remains a crop-only fallback and is protected by one shared lock to
avoid duplicating its large model for every camera. Failure or first-run
migration of the operator-feedback table no longer interrupts ANPR.

RC10 preserves imperfect reads for operator correction. A complete
low-confidence hypothesis is stored as `suggested`; five-to-seven observed
positions are shown with `؟` only in missing positions. `ناخوانا` is reserved
for a physical plate crop with no usable character evidence. Corrections remain
durable labelled samples. Exact complete mistakes are reused immediately, and
character-level generalization requires at least two consistent confirmed
corrections and may only re-rank alternatives already emitted by OCR.

The live overlay is always green for a physical plate candidate and advances
on every displayed frame. Tracking uses forward/backward optical flow, affine
translation/scale estimation and a template-matching fallback. Synthetic
translation-plus-scale verification tracked `18/18` frames with zero misses
and mean/minimum IoU `1.000`. Full local regression is
`114 passed, 1 skipped`; real uploaded-video and Windows acceptance remain
release gates.

## RC11 whole-plate CRNN in ONNX Runtime

Version `2.2.0-rc11` adds a segmentation-free Iranian CRNN+CTC reader that
reads the complete `DD L DDD DD` plate crop in one inference. ONNX Runtime is
the execution backend for that CRNN model; it is not an alternative model
architecture. The verified 10,452,525-byte model is stored under the
persistent data directory and accepted only when its fixed SHA-256 matches.
The model and its MIT notice are included in the offline Windows model seed.

RC11 keeps the RC10 character detector as an independent reader. Agreement
boosts the selected OCR confidence. On disagreement both valid hypotheses are
retained for position-wise multi-frame voting; a CRNN result replaces the
primary one only when its confidence exceeds the character read by at least
0.08, and the event remains explicitly reviewable. EasyOCR/Tesseract run only
when both dedicated readers fail on a credible plate crop.

The selected OCR engine, alternative complete read and disagreement flag are
stored in additive `plate_events` columns. Per-camera runtime status also
reports whole-plate attempts, agreements, disagreements and selection counts,
so the A/B result can be measured without replacing or deleting old events.

Each of the default three camera keys retains a distinct bounded ONNX session.
Session options enforce at most two intra-operation threads, one
inter-operation thread, sequential execution and disabled thread spinning.
Concurrent inference remains separately limited by available CPU capacity;
the bounded LRU cache prevents modulo collisions between camera IDs on
small-core Windows hosts.

Local verification after RC11 implementation is `121 passed, 1 skipped`.
Unit tests use deterministic CTC logits and a controlled ONNX session double;
an additional real ONNX Runtime `1.28.0` smoke executed a generated ONNX graph
through preprocessing, inference and CTC formatting with the two-thread cap.
Loading the production model on Windows, packaging, clean installation, update
persistence and accuracy comparison on `01.mp4` remain release gates.

## Uploaded video live-source fix

The `agent/video-upload-live-source` branch fixes the blocking camera-video
upload flow. The upload request no longer runs full-video ANPR synchronously.
After validation and storage, the file is registered as the single active
`video://` virtual camera, starts through the normal stream manager and appears
in the dashboard live grid. Live ANPR remains asynchronous through the existing
latest-frame worker. Reaching the end of the file seeks back to frame zero
without taking the stream offline. The browser form reports upload progress and
redirects to the live dashboard when registration succeeds.

The verified regression result for this combined branch is
`66 passed, 1 skipped`; focused upload, looping-stream and web-security tests
reported `16 passed`. Compile and whitespace checks also passed. Windows
packaging and installed-build acceptance are still required before release.

## Repository security observation

GitHub reported the repository visibility as public on 2026-07-27, while the
recorded project requirement is private. This external setting must be
corrected separately with explicit visibility-change authority.
