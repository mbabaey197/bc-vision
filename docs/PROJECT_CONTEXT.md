# BC Vision — Project Context

## Repository

- GitHub repository: `mahdibabaey197/bc-vision`
- Default branch: `main`
- Product: BC Vision
- Current application version in source: `2.2.0-rc8`
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

## ANPR model execution contract

The verified `best.pt` asset is a combined Iranian plate/character model, not
a single-class plate-only detector. Full-frame inference must select class
`30` (plate), followed by a character-only inference on the cropped plate.
Generic EasyOCR is reserved for the OpenCV/no-model fallback. Vehicle
attributes are calculated only after stable plate consensus, immediately
before event persistence.

The CPU-oriented defaults are 640px for plate localization, 416px for
character recognition, no test-time augmentation, and at most four plate
candidates per processed frame. A successful zero-result model inference
triggers the geometry fallback for difficult small, oblique or overexposed
plates. The live worker rate-limits this second pass adaptively from measured
processing latency so it does not run continuously on a slow CPU.

The model loader reapplies a six-thread maximum CPU budget after Ultralytics
model construction, because Ultralytics can otherwise reset Torch to use
nearly all logical processors. At least one logical processor is left
available on smaller systems. Deployments can override this with
`BCVISION_CPU_THREADS=1..8`.

Ultralytics predictor state is not shared safely across simultaneous
`predict()` calls. Live-camera workers therefore serialize the complete
plate-localization and character-read transaction on the shared model while
continuing to replace stale queued frames. This prevents silent empty
detections and avoids concurrent CPU oversubscription.

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

Live streams now draw a green box around valid plate candidates and an amber
box around unreadable candidates. Dashboard video cards are capped at a
smaller width, and the old system-status card beside recent plate events was
removed; technical status remains available from Settings.

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
