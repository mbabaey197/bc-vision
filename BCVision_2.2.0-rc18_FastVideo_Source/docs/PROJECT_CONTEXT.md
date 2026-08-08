# BC Vision — Project Context

## Repository

- GitHub repository: `mahdibabaey197/bc-vision`
- Default branch: `main`
- Product: BC Vision
- Current application version in source: `2.2.0-rc17`
- Windows persistent data root: `C:\ProgramData\BCVision\data`

## Release contract

Every production release must deliver all three outputs together:

1. Complete source archive
2. Windows installer
3. One-click updater that preserves the previous database, settings, snapshots, videos and AI models

Application data and AI models must remain outside the replaceable application directory.

## Current ANPR implementation

The Iranian plate-recognition subsystem now includes:

- verified lightweight Iranian YOLOv8 detectors through ONNX Runtime
- hardened OpenCV detector fallback
- low-light, high-exposure, blur and contrast preprocessing
- perspective correction for angled plates
- Persian, Arabic and English digit normalization
- spatial reconstruction of split OCR tokens
- position-aware OCR confusion repair
- segmentation-free Iranian CRNN+CTC recognition
- Iranian character CNN fallback
- image quality scoring
- ByteTrack-style two-pass association, Kalman filtering, Optical Flow and
  weighted multi-frame consensus
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
- independent bounded per-camera ONNX sessions
- conditional mild-motion deblurring with dedicated-AI reread
- per-camera ONNX sessions with a hard two-thread ceiling
- persistent feedback datasets and gated CRNN candidate promotion
- Jalali visible dates and Persian digits for all visible dates and times
- dormant signed RC14 OBB/CCT engine with baseline, shadow and next modes
- photo-calibrated private-plate renderer with fixed glyph cells, versioned
  geometry and renderer/font SHA-256 provenance
- mandatory OBB perspective rectification before candidate OCR
- five-best-frame probability voting and strict ambiguity rejection
- geometric-only track association; OCR text never links vehicle tracks
- Golden Dataset promotion gate and atomic runtime rollback
- temporal proof for operator-assisted confirmation: at least three
  independent full-plate frames over a non-zero confirmation span
- low-cost motion wake-up that bypasses idle backoff for entering vehicles
- temporal fixed-overlay suppression for NVR clock/name/date text
- globally optimized multi-vehicle association with trajectory direction
- immutable per-run training snapshots and active-checkpoint continuation
- fail-closed 40-sample/multi-slice Golden Dataset admission contract
- atomic Unicode-safe plate and vehicle evidence storage
- event media health/error tracking and historical storage-root access
- dashboard and event-report pagination with configurable dashboard row count
- canonical one/two-character partial plate search
- SQL-side Jalali date, Tehran-local time, camera-city and plate-region filters
- immutable per-event observation-city snapshots

## ANPR model execution contract

The production path uses a 12.6 MB primary ONNX plate detector at 416px. A
12.3 MB secondary ONNX detector at 640px runs only after a zero-result primary
pass, followed by hardened OpenCV geometry if needed. OCR first uses the
segmentation-free Iranian CRNN+CTC model; the 2.2 MB Iranian character CNN runs
only when CRNN has no valid result and eight real glyph regions exist.
The retired 119 MB combined `best.pt`, Ultralytics, EasyOCR and Tesseract are
not loaded or packaged. Vehicle attributes are calculated only after stable
plate consensus, immediately before event persistence.

Each camera inference is hard-capped to two native compute threads across
OpenMP, BLAS, OpenCV, ONNX Runtime and Torch. Deployments may reduce the per-camera budget
to one with `BCVISION_CPU_THREADS=1`; larger values are clamped to two.
Concurrent camera capacity is calculated from logical CPU count while
reserving two logical processors for decoding, the dashboard and Windows.
Each of the default three camera keys keeps a distinct bounded ONNX Runtime
session even when a small host permits only one simultaneous inference.
Every session uses `intra_op_num_threads <= 2`,
`inter_op_num_threads = 1`, sequential graph execution and disabled worker
spinning.

The detector and both OCR readers own a separate bounded Session and lock for
each camera key. Calls for one camera are serialized; different cameras can
infer concurrently within the global capacity. Additional cameras queue
safely and continue replacing stale frames.

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

The RC11 Windows Python 3.13 environment successfully imported and executed:

- Torch `2.13.0+cpu`
- TorchVision `0.28.0+cpu`
- OpenCV `5.0.0`
- EasyOCR `1.7.2`
- Ultralytics `8.4.106`

Persian EasyOCR reader creation and inference were verified with the persistent Arabic recognition and CRAFT detector models.
ONNX Runtime `1.28.0` and the verified CRNN model are RC11 Windows acceptance
dependencies; packaged inference remains a release gate until the candidate
workflow completes.

RC12 removes TorchVision, EasyOCR and Ultralytics from inference. Torch remains
packaged only for an administrator-initiated controlled CRNN training job;
normal camera processing uses ONNX Runtime.

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
- per-camera ONNX serialization and cross-camera concurrency
- conservative blur-recovery gating and result selection
- ByteTrack low-confidence association and Kalman identity continuity
- immutable operator sample capture and SHA-256 verification
- additive training-run database migration and gated candidate promotion
- Persian digits for all visible date/time fields
- traditional and end-to-end YOLO26-OBB output contracts
- signed candidate manifest verification and tamper rejection
- candidate shadow isolation and runtime rollback
- group-aware dataset splitting without track/plate leakage

The latest full-suite and system optimization result is recorded in
`agent-results/latest/PROJECT_OPTIMIZATION_PHASE_1.md`: `64 passed, 1 skipped`,
plus a successful headless runtime smoke and a 1,000-write SQLite concurrency
and integrity check.

RC16 engine hardening is recorded in
`agent-results/latest/ANPR_ENGINE_HARDENING_RC16.md`. Its locally executable
full regression result is `209 passed, 1 skipped`; the skipped check is the
existing opt-in real AI integration runtime gate. No new real-camera accuracy
claim was made.

RC17 completes the event evidence and report workflow: live and uploaded-video
events retain verified plate/vehicle images, the dashboard and report pages
are paginated, and report filtering operates in SQL over partial canonical
plates, observation city, plate-region code and Tehran-local Jalali time.
Media writes fail independently and atomically, existing event/database rows
are preserved by additive migrations, and retired media roots remain readable
only through validated containment plus an exact event reference.
The implementation and validation record is
`agent-results/latest/ANPR_EVENT_ARCHIVE_SEARCH_RC17.md`; its full local
regression result is `238 passed, 1 skipped`.

The latest synthetic private-plate fidelity work is recorded in
`agent-results/latest/ANPR_PLATE_REFERENCE_FIDELITY_V3.md`. Real reference
photos are measurement-only and never enter a dataset. Manifest schema 3 binds
each corpus to the exact renderer and separately licensed font stack. The
focused renderer/provenance/security result is `60 passed`, and the complete
local regression result is `329 passed, 1 skipped`; no real-camera accuracy or
activation claim is made.

A later severely overexposed and defocused rear-plate photograph was rejected
as a glyph/label reference and used only to calibrate the anonymous
`rear-plate-overexposed-defocus-v1` synthetic condition. The source image and
its identity remain outside Git and every dataset split. The bounded profile,
quality gate and scope are recorded in
`agent-results/latest/ANPR_REFERENCE_DEGRADATION_V4.md`; it affects only future
dataset builds and does not change or retrain the completed 30K model.

## Security hardening status

Phase 1 security hardening is included in the current optimization branch:

- uploaded video names are never used as filesystem paths
- video uploads are streamed with a 2 GiB limit and partial files are removed
- stored watchlist fields are HTML-escaped in event reports
- media access uses resolved path containment instead of string prefixes
- snapshot, plate, video and backup folders must be distinct children of the
  configured storage root
- retention cleanup refuses any folder outside the configured storage root
- camera, system, watchlist and video operations enforce server-side
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
events and settings. The empty dashboard contains instructions
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
Windows PR run `30348185238` loaded and executed the verified YOLO, EasyOCR,
ONNX Runtime and production CRNN assets successfully. Windows candidate run
`30348180982` built the portable executable, installer and updater, verified
the GUI subsystem, exercised clean install and in-place update, ran offline
ANPR self-tests before and after update, preserved SQLite and model markers,
and uploaded the release bundle. Accuracy comparison on labelled frames from
`01.mp4` and real multi-camera performance measurement remain release gates.

## RC12 ONNX modernization, tracking and controlled training

Version `2.2.0-rc12` replaces the remaining production ANPR components. Plate
localization now uses verified 12.6 MB and 12.3 MB Platrix YOLOv8 ONNX models;
the secondary model runs only on a primary miss. Full-crop CRNN remains the
first OCR reader, and a verified 2.2 MB Iranian character CNN is the only OCR
fallback. The 119 MB combined model, Ultralytics, EasyOCR and Tesseract are no
longer loaded, installed or collected by PyInstaller.

Track association is two-pass in the ByteTrack style: high-confidence
detections are matched first and remaining low-confidence detections can keep
an existing identity in the second pass. A constant-velocity Kalman filter
predicts and smooths boxes at detector times. Existing forward/backward
Optical Flow and template recovery continue to advance that filtered box on
every display frame between detector runs.

Every confirmed correction copies its plate crop into persistent immutable
training storage, records SHA-256 and assigns a deterministic train/validation
split. An administrator can start a bounded background CRNN training job only
after minimum sample and diversity thresholds are satisfied. Candidate models
are evaluated against the active model on the isolated validation split and
cannot be applied after a regression or below the minimum score. Promotion
re-verifies SHA-256, copies the model into persistent custom storage and
atomically updates the active manifest. Existing vendor and custom models,
events and feedback rows remain recoverable.

Visible application dates remain Jalali and every visible date/time digit is
Persian, including event time, replay timing and the demo
overlay. Database timestamps, filenames, API contracts and JavaScript numeric
control values remain machine-stable ASCII.

The final local RC12 regression is `130 passed, 1 skipped`; the skip is the
real-model integration test, which is enabled in the Windows AI acceptance
job. PR validation run `30359483088`, main run `30359720847` and Windows
release-candidate run `30359478238` passed. The candidate built the windowless
application, Setup and one-click Update, then verified clean install, offline
ANPR, in-place update, database/model preservation and standard uninstall.
PRs `#21` and `#22` were merged. Tag `v2.2.0-rc12` points to application merge
commit `273ebf43b7f12b20fd46e68f30ebcfab3784c113`; complete-release run
`30361052399` published the verified Setup, Update, exact Source ZIP and
SHA-256 manifest. Accuracy comparison on the customer's real `01.mp4` remains
a field validation gate and is not inferred from the model smoke tests.

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

## RC13 next ANPR engine scaffold

Version `2.2.0-rc13` adds the complete runtime boundary needed for the new
detector/OCR pair while preserving RC12 as the default customer path. The
candidate detector accepts both traditional YOLO OBB tensors and the
end-to-end `N x 7` contract, maps the rotated corners back through letterbox
geometry, and applies a perspective warp before OCR. The candidate OCR keeps
multiple constrained CTC hypotheses, calculates per-position margins and
returns `ناخوانا` when confidence or any character margin is insufficient.

Engine mode is `baseline`, `shadow` or `next`. Shadow executes the candidate
without changing persisted or displayed customer results. A runtime exception
in `next` atomically selects baseline. The candidate bundle requires a signed
Ed25519 manifest plus exact SHA-256 and size verification for both ONNX files.
No untrained or placeholder model file is included.

Track association no longer uses OCR text. It relies on predicted overlap,
normalized center distance and size consistency, preventing the same wrong OCR
string from joining two separate vehicles. Consensus uses the five
highest-quality observations and votes by character probability. A short
three-frame burst follows first plate visibility; exponential idle backoff
continues only while no plate is visible.

The Golden Dataset gate compares exact full-plate accuracy, false accepts,
latency and scenario slices. Related frames are grouped during
train/validation splitting so one vehicle or confirmed plate cannot leak
across both sets. Final promotion remains blocked until signed trained weights
and labelled real-camera videos are available.

Local RC13 regression is `143 passed, 1 skipped`. The skip is the existing real
ONNX integration test that requires the Windows model bundle. Windows
installer/update validation and real-video accuracy remain pending release
gates and are not claimed by this local test.

The real Hezar v2 preprocessing/decoder contract was subsequently completed:
blank index zero, full id-to-label mapping, mirrored `32x384` input,
reverse-time constrained decoding and normalized character margins. The
official checkpoint was exported locally to ONNX with maximum PyTorch parity
error `0.0000057220458984375` and tested on all 546 frames of `01.mp4`.
Baseline and the Hezar hybrid each matched 1 of 3 known truth plates; candidate
elapsed time was 92.751 seconds versus 67.542 seconds. The candidate therefore
failed promotion and RC12 remains the active customer engine. No model weight
was committed. RC13 regression after the decoder/export additions is
`145 passed, 1 skipped`.

## RC14 FastPlateOCR CCT candidate

Version `2.2.0-rc14` replaces Hezar as the intended next OCR architecture with
custom FastPlateOCR CCT-XS-v2 and CCT-S-v2 candidates. Hezar remains available
only for reproducing its failed benchmark. The CCT production reader uses
NumPy, OpenCV and ONNX Runtime only; Keras and Torch are confined to the
offline training environment.

The signed model manifest selects the OCR runtime. A CCT manifest entry must
also bind the alphabet, eight fixed output positions, input dimensions,
fixed NHWC layout, uint8 dtype, RGB conversion, resize policy and strict
ambiguity thresholds. The ONNX graph must expose the exact signed
`1x64x128x3 -> 1x8x37` contract. Unsupported runtimes, legacy engine IDs used
with CCT, and incomplete contracts fail closed to baseline.

Training data must carry a machine-readable provenance manifest. The
commercial path accepts explicitly attested company-owned or CC0 crops,
keeps complete vehicle/plate groups in one split, and rejects rows marked as
Golden, benchmark or test data. CC-BY is rejected until attribution can be
preserved end to end. Operator confirmation proves the label, not image
ownership: un-attested operator exports are therefore non-distributable
Shadow data until a separate rights attestation is supplied. The GPL-3.0
IR-LPR repository is
still excluded from the proprietary production model path, but the owner
requested an isolated research comparison on 2026-07-29. The dedicated
`prepare_ir_lpr_dataset.py` adapter therefore marks all IR-LPR derivatives
non-distributable and Shadow-only, preserves its official test split and
removes cross-split image/plate-identity leakage. A signed research bundle is
blocked from activating `next`. A reproducible synthetic Iranian plate
generator remains available to bootstrap commercial training without
importing a third-party plate dataset. Transfer learning copies only
shape-compatible feature backbone tensors from released FastPlateOCR models;
OCR/region heads and incompatible slot-query tensors are deliberately
reinitialized for the Iranian eight-position alphabet. Python 3.12 and
TensorFlow CPU form the validated offline training environment.

Promotion still requires both CCT variants to run on the exact same
operator-labelled Golden Dataset. The winner must improve exact full-plate
accuracy over RC12, avoid a false-accept or scenario-slice regression, and
remain faster than the recorded `67.542` second all-frame baseline on
`01.mp4`. A synthetic validation result alone can never activate `next`.

The exact fixed video was supplied again and verified by SHA-256 on
2026-07-28. Both Stage-2 candidates processed all 546 frames but matched
`0/3` known truth plates. CCT-XS emitted 214 rows containing 99 unverified
unique strings in `82.165` seconds; CCT-S emitted 204 rows containing 113
unverified unique strings in `153.530` seconds. Both are rejected and RC12
remains active. Inspection of the saved OCR crops showed that the source is a
four-camera composite and the fallback detector repeatedly selects OSD date,
clock and camera-name text. The next iteration needs a dedicated plate
detector plus labelled real company footage; synthetic OCR accuracy is not a
substitute.

The legacy `/ai/video-test` preparation-only page now runs actual full-frame
ANPR. It displays one row per tracked passage with the exact saved plate crop,
recognized text or `ناخوانا`, confidence, video time and OCR engine. The
standalone Golden benchmark can also persist every crop and render a
self-contained RTL HTML report with the crop and its recognized text on the
same row. Golden/test crops remain prohibited from entering training.

## RC15 PP-YOLOE-R and observable raw guesses

Version `2.2.0-rc15` implements the selected next detector runtime instead of
renaming the RC14 YOLO scaffold. A signed `bcvision-rc15` bundle can declare
`ppyoloe-r-onnx`; preprocessing supplies the three official PaddleDetection
inputs and the decoder consumes the official rotated boxes/scores tensors.
The contract, preprocessing thresholds and every model file remain covered by
the signed manifest and SHA-256 verification.

The detector data preparation tool accepts only company-owned/operator
confirmed, CC0 or CC-BY-4.0 images. It writes rotated COCO annotations for
PP-YOLOE-R and includes unlabelled hard-negative images so timestamp, camera
name, signs and headlights can be taught as background. Golden data and
train/validation source leakage fail closed.

Raw OCR hypotheses are now first-class diagnostic data. A rejected full-plate
hypothesis is shown together with confidence, engine, signed model revision
and rejection reason. In the owner-approved operator-assisted policy, a
complete multi-frame guess can become an operational
`review_status=auto-confirmed` event with a canonical plate value. This status
is distinct from strict multi-frame consensus and human confirmation:
`confirmation_source=ai-auto-guess`, `operator_reviewed=0` and
`experimental=1` remain set until an authorized operator confirms or corrects
the complete Iranian plate. AI-only guesses never become training labels.

Baseline and candidate Shadow results run through separate trackers on the
video-test page and are labelled by lane. In live operator-assisted mode, an
overlapping strict Baseline read always wins; a complete Shadow guess may
replace only an unreadable Baseline row. The resulting event is visibly
AI-confirmed and remains operator-reviewable. This does not activate the
research model as `next`, does not include it in an installer and does not
turn its output into ground truth. Operator feedback records exact-match and
Levenshtein character distance; Settings reports exact full-plate accuracy
and mean character error overall and per immutable model revision.

## Repository visibility authorization

The repository is private as of the 2026-07-30 GitHub audit. On 2026-07-28
the owner explicitly authorized publishing the RC13 branch and Draft PR;
that historical authorization does not make model weights, Golden evidence
or operator data suitable for public distribution.

## RC15 IR-LPR research result

The three official IR-LPR plate-crop archives were uploaded on 2026-07-29 and
passed ZIP integrity and SHA-256 checks. The importer produced 17,371 train,
2,007 validation and 3,903 test OCR crops with zero cross-split image or plate
identity overlap. Golden video data was not imported.

CCT-XS-v2 was fine-tuned in four two-epoch CPU stages from only the compatible
feature layers of the official FastPlateOCR global checkpoint. The final
Stage-4 ONNX achieved 87.45% raw exact accuracy and 97.61% character accuracy
on the untouched 3,903-image IR-LPR test split. Its SHA-256 is
`AD8D77D69CD0C914CB0CB3E0AC4E18709C446F78625A440D8F2D7AD2FB669482`.

The verified fixed Golden video then produced 807 detector candidates across
all 546 frames. Strict Tracker output matched `1/3`, equal to RC12; observable
raw OCR matched `2/3`. `55-ط-639-74` was exact in 19 observations and
confirmed. `84-ب-571-33` was exact in only one observation and correctly
failed the three-vote gate. The closest read for `31-ط-556-74` was
`31-ط-566-74`, one character away. There were 39 unverified emitted unique
strings, so promotion failed.

The derived model remains `research-shadow-only`, non-distributable and
excluded from installers because IR-LPR is GPL-3.0 research data. RC12 remains
active. The detailed record is
`agent-results/latest/ANPR_IR_LPR_RESEARCH_RC15.md`.

After reviewing the visual guesses, the owner approved using complete guesses
as operator-assisted events on 2026-07-29. The strict Golden result remains
`1/3`; the new label describes workflow state, not measured truth. Operators
can confirm an unchanged guess or correct it from the dashboard/event detail.
Only that human action creates `anpr_feedback`, captures the immutable crop and
allows later controlled training. The implementation record is
`agent-results/latest/ANPR_OPERATOR_ASSISTED_RC15.md`.

For private internal evaluation, RC15 also supports a signed
`baseline-yolov8-onnx` detector-reuse contract. Shadow CCT OCR runs on the
already verified Baseline detector crops, exactly matching the detector/OCR
pairing used for the Golden result and avoiding an untrained candidate
detector. `tools/build_internal_cct_model_installer.py` creates a private
one-click model pack outside Git; it verifies the embedded detector and OCR,
installs under the configured persistent data root and selects Shadow mode.
The public Setup, Updater, source archive and GitHub repository still exclude
the IR-LPR-derived weight.

## Commercial-clean synthetic plate generator v2

On 2026-07-30 the owner selected synthetic Iranian plate generation as the
first step toward replacing the GPL-derived research model. The procedural
generator now writes the exact CCT `128x64 RGB` / eight-slot contract and
records deterministic per-sample provenance without copying or transforming
any third-party plate image.

Ten balanced profiles cover clean, daylight, night, directional motion blur,
perspective, headlight glare, rain, dirt, simulated 68–155 pixel source width
and mixed hard conditions. Special plate colour families are represented and
known weak or missing classes `ژ، ث، ا، ف، ک، گ، D، S` receive extra sampling
only after all configured letters receive baseline coverage. Train and
validation plate identities are disjoint.

The same seed and inputs produce byte-identical annotations and JPEGs. The
manifest binds the font SHA-256 and approved license; a missing or unapproved
license fails closed. DejaVu Sans is only the permissively licensed bootstrap
font and is not claimed to be an exact production plate font.

A local 60-train/20-validation visual pilot covered all ten profiles with zero
identity overlap and no unusable-quality rescues. Focused regression is
`17 passed`. The implementation record is
`agent-results/latest/ANPR_SYNTHETIC_PLATE_GENERATOR_V2.md`.

The completed pilot contains 24,000 train, 3,000 validation and 3,000 held-out
test images from 3,000 unique plate identities. All ten profiles are balanced,
every image has the exact CCT shape and Train/Validation/Test plate identities
have zero overlap. CCT-XS-v2 was trained from random initialization for 30
epochs; epoch 24 produced the selected checkpoint. The ONNX SHA-256 is
`DE47B20DE2CF545276977E230EF07BD88657D1DFB9C7956C62873A7C731A966F`.

The synthetic Test split reached 99.57% raw exact-plate accuracy, 99.94%
character accuracy, 99.73% accepted precision and 0.20% rejection. Nine
profiles reached 100% raw exact accuracy; mixed-hard reached 95.67% and remains
the principal synthetic weakness.

The exact fixed `01.mp4` Golden video was recovered and verified by SHA-256.
An initial diagnostic run without the verified detector used the explicit
OpenCV fallback and correctly remained a rejection. The production primary
and secondary ONNX detectors were then restored from the pinned model
contract and independently verified by size and SHA-256. A strict rerun,
with OpenCV disabled, produced 807 candidates across all 546 frames in
62.163 seconds. The synthetic model matched `0/3` trusted plates through the
Tracker and only `1/3` as a one-frame raw exact guess; it emitted 23 rows with
21 unmatched unique strings. It therefore still fails activation. Metadata remains
`production-candidate` and distributable, but `activation_allowed=false` with
the `independent-real-camera-pass` gate. Runtime mode selection now enforces
that explicit activation flag. The benchmark also fails closed unless both
ONNX detector files pass their pinned size/hash contract; an OpenCV diagnostic
run or model preparation now requires an explicit command-line flag. A
tiny-contour fallback-detector crash found by the first benchmark has a
regression test.

RC12 remains active. The local application database contains no confirmed
operator samples, so no real-camera fine-tune was fabricated from Golden
truth or AI guesses. The operator-feedback schema-2 export can now be consumed
directly by the CCT dataset preparer, duplicate crops and conflicting labels
fail closed, and its default rights state is non-distributable/unverified.
Only a separate explicit ownership attestation can enter the commercial path;
all real-data candidates remain activation-locked until independent Golden
and real-camera gates pass. The internal start gate is aligned with promotion
at 24 Train samples, 12 Validation samples and eight unique plate identities.

The final 100,000–300,000 synthetic corpus remains blocked until
plate-faithful licensed glyphs are available and operator-labelled
real-camera transfer improves. The detailed record is
`agent-results/latest/ANPR_SYNTHETIC_30K_TRAINING_V1.md`.

## Company-owned crop batch 01 review

On 2026-07-30 the owner explicitly attested that the 505 plate-crop images in
private source archive `01.zip` were collected by the company. The archive
SHA-256 is
`83307A71EF3428739C20C286FB62DF0759C0572F803B77361D0CDCC95FFC5628`.
The source remains outside Git and is classified as
`operator-confirmed-company-owned`; the attestation evidence identifier is
`user-attestation-chat-2026-07-30-company-owned-01`.

The archive contains OCR crops rather than full vehicle frames, so it can
support OCR fine-tuning and hard-condition evaluation but cannot train the
plate detector or tracker. The established triage contains 204 good crops,
217 crops requiring careful review and 84 hard/test candidates.

`tools/build_plate_label_review.py` now creates a self-contained offline RTL
review page. Source images are embedded locally, browser progress can be
exported/imported, and the operator must explicitly choose Confirmed,
Unreadable or Excluded. Shadow OCR output is stored only as a visibly
untrusted draft suggestion and can never become a training label by itself.
The generated private review artifact is not committed.

A first partial operator export has now been validated: 82 Confirmed,
20 Unreadable, two Excluded and 401 Pending. The confirmed subset represents
79 unique plate identities. `tools/prepare_plate_review_dataset.py` binds the
CSV to the exact source archive and review-page provenance, verifies every
image digest, excludes all non-confirmed rows, and groups repeated plate
identities before creating the private 66-Train/16-Validation split. Plate and
digest overlap are both zero. Source crops and labels remain outside Git.

Two CCT-XS fine-tunes resumed the company-owned 30K synthetic checkpoint. The
stronger `5e-5` candidate improved development-crop character accuracy from
57.03% to 71.88% and full-plate exact accuracy from `1/16` to `2/16`.
Synthetic Test exact accuracy moved from 99.57% to 99.47%. The fixed 546-frame
real video still produced `0/3` Tracker matches and `1/3` raw exact truth, the
same as the synthetic base, while unmatched emitted unique strings rose from
21 to 23. Because the confirmed subset is tiny and 79 of 82 samples use one
letter class, no general OCR claim is made. The candidate remains private
Shadow-only and is not copied into the runtime, installer or `next` slot.
RC12 remains active.

The stronger candidate was then repeated with the exact same dataset
fingerprint, 66/16 split, source checkpoint, seed `20260730`, 30 epochs,
batch size 8 and learning rate `5e-5`. The development-crop result reproduced
exactly at `2/16` full plates and 71.88% character accuracy. The random
augmentation stream produced a different ONNX artifact; Synthetic Test exact
was slightly lower at `2982/3000` (99.40%), the fixed video again returned
`0/3` Tracker and `1/3` raw exact, and unmatched emitted strings increased
from the first candidate's 23 to 27. The repeat is therefore not promoted and
remains private Shadow-only.

The focused importer/review/security/dataset suite passes `64` tests. Detailed
records are
`agent-results/latest/ANPR_COMPANY_CROPS_01_REVIEW.md` and
`agent-results/latest/ANPR_COMPANY_CROPS_01_FINE_TUNE.md`.

## OCR geometry and temporal-safety correction

The earlier `2/16` metric described exact OCR reconstruction, not human
readability. Conservative review found at least 14 of the 16 development crops
human-readable, confirming an engine/domain problem.

The CCT runtime now supports a signed
`stretch-letterbox-geomean-v1` profile. It runs a stretched and an
aspect-preserving view, deduplicates identical tensors, fuses normalized
probabilities in log space, and requires both view strings plus agreement of
at least `0.75`. Rejected strings are review-only and cannot re-enter strict
temporal consensus or strong identity association.

With the first `5e-5` candidate, the company development split improves from
`2/16` to `8/16` exact and from 71.88% to 82.03% character accuracy. Four
reads pass the strict gate and all four are correct. The 3,000-image synthetic
raw exact count remains `2984/3000`. This is a tuning result, not an
independent test: the split has already been reused for checkpoint,
learning-rate and preprocessing selection.

Tracker confirmation now requires the same complete plate in at least three
independent observations over the minimum time span. Positional hybrids cannot
confirm, emitted identities are immutable, similar later vehicles can start a
new track, confirmed database events cannot be downgraded or overwritten, and
a clearer frame of the same identity refreshes the existing event. Live track
age is capped at six seconds.

On the fixed 546-frame / 807-detection video, the corrected path remains `0/3`
Tracker exact and `1/3` raw exact, but unmatched emitted unique strings fall
from the previously recorded 23 to one. This is a major false-emission
reduction but still fails activation.

A separate 30-epoch aspect-preserving Fine-tune used the same private 66/16
split and produced ONNX SHA-256
`61C9A8426D04EA4E2FCEE4E547364F60AEB038FE4D08D999A15CCBE26E07412D`.
It reached only `5/16` development exact, `2976/3000` synthetic exact and
`0/3` video Tracker exact with two unmatched strings, so it was rejected.

RC12 remains active; both candidates are Shadow-only and
`activation_allowed=false`. Full verification is `360 passed, 1 skipped`
(AI integration runtime disabled). The detailed record is
`agent-results/latest/ANPR_OCR_ENGINE_GEOMETRY_V1.md`.
