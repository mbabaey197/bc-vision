# BC Vision — Project Context

## Repository

- GitHub repository: `mahdibabaey197/bc-vision`
- Default branch: `main`
- Product: BC Vision
- Current application version in source: `2.2.0-rc1`
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
candidates per processed frame. A successful zero-result model inference does
not trigger the expensive OpenCV fallback.

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

The latest full-suite result is recorded in
`agent-results/latest/ANPR_VIDEO_PERFORMANCE_FIX.md`.

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
`2.2.0-rc1` includes an isolated executable self-test plus installer/updater
data-preservation verification. A production release has not yet been
declared.
