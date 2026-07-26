# BC Vision — Project Context

## Repository

- GitHub repository: `mahdibabaey197/bc-vision`
- Default branch: `main`
- Product: BC Vision
- Current application version in source: `2.1.0`
- Windows persistent data root: `C:\ProgramData\BCVision\data`

## Release contract

Every production release must deliver these two outputs together:

1. Complete source ZIP: `BC_Vision_Source_vX.Y.Z.zip`
2. Standard Windows installer: `BC_Vision_Setup_vX.Y.Z.exe`

Portable, separate updater, one-click update and patch-installer artifacts are not release outputs.

The standard installer must upgrade an existing installation without deleting the previous database, users, settings, license data, snapshots, videos or verified AI models. Application data and AI models must remain outside the replaceable application directory. Every delivered output must have a verified SHA-256 entry in `SHA256SUMS.txt`.

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
- per-character voting with ambiguity rejection
- canonical duplicate suppression
- vehicle direction estimation
- ROI processing for uploaded videos and live cameras
- asynchronous nonblocking live-camera recognition
- automatic background startup for every enabled camera
- safe shutdown and reconnect behavior
- backward-compatible SQLite migrations and canonical `plate_norm`
- persistent, SHA-256-verified AI model management
- AI-aware PyInstaller build configuration

## Important commits

- Iranian plate rules: `16f0a5354125ff90c9d45b253852e5dc56b239dc`
- Robust ANPR pipeline merge: `06320a424837cce9044e52531b530fe302b10be4`
- Continuous background camera ANPR merge: `d04a2bc64e5ecf546754f0d069cb971483329333`
- Strict character-voting and ambiguity fix: `38d921175cb4e8716e7bf84d72bd9c66cfb1a0c7`
- Real-video fixture metadata registration: `e9286d23fc59ba610c067e88d7c62eb211f58fbd`

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
- per-character minimum votes, agreement and ambiguity margin
- single-frame event rejection
- duplicate cooldown
- image quality scoring
- video event generation
- live worker nonblocking behavior
- background camera auto-start and clean shutdown
- backward-compatible database migration
- model hash verification

Focused strict-voting regression result: `17 passed / 17 total`.
Randomized consensus result: `298 exact`, `2 safely rejected`, `0 incorrect emitted` across `300` scenarios.

## Real-video fixture status

Expected path: `tests/fixtures/anpr/01.mp4`

Registered identity:

- Size: `24097556` bytes
- Duration: `68.25` seconds
- Resolution: `1920x1080`
- FPS: `8`
- Codec: `H.264`
- SHA-256: `b5193d8cf32d79daf17e15bea0b1c74e05156a70eddf49ca9e5d0466e568705d`

The metadata file exists at `tests/fixtures/anpr/README.md`, but the MP4 fixture itself must be verified in GitHub before any real-video result is accepted.

## Operational truth

No software can recover a plate that contains no usable pixels, is completely hidden, is outside the frame, or is destroyed by severe motion blur/overexposure. BC Vision uses multiple software recovery layers, but camera placement, optical focus, shutter speed, resolution, night illumination and plate pixel height still determine the upper accuracy limit.

## Remaining acceptance work

Before declaring a production release, perform these validations with real customer-like data:

- verify and process the complete registered `01.mp4` fixture
- evaluate all confirmed ground-truth plates with zero hidden false positives
- real RTSP camera during day and night
- fast-moving vehicle and motion blur
- angled entry/exit lanes
- dirty, damaged and partially occluded plates
- multiple simultaneous cameras
- long-duration CPU and memory stability
- build and isolated-test the standard Windows installer
- reinstall/upgrade through the standard installer while preserving the database and models
- build and verify the complete source ZIP

The ANPR source is merged into `main`; the registered real-video fixture is not currently available at its required repository path, so real-video validation is not yet complete.
