# BC Vision — Project Context

## Repository

- GitHub repository: `mahdibabaey197/bc-vision`
- Default branch: `main`
- Product: BC Vision
- Current application version in source: `2.1.0`
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

Latest isolated full-suite result after background auto-start: `22 passed`.

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

The ANPR source is merged into `main`; a new installer/release has not yet been produced from these commits.
