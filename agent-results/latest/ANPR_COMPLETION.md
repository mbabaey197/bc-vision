# Latest Agent Result — ANPR Completion

Date: 2026-07-26
Status: source merged to `main`; production release pending

## Applied

- Robust Iranian plate detection with verified YOLO model support
- Multi-strategy OpenCV fallback for hard scenes
- Perspective correction
- Persian multi-part OCR reconstruction
- Position-aware OCR error repair
- Exposure, denoise, sharpen and threshold variants
- Quality-aware confidence calculation
- Multi-frame voting and tracking
- Duplicate suppression and direction estimation
- Video ROI processing
- Nonblocking live camera worker
- Automatic startup of all enabled cameras
- Persistent live event storage
- Database migration preserving existing data
- Persistent verified AI model storage
- Windows AI runtime/build configuration
- Regression and AI smoke-test workflows

## Verification evidence

- Plate-rule suite: 52/52 passed
- OCR/ANPR isolated suite before auto-start: 20 passed
- Full isolated suite after auto-start: 22 passed
- Windows fast ANPR CI passed on the completed core implementation
- Torch operation result: `10.0`
- EasyOCR Persian reader created successfully
- EasyOCR inference completed successfully
- Observed split OCR tokens were reconstructed to the expected Iranian plate layout

## Merge commits

- Core ANPR: `06320a424837cce9044e52531b530fe302b10be4`
- Background continuous recognition: `d04a2bc64e5ecf546754f0d069cb971483329333`

## Not yet claimed

- No claim of 100% recognition under physically unrecoverable imagery
- No new Windows installer has been built from these commits yet
- No new one-click updater has been built from these commits yet
- Final accuracy still requires real day/night RTSP video from the deployment camera and lane

## Next production step

Build and validate the next BC Vision release as three coordinated artifacts:

1. Source archive
2. Windows installer
3. Database/model-preserving one-click updater

Run real-camera acceptance tests before tagging the release.
