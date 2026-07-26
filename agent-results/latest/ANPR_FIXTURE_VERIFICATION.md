# ANPR Real Video Fixture Verification

## Status

The user-supplied video was independently verified before any acceptance claim.

- Expected repository path: `tests/fixtures/anpr/01.mp4`
- Repository status at verification time: file not yet committed
- Verification source: user-uploaded binary file
- Verification result: exact metadata match

## Verified metadata

- File name: `01.mp4`
- Size: `24097556` bytes
- SHA-256: `b5193d8cf32d79daf17e15bea0b1c74e05156a70eddf49ca9e5d0466e568705d`
- Container: ISO Base Media / MP4
- Video codec: H.264
- Resolution: `1920x1080`
- Frame rate: `8/1` FPS
- Duration: `68.250000` seconds
- Frame count: `546`

## Validation performed

- Binary size checked with the operating system
- SHA-256 calculated from the complete binary
- Container signature identified as MP4, not a Git LFS pointer
- Stream metadata read with `ffprobe`
- Existing regression suite executed against the verified source snapshot: `28 passed, 1 skipped`

## Acceptance status

Real-video ANPR acceptance is **not complete**.

The production acceptance run requires the exact binary to be committed at `tests/fixtures/anpr/01.mp4` so the Windows AI runner can use the real YOLO model, EasyOCR models, production preprocessing, tracking and consensus pipeline. No plate result, Ground Truth match, False Positive count or False Negative count is claimed by this verification step.

A local fallback-only diagnostic was intentionally stopped because the isolated environment did not contain Ultralytics, EasyOCR or the verified detector model. Treating that degraded run as a production ANPR result would be invalid.
