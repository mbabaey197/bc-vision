# ANPR Video Processing Visibility RC5

Date: 2026-07-27
Branch: `agent/anpr-video-processing-visible-rc5`
Version: `2.2.0-rc5`
Status: published in draft PR #11; Windows candidate build pending

## Root causes corrected

- Uploaded CCTV files that OpenCV could open without returning frames now
  fall back to the bundled FFmpeg/PyAV decoder.
- Dashboard camera cards now expose received and processed frame counts,
  detector candidates, emitted events, model readiness and processing errors.
- The live consensus window expands from measured processing latency, so three
  observations remain available on slower CPU-only systems.
- Detector and EasyOCR models are copied from a verified seed embedded in the
  Windows package before any network download is considered.
- The installed executable's release gate loads YOLO, runs a real inference,
  initializes EasyOCR offline, then repeats the check after one-click update.

## Validation

- Full local regression: `78 passed, 1 skipped`
- Uploaded-video end-to-end route:
  - upload endpoint accepted the file
  - virtual camera was registered
  - a real JPEG frame was produced
  - frames reached the background ANPR worker
  - three observations reached consensus
  - the plate event was inserted in SQLite
- OpenCV decode failure exercised the FFmpeg/PyAV fallback.
- Three-second simulated CPU processing preserved consensus observations.
- Packaged detector and EasyOCR seed installation was tested without download.

## Safety

- Unreadable or ambiguous plates remain unreadable; no plate character is
  guessed.
- Database, settings, license files, media and existing AI model data remain
  under the persistent data directory and are not replaced by the updater.
- `main` is unchanged. Windows installer/updater artifacts are not deliverable
  until the GitHub Actions Windows gate passes for this exact commit.
