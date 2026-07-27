# BC Vision ANPR multi-frame capture — RC8

Version: `2.2.0-rc8`

## Implemented

- Added quality-aware latest-frame selection across every received frame.
- Preserved adaptive CPU throttling for expensive detector/OCR inference.
- Added optical-flow plate-box tracking on every displayed live frame.
- Compensated new detector boxes from their source frame to the current frame.
- Removed overlays when visual tracking evidence is lost instead of showing a
  stale box.
- Added immediate unreadable detection events with an empty canonical plate
  value, a real cropped plate image and a clear cropped vehicle image.
- Upgraded an unreadable event in place when multi-frame OCR later reaches a
  valid consensus, preserving a single database row and stable image paths.
- Added vehicle and plate thumbnails beside the recognized value in the
  dashboard's live recent-events report.
- Applied the same one-event upgrade behavior to offline video processing.
- Repaired the two pre-existing ANPR validation workflows by moving
  `runner.temp` model paths from unsupported job-level expressions to the AI
  test steps where the runner context is available.

## Safety and compatibility

- Ambiguous OCR remains `ناخوانا`; no plate value is inferred or guessed.
- SQLite schema is unchanged and no destructive migration is required.
- Existing databases, settings, users, licenses, videos, images and AI models
  remain compatible with the one-click updater.
- Full-frame detector/OCR calls remain serialized and adaptively rate-limited.

## Verification

- Focused ANPR, live-worker, stream, video and dashboard tests:
  `42 passed`.
- Full regression suite before the version/documentation update:
  `89 passed, 1 skipped`.
- Dedicated tests cover unreadable-to-recognized in-place upgrade, clear-frame
  selection from all received frames, optical-flow box motion, cropped media
  persistence and end-to-end uploaded-video processing without duplicate
  database events.
