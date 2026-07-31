# BC Vision RC18 — Visual ROI and still-only evidence

## Scope

- Per-camera hollow green ROI editor on the dashboard live image.
- Pointer drag and resize with explicit save/cancel controls.
- Normalized percentage storage in the existing camera ROI columns.
- Immediate live-worker configuration invalidation.
- Rejection of in-flight detections computed against the previous ROI.
- Full-resolution vehicle frame plus separate plate crop for every new live
  event when image persistence is enabled.
- Full-screen vehicle/plate preview on the dashboard and event pages.
- No video path attached to new live events and no event-detail video player.

## Persistence and compatibility

The feature reuses `cameras.roi_x`, `roi_y`, `roi_w` and `roi_h`; no destructive
schema migration is required. SQLite accepts the new two-decimal normalized
values in existing installations. Database, users, settings, licenses, models
and historical event/media rows remain untouched.

## Safety boundaries

- ROI values must be finite, fully inside the image and at least two percent
  wide/high.
- ROI editing requires `camera.manage` permission.
- Activity analysis and detector/OCR receive only the ROI crop.
- Saving a new ROI clears stale tracks, pending frames and overlays.
- A result already computing against an older ROI generation is discarded
  before consensus or disk persistence.
- Still evidence failures remain independent: a failed plate encode cannot
  erase the vehicle image or textual event, and vice versa.

## Verification state

`python -m compileall -q app tests` passed locally. Focused and full Pytest are
required on the GitHub Windows runner because the transient Codex environment
did not contain the pinned test packages and package installation was blocked
by the environment usage limit. Do not treat RC18 as release-verified until
those CI gates and the packaged updater preservation test complete.
