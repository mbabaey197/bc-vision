# BC Vision RC17 — event evidence, pagination and search

## Outcome

RC17 completes the storage and retrieval path around an ANPR decision. A live
or uploaded-video event can retain a verified plate crop, a vehicle evidence
image and its source-video reference. The dashboard and event archive expose
those files with bounded pagination and server-side filters.

This release does not claim a new recognition-accuracy result and does not
change the RC16 Golden Dataset admission policy.

## Media persistence

- Plate and vehicle JPEGs are encoded in memory and published with a temporary
  file, `fsync` and atomic replacement.
- Unicode/Persian Windows paths do not pass through OpenCV's filename API.
- A missing detector crop is reconstructed from the clipped plate bounding
  box.
- Vehicle evidence uses the vehicle ROI produced by the lightweight vehicle
  analysis stage and marks the detected plate.
- Plate and vehicle writes fail independently; text events are retained even
  when one file cannot be written.
- `media_status` and `media_error` distinguish complete, partial, missing,
  disabled and failed evidence.
- Legacy migration verifies file existence and non-zero size instead of
  trusting a non-empty path string.
- Uploaded-video results are archived in `plate_events` with image, plate and
  video paths plus the event time inside the video.

## Dashboard and reports

- The dashboard defaults to 12 events per page and can be configured from 6
  through 50.
- Previous, next, numbered-page and record-count controls are rendered below
  both the dashboard events and the full event report.
- Dashboard pages use an event-ID snapshot so a new event cannot shift older
  pages while an operator is reviewing them.
- Polling refreshes a newly inserted event and an update to the current event,
  while avoiding replacement during operator input.
- The event report supports 25, 50 or 100 rows per page.

## Search contract

All archive filters execute in SQLite before `LIMIT/OFFSET`:

- partial canonical plate text, including one or two Persian, Arabic or Latin
  digits;
- explicit observation city captured from the camera at event time;
- two-digit plate-region code, kept separate from observation city;
- camera, watchlist status, vehicle type and vehicle color;
- Jalali date and Tehran-local time boundaries converted to UTC for indexed
  comparison.

Location descriptions such as a gate or parking entrance are never inferred
to be a city. No city name is guessed from a plate-region code.

## Compatibility and security

- Database migration is additive and preserves users, cameras, events,
  settings and existing paths.
- A storage-root change records only validated former media roots. Historical
  files require both containment in such a root and an exact event reference.
- Media delivery accepts only supported image/video extensions.
- A failed virtual-camera startup removes only the new staged camera and file;
  previous virtual cameras remain intact.
- Watchlist matching consistently uses canonical `plate_norm`, including in
  event details with Persian or Arabic digits.

## Validation

- Focused RC17 storage, migration, dashboard, search, timezone, video and
  security checks: `75 passed`.
- Full local regression suite: `238 passed, 1 skipped`.
- The skipped test is the existing opt-in real-model integration gate.
- Python compilation and `git diff --check` both completed successfully.
