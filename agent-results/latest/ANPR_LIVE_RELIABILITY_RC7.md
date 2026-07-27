# ANPR Live Reliability RC7

Date: 2026-07-27
Branch: `agent/anpr-live-performance-rc7`
Version: `2.2.0-rc7`

## Confirmed root causes

- Live detection rectangles were drawn on a display copy, but stream resizing
  encoded the original unannotated frame. Resizing now preserves overlays.
- The dashboard recent-events table was rendered only on initial page load.
  It now checks for new committed events every 750 ms and replaces the table
  only when its latest event ID changes.
- A successful YOLO inference with zero boxes incorrectly prevented the
  geometry detector from trying difficult small, oblique or overexposed
  Iranian plates. Geometry fallback now runs after a YOLO miss.
- Continuous inference could keep a slow CPU saturated. The worker now derives
  a bounded minimum interval from measured inference latency while retaining
  the newest selected frame and the original image resolution.

## Validation

- Full local regression: `85 passed, 1 skipped`.
- Added regression coverage proving the live rectangle survives stream resize.
- Added regression coverage proving a YOLO miss reaches geometry fallback.
- Added regression coverage proving slow inference is adaptively spaced.
- Added regression coverage proving committed events are returned to the
  dashboard immediately without a full-page reload.
- Python compilation of application and tests completed without error.
- Packaging version metadata and both Inno Setup outputs are aligned to
  `2.2.0-rc7`.

## Safety and compatibility

- Three-observation character consensus remains mandatory; ambiguous plates
  are still reported as unreadable rather than guessed.
- Database schema and persistent user data are unchanged.
- The one-click updater remains sufficient; a full reinstall is not required.
