# BC Vision video upload/live-source verification

Date: 2026-07-27
Branch: `agent/video-upload-live-source`
Base optimization commit: `811859a`

## Root cause

`POST /cameras/video-upload` saved the upload and then called
`process_video(...)` synchronously inside the web request. Full AI inference
therefore blocked the request-serving thread until all selected frames had
been processed. The saved file was never registered with `StreamManager`, so it
could not appear in Live View.

## Implemented behavior

- The browser displays actual upload progress.
- The endpoint validates and stores the video without batch ANPR inference.
- One active `video://` virtual camera is registered using the selected
  camera's ANPR settings.
- A new upload replaces the previous uploaded-video source, preventing
  accumulation of virtual streams.
- The virtual camera starts immediately in `StreamManager` and appears in the
  dashboard live grid.
- The existing nonblocking live ANPR worker receives the frames.
- End-of-file seeks to frame zero and playback continues.

## Verification

- Focused suite: `16 passed`
- Full regression: `66 passed, 1 skipped`
- Python compileall: passed
- Git whitespace check: passed
- Endpoint test confirmed JSON success, virtual-camera persistence and stream
  startup without invoking full-video batch processing.
- Real OpenCV stream test read beyond the number of source frames, proving the
  uploaded video looped, and then stopped its thread cleanly.

## Remaining release gate

Build and test the Windows installer/updater from the final commit. Installation
acceptance must include upload progress, dashboard playback, responsive
navigation during playback, clean application shutdown and preservation of the
existing database and settings.
