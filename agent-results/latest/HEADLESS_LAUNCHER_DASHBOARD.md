# Headless Launcher and Clean Dashboard

Date: 2026-07-26
Status: implemented and validated on
`agent/headless-launcher-clean-dashboard`; merge pending

## Applied

- Removed the persistent Tkinter service window.
- Kept Uvicorn alive in the main process.
- Retained automatic browser opening after verified startup.
- Added an unauthenticated, identity-bearing `/api/health` endpoint.
- Rejected unrelated applications occupying port 8000.
- Preserved clean camera-worker shutdown when the process stops.
- Stopped creating an enabled synthetic camera in new databases.
- Added a one-time exact-match migration for the historical built-in sample.
- Preserved real cameras, user-created demos, events, settings and licenses.
- Removed the camera glyph and sample-animation wording from the empty
  dashboard.

## Verification evidence

- Launcher/database targeted tests: `7 passed`
- Full regression suite: `34 passed, 1 skipped`
- Python compile-all: passed
- Git whitespace validation: passed
- Isolated runtime smoke test:
  - Uvicorn started on `127.0.0.1:8000`
  - `/api/health` returned HTTP 200 with service `bc-vision`
  - SIGINT produced a clean Uvicorn shutdown

## Not claimed

- The Windows PyInstaller executable and installer have not yet been built on
  a Windows runner for this branch.
- Other previously identified security, storage, licensing and reporting work
  remains separate from this focused change.
