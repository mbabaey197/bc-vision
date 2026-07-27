# Windowless Windows Service Host

Date: 2026-07-27
Branch: `agent/windows-background-service-rc3`
Version: `2.2.0-rc3`
Status: implemented and locally validated; Windows build validation pending

## Applied

- Kept Uvicorn and enabled-camera ANPR workers running in the packaged
  background process without a Tkinter keep-alive window.
- Added a Windows runtime safeguard that hides any accidentally attached
  console in frozen builds.
- Changed the source launcher to hand off to `pythonw.exe` and exit its batch
  process instead of keeping `python launcher.py` in a visible command window.
- Applied `CREATE_NO_WINDOW` and hidden startup flags to PowerShell/WMIC
  hardware-identity probes.
- Added a PE-header verifier that requires the packaged executable to use
  `IMAGE_SUBSYSTEM_WINDOWS_GUI`.
- Added the PE verification as a blocking Windows release-candidate gate.

## Validation

- Focused launcher, licensing and packaging tests: `10 passed`
- Full regression: `70 passed, 1 skipped`
- Python compile-all: passed
- Headless runtime smoke:
  - `/api/health` returned service `bc-vision`, status `ok`, version
    `2.2.0-rc3`
  - `SIGINT` shutdown completed with exit code `0`
- Full regression and Windows installer/updater acceptance: pending

## Preserved behavior

- The browser remains the only visible application window.
- Enabled cameras and ANPR start automatically with BC Vision.
- A second launch opens the existing service panel instead of starting a
  duplicate server.
- Shutdown still stops all camera workers cleanly.
- Database, settings, license data, media and AI models remain outside the
  replaceable application directory.
