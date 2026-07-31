# BC Vision RC17 — fast Windows updater pipeline

## Problem

The release-candidate workflow performed a complete Windows release for every
agent-branch push.  Each run deleted PyInstaller's `build` and `dist` trees,
used `--clean`, recreated the model seed, installed Inno Setup into temporary
storage, compressed both the full Setup and the updater, ran the complete
clean-install/update/uninstall gate and uploaded the full multi-file release.
On the HDD-backed self-hosted runner, retries and cancellations turned a small
application change into a multi-hour wait.

## Implemented split

- `.github/workflows/windows-fast-updater.yml` is the automatic pull-request
  path.  It retains ignored build state with `actions/checkout clean: false`,
  enables `BCVISION_INCREMENTAL_BUILD=1`, reuses the Python/PyInstaller/model
  caches and the verified Inno Setup installation, builds only the one-click
  updater, and uploads only the updater plus SHA-256.
- `.github/workflows/windows-release-candidate.yml` is now a manual final
  release gate.  It still performs the clean PyInstaller build, full Setup and
  updater build, clean installation, in-place update, offline ANPR checks,
  persistence checks and uninstall verification.  Its active run is not
  cancelled by a later commit.
- `BUILD_PORTABLE_EXE.bat` remains clean by default.  Incremental behavior is
  opt-in and cannot weaken the final release workflow accidentally.
- `scripts/ensure_inno_setup.ps1` caches the signed Inno Setup tool under the
  runner tool cache and verifies the Pyrsys B.V. Authenticode signature before
  first use.
- The fast update smoke gate runs packaged detector/OCR inference before and
  after the updater and verifies preservation of SQLite and AI model markers.
- Updater compression is `lzma2/fast` and non-solid; the full installer keeps
  its original maximum solid compression.

## Safety boundary

The fast artifact is an engineering/test updater, not a replacement for the
final full release gate.  Customer delivery still requires the manual clean
release workflow.  Database, settings, license data, snapshots, videos and
persistent AI models remain outside the application directory and are not
removed or replaced by either path.

## Expected effect

The first run may be cold.  Later runs on the same self-hosted runner avoid
re-downloading/reinstalling the AI runtime, avoid forced PyInstaller cache
destruction, avoid building the full Setup, avoid the uninstall gate and avoid
uploading the Setup/source bundle.  Actual elapsed time must be recorded from
the first warm GitHub Actions run before making a numeric speed-up claim.
