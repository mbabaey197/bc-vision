# RC28 validation checkpoint

## Recovery base

- Branch: `agent/anpr-detection-recovery-rc25`
- Head used for the patch: `c91f4afa8efa51435aefbff8334d7b44a9244a1d`
- Verified RC25 updater size: `292482606` bytes
- Verified RC25 updater SHA-256:
  `60b431045558106cff4d9e435bcbe90eb8ee571206f1468adf833728a6b72acc`

The previously downloaded 114,556,928-byte executable is not the verified
recovery build and must not be used as the RC28 base.

## Completed checks

- RC25 Windows workflow run `31251883940` completed `430 passed, 19 skipped`
  source tests, built the application and updater, and printed
  `Fast updater smoke test passed for 2.2.0-rc25` after the packaged-ANPR and
  persistent-data checks.
- That workflow failed only while uploading the finished artifact because the
  repository's Actions artifact storage quota was full.
- RC28 Python sources compile successfully, and all local no-license/UI/build
  contract gates pass.
- RC28 starts in no-license mode without creating `license.dat`.
- Feature checks, runtime-camera checks, and camera-capacity checks are
  permissive in the default packaged mode.
- The license menu, activation endpoints, activation forms, plan display, and
  camera-count gate are absent from `app/main.py`.
- The legacy implementation can be exposed only inside a non-frozen pytest
  process with `BCVISION_LICENSE_REGRESSION=1`; customer environment variables
  cannot re-enable enforcement.
- A stale invalid license and state files are ignored without being modified.
- The packaged build does not require or copy a public license key.

## Final release gate

The Windows updater must still be built from this overlay, pass
`scripts/verify_fast_update.ps1`, and have its final SHA-256 recorded before it
is distributed as `BCVision_Update_v2.2.0-rc28.exe`.
