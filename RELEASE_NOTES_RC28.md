# BC Vision 2.2.0-rc28 — No-license recovery build

Base: `agent/anpr-detection-recovery-rc25`

- Runtime license enforcement is disabled unconditionally in the packaged app.
- No `license.dat` is generated or required.
- The activation menu, activation routes, plan label, and camera-count gate are
  removed from the application UI.
- All product features and runtime cameras are enabled. The compatibility
  status API advertises a 4096-camera ceiling, but the application no longer
  enforces that value when cameras are added.
- Existing license/state files are ignored and left untouched.
- The original offline implementation remains only for source regression tests;
  environment variables cannot restore it in the packaged customer runtime.
- The updater no longer requires or copies `license_public_key.pem`.
- The packaged updater gate runs with no license file and fails unless the
  installed EXE reports `no_license_ready=true`.
- Database, media, settings, model files, and storage configuration remain in
  ProgramData and are not replaced by the updater.
- The verified RC25 recovery base is 292,482,606 bytes with SHA-256
  `60b431045558106cff4d9e435bcbe90eb8ee571206f1468adf833728a6b72acc`.
