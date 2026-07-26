# Latest Agent Result — Security Hardening Phase 1

Date: 2026-07-26
Status: implemented and validated on `agent/security-hardening-phase-1`;
review/merge pending

## Applied

- Replaced client filenames with generated server-side video filenames
- Added chunked uploads with a 2 GiB limit and partial-file cleanup
- Escaped stored owner and vehicle values in the event table
- Replaced media path prefix checks with resolved path containment
- Required all storage data folders to be distinct children of the storage root
- Blocked retention cleanup outside the configured storage root
- Added permission checks for camera, settings, storage, AI, license, backup,
  watchlist and video-processing routes
- Defined explicit permissions for admin, system, operator and guard roles

## Verification evidence

- Full suite: `36 passed, 1 skipped`
- Python compile check: passed
- Git whitespace/error check: passed
- New security regression tests: 8 passed
- The skipped test requires the real external ANPR model fixture

## Files changed

- `app/main.py`
- `tests/test_web_security.py`
- `docs/PROJECT_CONTEXT.md`
- `agent-results/latest/SECURITY_HARDENING_PHASE_1.md`

## Not yet claimed

- The changes are not merged into `main` yet
- This phase does not fix the default administrator password or trial-license
  tamper resistance
- Backup consistency and live database migration remain phase 2 work
