# BC Vision — Project Optimization Phase 1

Date: 2026-07-27
Branch: `agent/project-optimization-phase-1`
Version: `2.2.0-rc2`
Status: locally implemented and validated; Windows build and merge pending

## Baseline

- Starting commit: `50cba8e3d01dd7897a479ed673422fb3e4f4b407`
- Starting suite: `41 passed, 1 skipped` in 1.19 seconds
- Reference video: 546/546 decodable frames, 1920x1080, 68.25 seconds
- Reference video SHA-256:
  `B5193D8CF32D79DAF17E15BEA0B1C74E05156A70EDDF49CA9E5D0466E568705D`
- The ANPR implementation under `app/ai` is unchanged from the validated
  starting commit.

## Applied

### Web and authorization

- Enforced explicit server-side role permissions for camera, storage, system,
  license, watchlist, backup and video-processing operations.
- Replaced media path prefix checks with resolved path containment.
- Generated upload names server-side, streamed video uploads, enforced a
  2 GiB limit and removed partial files after failure.
- Escaped stored watchlist fields in event reports.
- Added anti-framing, MIME-sniffing, referrer and browser-feature headers.
- Removed license and customer data from the unauthenticated legacy health
  endpoint.
- Neutralized spreadsheet formula prefixes in exported CSV text cells.

### Data safety and performance

- Enabled SQLite WAL mode, a 20-second busy timeout, foreign-key enforcement,
  NORMAL synchronous mode and bounded WAL auto-checkpointing.
- Replaced live database file copying with SQLite's online backup API.
- Added an integrity check and atomic publication for every database backup.
- Made storage-root migration build and update the destination database before
  atomically switching the bootstrap configuration.
- Refused a destination that already contains BC Vision persistence files,
  preventing accidental replacement or switching to stale data.
- Kept the previous database unchanged so a failed or abandoned storage move
  remains recoverable.

### Runtime and first run

- Runs Uvicorn in the main process without a Tkinter keep-alive window.
- Opens the browser only after the identity-bearing health endpoint succeeds.
- Rejects an unrelated application already using port 8000.
- Stops creating an enabled synthetic camera in new databases.
- Removes only the exact historical built-in demo camera through a one-time
  migration while preserving user cameras and existing data.

### Build and CI

- Advanced the release candidate consistently to `2.2.0-rc2`.
- Added a regression test that keeps VERSION, application and Inno Setup
  metadata synchronized.
- Made the Windows candidate workflow work for any `agent/**` branch.
- Made the artifact name derive from VERSION.
- Added concurrency cancellation for superseded runs.
- Removed duplicate PR execution from the legacy main-branch workflow.
- Expanded PR validation to all application, installer and packaging changes.
- Added the exact web-test dependencies required by the locked Starlette
  runtime.

## Verification evidence

- Full local regression: `64 passed, 1 skipped` in 1.46 seconds
- Targeted database, security, launcher and packaging regression: passed
- Python compile-all: passed
- Workflow YAML parse: passed
- Git whitespace check: passed
- Tracked-file credential/private-key signature scan: no matches
- Headless runtime smoke:
  - `/api/health` returned service `bc-vision`, status `ok`, version
    `2.2.0-rc2`
  - SIGINT shutdown completed with exit code 0
- SQLite concurrency smoke:
  - 4 writer threads
  - 1,000/1,000 committed events
  - 3,910.6 writes/second on this test host
  - `PRAGMA quick_check=ok`
  - `journal_mode=wal`

The throughput number is host-specific and is evidence of this isolated test,
not a customer performance guarantee.

## Outstanding high-priority work

- GitHub currently reports the repository visibility as **public**, while the
  recorded project requirement is private. Visibility was not changed
  automatically because it affects repository access and integrations.
- A fresh installation still exposes the documented default administrator
  password until the operator changes it. A forced first-login password change
  should be implemented before production.
- Trial-license rollback/tamper resistance still needs a hardened design.
- `app/main.py` remains a large mixed UI/routing/service module and should be
  split incrementally only after route-contract tests cover every page.
- Production acceptance still needs day/night RTSP, multi-camera endurance,
  customer-like ground truth and a real Windows installer/update run for this
  exact commit.

## Not claimed

- No production release or merge to `main` has been performed.
- No new ANPR accuracy claim is made in this phase.
- The full AI video benchmark was not rerun locally because this phase does not
  change `app/ai`; the real-model Windows gate remains required before merge.
