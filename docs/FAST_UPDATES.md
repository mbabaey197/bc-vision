# BC Vision transactional fast updates

RC30 is the immutable full-runtime base. It installs Python, native libraries,
AI runtimes, `BCVision.exe`, the verified `app` payload and the stable runtime
loader once. A normal update after RC30 contains only the versioned `app`
payload inside a small one-click updater.

## Release paths

- `Windows Hosted Build and Release` is the one-time full-base path. It is
  main-only, requires `VERSION == FAST_UPDATE_BASE_VERSION`, and starts
  automatically when a push to `main` changes a full-runtime contract input.
  Routine changes that only advance `VERSION` and `app` do not trigger this
  heavy build. It performs the full Windows clean-install gate plus an RC28.1
  upgrade/data-preservation gate pinned to commit
  `5032e5a5801af5368af4c85476cbefbd1cd563e1` and Setup SHA-256
  `E64DFCC90D8C9D17742591C7254D7176CB088C60F38C58C5930D42CAA529C2EC`,
  then publishes `RUNTIME_CONTRACT.json` and `RUNTIME_CONTRACT_ID.txt` as base
  assets. Manual dispatch remains an optional recovery path.
- `ANPR tests` validates each exact commit pushed to `main`, including the real
  detector/OCR runtime. The fast workflow requires a successful run for its
  exact SHA and does not repeat those expensive jobs.
- `Fast one-click update` is the routine path. Every push to `main` that changes
  `VERSION` starts it automatically at the same time as `ANPR tests`. A base
  version is detected as a successful no-op. A fast child such as
  `2.2.0-rc30.1` downloads the contract from release `v2.2.0-rc30`, verifies
  that the base tag is an ancestor of the exact update SHA, rejects a version
  that is not newer than every existing valid fast release (except an exact-tag
  rerun of the same SHA), builds a payload below 10 MiB and an updater below
  15 MiB, and publishes only the updater and `SHA256SUMS.txt`. Source remains
  available at the immutable release tag.

For routine fast children, the four-minute payload build and the exact-SHA ANPR
validation run in parallel. When the already-cached validation finishes during
that build, publication, remote asset download and checksum verification share
one final one-minute job, so the warm-path target is about five minutes after a
runner starts. This is a target, not an end-to-end SLA: GitHub queue time, cold
ANPR dependency/model installation, or a slow validation can take longer. The gate
waits up to 18 minutes (inside a 20-minute job timeout) rather than rejecting a
healthy commit at an unmeasured 210-second deadline, and it never publishes
before the exact SHA succeeds. Commit statuses `bcvision/fast-release` and
`bcvision/full-release` expose the final result to connected tooling. The first
fast child additionally runs the slower mandatory real RC30
Setup-to-fast-update integration job; it is a one-time bootstrap gate. Manual
runs may request it again with the `verify_base_install` input.

## Transaction and rollback

Payloads live under `runtime/<version>` and contain SHA-256 for every file.
Only the verified external `app` package is exposed to the import system; the
payload root is not added to `sys.path`, so an unmanifested top-level module
cannot shadow or extend the frozen runtime.
The runtime directory uses these atomic markers:

- `current.txt`: selected runtime;
- `previous.txt`: runtime active before the update;
- `last-known-good.txt`: most recently confirmed runtime;
- `pending.txt`: transaction being tested;
- `failed.txt`: last candidate rejected or recovered from.

Before `[InstallDelete]`, the elevated updater refuses candidates older than
either `current` or `last-known-good`, selects a different intact rollback
payload, proves it with the installed executable's isolated self-test, and
atomically records it as `previous`. This also makes reinstalling an exact fast
version safe: its own directory is never used as its rollback copy. The updater
then stages all files, writes `pending` and `current`, and runs the installed
`BCVision.exe` with `--self-test` and an explicit `--runtime-candidate`. That
switch is rejected on every normal invocation and requires both `--self-test`
and an explicit isolated `--self-test-data-dir`. Success is possible only when
the imported application version is the requested payload version. The updater
then atomically confirms `current` and `last-known-good` and clears `pending`.
On failure it restores the different verified runtime, records `failed`, and
returns a nonzero installer result.

If power loss or process termination leaves `pending.txt`, the normal
unelevated launcher treats Program Files as read-only. A transaction that
reached the confirmed last-known-good state may run; every other pending
candidate is skipped in memory in favor of the newest verified
last-known-good/previous payload. If no external payload is valid, the bundled
full-base application remains the final fallback. The launcher writes only a
diagnostic to the normal writable application log. Marker cleanup/commit stays
owned by an elevated installer, so a denied Program Files write cannot prevent
the application from starting on its fallback.

The full RC30 Setup/Update installer also excludes runtime pointers from its
recursive file copy. When used as a repair after a confirmed fast release, it
isolates and verifies the newest compatible fast payload and recommits that
version after replacing base files. It refuses an incompatible or unverifiable
newer pointer instead of activating bundled RC30 against a potentially migrated
database.

## Persistent customer data

Fast update files are restricted to `{app}/runtime/<version>`. SQLite data,
settings, `storage_config.json`, custom storage roots, videos, snapshots, plate
images, downloaded/trained models, YOLO11n and YOLO8n model files are never part
of the updater. The full Windows release gate and first-fast integration place
markers in each of these locations and verify them after update; the full gate
also verifies they survive uninstall.

## Runtime compatibility rules

After RC30 is published, do not modify any runtime-contract file for a fast
release. Changes to these areas require a new full base, a new ABI and new
immutable contract assets:

- `launcher.py` or `runtime_payload.py`;
- `app/database.py`, database schema, or data-migration behavior;
- `BUILD_PORTABLE_EXE.bat` or the executable entry point;
- `requirements-lock.txt` or `requirements-ai-lock.txt`;
- Python, PyInstaller, native DLLs or collected dependencies.

Fast updates self-test against a fresh isolated database, so they must never
introduce schema/data migrations that could behave differently on customer
history. Any database change requires a new full base/ABI and its own real
upgrade integration gate.

Fast versions do not change the static full-installer metadata. The full
installer remains bound to `FAST_UPDATE_BASE_VERSION`; only `VERSION` and the
application version advance for a routine update.

## Release recovery

Both release paths are exact-SHA safe. Publishing starts as a draft. A rerun may
resume or replace missing assets only while that release remains a draft and
its tag targets the same commit. A complete non-draft release is accepted as
already finished; an incomplete public release is never mutated and requires a
new version. A tag pointing to any other commit is refused. Before accepting
either a resumed or
already-complete release, the workflow downloads the exact asset set, compares
the released checksum metadata with the exact-SHA build, and verifies every
payload asset byte against `SHA256SUMS.txt`. This allows recovery from an
interrupted asset upload without trusting names alone or mutating a release from
different source.
