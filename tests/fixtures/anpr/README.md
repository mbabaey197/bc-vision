# Real CCTV ANPR fixture

This directory is reserved for the real CCTV regression video shared by the project owner.

## Expected file

- Path: `tests/fixtures/anpr/01.mp4`
- Size: `24097556` bytes
- Duration: `68.25` seconds
- Resolution: `1920x1080`
- Frame rate: `8 fps`
- Codec: `H.264`
- SHA-256: `b5193d8cf32d79daf17e15bea0b1c74e05156a70eddf49ca9e5d0466e568705d`

## Ground-truth regression plates

The current trusted reference labels from this video are:

- `31-ط-556-74`
- `55-ط-639-74`
- `84-ب-571-33`

The video must not be accepted by automated tests unless its SHA-256 matches the value above.

## Test policy

- Per-character multi-frame consensus is required.
- A single frame must never create a definitive event.
- Ambiguous character positions must be reported as unreadable instead of guessed.
- The common `ط/ل` confusion and wrong middle/final digits are explicit regressions for this fixture.
