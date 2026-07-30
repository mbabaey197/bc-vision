# BC Vision RC15 — operator-assisted automatic confirmation

Date: 2026-07-29

## Owner decision

Complete multi-frame ANPR guesses are useful enough to appear in the normal
event workflow. They are recorded as `تأیید خودکار مدل` and remain editable.
Training continues only from explicit operator confirmations/corrections.

## Implemented contract

- `review_status=auto-confirmed`
- `confirmation_source=ai-auto-guess`
- `operator_reviewed=0`
- `experimental=1`
- complete canonical `plate_norm` for search/watchlist/event display
- visible badge on dashboard, event list, video-test and event detail
- one action for confirming an unchanged guess or submitting a correction
- operator submission changes the event to `confirmed`, clears the
  experimental marker and creates durable `anpr_feedback`
- the confirmed crop is copied into the immutable, hash-verified local
  training dataset

In live Shadow assistance, an overlapping strict Baseline read always has
priority. A complete candidate guess can replace only an unreadable Baseline
row. The behavior is controlled by `anpr_auto_confirm_guesses` in AI Settings.

## Accuracy boundary

This workflow does not change the fixed Golden measurements. RC15 Stage-4
strict Tracker accuracy remains `1/3`; raw OCR observed `2/3`. Automatic
confirmation is a review state, not a ground-truth or accuracy claim.

The IR-LPR-derived weight remains `research-shadow-only` and excluded from
public Setup/Update packages. After the owner approved internal evaluation, a
separate private model installer was generated outside the repository. It
installs the signed Stage-4 CCT bundle, selects Shadow mode and never changes
the application's Baseline priority or the operator-training boundary.

The Shadow runtime now reuses each verified Baseline detector crop for CCT
OCR. This is the same detector/OCR pairing used in the fixed Golden benchmark,
avoids a second detector pass, and means the private model pack does not
require an untrained OBB detector. The signed manifest binds the fixed
Baseline detector SHA-256 and size and fails closed if that detector is
missing or modified.

## Verification

- focused policy/pipeline/video regression: `29 passed`
- database/feedback/training/live-worker regression: `24 passed`
- broad executable AI regression: `113 passed`
- operator confirmation/correction route integration: `1 passed`
- RC15 signed model-pack contract/baseline-crop reuse: `15 passed`
- extracted private pack manifest/signature/SHA-256 verification: passed
- extracted CCT inference on Golden crop `55-ط-639-74`: passed at `0.9968`
- broad locally executable AI regression after packaging changes:
  `123 passed, 1 skipped`; five database/feedback tests were not collected
  successfully because FastAPI is absent from the isolated CCT environment
- Python compile for changed application/test files: passed
- `git diff --check`: passed

The isolated CCT environment lacks its own FastAPI, PyAV and Cryptography.
The operator route was exercised with the available local runtime
dependencies, but a full web/packaging regression is not claimed from this
environment.
