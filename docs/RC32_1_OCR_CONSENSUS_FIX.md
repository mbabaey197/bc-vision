# RC32.1 OCR consensus correction

Status: source regression fix; not a real-camera accuracy certification or a
claim that a Windows installer has been built. Runtime base remains RC32;
`VERSION` and the visible application version identify this payload as RC32.1.
No model weights, detector selection, database schema, or gate permissions change.

## Reproduced defects and fixes

- RC32 accepted a layout-valid character-CNN output even with confidence 0.10.
  The adapter now requires a finite score in [0.55, 1]. The CNN decoder also
  rejects invalid probability arrays and requires every glyph to score at least
  0.55 with a 0.12 margin over all competing classes. A valid plate-shaped string
  is not evidence that the image was readable.
- Repetition and detector/quality scores could raise a weak read's final score.
  Strict primary evidence now requires an absolute OCR floor, and consensus
  confidence is capped by the weighted supporting OCR confidence. Explicit zero
  and nonfinite OCR scores cannot fall back to detector confidence for admission.
  Existing explicit Hezar temporal-evidence policy remains separate.
- A one-character OCR change after emission could create a second vehicle at
  the same continuously tracked box. With at least 0.50 predicted-box IoU and
  fewer than two missed detections, it now stays on the track. Conflicting live
  reads carry `identity_conflict` and `needs_review`; the emitted identity is
  not silently overwritten. Two observed misses still permit a new vehicle.
- Whole-plate repetition previously blocked distributed single-character error
  recovery. A new `position-recovery` path requires five strong observations,
  OCR confidence >=0.75, crop quality >=0.20, at most one differing slot per
  observation, and at least four primary votes at every position. The weighted
  position ratio must be >=0.80 and margin >=0.60. The existing confirmation
  timespan and camera confidence gate still apply. A disagreement also discounts
  the supporting OCR score. Weak/ambiguous hybrid strings cannot use this path.
- Duplicate, out-of-order, and nonfinite timestamps are rejected before updating
  association, misses, expiry, Kalman state, or consensus votes.

## Verification

Run from repository root:

```sh
PYTHONPATH=. uv run --with-requirements requirements-test.txt pytest -q --tb=short
python scripts/verify_runtime_contract.py --validate-update-version 2.2.0-rc32.1
```

`tests/test_ocr_consensus_guards.py` covers the reproductions and negative cases.
Local Linux/Python 3.12 verification: **1,148 passed, 1 skipped** (the opt-in
real-model integration test), plus one existing dependency deprecation warning.
Compilation and runtime-contract verification passed. The new regression file
passes Ruff; critical-error checks pass for changed AI modules. Full Ruff still
reports pre-existing style/broad-exception findings in the surrounding code.
The pre-fix reproductions had nine failing cases and one passing boundary case.
Existing persistence-retry tests still exercise a valid OCR read below the
camera's stricter emission gate, now using OCR 0.60 rather than rejected 0.45.

These thresholds are engineering guards, not calibrated accuracy percentages.
Unit tests use synthetic observations/model stubs, not camera inference. They
cannot establish real-world precision, recall, night performance, or distinguish
two similar plates occupying the same box without an observed temporal boundary.
Frame timestamps also cannot detect frozen camera pixels with fresh timestamps.
A Windows build/startup check and labeled real-camera runs remain necessary
before claiming that the installed reader is fixed in the field.
