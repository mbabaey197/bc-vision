# BC Vision RC6 — Dashboard and plate feedback

Date: 2026-07-27

Version: `2.2.0-rc6`

Branch: `agent/dashboard-plate-feedback-rc6`

## Implemented

- smaller dashboard live-video cards
- live green/amber plate candidate overlays
- Persian digits in plate and recent-event presentation
- Iranian physical plate-style component
- operator correction form beside every recent plate
- strict full Iranian plate validation
- backward-compatible `anpr_feedback` migration
- preservation of the original OCR observation, media references and operator
- exact confirmed correction reuse in later OCR output
- removal of the dashboard system-status card beside recent events

## Validation

- `python -m compileall -q app`: passed
- `.venv/bin/python -m pytest -q`: `82 passed, 1 skipped`
- `git diff --check`: passed

## Accuracy boundary

Feedback is persisted as labelled training data and exact repeated OCR errors
can be corrected locally. Model weights are not trained inside the live server.
A reviewed offline training and held-out accuracy evaluation are still required
before a new detector/OCR model can be promoted.
