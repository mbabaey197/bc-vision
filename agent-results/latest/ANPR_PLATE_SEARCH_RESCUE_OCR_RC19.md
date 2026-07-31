# BC Vision RC19 — segmented plate search and rescue OCR

Date: 2026-07-31

## Scope

- Replace the generic archive plate search box with a position-aware Iranian
  plate control.
- Support direct typing and a browser-native searchable dropdown for plate
  letters.
- Preserve partial search: each of the four plate sections can be left empty.
- Make the secondary whole-plate OCR conditional instead of always-on for
  credible crops.

## Safety rules preserved

- Strict multi-frame, per-character consensus remains the automatic-confirm
  gate and requires at least three independent observations.
- Rejected single-frame hypotheses do not become confirmed truth.
- Operator corrections remain durable labelled feedback and never teach one
  generic unreadable result as a global plate replacement.
- No existing database, user, camera, setting, license, image or model file is
  deleted or reset by this change.
- Research/shadow model weights are not promoted to the commercial runtime.

## Validation coverage added

- segmented prefix, letter, serial and Iran-region archive filtering;
- Persian-digit and direct `الف` entry;
- invalid segmented filters fail closed;
- strong unambiguous primary OCR skips the rescue reader;
- known `2↔3` ambiguity forces rescue OCR even at high primary confidence.
