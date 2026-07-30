# BC Vision RC16 — ANPR engine hardening

Date: 2026-07-30

## Implemented

1. Automatic confirmation now requires at least three independently observed
   full-plate frames and at least 0.12 seconds of temporal support. A single
   complete guess followed by unreadable frames remains an experimental
   operator suggestion.
2. A low-resolution per-camera activity analyzer learns stable CCTV/NVR text
   edges in conservative overlay bands. Current motion clears the learned mask
   before it can suppress a detection.
3. Motion immediately bypasses empty-scene inference backoff and starts a
   four-frame burst. The previous maximum 3.2-second idle delay remains
   available for static empty scenes without making entry detection wait for
   that delay.
4. Multi-vehicle association uses global assignment, raw trajectory direction,
   size and geometry. OCR text still has no role in track association.
5. Perspective fallback crops retain an expanded physical-plate border and
   export their quadrilateral. ROI translation also translates the
   quadrilateral.
6. Operator training creates an immutable, hash-bound per-run dataset snapshot.
   Promoted CRNNs retain a verified weights-only state dictionary so subsequent
   runs can initialize from the active checkpoint.
7. Candidate promotion now checks validation size, exact accuracy, mean
   character error, per-sample regressions, active baseline identity, candidate
   identity and a complete Golden comparison.
8. The Golden manifest requires at least 40 operator-labelled samples, 20
   unique readable plates and at least three examples in every required slice:
   day, night, fast, angle, blur, glare, unreadable and multi-vehicle.
   `tools/prepare_golden_dataset.py` copies and hashes the media while marking
   it permanently forbidden for training.

## Verification

- Python compilation for application AI modules and tools: passed
- `git diff --check`: passed
- Full locally executable regression: `209 passed, 1 skipped`
- Skip reason: the existing opt-in real AI integration runtime gate is disabled
  in this Linux audit environment

## Accuracy boundary

No new real-camera accuracy number is claimed by RC16. The trusted fixed video
still has only three operator-supplied truth plates. Those three labels remain
useful regression truth but intentionally fail the new large-Golden admission
contract. New day/night/fast/angle/blur/glare/unreadable/multi-vehicle media
must be labelled by the operator and evaluated through the real BC Vision
detector/OCR/tracker path before any candidate can be promoted.
