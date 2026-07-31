# BC Vision — OCR geometry and temporal-safety correction

Date: 2026-07-30

## Decision

The operator was correct that `2/16` did not describe image readability.
Conservative visual review found at least 14 of the 16 development crops
human-readable, while the legacy OCR reconstructed only two complete strings.
The failure was therefore treated as an OCR/runtime defect rather than as
evidence that only two images were usable.

The runtime correction materially improves the current development crops and
reduces false video emissions, but it does not pass activation:

- development exact reads improve from `2/16` to `8/16`;
- four development reads pass the new strict acceptance gate and all four are
  correct;
- the fixed video remains `0/3` at Tracker level;
- one incorrect complete string is still emitted on that video;
- the confidence score is not a calibrated full-plate probability;
- the 16-crop split has already been used for checkpoint and preprocessing
  selection and is therefore development/tuning data, not an independent
  test.

RC12 remains active. No new ONNX is installed, copied to the `next` slot or
included in a commercial package.

## Root causes reproduced

1. Training, offline evaluation and runtime all forced variable-ratio crops
   into `128x64`. The 16 development crops range from aspect ratio 1.77 to
   4.58, with median 3.44; forcing the median crop to ratio 2.0 compresses its
   horizontal geometry by about 42% relative to vertical geometry.
2. The CCT model remains strongly biased toward its synthetic domain. Its
   synthetic accuracy is near 99%, while many visually clear real crops are
   wrong.
3. The reported OCR confidence is the geometric mean of position scores. It
   is a ranking/reliability score, not a calibrated probability that the
   complete plate is correct.
4. The old temporal path could construct a character-wise hybrid that never
   appeared as a complete hypothesis, reuse one physical track for a later
   vehicle, and overwrite a confirmed event with another identity.

## Runtime correction

`app/ai/onnx_cct.py` now owns one shared, signed inference transaction used by
runtime, crop evaluation and fixed-video evaluation:

- legacy `stretch-v1` remains backward compatible;
- `stretch-letterbox-geomean-v1` prepares a stretched view and an
  aspect-preserving letterbox view;
- byte-identical inputs are deduplicated;
- ONNX calls remain sequential batch-one transactions;
- normalized probabilities are fused in log space with a geometric mean;
- both view top strings must be structurally valid;
- per-position view agreement must be at least `0.75`;
- rejected strings remain operator-review evidence and can never enter strict
  temporal consensus or strong track association;
- per-view diagnostics, raw model score and acceptance reason are retained.

The signed model contract validates the closed preprocessing/fusion enums and
rejects a dual-view agreement threshold below `0.75`.

## Geometry correction

Perspective crops now declare their geometry explicitly. The shared
quadrilateral rectifier:

- rejects nonfinite, duplicate, nonconvex, tiny, extreme-ratio and singular
  geometry;
- expands margins in the plate's local horizontal/vertical basis rather than
  global image axes;
- keeps one uniform target scale rather than forcing a target aspect ratio;
- exposes both `corners` and `quadrilateral` for OBB diagnostics;
- falls back to the original axis-aligned crop when a safe perspective crop
  cannot be produced.

The fixed benchmark currently uses the verified axis-aligned ONNX detector, so
this OBB correction is covered by focused tests but does not change the 546
frame result.

## Tracker and persistence correction

Strict confirmation now requires the same complete top plate on at least
three independent observations and the configured minimum time span.
Character-position voting alone may not confirm a string.

Additional safeguards:

- rejected OCR hypotheses are review-only;
- a confirmed track identity is immutable;
- a later strong different identity starts a new track, including a
  one-character change after an event has been emitted;
- two similar vehicles separated by missed detections do not share a track;
- tracker age is capped at six seconds while preserving consecutive
  observations under slow CPU processing;
- temporal weights prefer OCR confidence over detector-mixed confidence;
- a clearer frame of the same identity refreshes the same event without
  creating a duplicate;
- a reviewable/unreadable row cannot downgrade a confirmed database event;
- a stale `event_id` carrying another confirmed identity creates a separate
  reviewable identity-conflict row instead of overwriting the event.

## Development-crop result

The source model is the first `5e-5` company-crop candidate:

`5464B132CE2F28FF0FFA45BE196B55664BB908078FD44F4FA2B604F32E9E9820`

| Inference profile | Exact | Character | Accepted correct / accepted | Rejection |
|---|---:|---:|---:|---:|
| Legacy stretch | 2/16 | 71.88% | 2/9 (22.22%) | 43.75% |
| Dual-view strict | 8/16 | 82.03% | 4/4 (100%) | 75.00% |

This is an exploratory development result. The split has zero plate-identity
overlap with Train, but it is not camera/session/time independent and has now
been reused for learning-rate, repeatability, checkpoint and preprocessing
selection.

## Synthetic regression

The 3,000 synthetic images are already exactly `128x64`, so the two prepared
views are byte-identical and the runtime correctly deduplicates them.

| Profile | Raw exact | Accepted correct / accepted | Accepted precision |
|---|---:|---:|---:|
| Legacy stretch | 2984/3000 | 2977/2988 | 99.6319% |
| Dual-view strict | 2984/3000 | 2975/2985 | 99.6650% |

Raw regression is zero. The stricter gate rejects three additional samples and
reduces accepted exact count by two. This split contains repeated synthetic
views of fewer underlying identities, so image count is not an independent
identity count.

## Fixed-video result

Input:

- video SHA-256:
  `B5193D8CF32D79DAF17E15BEA0B1C74E05156A70EDDF49CA9E5D0466E568705D`
- 546 frames;
- 807 detections;
- both ONNX detectors verified by pinned size and SHA-256;
- OpenCV detector fallback disabled.

| Metric | Previously recorded candidate | Corrected dual-view runtime |
|---|---:|---:|
| Tracker exact truth | 0/3 | 0/3 |
| Raw exact truth seen in any frame | 1/3 | 1/3 |
| Unmatched emitted unique strings | 23 | 1 |

The remaining emitted error is `31-ط-566-74`, one character from the trusted
`31-ط-556-74`. It appeared as the accepted complete read in three observations
with view agreement `0.875`. The exact raw `55-ط-639-74` appeared five times
but failed the cross-view acceptance gate at agreement `0.6875`, so it
correctly remained review-only and was not promoted by temporal repetition.

The video metric is a string-set comparison and lacks time/bbox/track ground
truth. It is a useful fixed regression, not a statistically sufficient
event-level recall estimate.

## Aspect-preserving retrain

A new 30-epoch experiment reused the same private 66 Train / 16 development
split, source checkpoint, seed `20260730`, batch size 8 and learning rate
`5e-5`, but changed the FastPlateOCR training loader to
`keep_aspect_ratio: true`. Runtime evaluation used the strict dual profile.

- selected human epoch: 25;
- ONNX SHA-256:
  `61C9A8426D04EA4E2FCEE4E547364F60AEB038FE4D08D999A15CCBE26E07412D`
- training plate-config SHA-256:
  `FA322FFDDFD79975A0FC1EAB8D6259435D70D29614E548E157F9AAD530500F15`

| Gate | Existing dual runtime | Letterbox-trained candidate |
|---|---:|---:|
| Development exact | 8/16 | 5/16 |
| Development character | 82.03% | 82.03% |
| Accepted development | 4/4 correct | 3/3 correct |
| Synthetic raw exact | 2984/3000 | 2976/3000 |
| Video Tracker exact | 0/3 | 0/3 |
| Video unmatched emitted strings | 1 | 2 |

The retrain is worse on every decisive full-string gate and is rejected. Its
metadata remains `activation_allowed=false`.

## Verification

- full repository suite: `360 passed, 1 skipped`;
- skip reason: AI integration runtime is disabled;
- Python compilation: passed;
- diff whitespace validation: passed;
- legacy one-view compatibility: passed;
- dual-view tensor shape, padding, deduplication and disagreement gates:
  passed;
- rejected-to-temporal-confirmation regression: passed;
- unseen positional hybrid regression: passed;
- similar sequential vehicle identity regression: passed;
- confirmed-event immutability and clearer-capture refresh: passed;
- invalid quadrilateral and shuffled-corner rectification: passed.

## Activation gate

Before activation, collect a new hash-bound holdout grouped by plate identity,
camera, capture session and time. It must not be used for checkpoint,
threshold or preprocessing selection. Confidence must then be calibrated on
separate real data, and the fixed video should receive temporal/spatial track
annotations rather than string-only truth.

Until those gates pass, all complete but rejected reads remain suggestions,
ambiguous images remain `ناخوانا`, and the candidate stays Shadow-only.
