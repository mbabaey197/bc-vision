# BC Vision — company-owned crop batch 01 partial fine-tune

Date: 2026-07-30

## Decision

The first partial operator export is technically valid and can continue the
training workflow, but neither candidate is eligible for activation. The
stronger candidate remains a private experimental Shadow artifact:

- the company development/tuning split is only 16 crops;
- exact full-plate accuracy is only `2/16`;
- the fixed real-video Tracker result remains `0/3`;
- 79 of 82 confirmed crops use the same plate-letter class;
- 401 source crops are still Pending.

RC12 remains active. No source crop, confirmed plate text, private training
dataset, checkpoint or ONNX candidate is committed to Git.

## Operator export

The reviewed CSV contains 505 integrity-bound rows:

| State | Samples |
|---|---:|
| Confirmed | 82 |
| Unreadable | 20 |
| Excluded | 2 |
| Pending | 401 |

The 82 confirmed crops represent 79 unique plate identities and 82 unique
image digests. Three repeated identities are kept in one split. The confirmed
letter-class distribution is heavily skewed: 79 `ع`, two `د` and one `ز`.
This export can test the pipeline and specialize a yellow-plate slice, but it
cannot support a general Iranian OCR accuracy claim.

## Secure preparation

`tools/prepare_plate_review_dataset.py` performs the import:

1. verifies review-page provenance and the exact source-archive SHA-256;
2. validates all 505 CSV names, statuses, labels and image digests;
3. accepts only explicitly Confirmed canonical Iranian labels;
4. rejects model suggestions, Pending, Unreadable and Excluded rows;
5. verifies each source image with Pillow;
6. groups repeated plate identities before Train/Validation splitting;
7. writes an integrity manifest with zero cross-split identity overlap.

Prepared private split:

| Split | Crops | Unique identities |
|---|---:|---:|
| Train | 66 | 63 |
| Validation | 16 | 16 |

- plate-identity overlap: `0`
- image-digest overlap: `0`
- dataset fingerprint:
  `308C6778F4213997ECD51EB802D8257ED174322F91821DAE8C742806740D3627`
- activation gate: `independent-golden-and-real-camera-pass`

## Fine-tune experiment

Both candidates resume the company-owned 30K synthetic CCT-XS checkpoint.
The source checkpoint SHA-256 is
`7793090A215EAB464475202E7FA02AA9223BE50ECF0987E9C85330838F05F19C`.
The same deterministic split and seed `20260730` are used throughout.

| Model | Learning rate | Company Dev exact | Company Dev character | Accepted precision | Rejection |
|---|---:|---:|---:|---:|---:|
| Synthetic base | — | 6.25% (1/16) | 57.03% | 14.29% | 56.25% |
| Candidate 1 | `1e-5` | 12.50% (2/16) | 58.59% | 14.29% | 56.25% |
| Candidate 2 | `5e-5` | 12.50% (2/16) | 71.88% | 22.22% | 43.75% |

Candidate 2 is the stronger crop-level experiment. Its Train exact accuracy
is `22/66` while development exact is `2/16`, so the small,
skewed batch already shows substantial memorization risk.

## Regression checks

Candidate 2 retains most synthetic performance but does regress slightly:

| Model | Synthetic Test exact | Synthetic Test character | Accepted precision | Rejection |
|---|---:|---:|---:|---:|
| Synthetic base | 99.57% | 99.94% | 99.73% | 0.20% |
| Candidate 2 | 99.47% | 99.93% | 99.63% | 0.40% |

Its ONNX SHA-256 is
`5464B132CE2F28FF0FFA45BE196B55664BB908078FD44F4FA2B604F32E9E9820`.

The fixed company video was then rerun with both pinned ONNX detectors
verified by size and SHA-256 and OpenCV fallback disabled:

| Metric | Synthetic base | Candidate 2 |
|---|---:|---:|
| Frames | 546 | 546 |
| Detector candidates | 807 | 807 |
| Tracker exact truth | 0/3 | 0/3 |
| Raw exact truth | 1/3 | 1/3 |
| Unmatched emitted unique strings | 21 | 23 |

Candidate 2 therefore fails the real-video activation gate and is not copied
into the runtime, installer or `next` model slot.

## Repeatability run

Candidate 2 was trained a second time from the exact same source checkpoint,
dataset split and configuration:

- dataset fingerprint:
  `308C6778F4213997ECD51EB802D8257ED174322F91821DAE8C742806740D3627`
- source checkpoint SHA-256:
  `7793090A215EAB464475202E7FA02AA9223BE50ECF0987E9C85330838F05F19C`
- seed: `20260730`
- epochs: `30`
- batch size: `8`
- learning rate: `5e-5`
- checkpoint metric: validation character accuracy

The random augmentation stream produced a different ONNX artifact, but the
main company development result was reproduced exactly:

| Metric | Candidate 2 first run | Candidate 2 repeat |
|---|---:|---:|
| Company Train exact | 22/66 (33.33%) | 24/66 (36.36%) |
| Company Train character | 80.87% | 82.77% |
| Company development exact | 2/16 (12.50%) | 2/16 (12.50%) |
| Company development character | 71.88% | 71.88% |
| Accepted development precision | 22.22% (2/9) | 28.57% (2/7) |
| Synthetic Test exact | 2984/3000 (99.47%) | 2982/3000 (99.40%) |
| Synthetic Test character | 99.93% | 99.92% |
| Fixed-video Tracker exact | 0/3 | 0/3 |
| Fixed-video raw exact | 1/3 | 1/3 |
| Fixed-video unmatched emitted strings | 23 | 27 |

The three nearest raw video observations also remained functionally
unchanged:

| Truth | Repeat nearest raw read | Distance | Accepted |
|---|---|---:|---|
| `31-ط-556-74` | `31-ط-566-74` | 1 | Yes |
| `55-ط-639-74` | `55-ط-639-74` | 0 | Yes |
| `84-ب-571-33` | `84-ا-577-32` | 3 | No |

- first-run ONNX SHA-256:
  `5464B132CE2F28FF0FFA45BE196B55664BB908078FD44F4FA2B604F32E9E9820`
- repeat ONNX SHA-256:
  `A41EE0DF9129381AB3DE95DAF8448D51F69F336CDA918CE32B3F9093CF009786`
- first-run best epoch: `29`
- repeat best epoch: `30`

The repeat therefore confirms that the crop-level improvement is broadly
repeatable, but it also confirms that the 82-crop batch is too small and
skewed to improve the real-video system. The repeat is slightly worse on
synthetic exact accuracy and emitted four more unmatched video strings, so it
is not promoted over the first candidate and remains private Shadow-only.

## Verification

- actual dataset preparation: passed
- source/archive/image integrity checks: passed
- zero Train/Validation identity leakage: passed
- focused pytest:
  `64 passed`
- Candidate 1 ONNX export and benchmark: passed
- Candidate 2 ONNX export and benchmark: passed
- Candidate 2 synthetic Validation/Test regression benchmark: passed
- Candidate 2 fixed-video strict ONNX benchmark: completed, promotion failed
- Candidate 2 exact-configuration repeat: completed, promotion failed
- repeat focused security/integrity tests: `64 passed`

## Next safe iteration

The 16-crop split is plate-identity disjoint from Train, but it has been reused
for checkpoint, learning-rate and preprocessing selection. It is therefore
development/tuning data and must not be described as an independent test.
The geometry/runtime correction and aspect-preserving retrain are recorded in
`agent-results/latest/ANPR_OCR_ENGINE_GEOMETRY_V1.md`.

Complete the remaining 401 reviews, keep ambiguous crops Unreadable, and add
white, night and IR plates with broader letter coverage. Rebuild the split
from the complete export, train a mixed synthetic-plus-real schedule, and
evaluate a new untouched camera/session/time holdout plus the fixed real-camera
Golden gates.
