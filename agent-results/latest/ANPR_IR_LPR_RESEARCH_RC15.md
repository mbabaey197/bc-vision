# BC Vision RC15 — IR-LPR research training result

Date: 2026-07-29

## Decision

The uploaded official IR-LPR plate archives were imported, audited, used to
fine-tune CCT-XS-v2, evaluated on an independent IR-LPR test split and then
tested on the fixed BC Vision Golden video.

The final Stage-4 model is a meaningful OCR improvement over the previous
untrained/synthetic CCT candidate, but it is **not promotable**:

- Golden confirmed output remains `1/3`, equal to the RC12 Baseline;
- raw OCR guesses see `2/3`, but one exact guess occurs in only one frame and
  cannot satisfy three-frame consensus;
- 39 emitted unique strings remain outside the labelled Golden set;
- IR-LPR is GPL-3.0 research data, so the derived model is Shadow-only and
  non-distributable.

RC12 remains active. The trained model is not committed, packaged, signed for
production, or copied into a customer installer.

## Source archives

| Official archive | Source rows | SHA-256 |
|---|---:|---|
| `plate_img-train(1).zip` | 19,381 | `A7A8B6EA39EEF74BECBEBF96709BD077C8A5F3FDE5D38AB3052C7582A10ADDD5` |
| `plate_img-validation.zip` | 2,805 | `7C2448C04978BB456E7C696957C7BBFDF2F9FAB415DC163BBC4D5B5EC87E4587` |
| `plate_img-test.zip` | 5,559 | `8091138FFBA32F95E83489A13064329994B7777AECCE31F5E7AA9DF4DE7BBA9A` |

All three ZIP central directories and payloads passed integrity validation.
The official source is https://github.com/mut-deep/IR-LPR and its repository
license is GPL-3.0.

## Import and isolation

The importer keeps distinct views of the same plate inside its original split,
but excludes exact duplicate images and any plate identity already present in
an earlier split. It also maps the verbose disabled-veteran label
`ژ (معلولین و جانبازان)` to the canonical single OCR character.

| Split | Imported OCR crops |
|---|---:|
| Train | 17,371 |
| Validation | 2,007 |
| Test | 3,903 |

- train/validation/test plate-identity overlap: `0`
- cross-split image overlap: `0`
- missing imported images: `0`
- invalid imported labels: `0`
- Golden video/crops in training: `0`
- full FastPlateOCR dataset validation errors: `0`

IR-LPR lacks examples for some configured classes, including `D`, `S`, `ف`,
`ک` and `گ`. The trained model therefore cannot claim complete coverage of
every Iranian plate class.

## Training

Training used Python 3.12, TensorFlow CPU, FastPlateOCR 1.1.0, batch size 64,
moderate geometric/photometric augmentation and four two-epoch stages.
Golden data was never used for checkpoint selection.

Random initialization was rejected after an exact-image overfit diagnostic
failed to learn beyond approximately 12.8% character accuracy. The accepted
run initialized only compatible CCT-XS-v2 feature-backbone layers from the
official FastPlateOCR global checkpoint. OCR and region heads remained
Iran-specific and were not transferred.

| Artifact | SHA-256 |
|---|---|
| Official CCT-XS-v2 backbone initialization | `0716717772B1F8D25B3C227E1E65E7F42E63900EC017059B4A32155488735FFD` |
| Final Stage-4 ONNX | `AD8D77D69CD0C914CB0CB3E0AC4E18709C446F78625A440D8F2D7AD2FB669482` |

Final ONNX size is 2,394,622 bytes and its verified contract is
`1x64x128x3 uint8 NHWC -> 1x8x37`.

## Independent IR-LPR result

The final model was selected on Validation. The untouched Test split was then
evaluated once by the training tool.

| Metric | Validation (2,007) | Test (3,903) |
|---|---:|---:|
| Raw exact plate accuracy | 89.04% | 87.45% |
| Raw character accuracy | 98.03% | 97.61% |
| Mean character error | 0.158 | 0.191 |
| Accepted-output precision | 92.41% | 90.66% |
| Rejection rate | 4.83% | 4.59% |
| Mean CPU OCR latency/crop | 4.22 ms | 3.89 ms |

Test per-position accuracies are:

```text
97.36%, 98.33%, 96.87%, 98.46%, 98.44%, 98.31%, 97.57%, 95.52%
```

The final two-digit region position remains the weakest.

## Fixed Golden video

The exact fixed video was verified before execution:

- SHA-256:
  `B5193D8CF32D79DAF17E15BEA0B1C74E05156A70EDDF49CA9E5D0466E568705D`
- 546 frames, 8 FPS, 1920x1080, 68.25 seconds
- verified RC12 ONNX detector and fallback model loaded
- 807 detector candidates, matching the earlier corrected CCT comparison
- fallback detector was loaded but not used
- all 546 frames processed

| Golden truth | Closest raw OCR | Character error | Exact raw observations | Tracker-confirmed |
|---|---|---:|---:|---|
| `31-ط-556-74` | `31-ط-566-74` | 1 | 0 | No |
| `55-ط-639-74` | `55-ط-639-74` | 0 | 19 | Yes |
| `84-ب-571-33` | `84-ب-571-33` | 0 | 1 | No |

End-to-end Stage-4 run:

- confirmed Golden matches: `1/3`
- raw exact Golden guesses: `2/3`
- emitted track rows: `53`
- unverified emitted unique strings: `39`
- elapsed CPU time: `75.128` seconds

The Stage-3 timing was `57.859` seconds on the same environment, so the
Stage-4 wall time is treated as a noisy system-load measurement, not an
architecture slowdown claim. Both exceed the historical `67.542` second RC12
run in at least one measurement, and neither improves confirmed accuracy.

Rejected or single-frame guesses remain visible as experimental diagnostics.
They do not receive a canonical confirmed plate value, cannot enter strict
consensus, and never become a training label without operator correction.

## Later operator-assisted workflow decision

After this fixed Golden result, the owner approved showing complete
multi-frame guesses in the operational event list as `تأیید خودکار مدل`.
This is a workflow change, not a retroactive accuracy improvement: strict
Golden accuracy remains `1/3`, and the one-frame exact read is still not
claimed as strict consensus.

AI-confirmed events remain visibly sourced from the model and are excluded
from training. An operator must explicitly confirm the unchanged plate or
correct it; only then is feedback and its crop admitted to controlled
training. The research-derived model remains Shadow-only, non-distributable
and absent from installers.

## Code corrections made during the run

- fixed the GPL acceptance CLI destination;
- fixed over-aggressive within-split plate-identity deduplication;
- added the verbose `ژ` class alias;
- added the missing OpenCV training dependency;
- added resumable checkpoints, explicit checkpoint metric and learning-rate
  controls;
- added augmentation provenance;
- added character accuracy, mean character error and per-position accuracy;
- extended the Golden benchmark and RTL report with nearest raw guesses,
  character distance, exact-guess frequency and their exact crops.

## Verification

- focused AI/runtime/training regression: `113 passed, 1 skipped`
- Python compileall for `app`, `tools` and `tests`: passed
- CCT/IR-LPR/raw-guess focused subset: `30 passed`
- ONNX export, ONNX Runtime load and Keras/ONNX parity: passed

The skip is the existing Windows real-model integration gate. The isolated
training environment does not contain FastAPI, PyAV or Cryptography, and
network policy blocked installing them. Therefore the web, database and
packaging test groups were not claimed as a clean full regression in this run.

## Publication boundary

Dataset files, customer video, extracted crops, training runs, Keras
checkpoints, ONNX weights and generated visual reports are ignored and remain
outside Git. Only source code, tests, configuration and this result record are
eligible for a branch update.
