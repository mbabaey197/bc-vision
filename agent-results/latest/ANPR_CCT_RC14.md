# BC Vision RC14 — FastPlateOCR CCT candidate

Date: 2026-07-28

## Outcome

RC14 adds a fail-closed FastPlateOCR CCT runtime, reproducible offline
training/export tools, licensed-data admission checks and an all-frame video
benchmark. It does not replace or automatically activate the verified RC12
baseline.

Two Iranian-layout candidates were trained only to validate the pipeline:

| Candidate | Raw exact | Accepted exact | Accepted precision | Rejected | CPU latency/crop |
|---|---:|---:|---:|---:|---:|
| CCT-XS-v2 | 187/200 (93.5%) | 180/200 (90.0%) | 95.2% | 5.5% | 3.78 ms |
| CCT-S-v2 | 194/200 (97.0%) | 189/200 (94.5%) | 97.4% | 3.0% | 19.79 ms |

These measurements are synthetic-pipeline evidence, not a claim of customer
camera accuracy. The later fixed real-camera test rejected both candidates.

## Training evidence

- Training data: 1,000 synthetic train images and 200 synthetic validation
  images.
- Unique identities: 500 train and 100 validation plates.
- Train/validation identity overlap: zero.
- Every supported Persian and diplomatic letter, including `D` and `S`, is
  represented in both splits.
- Golden video or crops in training: none.
- Third-party plate datasets: none.
- Generator seed: `20260728`.
- Font: DejaVu Sans Bold under the DejaVu font licence.
- Epochs: 20 for each candidate.
- Batch size: 32.
- Backend: TensorFlow CPU.
- Input: fixed `1x64x128x3`, NHWC, RGB, uint8.
- Output: fixed `1x8x37`.

The optional official FastPlateOCR global checkpoints were used only for
shape-compatible feature-backbone initialization. OCR/region heads and
incompatible slot-query tensors were excluded. The training command requires
at least 95% compatibility for the eligible backbone layers.

| Artifact | SHA-256 |
|---|---|
| Official CCT-XS-v2 initialization | `0716717772b1f8d25b3c227e1e65e7f42e63900ec017059b4a32155488735ffd` |
| Official CCT-S-v2 initialization | `dc6db494714d88e1fb0a21b4e4e96089842dd5440418262abfc382454453c2c8` |
| Trained CCT-XS-v2 candidate | `3ed1dac2cc83185cc6d5b84a4179139613e6d5517f3d829d617cc5b7922fccbf` |
| Trained CCT-S-v2 candidate | `7796cf9635e2f42ddb2908d92a2528ed8e80295cda18e8a301801d69ce33ec4c` |
| Dataset licence manifest | `70cabfa6d1a6e670e719e7fabb500956a9a336ac88369dda67650caccf1e302f` |

The ONNX exporter used the unsimplified graph because the tested ONNX
simplifier produced an invalid CCT-v2 `Gemm` bias. The retained graphs passed
ONNX validation, ONNX Runtime execution and exact Keras/ONNX output comparison.

## Runtime safety

- The signed manifest binds the runtime, alphabet, eight slots, input
  preprocessing, decoder thresholds, beam width and top-k.
- Missing, unknown or permissive contracts fail closed before a session opens.
- Inference sessions retain the bounded CPU/thread settings used by BC Vision.
- The decoder constrains digit/letter positions to the Iranian plate layout.
- Ambiguous crops are rejected rather than completed or guessed.
- Tracker association remains geometric and receives a positive probability,
  never the decoder's negative log score.
- Candidate model files are not committed to the repository or installers.

## Real-video gate

The fixed `01.mp4` benchmark must process all 546 frames and verify this
pre-recorded file digest before making a promotion decision:

```text
b5193d8cf32d79daf17e15bea0b1c74e05156a70eddf49ca9e5d0466e568705d
```

Known truth plates:

```text
31-ط-556-74
55-ط-639-74
84-ب-571-33
```

The operator supplied the exact fixed video again on 2026-07-28. Its SHA-256,
resolution, frame count, FPS and duration matched the recorded fixture. Both
candidates processed every frame:

| Candidate | Exact known plates | Detector candidates | Track emissions | Unverified unique strings | CPU time |
|---|---:|---:|---:|---:|---:|
| CCT-XS-v2 | 0/3 | 4,326 | 214 | 99 | 82.165 s |
| CCT-S-v2 | 0/3 | 4,326 | 204 | 113 | 153.530 s |

Neither candidate is promotable. The CCT-S accuracy advantage on synthetic
data disappeared on the real camera and it was also slower. CCT-XS was faster
than CCT-S but slower than the `67.542` second RC12 baseline.

Visual inspection of the exact OCR crops exposed the dominant detector error:
the source is a four-camera composite and the OpenCV fallback repeatedly
localized the fixed date, clock and camera-name overlays as if they were
plates. Genuine plate crops were present as well, but neither CCT candidate
produced an exact Golden match. The next training iteration therefore requires
both a dedicated plate detector and separate company-owned labelled field
crops; OCR-only retraining cannot fix the overlay detections.

The benchmark now optionally stores the exact crop behind every emitted row.
The self-contained RTL HTML renderer places each crop, recognized text,
confidence, timestamp and track ID on one row. Strings outside the currently
labelled Golden set are explicitly marked as requiring operator labels and are
not silently scored as correct or as false positives.

Promotion requires:

1. More exact truth matches than the RC12 result of 1/3.
2. No unverified output silently classified as a false positive or success.
3. End-to-end runtime measured on the same CPU and all 546 frames.
4. No regression in day/night/blur/glare/unreadable Golden slices.
5. A signed bundle installed first in `shadow` mode.

Until all five gates pass, `baseline` remains the active mode. The current
Golden result fails gates 1, 3 and 4.

## Source verification

```text
169 passed, 1 skipped
```

The skipped check is the existing Windows real-model integration gate.
Compilation, whitespace validation and the focused crop/report UI checks also
passed. Candidate ONNX files, customer video, crops and generated HTML reports
remain outside repository scope.
