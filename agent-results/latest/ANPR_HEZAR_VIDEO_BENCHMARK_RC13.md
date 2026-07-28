# RC13 Hezar v2 real-video benchmark

Date: `2026-07-28`

## Fixed inputs

- Video: `01.mp4`
- Size: `24097556` bytes
- SHA-256:
  `B5193D8CF32D79DAF17E15BEA0B1C74E05156A70EDDF49CA9E5D0466E568705D`
- Frames: `546`
- Resolution: `1920x1080`
- FPS: `8`
- Known truth set:
  - `31-ط-556-74`
  - `55-ط-639-74`
  - `84-ب-571-33`

No other visible plate in the video was treated as truth without an operator
label.

## Candidate artifact

The official `hezarai/crnn-fa-license-plate-recognition-v2` model was exported
locally to fixed-batch ONNX and was not committed to the public repository.

- ONNX size: `37146355` bytes
- SHA-256:
  `D2E2A7BB9001DB01D41AF396215E6D3357F741AC2578D50BEAF63DEAA08B93B0`
- Input: `1x1x32x384`
- Output: `1x96x45`
- PyTorch/ONNX maximum absolute error:
  `0.0000057220458984375`
- Runtime: ONNX Runtime CPU

RC13 was corrected for the real Hezar v2 contract:

- CTC blank at class index `0`;
- full Hezar `id2label` mapping without a one-character shift;
- mirrored grayscale input;
- Hezar mean `0.6595` and standard deviation `0.1501`;
- reverse-time decoding before Iranian-layout constraints;
- normalized per-position hypothesis margins.

## Fair all-frame comparison

Both engines processed every frame (`frame_step=1`) with the same current
Iranian plate detector and three-vote geometric tracker.

| Metric | RC12 baseline | Baseline detector + Hezar v2 ONNX |
| --- | ---: | ---: |
| Frames processed | 546 | 546 |
| Exact known plates | 1 / 3 | 1 / 3 |
| Elapsed CPU time | 67.542 s | 92.751 s |
| Emitted tracks/events | 52 | 65 |

Both engines exactly confirmed `55-ط-639-74`.

Known mismatches:

- Baseline: `31-ط-566-74`, `84-ب-578-32`
- Hezar v2: `31-ط-566-74`, `84-ب-579-32`

The other emitted strings have no operator-provided labels and are therefore
recorded as unverified, not as correct or false.

## Decision

The ready-made Hezar v2 model is **not promoted**:

- exact known-plate accuracy did not improve over baseline;
- elapsed time increased by `37.3%`;
- the `556/566` systematic substitution remained;
- the third known plate still had two wrong digits.

The successful ONNX integration remains useful for the next fine-tuned model.
The current RC12 engine stays customer-visible. Hezar may run only in
development benchmark or signed shadow mode until a fine-tuned model passes
the Golden Dataset gate.

No trained YOLO26-OBB Iranian plate weight was available for this run.
An official generic OBB checkpoint is not a license-plate detector and is not
substituted for a trained Iranian model. OBB accuracy remains blocked on
labelled training data and GPU training.
