# Engine V2 same-input partial real-media evidence — 2026-08-11

This is non-promotional evidence from one real 68.25-second H.264 clip. It
exercises the concrete unchanged-V1 and V2 offline adapters on the same verified
input bytes. It does **not** establish production accuracy or camera capacity.

## Evidence identity

- Dataset: `bcvision-01mp4-trusted-regression-partial-v1`
- Dataset fingerprint: `298cf6fdba946f7ab1a11249ea6c9a10a0b755dc8fa05a8904a5d1170b599c21`
- Manifest SHA-256: `765977702d57fa8de3e515f32887820e0d10d6b1c244b7dae8aef5dfc531e054`
- Media SHA-256: `b5193d8cf32d79daf17e15bea0b1c74e05156a70eddf49ca9e5d0466e568705d`
- Media size: 24,097,556 bytes; H.264, 1920×1080, 8 FPS, 546 frames
- Input-bytes fingerprint at all three run boundaries:
  `b4b6a794eea5769bfeb535a5d2475eac34b348beaf6b9dad9eec8c9c7bbe0751`

The manifest, dataset fingerprint and every media hash/size were checked
immediately before V1, between V1 and V2, and immediately after V2. All three
checkpoints matched, so `same_input_bytes_for_both_engines=true`.

## Label boundary

The only trusted labels available for this clip are three regression-known
positive plates. They are not an exhaustive inventory of every vehicle/event in
the clip, so the sample is explicitly `label_scope=known_positives`. It covers
only `multiple_vehicles`; it has no verified negative sample and lacks the other
seven required categories.

Consequently, unmatched predictions are **unscored**, not false positives.
Exact-set accuracy, precision, false-positive count and duplicate count are
unavailable (`null`) for both engines.

## Results

| Metric | V1 legacy | V2 ORT/CPU |
| --- | ---: | ---: |
| Known positive events | 3 | 3 |
| Matched / missed | 1 / 2 | 1 / 2 |
| Event recall | 0.333333 | 0.333333 |
| Mean character error rate | 0.083333 | 0.208333 |
| Raw accepted predictions | 41 | 31 |
| Unmatched, unscored predictions | 40 | 30 |
| Exact / precision / FP / duplicate | unavailable | unavailable |
| Single-sample wall processing time | 43,575.4406 ms | 33,698.6979 ms |

The processing times are recorded for reproducibility only. V1 initializes its
lazy legacy stack inside a different lifecycle boundary from V2's pre-created
shared sessions, and this is a single run, so the timing delta is not a speed or
capacity claim.

## Runtime facts

- V1 invoked the unchanged `app.ai.video_test.process_video` path with the
  `yolov8n` detector variant.
- V2 used ONNX Runtime 1.28.0 with `CPUExecutionProvider`, inference fallback
  disabled, one shared detector session and one shared OCR session, and zero
  sessions per camera.
- V2 decoded all 546 frames through FFmpeg software decode, used PTS timestamps
  for all frames, and invoked the EOF camera finalizer.
- Both adapters' model SHA/runtime-provider metadata validated, and their
  effective input options were proven symmetric (`frame_step=1`, no frame cap,
  no ROI).
- Detector model SHA-256:
  `a54e475c402e6036bb5c70f1a6ff75179e76098a5c8039bb5d148c0b6421f5c6`
- OCR model SHA-256:
  `45f8c45f29eb1ee91f6274cb8d9c328da1a2050ea7d8596bae61f4a6b9f9fb1e`

This host exposed no `/dev/dri` device and ONNX Runtime exposed no OpenVINO
provider. There was also no real RTSP source. Therefore no Intel QSV/VAAPI or
real-RTSP capacity evidence could be produced here; software-file decode must
not be presented as a hardware-decode benchmark.

## Verdict

`accuracy_gate_evaluable=false`, `v2_accuracy_not_worse=false`, and
`production_decision_allowed=false`. The explicit blockers are:

- `non-exhaustive-label-scope`
- `incomplete-required-coverage`
- `unavailable-gate-metrics`

V1 remains the production engine. The complete eight-category exhaustive set,
verified negative samples, real RTSP/QSV target hardware runs, repeated
process-isolated measurements and soak testing are still required.
