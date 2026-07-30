# BC Vision — commercial-clean synthetic 30K OCR pilot

Date: 2026-07-30

## Decision

The 30,000-image synthetic pilot trained successfully and produced a valid
FastPlateOCR CCT-XS-v2 ONNX model. It is commercially clean with respect to
the image corpus, but it is **rejected for automatic activation** because it
matched none of the three trusted plates in the fixed real `01.mp4` benchmark.

The artifact is distributable for internal evaluation and Shadow testing.
Its metadata is fixed to:

- `usage_scope=production-candidate`
- `distribution_allowed=true`
- `activation_allowed=false`
- `activation_gate=independent-real-camera-pass`

RC12 remains the active customer engine.

## Dataset

| Split | Images | Unique plate identities |
|---|---:|---:|
| Train | 24,000 | 2,400 |
| Validation | 3,000 | 300 |
| Test | 3,000 | 300 |
| Total | 30,000 | 3,000 |

Every identity has ten deterministic views. Each split is exactly balanced
across clean, daylight, night, motion blur, perspective, headlight glare,
rain, dirt, low resolution and mixed-hard profiles.

Audit results:

- Train/Validation/Test plate-identity overlap: `0/0/0`
- duplicate image SHA-256 across all splits: `0`
- missing or unreadable images: `0`
- invalid labels or shapes: `0`
- image contract: RGB `128x64`
- label contract: eight slots, 37-class alphabet
- third-party plate pixels used: `false`
- IR-LPR images or derived weights used: `false`
- random initialization: `true`
- Golden video frames used for training: `false`

The source license is `synthetic-bcvision-company-owned`. The bootstrap font
is DejaVu Sans Bold under the DejaVu font license. Its SHA-256 is
`5C1247ACEF7F2B8522A31742C76D6ADCB5569BACC0BE7CEAA4DC39DD252CE895`.
DejaVu remains a bootstrap font and is not a fidelity claim for the physical
Iranian plate typeface.

## Training

| Setting | Value |
|---|---|
| Architecture | FastPlateOCR CCT-XS-v2 |
| Backend | TensorFlow CPU |
| Initialization | Random; no pretrained or resume checkpoint |
| Epochs | 30 |
| Batch size | 32 |
| Seed | 20260730 |
| Learning rate | 0.0005 |
| Checkpoint metric | Validation character accuracy |
| Best epoch | 24 |
| Augmentation | FastPlateOCR default |

At the selected checkpoint, Validation exact-plate accuracy was `99.50%`,
character accuracy was `99.94%` and loss was `0.09316`.

## Held-out synthetic results

| Metric | Validation | Test |
|---|---:|---:|
| Samples | 3,000 | 3,000 |
| Raw exact-plate accuracy | 99.50% | 99.57% |
| Raw character accuracy | 99.94% | 99.94% |
| Accepted exact accuracy over all samples | 99.40% | 99.53% |
| Precision among accepted results | 99.63% | 99.73% |
| Rejection rate | 0.23% | 0.20% |
| Mean CPU latency | 3.65 ms | 4.53 ms |

### Test result by condition

| Condition | Samples | Raw exact | Accepted precision | Rejected |
|---|---:|---:|---:|---:|
| Clean | 300 | 100.00% | 100.00% | 0.00% |
| Daylight | 300 | 100.00% | 100.00% | 0.00% |
| Night | 300 | 100.00% | 100.00% | 0.00% |
| Motion blur | 300 | 100.00% | 100.00% | 0.00% |
| Perspective | 300 | 100.00% | 100.00% | 0.00% |
| Headlight glare | 300 | 100.00% | 100.00% | 0.00% |
| Rain | 300 | 100.00% | 100.00% | 0.00% |
| Dirt | 300 | 100.00% | 100.00% | 0.00% |
| Low resolution | 300 | 100.00% | 100.00% | 0.00% |
| Mixed hard | 300 | 95.67% | 97.28% | 2.00% |

The mixed-hard profile is the only meaningful synthetic failure slice. Across
all 1,200 hard samples, raw exact accuracy is `98.92%`, accepted precision is
`99.33%` and rejection is `0.50%`.

### Test result by plate style

| Style | Samples | Raw exact |
|---|---:|---:|
| Diplomatic | 210 | 99.52% |
| Government | 200 | 100.00% |
| Military | 270 | 99.26% |
| Private | 1,750 | 99.60% |
| Public | 360 | 99.44% |
| Service | 210 | 99.52% |

## ONNX verification

- Model size: `2,395,227` bytes
- SHA-256:
  `DE47B20DE2CF545276977E230EF07BD88657D1DFB9C7956C62873A7C731A966F`
- Input: fixed `uint8 [1,64,128,3]`, RGB/NHWC
- Output: `[1,8,37]`
- ONNX checker: passed
- Keras/ONNX export parity: passed
- Runtime provider: CPUExecutionProvider

ONNX Runtime reports a non-fatal `UnsqueezeElimination` optimization warning;
the graph checker, inference contract and Keras parity all pass.

## Fixed real-video result

The benchmark used the exact trusted fixture:

- File: `01.mp4`
- SHA-256:
  `B5193D8CF32D79DAF17E15BEA0B1C74E05156A70EDDF49CA9E5D0466E568705D`
- Frames: 546/546
- Trusted plate identities: 3 (retained only in the private Golden fixture)
- Detector mode: verified primary/fallback ONNX, strict; OpenCV disabled
- Primary detector SHA-256:
  `A54E475C402E6036BB5C70F1A6FF75179E76098A5C8039BB5D148C0B6421F5C6`
- Fallback detector SHA-256:
  `A6974FCB0A79755C270D50F1EBEFD4D96D765C879A29051A19AAC00DFDA8B5AF`
- Detections: 807
- Elapsed: 62.163 seconds
- strict matched truth: `0/3`
- raw exact truth: `1/3`
- emitted results: 23
- unmatched emitted unique strings: 21

Both detector files passed pinned size and SHA-256 preflight before any video
frame was processed. The benchmark then required the ONNX runtime to report a
loaded primary model after every inference. Missing or changed detector
files now fail the benchmark; downloading/preparing models and permitting the
OpenCV diagnostic path each require a separate explicit flag.

Restoring the production detector removed most irrelevant candidates and
produced one exact single-frame raw hypothesis, but it did not produce the
three consistent votes required for a trusted event. The result is therefore
a detector-equivalent rejection, not evidence for activation.

## Required next step

Do not activate this model and do not use its AI guesses as training labels.
The local checkout was audited after the strict rerun:

- confirmed operator samples: `0`
- Train samples: `0`
- Validation samples: `0`
- unique verified plate identities: `0`
- Golden/test samples admitted to training: `0`

No real-camera fine-tune was run because doing so would require inventing
labels or leaking the Golden benchmark. The application now exports its
schema-2 operator-confirmed dataset directly into the CCT preparation path.
Image hashes are rechecked, identical crops are deduplicated, conflicting
labels fail closed, and plate identities remain isolated across Train and
Validation. The aligned minimum gate is 24 Train samples, 12 Validation
samples and eight unique identities. Operator confirmation establishes the
label but not ownership, so these exports default to
`operator-confirmed-rights-unverified`, `distribution_allowed=false`; an
explicit, evidenced ownership attestation is required before commercial
distribution.

The next useful experiment is:

1. collect operator-confirmed real plate crops from company-owned cameras;
2. keep vehicle/plate identities isolated across Train/Validation/Test;
3. fine-tune from this commercially clean checkpoint;
4. rerun the independent Golden and real-camera gates;
5. replace the bootstrap font with approved plate-faithful glyph sources; and
6. scale synthetic data toward 100,000–300,000 only after real-camera transfer
   improves rather than regresses.
