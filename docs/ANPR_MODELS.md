# BC Vision ANPR models

BC Vision stores AI models in the persistent data directory. Installer and
one-click updates therefore preserve models, promoted custom CRNNs, training
samples, settings, events, snapshots and the SQLite database.

## Lightweight Iranian plate detectors

Both detector models come from the MIT-licensed Platrix model repository:
`https://huggingface.co/Dibachain/Platrix`.

- `plate_yolo.onnx`
  - Input size: `416`
  - Expected size: `12608775` bytes
  - SHA-256:
    `A54E475C402E6036BB5C70F1A6FF75179E76098A5C8039BB5D148C0B6421F5C6`
- `plate_yolo_fallback.onnx`
  - Input size: `640`
  - Expected size: `12265080` bytes
  - SHA-256:
    `A6974FCB0A79755C270D50F1EBEFD4D96D765C879A29051A19AAC00DFDA8B5AF`

The primary detector runs first. RC22 runs the 640 detector adaptively when the
primary result is absent or weak, merges both result sets with cross-model NMS,
and applies rate-limited overlapping landscape tiles when a high-resolution
frame still has insufficient evidence. A reliable local plate contour is also
exposed as a perspective-normalized OCR variant; the raw detector crop remains
available so geometry refinement cannot erase evidence. Hardened OpenCV
geometry is the final localization fallback. The retired 119 MB combined
`best.pt` model and Ultralytics are not loaded or packaged.

## Iranian OCR models

- `ocr_crnn.onnx`
  - Segmentation-free whole-plate CRNN+CTC reader
  - Expected size: `10452525` bytes
  - SHA-256:
    `45F8C45F29EB1EE91F6274CB8D9C328DA1A2050EA7D8596BAE61F4A6B9F9FB1E`
- `ocr_cnn.onnx`
  - Eight-glyph Iranian character fallback
  - Expected size: `2226402` bytes
  - SHA-256:
    `7D573C51CC855A8E080F1F88597477F4FB5A2B9CAFA1BB125BD6038E441F5BCA`

CRNN reads the full crop first with a plate-layout-constrained CTC prefix beam.
A plausible string is no longer accepted solely because it happens to match the
Iranian layout: sequence confidence, per-position margin and decoder/view
agreement are checked. Weak crops can use a perspective-refined view and one
adaptive illumination/contrast view, while decisive crops still require one
inference. Rejected beam hypotheses remain visible for operator review and are
not silently converted into confirmed truth. The CNN is attempted only if CRNN
has no accepted complete Iranian plate and exactly eight real glyph regions can
be segmented. Missing characters are never invented. EasyOCR and Tesseract are
not in the production path or Windows package.

## Persistent Windows paths

```text
C:\ProgramData\BCVision\data\models\plate\plate_yolo.onnx
C:\ProgramData\BCVision\data\models\plate\plate_yolo_fallback.onnx
C:\ProgramData\BCVision\data\models\crnn\ocr_crnn.onnx
C:\ProgramData\BCVision\data\models\cnn\ocr_cnn.onnx
C:\ProgramData\BCVision\data\models\crnn\custom\...
C:\ProgramData\BCVision\data\anpr-training\...
```

Environment variables can override vendor model locations:

```text
BCVISION_PLATE_MODEL
BCVISION_PLATE_FALLBACK_MODEL
BCVISION_CRNN_MODEL
BCVISION_CNN_MODEL
BCVISION_MODEL_SOURCE_DIR
BCVISION_CRNN_SOURCE_DIR
BCVISION_CNN_SOURCE_DIR
BCVISION_ONNX_DETECTOR_SIZE
BCVISION_DETECTOR_CASCADE=off|adaptive|accuracy
BCVISION_TILE_RESCUE_INTERVAL
BCVISION_CRNN_BEAM_WIDTH
BCVISION_CRNN_RESCUE_VIEWS
BCVISION_CRNN_MIN_CONFIDENCE
BCVISION_CRNN_MIN_MARGIN
BCVISION_CNN_MIN_CONFIDENCE
BCVISION_CPU_THREADS
```

Every model is accepted only after exact size and SHA-256 verification.
Per-camera ONNX Runtime sessions use at most two intra-operation threads, one
inter-operation thread, sequential execution and disabled worker spinning.

## RC15 selected candidate bundle

RC15 implements the selected next-generation pairing while preserving RC12 as
the active customer-visible engine:

- custom Apache-2.0 PP-YOLOE-R-s rotated detector;
- custom MIT FastPlateOCR CCT-XS-v2 Iranian OCR;
- current CRNN as the independent control reader;
- strict multi-frame consensus and separately visible raw guesses.

The signed `bcvision-rc15` detector entry uses runtime
`ppyoloe-r-onnx`. Its manifest binds input width/height, aspect-ratio resize,
stride-32 padding, RGB ImageNet normalization, score threshold, rotated-NMS
threshold and maximum result count. The ONNX graph must expose PaddleDetection
inputs `image`, `im_shape` and `scale_factor`, with outputs `B x N x 8`
quadrilaterals and `B x C x N` scores. Any incomplete or unknown contract fails
closed before the model can affect a result.

`tools/prepare_ppyoloe_r_dataset.py` converts explicitly licensed,
operator-labelled company images to rotated COCO annotations. It supports
empty hard-negative images for timestamps, camera names, signs, headlights and
other false localizations. Golden/test frames, unapproved licenses, invalid
corners and source-image leakage across train/validation are rejected.
The matching one-class PaddleDetection configuration lives under
`training/ppyoloe_r/`.

An OCR hypothesis rejected by its signed threshold is still exposed to the
operator as an experimental raw guess with confidence and rejection reason.
It is excluded from definitive consensus. Repetition cannot promote that
rejected guess to `confirmed-ai`; only independently accepted OCR evidence can
do so. An operator correction stores the original guess, engine, model
revision, exact-match result and character distance. Aggregate measurements
therefore show whether a later trained revision improves over earlier ones.

The video-test path runs Baseline and a verified candidate in separate lanes.
Shadow rows are labelled experimental even when the candidate accepts them,
and never replace Baseline events. Live shadow boxes are amber and labelled
`GUESS`; confirmed Baseline boxes remain green. Candidate weights are still
blocked on licensed real field data and the Golden promotion gates.

## RC14 candidate bundle

RC14 extends the fail-safe engine contract without replacing the verified
RC12 models:

- a rotated YOLO26-OBB detector exported to ONNX;
- a custom FastPlateOCR CCT-S/XS Iranian OCR exported to ONNX;
- mandatory perspective rectification from the four OBB corners;
- fixed-position Iranian-layout decoding with multiple plate hypotheses;
- `baseline`, `shadow` and `next` runtime modes.

The candidate engine is not enabled merely because two ONNX files exist. Its
`active-models.json` must use schema `1`, identify a compatible
`bcvision-rc13`, `bcvision-rc14` or `bcvision-rc15` engine, list the
exact filename, size and SHA-256 of both files, and carry an Ed25519 signature
verified by `model_public_key.pem`. Missing, modified or unsigned bundles fail
closed to `baseline`.

```text
C:\ProgramData\BCVision\data\models\next\active-models.json
C:\ProgramData\BCVision\data\models\next\model_public_key.pem
C:\ProgramData\BCVision\data\models\next\<detector>.onnx
C:\ProgramData\BCVision\data\models\next\<ocr>.onnx
C:\ProgramData\BCVision\data\models\next\runtime-state.json
```

Optional environment overrides:

```text
BCVISION_NEXT_MANIFEST
BCVISION_ANPR_MODEL_PUBLIC_KEY
BCVISION_ANPR_MODE=baseline|shadow|next
```

No trained YOLO26-OBB or promoted CCT weights are committed in RC14. Those
weights must be produced from licensed data, signed, and pass the fixed real
camera Golden Dataset in shadow mode before `next` can be activated.

The signed OCR entry must declare its runtime. RC14/RC15 accept the historical
`hezar-ctc-onnx` contract and the new `fast-plate-ocr-cct` contract. A CCT
entry also signs its alphabet, eight output slots, input dimensions, layout,
dtype, colour mode and rejection thresholds. Unknown runtimes or incomplete
CCT contracts fail closed before ONNX Runtime is started. CCT entries require
the `bcvision-rc14` or `bcvision-rc15` engine identifier; legacy RC13 bundles
remain Hezar-only.

### FastPlateOCR CCT-S/XS training and benchmark

`fast-plate-ocr` v1.1.0 is MIT licensed and is used only by the offline
training/export environment. Installed BC Vision systems continue to run only
the exported ONNX file through the existing bounded ONNX Runtime sessions.

- `tools/generate_cct_synthetic_dataset.py` creates a reproducible synthetic
  Iranian-plate dataset with disjoint train/validation plate identities and
  balanced coverage of every supported Persian and diplomatic letter.
- `tools/prepare_cct_dataset.py` imports only explicitly licensed,
  operator-labelled company crops and rejects Golden Dataset rows.
- `tools/train_fastplate_cct.py` trains CCT-XS-v2 or CCT-S-v2, exports a
  fixed-batch uint8 NHWC ONNX file, verifies its SHA-256 and measures exact
  held-out accuracy, character accuracy, mean character error, per-position
  accuracy and CPU latency. Long CPU runs can resume from a verified Keras
  checkpoint with an explicit learning rate and checkpoint metric. The
  validated training environment is Python 3.12 with TensorFlow CPU. An
  optional released FastPlateOCR model can initialize only shape-compatible
  backbone tensors; OCR, region and incompatible slot-query tensors are never
  transferred.
- `tools/benchmark_cct_video.py` processes every selected video frame with the
  current detector and multi-frame tracker. It can require the exact video
  SHA-256 before execution. It separately records the closest raw guess to
  each labelled truth, character distance, exact-guess frequency and crop, so
  rejected/single-frame hypotheses remain visible without being promoted to
  strict confirmed consensus.

### Operator-assisted automatic confirmation

When `anpr_auto_confirm_guesses=1`, a complete guess supported by the live
multi-frame review path is stored with `review_status=auto-confirmed`. A
complete Shadow guess may replace only an overlapping unreadable Baseline
result; a strict Baseline read keeps priority. The UI shows the source
explicitly and offers `تأیید/اصلاح و آموزش`.

Automatic confirmation is not a training label. The event remains
`experimental=1`, `confirmation_source=ai-auto-guess` and
`operator_reviewed=0`. Only an authorized operator submission changes it to
`review_status=confirmed`, records exact-match/character distance and captures
the crop for the controlled training dataset. The setting can be disabled
from AI Settings without deleting existing events.

The public IR-LPR dataset remains excluded from the proprietary production
model path because its repository declares GPL-3.0. RC15 adds a separate,
explicitly non-distributable research adapter:

- `tools/prepare_ir_lpr_dataset.py` reads the official VOC/XML train,
  validation and test ZIPs or extracted directories;
- standard eight-slot Iranian plate labels are reconstructed from the
  published IR-LPR character classes without inventing missing characters;
- duplicate images and plate identities are removed from later splits;
- the official test split remains held out from training;
- every resulting manifest is fixed to `research-shadow-only`, sets
  `distribution_allowed=false`, and cannot activate the `next` mode.

Commercial candidate training still accepts only company-owned,
operator-confirmed, CC0 or CC-BY-4.0 data. The fixed `01.mp4` Golden Dataset
remains benchmark-only and must not enter either training path. Synthetic
generation also requires an approved commercial font license and does not
accept a third-party plate dataset hidden behind the synthetic manifest.

For the owner's private internal RC15 evaluation only,
`tools/build_internal_cct_model_installer.py` can build a separate one-click
model pack. The pack is never committed, attached to a public GitHub release,
or included in Setup/Update. It binds the fixed verified Baseline detector,
installs the Stage-4 CCT weight under the persistent data root, verifies every
payload by exact SHA-256, writes a signed `research-shadow-only` manifest and
selects Shadow mode. The live worker reuses the Baseline detector crop for CCT
OCR, matching the detector/OCR pairing used in the Golden benchmark and
avoiding a second detector pass. Operator confirmation remains mandatory for
training.

### Hezar v2 export and isolated benchmark

`tools/export_hezar_onnx.py` exports the official Hezar v2 Persian
license-plate CRNN to the exact ONNX contract used by RC13. It also writes the
label map and preprocessing metadata beside the ONNX file and rejects an
export whose ONNX Runtime output differs materially from PyTorch.

`tools/benchmark_hezar_video.py` pairs an exported OCR with the current
detector for an isolated OCR comparison before an OBB weight exists. Truth
plates must be supplied explicitly; other outputs remain unverified.

The historical `01.mp4` benchmark on 2026-07-28 processed all 546 frames. Baseline
and ready-made Hezar v2 each matched only 1 of the 3 known plates, while Hezar
was 37.3% slower. The checkpoint was therefore not promoted. See
`agent-results/latest/ANPR_HEZAR_VIDEO_BENCHMARK_RC13.md`.

## Operator training and promotion

Confirmed corrections copy the corresponding plate crop into an immutable
local dataset with a SHA-256 digest and deterministic group-aware
train/validation split. Frames with the same confirmed plate never cross the
split boundary.
Training runs outside live inference and create a candidate CRNN. The candidate
is evaluated against the currently active model on an immutable per-run
validation snapshot. RC16 stores a hash-verified PyTorch state dictionary next
to every promoted ONNX model, so later runs start from the active checkpoint;
the vendor-only first run remains explicitly identified as active-model
distillation.

Promotion now fails closed unless all of these conditions hold:

- at least 12 independent validation samples;
- no exact-plate, mean-character-error or per-sample regression;
- immutable identity of the active baseline and candidate SHA-256;
- a complete Golden comparison with more exact reads, no slice regression,
  no false-accept regression and bounded CPU latency;
- a Golden manifest containing at least 40 operator-labelled samples,
  20 unique readable plates and at least three samples in every required
  slice: day, night, fast, angle, blur, glare, unreadable and multi-vehicle.

`tools/prepare_golden_dataset.py` copies operator-labelled media into a
hash-verified, training-forbidden Golden directory and reports missing
coverage. The three labels from `01.mp4` remain trusted regression truth, but
do not satisfy this larger admission contract by themselves. Until real media
fills every slice and the same end-to-end pipeline passes the comparison, a
trained candidate remains `awaiting-golden` or `rejected` and cannot be
applied.

## Operational limitation

Software cannot recover a plate whose pixels are absent, fully hidden,
destroyed by severe motion blur or saturation, or outside the frame. Camera
placement, focus, shutter speed, resolution, plate pixel height and lighting
remain part of end-to-end accuracy.
