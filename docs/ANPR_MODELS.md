# BC Vision ANPR models

BC Vision stores AI models in the persistent data directory. Installer and
one-click updates therefore preserve models, promoted custom CRNNs, training
samples, settings, events, snapshots and the SQLite database.

## Active Iranian plate detectors

BC Vision installs two independently verified, single-class ONNX plate
detectors and can register one hash-verified customer YOLOX export. The
persisted `anpr_detector_model` setting selects exactly one of them for live
cameras and uploaded-video tests; `yolo11n` is the default. The YOLOX manifest,
runtime contract and installation command are documented in
[`ANPR_ENGINE_V3.md`](ANPR_ENGINE_V3.md).
Selectable inference does not cascade to the other detector after a miss or a
load failure, so counts from the same video remain attributable to the model
chosen by the operator. These customer-facing paths are forced to effective
Baseline mode: a legacy Shadow/Next runtime setting cannot start a candidate
detector beside, or instead of, the selected YOLO graph.

The YOLO11n choice is:

- `plate_yolo11n.onnx`
  - Source: `morsetechlab/yolov11-license-plate-detection`
  - Pinned revision: `0f8dc030388b3660418ac7d8c37d3a40148064c1`
  - Input: dynamic NCHW; BC Vision uses `1 x 3 x 640 x 640`
  - Output: `1 x 5 x 8400` at 640 pixels
  - Expected size: `10481682` bytes
  - SHA-256:
    `693133A1DB97A3BA1E90068986F80AFB72C3FCDDB681E57181A89A9A3DC351D6`
  - Upstream license: AGPL-3.0

The YOLOv8n choice is the verified Platrix graph formerly stored under its
upstream `plate_yolo.onnx` filename:

- `plate_yolov8n.onnx`
  - Source: `Dibachain/Platrix`, upstream file `plate_yolo.onnx`
  - Pinned revision: `4f5a43eae683e0b6ad977d4001e3967dcb96e295`
  - Architecture provenance: the Platrix training script initializes
    `yolov8n.pt`
  - Input: dynamic NCHW; BC Vision uses `1 x 3 x 416 x 416`
  - Expected size: `12608775` bytes
  - SHA-256:
    `A54E475C402E6036BB5C70F1A6FF75179E76098A5C8039BB5D148C0B6421F5C6`
  - Embedded exporter metadata: Ultralytics `8.4.104`, `AGPL-3.0 License`
    (`https://ultralytics.com/license`)
  - Platrix repository code license: MIT. That repository license does not
    override the license metadata embedded in this ONNX graph.

An existing verified
`C:\ProgramData\BCVision\data\models\plate\plate_yolo.onnx` is migrated
atomically to the explicit YOLOv8n filename, so an upgraded offline
installation can reuse the identical graph.

The historical Platrix `plate_yolo_fallback.onnx` is still verified, seeded,
and reported for compatibility, but normal selectable live/video inference
does not execute it:

- Input size: `640`
- Expected size: `12265080` bytes
- SHA-256:
  `A6974FCB0A79755C270D50F1EBEFD4D96D765C879A29051A19AAC00DFDA8B5AF`

Hardened OpenCV geometry remains available only to legacy diagnostic calls
that do not request a camera session or detector variant. Selectable live and
video paths fail closed instead of changing algorithms. Ultralytics is not
imported by the camera runtime; ONNX Runtime executes the selected graph. The
YOLO11n model card reports possible train/test contamination in its source
dataset, so it must be measured on BC Vision's independent field/Golden data
before accuracy claims are made. Both selectable graphs carry AGPL-related
provenance. Their model-weight, execution, redistribution and commercial-use
terms require legal/licensing review before commercial distribution; the MIT
license on Platrix repository code is not evidence that its exported graph is
MIT-licensed.

## Active Iranian OCR models

The official Hezar v2 CRNN is the first whole-plate OCR reader:

- `crnn_fa_v2.onnx`
  - Source: `hezarai/crnn-fa-license-plate-recognition-v2`
  - Pinned revision: `0c48a86abe5bfb140ceeb160c028701028d236b9`
  - Input: mirrored grayscale `1 x 1 x 32 x 384`
  - Output: CTC logits `1 x 96 x 45`
  - Expected size: `37146355` bytes
  - SHA-256:
    `57CB02BC10BDEBD14BE2AC50CD7C25D657BDCDEE6EFE77A37A561B832206B0C8`
  - Blank label index: `0`; Persian output digits are normalized to ASCII
    internally before the Iranian plate-layout gate.

The build downloads the immutable Hezar revision, verifies all three source
files, exports opset-18 ONNX, checks PyTorch/ONNX Runtime numerical parity and
accepts only the fixed ONNX hash above. Portable builds ship that ONNX file in
`model-seed`; the live camera path does not load PyTorch or the Hezar Python
package.

The fixed Platrix CRNN remains the only production fallback:

- `ocr_crnn.onnx`: full-plate CRNN, 10452525 bytes,
  SHA-256
  `45F8C45F29EB1EE91F6274CB8D9C328DA1A2050EA7D8596BAE61F4A6B9F9FB1E`
- `ocr_cnn.onnx`: diagnostic-only eight-glyph Iranian CNN, 2226402 bytes,
  SHA-256
  `7D573C51CC855A8E080F1F88597477F4FB5A2B9CAFA1BB125BD6038E441F5BCA`

Production OCR order is immutable: Hezar v2, then the fixed Platrix CRNN only
after Hezar rejects or errors. Every candidate must still satisfy the
eight-position Iranian plate layout and the Platrix confidence floor. Missing
characters are never invented. The promoted custom CRNN and character CNN are
available only to explicit training/diagnostic APIs. Legacy
`BCVISION_OCR_ENGINE` values are ignored by the production route.

## Persistent Windows paths

```text
C:\ProgramData\BCVision\data\models\plate\plate_yolo11n.onnx
C:\ProgramData\BCVision\data\models\plate\plate_yolov8n.onnx
C:\ProgramData\BCVision\data\models\plate\plate_yolo_fallback.onnx
C:\ProgramData\BCVision\data\models\plate\yolox-<sha-prefix>.onnx
C:\ProgramData\BCVision\data\models\plate\yolox-custom.json
C:\ProgramData\BCVision\data\models\hezar\crnn_fa_v2.onnx
C:\ProgramData\BCVision\data\models\crnn\ocr_crnn.onnx
C:\ProgramData\BCVision\data\models\cnn\ocr_cnn.onnx
C:\ProgramData\BCVision\data\models\crnn\custom\...
C:\ProgramData\BCVision\data\anpr-training\...
```

Environment variables can override vendor model locations:

```text
BCVISION_PLATE_MODEL
BCVISION_PLATE_YOLOV8N_MODEL
BCVISION_PLATE_FALLBACK_MODEL
BCVISION_YOLOX_MODEL
BCVISION_YOLOX_MANIFEST
BCVISION_HEZAR_MODEL
BCVISION_CRNN_MODEL
BCVISION_CNN_MODEL
BCVISION_MODEL_SOURCE_DIR
BCVISION_HEZAR_SOURCE_DIR
BCVISION_HEZAR_CACHE_DIR
BCVISION_CRNN_SOURCE_DIR
BCVISION_CNN_SOURCE_DIR
BCVISION_ONNX_DETECTOR_SIZE
BCVISION_CPU_THREADS
```

`BCVISION_PLATE_YOLO8N_MODEL` is retained as a compatibility alias for the
YOLOv8n override. Every model is accepted only after exact size and SHA-256
verification. Production passes one service-wide inference key, so detector,
Hezar and Platrix sessions are shared across cameras and protected by per-model
run locks. They use at most two intra-operation threads, one inter-operation
thread, sequential execution and disabled worker spinning. Camera queues and
trackers remain isolated. Changing the persisted detector selection invalidates
the live status/session and temporal-consensus cache.

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

Candidate evaluation remains available to explicit offline development and
Golden/benchmark tooling. The customer live-camera, uploaded-camera and
uploaded-video-test paths no longer start candidate Shadow/Next inference:
they run only the YOLO11n or YOLOv8n detector selected in settings. Candidate
weights remain blocked on licensed real field data and the Golden promotion
gates.

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

`BCVISION_ANPR_MODE` applies only to explicit candidate-engine tooling. It
does not override the persisted exclusive detector selection in production
live or uploaded-video paths.

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
multi-frame review path is stored with `review_status=auto-confirmed`. In the
current selectable production path that evidence comes only from the chosen
Baseline detector; candidate Shadow rows are not generated. The UI shows the
source explicitly and offers `تأیید/اصلاح و آموزش`.

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
or included in Setup/Update. It binds a fixed verified Baseline detector,
installs the Stage-4 CCT weight under the persistent data root, verifies every
payload by exact SHA-256 and writes a signed `research-shadow-only` manifest.
That candidate is evaluated through explicit offline Golden/benchmark tools;
the production live and uploaded-video workers remain exclusive-Baseline even
if a legacy runtime-state file requests Shadow mode. Operator confirmation
remains mandatory for training.

### Hezar v2 export and isolated benchmark

`tools/export_hezar_onnx.py` exports the official Hezar v2 Persian
license-plate CRNN to the exact ONNX contract used by RC13. It also writes the
label map and preprocessing metadata beside the ONNX file and rejects an
export whose ONNX Runtime output differs materially from PyTorch.

`tools/benchmark_hezar_video.py` pairs an exported OCR with the current
detector for an isolated OCR comparison before an OBB weight exists. Truth
plates must be supplied explicitly; other outputs remain unverified.

The historical `01.mp4` benchmark on 2026-07-28 processed all 546 frames.
Baseline and ready-made Hezar v2 each matched only 1 of the 3 known plates,
while Hezar was 37.3% slower. It was not auto-promoted at that checkpoint. The
Engine V3 now makes Hezar-first the explicit production policy, but that policy
decision does not turn the historical three-plate run into accuracy evidence.
It must be re-evaluated with the registered YOLOX crops and independent
field/Golden data before any comparative accuracy claim.
See `agent-results/latest/ANPR_HEZAR_VIDEO_BENCHMARK_RC13.md`.

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
