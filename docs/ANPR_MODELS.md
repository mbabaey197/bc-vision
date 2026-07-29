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

The primary detector runs first. The second ONNX detector runs only after a
zero-result primary pass. Hardened OpenCV geometry is the final localization
fallback. The retired 119 MB combined `best.pt` model and Ultralytics are not
loaded or packaged by RC12.

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

CRNN reads the full crop first. The CNN is attempted only if CRNN has no valid
complete Iranian plate and exactly eight real glyph regions can be segmented.
Missing characters are never invented. EasyOCR and Tesseract are no longer in
the RC12 production path or Windows package.

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
BCVISION_CPU_THREADS
```

Every model is accepted only after exact size and SHA-256 verification.
Per-camera ONNX Runtime sessions use at most two intra-operation threads, one
inter-operation thread, sequential execution and disabled worker spinning.

## RC14 candidate bundle

RC14 extends the fail-safe engine contract without replacing the verified
RC12 models:

- a rotated YOLO26-OBB detector exported to ONNX;
- a custom FastPlateOCR CCT-S/XS Iranian OCR exported to ONNX;
- mandatory perspective rectification from the four OBB corners;
- fixed-position Iranian-layout decoding with multiple plate hypotheses;
- `baseline`, `shadow` and `next` runtime modes.

The candidate engine is not enabled merely because two ONNX files exist. Its
`active-models.json` must use schema `1`, identify either the compatible
`bcvision-rc13` engine or the new `bcvision-rc14` engine, list the
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

The signed OCR entry must declare its runtime. RC14 accepts the historical
`hezar-ctc-onnx` contract and the new `fast-plate-ocr-cct` contract. A CCT
entry also signs its alphabet, eight output slots, input dimensions, layout,
dtype, colour mode and rejection thresholds. Unknown runtimes or incomplete
CCT contracts fail closed before ONNX Runtime is started. CCT entries require
the `bcvision-rc14` engine identifier; legacy RC13 bundles remain Hezar-only.

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
  held-out accuracy and CPU latency. The validated training environment is
  Python 3.12 with TensorFlow CPU. An optional released FastPlateOCR model can
  initialize only shape-compatible backbone tensors; OCR, region and
  incompatible slot-query tensors are never transferred.
- `tools/benchmark_cct_video.py` processes every selected video frame with the
  current detector and multi-frame tracker. It can require the exact video
  SHA-256 before execution.

The public IR-LPR dataset is not used by this path because its repository
declares GPL-3.0. BC Vision's proprietary candidate training accepts
company-owned, operator-confirmed, CC0 or CC-BY-4.0 data only. The fixed
`01.mp4` Golden Dataset remains benchmark-only and must not enter training.
Synthetic generation also requires an approved commercial font license and
does not accept a third-party plate dataset hidden behind the synthetic
manifest.

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
is evaluated against the currently active model on the isolated validation
split. It cannot be applied when it regresses or scores below the promotion
threshold. Applying a verified candidate copies it into persistent storage,
records its digest and run ID, and atomically changes the active manifest.

## Operational limitation

Software cannot recover a plate whose pixels are absent, fully hidden,
destroyed by severe motion blur or saturation, or outside the frame. Camera
placement, focus, shutter speed, resolution, plate pixel height and lighting
remain part of end-to-end accuracy.
