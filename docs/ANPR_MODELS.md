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

## Operator training and promotion

Confirmed corrections copy the corresponding plate crop into an immutable
local dataset with a SHA-256 digest and deterministic train/validation split.
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
