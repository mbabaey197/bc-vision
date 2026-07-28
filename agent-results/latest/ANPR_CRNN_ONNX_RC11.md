# BC Vision ANPR — whole-plate CRNN on ONNX Runtime

Version: `2.2.0-rc11`

## Implemented

- Added a segmentation-free Iranian CRNN+CTC whole-plate reader.
- Executes the CRNN model through ONNX Runtime on CPU.
- Stores the verified model under persistent application data.
- Pins model size and SHA-256 before load, download, seed or update.
- Keeps distinct bounded ONNX sessions for the default three camera keys.
- Hard-caps ONNX intra-operation threads at two per camera.
- Uses one inter-operation thread, sequential execution and no thread spinning.
- Keeps the RC10 character detector as an independent A/B reader.
- Boosts confidence on agreement and retains both hypotheses on disagreement.
- Marks a selected disagreement as reviewable instead of hiding uncertainty.
- Persists selected engine, alternative read and disagreement in additive event
  columns, and exposes per-camera A/B counters.
- Keeps EasyOCR/Tesseract as the final credible-crop fallback.
- Adds the model's MIT notice to source and Windows output.
- Preserves the database, users, settings, license, media and existing models.

## Verified locally

- Focused CRNN, OCR, model-manager, pipeline, CPU, worker and packaging tests:
  `51 passed`.
- Full source regression: `121 passed, 1 skipped`.
- Python compileall: passed.
- Whitespace validation: passed.
- CTC repeat/blank collapse: passed.
- Per-camera session separation: passed.
- A Windows two-core runner exposed a modulo collision between camera IDs 1
  and 2. The cache now uses the exact camera key, retains three distinct
  sessions independently of the concurrency limit, and has deterministic
  low-core regression coverage.
- Attempted thread override `9` was clamped to `2`: passed.
- Missing/unverified model failed closed and left legacy fallback available.
- Stronger CRNN disagreement kept both reads and required review.
- Real ONNX Runtime `1.28.0` smoke with a generated ONNX graph completed the
  full input, inference, CTC and Iranian-format path as `31-ط-556-74`, with
  reported thread limit `2`.

## Model contract

- Source: `Dibachain/Platrix`
- File: `ocr_crnn.onnx`
- Size: `10,452,525` bytes
- SHA-256:
  `45f8c45f29eb1ee91f6274cb8d9c328da1a2050ea7d8596bae61f4a6b9f9fb1e`
- License: MIT

## Pending release gates

- Load and execute the real verified model on Windows Python 3.13.
- Build the portable Windows directory, installer and one-click updater.
- Verify the installed and updated executable with offline detector, CRNN and
  EasyOCR inference.
- Confirm database, settings, media and model preservation across update.
- Compare RC10 and RC11 against labelled frames from `01.mp4`.
- Measure multi-camera CPU and memory use with two threads per camera.
