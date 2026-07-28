# BC Vision RC13 — next ANPR engine scaffold

Version: `2.2.0-rc13`

## Implemented

- YOLO26-OBB ONNX contract for traditional and end-to-end output
- four-corner mapping and mandatory perspective correction
- Hezar-compatible ONNX OCR contract
- Iranian-layout-constrained multi-hypothesis CTC beam search
- per-character probability margins and strict unreadable rejection
- five-best-frame consensus
- geometric-only ByteTrack/Kalman association
- short inference burst after first plate visibility
- baseline, shadow and next runtime modes
- shadow isolation and automatic next-to-baseline rollback
- Ed25519-signed model manifest and per-file SHA-256 verification
- Golden Dataset metrics and promotion gate
- group-aware train/validation splitting

## Intentionally not claimed

RC13 does not contain trained YOLO26-OBB or fine-tuned Hezar model weights.
Placeholder weights were not created. Activation remains impossible until a
real signed bundle is installed. Accuracy on customer cameras is not inferred
from synthetic unit tests.

## Verification

```text
143 passed, 1 skipped
```

The skipped test is the existing real-model integration check. It needs the
verified Windows model bundle and remains part of the Windows acceptance gate.
Source compilation, whitespace inspection, secret scanning, Windows packaging,
and the public Draft PR are separate gates recorded after they run.

## Remaining field gates

1. Collect original NVR video without recompression.
2. Label a fixed day/night/fast/glare/unreadable Golden Dataset.
3. Train the OBB detector and Hezar-derived OCR only from licensed data.
4. Export and sign the two ONNX models.
5. Run baseline and shadow on identical frames.
6. Promote only with exact accuracy gain and no false-accept or slice
   regression.
7. Run Windows Setup and one-click Update preservation tests before release.
