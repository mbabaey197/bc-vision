# BC Vision ANPR Engine V3

Engine V3 is the production integration contract for one explicitly selected,
single-class ONNX plate detector with Hezar v2 as the authoritative OCR
reader. The current production default and recommended detector is the pinned
YOLO11n plate model. The customer-trained YOLOX adapter remains an optional,
fail-closed integration for a later field comparison; it is not selected or
required by the current release.

```text
nonblocking camera submit -> one replaceable pending slot
  -> queued worker atomically claims the newest same-revision frame
  -> shared, hash-verified selected-detector ONNX session (YOLO11n by default)
  -> plate crop and quality gate
  -> shared Hezar v2 ONNX session
  -> only after Hezar rejects/errors: fixed Platrix CRNN
  -> exact full-plate temporal consensus (minimum 3 observations)
  -> existing deduplication, persistence and live overlay
```

## Production invariants

- A miss from the selected detector is authoritative. No hidden YOLO11,
  YOLOv8, YOLOX or OpenCV detector runs after it.
- YOLO11n is the persisted default and recommended production selection. Its
  graph, size and SHA-256 are pinned and its runtime contract is fixed at a
  `1 x 3 x 640 x 640` input.
- YOLOX is selectable only when its bytes match the size and SHA-256 in its
  persistent manifest.
- Hezar v2 is always called first. An accepted Hezar result is returned
  immediately and cannot be overridden by another OCR engine.
- The Platrix fallback always uses the fixed, vendor-pinned `ocr_crnn.onnx`.
  A promoted custom CRNN is never loaded by this production route.
- The legacy character CNN and detector-attached character hypotheses are
  excluded from production voting. They remain available to explicit
  diagnostics/training code.
- ONNX sessions are shared across cameras and protected by per-model run
  locks. Camera queues, trackers and duplicate state remain independent.
- Every inference carries its detector contract revision even when no plate is
  detected. Live trackers reset before consuming the first frame of a hot
  revision. An uploaded-video test pins one revision for the entire pass; for
  optional YOLOX this also means aborting if its manifest changes.
- Model installation writes a versioned model file and switches the manifest
  last. Application updates do not overwrite data-directory model files.

## Optional: installing a custom YOLOX export

The model itself is installation data and is not committed to the application
repository. Install it on the target machine with an explicit export contract:

```powershell
python -m app.ai.model_manager `
  --install-yolox D:\Models\plate_yolox.onnx `
  --yolox-input-size 640 `
  --yolox-output-format raw-grid `
  --yolox-output-index 0 `
  --yolox-class-count 1 `
  --yolox-plate-class-id 0 `
  --yolox-strides 8,16,32 `
  --yolox-color bgr `
  --yolox-input-scale 1.0 `
  --yolox-letterbox top-left
```

Those preprocessing defaults match a standard Megvii YOLOX export. They must
be changed if the training/export pipeline used RGB, `1/255` normalization or
centered letterboxing. If the selected graph output emits objectness and class
scores as logits, add `--yolox-scores-are-logits`; probability outputs must not
use that flag.

Supported output contracts are intentionally explicit:

- `raw-grid`: rows are `cx_offset, cy_offset, log_w, log_h, objectness,
  class_scores...`; coordinates are in grid space and Engine V3 applies the
  manifest strides.
- `decoded-cxcywh`: rows use decoded input-pixel `cx, cy, w, h` followed by
  objectness and class scores.
- `nms-xyxy`: the graph already emits `x1, y1, x2, y2, score[, class_id]`.

`decoded-cxcywh` and `nms-xyxy` coordinates must be pixels in the letterboxed
model input. Normalized coordinates or original-frame coordinates are not
accepted. `--yolox-output-index` selects the detection output explicitly for
multi-output graphs; the adapter never guesses it. For a multi-class raw or
decoded graph, a row is a plate only when `plate_class_id` is the highest class
score. Multi-class `nms-xyxy` outputs must include the class-id column.

The installer stores a schema-v1 JSON sidecar under the persistent data
directory. Before the manifest is activated it loads the graph in ONNX Runtime,
checks the input and selected output, performs a zero-input dry run and decodes
the result against the declared contract. Unknown shapes, transposed output,
hash mismatches and incomplete manifests fail closed instead of guessing a
decoder.

The current V3 contract returns axis-aligned bounding boxes and therefore uses
axis-aligned plate crops. Perspective rectification is deliberately not
claimed or enabled for this YOLOX path. It requires an explicit OBB or
four-corner output contract plus validation on the actual export and camera
set; adding it by guessing corners would make OCR results less attributable.

Only for an explicit comparison after installation, select **YOLOX اختصاصی**
in AI settings. The settings endpoint refuses the switch while the model is
missing or invalid, leaving the production YOLO11n selection unchanged.

## Validation before release

Unit tests prove routing and integrity properties, but they do not establish
field accuracy. Before enabling the current YOLO11n route on production
cameras, run its pinned ONNX against representative day/night and angled-plate
video and record:

- exact full-plate accuracy per vehicle pass;
- confirmed precision and false-confirm rate;
- duplicate-event rate;
- p50/p95 end-to-end latency;
- CPU, RAM and dropped-frame ratio on the target i5 host.

The current release gate is a same-input comparison of YOLO11n + Hezar v2
against the currently deployed engine. If custom YOLOX is evaluated later, it
must pass that same independent comparison; synthetic decoder tests alone are
not accuracy evidence.
