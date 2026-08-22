# BC Vision CPU and passage baseline

This is the mandatory evidence step before changing stream ingest, inference
cadence, preview JPEG generation or camera scheduling. It does not replace the
RC30 release gates and it cannot establish 99% accuracy from an unlabelled
video.

## Capacity matrix

Run the same immutable real video through the production stream and ANPR path
with exactly 1, 3 and 6 independent camera instances:

```powershell
python -m app.ai.capacity_baseline `
  --video C:\BCVision-Lab\gate-passages.mp4 `
  --output C:\BCVision-Lab\rc30-baseline.json `
  --camera-counts 1 3 6 `
  --detector yolo11n `
  --viewers-per-camera 0
```

The command refuses to overwrite an existing report. Each camera-count run
uses a fresh isolated data directory and the real selected detector, Hezar v2,
fixed Platrix fallback, activity gate, consensus, visit deduplication and
persistence path. `--download-models` is only for a connected development
machine; the RC30 offline installation already owns the verified models.

Repeat the matrix with `--viewers-per-camera 1` when measuring preview cost.
Use the same video SHA-256, host, power plan, detector, settings and viewer mode
when comparing two commits. Perform at least three complete repetitions and
compare medians; a single run is not a promotion decision.

The report contains:

- process CPU as 100%-per-core and whole-host percentages;
- decoded frames, decode time and aggregate/per-camera FPS;
- inference calls, time, aggregate/per-camera FPS and processed frames;
- JPEG attempts, time, FPS and bytes;
- decode shortfall plus stream-queue and live-worker frame coalescing;
- emitted and persisted event counts, errors and persistence backpressure;
- exact application/detector/OCR pipeline revision.

Application coalescing is intentional newest-frame sampling. It is reported as
frame-drop evidence because a coalesced frame never reaches inference, but it
must not be presented as RTSP packet loss. OpenCV does not expose portable
network packet-loss counters.

## Passage-level exact accuracy

Capacity event totals are diagnostics, not an accuracy denominator. An
accuracy evidence JSON may be attached with `--passage-evidence`. It must
contain independently labelled production passages in `passages` and a
non-empty `annotation_provenance` object. Every passage is scored by the RC30
fail-closed gate in `app.ai.pass_benchmark`.

The primary metric is exact plate accuracy per physical passage. The evidence
must retain misses, wrong reads, false accepts, duplicate rows and unreadable
passages, and cover day, night, speed, angle, blur, glare, multiple vehicles,
distance and image quality. The current claim gate also requires at least 400
readable passages, 800 negative passages, 100 unique plates, three cameras,
three sessions and adequate slice/provenance coverage. Its 95% Wilson lower
bound—not the point estimate—must reach 99%.

Without that independent labelled evidence, the generated report always keeps
`passage_accuracy.claim_ready` false. Synthetic plates, hand-selected crops,
the three historical known plates and capacity replay event counts must never
be used to claim 99%.

## Promotion rule

A performance change may move to a product PR only when all repeated matrices
are comparable and show a material CPU/capacity improvement without reducing
decoded or inference coverage, increasing frame coalescing unexpectedly,
changing event counts on the same input, or regressing passage-level results.
PR #70 remains separate until it has this before/after evidence with both zero
and one viewer per camera.
