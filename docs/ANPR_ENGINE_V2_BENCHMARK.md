# ANPR Engine V2 benchmark harness

This harness records evidence; it never enables V2 or replaces the production
engine. Synthetic results validate scheduling and report formats only. They are
written with `production_evidence=false` and cannot support a release decision.

## Scheduling/resource smoke run

```bash
python tools/benchmark_engine_v2.py performance \
  --output-dir benchmark-results/engine-v2 \
  --paced \
  --include-32
```

The default `standard` matrix runs two independent 1, 4, 8 and 16 camera sweeps
(optionally 32): a fixed-active sweep that isolates incremental idle-camera
cost, and an all-active sweep that measures busy-site scaling. Use
`--matrix fixed-active` or `--matrix all-active` to run only one side.
Without `--paced`, nominal time controls workload size only so CI remains fast;
that mode cannot be classified as production evidence.

Synthetic mode does not load a model or decoder. For a real in-process run,
provide a long-lived callable so model sessions remain shared:

```bash
python tools/benchmark_engine_v2.py performance \
  --adapter-callable my_benchmark_adapter:create_or_process \
  --adapter-name engine-v2-real \
  --output-dir benchmark-results/engine-v2-real
```

The symbol may expose `process(job)` or itself be callable. It returns a mapping
with these counters for that scheduled job:

```json
{
  "detector_inferences": 1,
  "ocr_inferences": 0,
  "plate_events": 0,
  "decode_utilization_percent": 37.2,
  "decode_utilization_kind": "measured",
  "decode_utilization_source": "intel_gpu_busy_counter"
}
```

Decode utilization must be `null`/`unavailable`, or carry both a `measured` or
`estimated` kind and a precise source. A callable can implement
`configure_scenario`, `prepare_scenario`, `start_scenario`, `observe_idle`,
`stop_scenario`, and `close`. Without configure/start/stop hooks, the report
states that RTSP/decode lifecycle cost was not measured. Model preparation is
outside timed inference but its resident memory is sampled.

Callable results are fail-closed: they are not production evidence by default.
An adapter requesting that label must supply `evidence_metadata` with verified
input/model file SHA-256 values, the execution provider, `resource_scope` set to
`current-process`, and `uses_child_processes=false`. The harness re-hashes those
files; self-asserted counters alone never become production evidence.
Per-scenario validation additionally requires real-time pacing, complete
configure/start/stop stream lifecycle hooks, and numeric `measured` decode
utilization with a precise source. Estimated or unavailable decode cannot pass.

Configured Active/Idle counts are separate from optional measured runtime
mean/max counts. If `resource.getrusage.maxrss` is the only RAM source, it is a
process-lifetime peak rather than an independent peak for each scenario; use
process isolation or a trustworthy current-RSS sampler for capacity evidence.

Command performance adapters are supported for contract tests, but are marked
non-production evidence because parent-process CPU/RAM counters do not include
the spawned child process.

## V1/V2 accuracy comparison

Copy `tests/fixtures/engine_v2_accuracy_manifest.template.json`, replace all
placeholder paths, add operator-verified labels and explicit provenance, set
`training_allowed=false`, set every `label_status` to `verified`, enable the
rows, keep every sample ID opaque (never embed a plate or label hint), and remove
`template=true`. Then materialize a new content-addressed,
tamper-evident dataset instead of benchmarking mutable source paths:

```bash
python tools/prepare_engine_v2_accuracy_dataset.py \
  --draft-manifest /review/anpr-accuracy-draft.json \
  --output /data/bcvision/anpr-accuracy-v1
```

The preparer refuses to overwrite an existing output, rejects symlinks and
unverified/invalid labels, copies every media file under its SHA-256 name,
records its exact byte size, writes a deterministic dataset fingerprint, and
marks both dataset and samples as training-forbidden. The complete manifest
schema is `docs/ANPR_ENGINE_V2_BENCHMARK_MANIFEST.schema.json`. The closed schema
rejects unknown sample, input and event fields; per-sample `adapter_input` is not
part of the evidence format.

The runner rejects templates, missing files, unverified labels and duplicate
IDs. By default it separately requires readable labels in all eight categories,
a verified negative sample, and at least two `expected_events` in one
`multiple_vehicles` sample. Those are coverage requirements, not the definition
of strict evidence. Strict evidence is the byte/provenance contract: explicit
inference-only policy and label provenance plus a local relative `input.path`,
`media_type`, `input.sha256`, and `input.size_bytes` for every sample. The runner
re-hashes the manifest and every input immediately before V1, between V1 and V2,
and immediately after V2. It aborts on any identity change and computes
`same_input_bytes_for_both_engines` from those checks rather than accepting an
adapter assertion. `--allow-missing-input-files` disables strict evidence for
contract tests and makes the accuracy promotion gate non-evaluable even if a
local file happens to be reachable at run time.

```bash
python tools/benchmark_engine_v2.py compare-accuracy \
  --manifest /data/bcvision/verified-manifest.json \
  --v1-callable adapters.legacy:predict \
  --v2-callable adapters.engine_v2:predict \
  --output-dir benchmark-results/v1-v2
```

The repository also includes concrete offline adapters, so a comparison does
not require a project-specific adapter module:

```bash
python tools/benchmark_engine_v2.py compare-accuracy \
  --manifest /data/bcvision/verified-manifest.json \
  --v1-builtin legacy-video \
  --v1-detector-variant yolo11n \
  --v2-builtin engine-v2-offline \
  --v2-detector-model /models/plate-detector.onnx \
  --v2-ocr-model /models/iran-plate-ocr.onnx \
  --v2-backend auto \
  --v2-device AUTO \
  --v2-detector-frame-size 640x360 \
  --output-dir benchmark-results/v1-v2
```

The V1 adapter lazily invokes the unchanged
`app.ai.video_test.process_video` with explicit settings and isolated temporary
output directories. Its operator-review-only `capture_only`/unreadable rows are
excluded because they are not accepted plate events. The V2 adapter creates one
shared detector session and one
shared OCR session for the whole manifest, resets camera/episode state between
samples, derives its 640x360 detector frame from each decoded main frame, and
uses the V2 EOF finalizer to process only already-harvested candidates. Video
decode requests FFmpeg software mode with one decoder thread; timestamps use
OpenCV PTS (`CAP_PROP_POS_MSEC`) with a recorded frame/FPS fallback. Images use
`imdecode` and timestamp zero.

The JSON report includes `adapter_reproducibility` with model SHA-256 values,
selected provider/device metadata, the complete V2/model configuration, decode
settings and per-sample timestamp-source counts. Capture-backend fallback is
disabled by default because silently changing decoders weakens reproducibility;
enable it explicitly with `--v2-allow-capture-backend-fallback` only when that
trade-off is acceptable.

The accuracy gate independently validates detector/OCR model SHA-256 identities
and runtime/provider identity for both adapters. It also normalizes and compares
effective-input options (`frame_step`, `max_frames`, and ROI). Missing metadata,
unproven symmetry, or any asymmetric option makes the gate non-evaluable.
Generic callable/command adapters must declare the same metadata contract; a
source hash or adapter name alone is insufficient.

Complete eight-category coverage and at least one verified negative sample
remain mandatory by default. Exploratory evidence can opt out explicitly with
`--allow-partial-coverage` and/or `--allow-no-negative`; those flags do not make
the result sufficient for a production decision. Every manifest containing
`label_scope=known_positives` requires `--allow-partial-coverage`, even if all
eight category names are present; add `--allow-no-negative` only when it also has
no verified exhaustive negative sample.

Built-in comparisons fail closed when effective input would differ. They reject
`--v2-max-frames`, require equal `--v1-frame-step` and `--v2-frame-step`, and
reject `--v1-roi` until V2 has an equivalent ROI contract. Both built-ins also
reject `input.start_ms`/`input.end_ms` instead of silently applying different
seek behavior. Create a pre-clipped, content-addressed media file and reference
that file from the manifest when a time window is required.

Each sample also has a `label_scope`. The default, `exhaustive`, asserts that
every event in the sample was reviewed and labelled. Use `known_positives` only
when the labels identify trusted positive events but are not a complete event
inventory. A known-positive sample must contain at least one readable positive
label and cannot be a negative sample. Its matched/missed counts, recall and
character error rate remain meaningful, but exact-set accuracy, event
precision, false-positive events and duplicate events are reported as JSON
`null` rather than fabricated zeroes. Any known-positive sample makes the
accuracy promotion gate non-evaluable and fail-closed.

Each callable receives one sample mapping and returns `null`, a plate string, or
a JSON-compatible mapping such as:

```json
{"plate": "12ب34567", "confidence": 0.91, "accepted": true}
```

For a multi-vehicle clip, return event objects. Labelled time windows prevent a
correct plate from matching the wrong passage:

```json
{
  "events": [
    {"plate": "12ب34567", "timestamp_ms": 840, "confidence": 0.91},
    {"plate": "34د76543", "timestamp_ms": 1920, "confidence": 0.88}
  ]
}
```

Ground-truth fields (`expected_plate`, `expected_events`, `label_status`, label
scope/provenance, category and notes) never cross the adapter/command boundary.
Adapters receive an opaque request identity plus an allowlisted inference input
that resolves to the prepared content-addressed media; arbitrary per-sample
configuration is not admitted. Opaque manifest IDs and hash-named media paths
provide defense in depth against labels leaking through names.

This filtering is an evidence API contract, not a security sandbox. Callable
and command adapters run as trusted local code and may otherwise have filesystem
access, so unreviewed or adversarial adapters cannot produce valid evidence.
Scoring records exact event-set accuracy, event precision/recall, missed events,
false-positive events and duplicate events.

The first real-media partial run is recorded in
`docs/benchmarks/engine_v2_01mp4_partial_2026-08-11/README.md`. It proves the
same-input integrity and concrete-adapter path on one known-positive clip, not
eight-category accuracy, false-positive behavior, capacity, or production
readiness.

Commands can be used instead of callables with `--v1-command` and
`--v2-command`. Each command receives one JSON request on stdin and must write
one JSON prediction to stdout. Its process startup time is included in sample
latency.

Outputs are a detailed JSON report plus flat CSV scenario/prediction rows. An
accuracy comparison alone never authorizes replacement: real resource results
and acceptable accuracy are both required outside this harness.
