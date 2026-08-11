# ANPR Engine V2 benchmark harness

This harness records evidence; it never enables V2 or replaces the production
engine. Synthetic results validate scheduling and report formats only. They are
written with `production_evidence=false` and cannot support a release decision.

## Scheduling/resource smoke run

```bash
python tools/benchmark_engine_v2.py performance \
  --output-dir benchmark-results/engine-v2 \
  --include-32
```

The default `standard` matrix runs two independent 1, 4, 8 and 16 camera sweeps
(optionally 32): a fixed-active sweep that isolates incremental idle-camera
cost, and an all-active sweep that measures busy-site scaling. Use
`--matrix fixed-active` or `--matrix all-active` to run only one side.

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

Command performance adapters are supported for contract tests, but are marked
non-production evidence because parent-process CPU/RAM counters do not include
the spawned child process.

## V1/V2 accuracy comparison

Copy `tests/fixtures/engine_v2_accuracy_manifest.template.json`, replace all
placeholder paths, add operator-verified labels, set every `label_status` to
`verified`, enable the rows, and remove `template=true`. The complete schema is
`docs/ANPR_ENGINE_V2_BENCHMARK_MANIFEST.schema.json`.

The runner rejects templates, missing files, unverified labels, duplicate IDs,
all-null datasets, and manifests without readable labels in all eight required
categories. It also requires a verified negative sample and at least two
`expected_events` in one `multiple_vehicles` sample. Add `input.sha256` whenever
the media is stable so its bytes can be verified. The exact same loaded manifest
is scored separately for V1 and V2.

```bash
python tools/benchmark_engine_v2.py compare-accuracy \
  --manifest /data/bcvision/verified-manifest.json \
  --v1-callable adapters.legacy:predict \
  --v2-callable adapters.engine_v2:predict \
  --output-dir benchmark-results/v1-v2
```

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

Ground-truth fields (`expected_plate`, `expected_events`, `label_status`, and
notes) never cross the adapter/command boundary. Adapters receive only sample
identity, category, inference input and explicit `adapter_input` configuration.
Scoring records exact event-set accuracy, event precision/recall, missed events,
false-positive events and duplicate events.

Commands can be used instead of callables with `--v1-command` and
`--v2-command`. Each command receives one JSON request on stdin and must write
one JSON prediction to stdout. Its process startup time is included in sample
latency.

Outputs are a detailed JSON report plus flat CSV scenario/prediction rows. An
accuracy comparison alone never authorizes replacement: real resource results
and acceptable accuracy are both required outside this harness.
