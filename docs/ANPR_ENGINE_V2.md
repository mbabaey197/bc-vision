# BC Vision ANPR Engine V2

Engine V2 is an independent, clean-room ANPR path. It does not import, refactor,
configure, or replace the legacy production engine. V2 remains evaluation-only
until the same verified inputs prove that it is both accurate enough and more
resource-efficient than V1.

## Architecture contract

```text
RTSP main + sub stream producers
        -> hardware/software decode selection
        -> low-cost motion/ROI gate
        -> central latest-only priority scheduler
        -> one shared detector session
        -> lightweight multi-object tracking
        -> detector-to-main coordinate mapping
        -> high-resolution plate crops
        -> quality-ranked candidate collection
        -> one shared OCR session/worker
        -> temporal confidence voting
        -> Iranian plate validation
        -> exact duplicate suppression
        -> PlateEvent callback/database adapter
```

Camera readers are producers only. A camera never constructs or owns a detector
or OCR session. `SharedModelBundle` owns exactly two model sessions for the
service: one detector and one OCR session, independent of camera count.

## Implemented components

### Shared inference and model adapters

- `SharedInferenceBackend` lazily selects a direct OpenVINO runtime or a ranked
  ONNX Runtime execution provider and exposes provider/device metadata.
- Direct OpenVINO ranks Intel GPU/iGPU before CPU when both are available. It
  falls back safely when compilation fails and records the reason.
- ONNX Runtime ranks OpenVINO, TensorRT/CUDA and other accelerator providers
  before CPU, with configurable provider options and fallback behavior.
- A single inference lock protects runtimes that do not guarantee concurrent use
  of one request/session. Queue-wait and inference latency are recorded.
- `YOLOPlateDetector` provides clean-room letterbox, YOLOv5/v8/v11 output
  decoding, class filtering, NMS, and mapping back to source coordinates.
- Ambiguous six-column single-class YOLO output defaults to raw decoding;
  end-to-end exports must be pinned explicitly, while heuristic guessing is an
  opt-in compatibility mode.
- `CTCPlateOCR` provides configurable resize/padding, normalization, tensor
  layouts, CTC blank/repeat collapse, and per-character confidence.
- `build_engine_v2()` creates the two shared sessions and wires their adapters to
  the independent V2 runtime. Failure during construction closes partial state.

OpenVINO and ONNX Runtime remain optional imports at module-load time. A clean
installation must include the intended runtime package before hardware evidence
can be collected; unit tests use injected fake runtimes and do not prove a real
device path.

### RTSP dual-stream producers

- `DualStreamRTSPProducer` reads main and detector/sub streams independently.
- Main frames are held in a one-slot replacement cache; sub-stream frames are
  paired only with a sufficiently close main frame.
- Main/sub skew is measured as `main_detector_skew_ms`; overly old or future
  main frames are rejected using the absolute skew limit.
- Decoder selection supports PyAV, FFmpeg, and OpenCV with hardware-to-software
  fallback after a verified first frame.
- Intel Quick Sync (`qsv`) is preferred when FFmpeg reports it and the device is
  usable. VAAPI, D3D11VA/DXVA2, CUDA, or VideoToolbox may be selected by OS.
- Reconnect uses bounded exponential backoff and redacts RTSP credentials from
  errors.
- A non-blocking admission controller continuously drains RTSP but forwards
  only the newest due sub-stream frames. Active policies interleave cheap
  tracking frames between detector frames; idle policies retain a motion-gate
  safety floor.
- Sequence numbers remain monotonic across stop/start. Every producer lifecycle
  also emits a unique `producer_epoch`, allowing the runtime to recover safely
  when a new reader instance starts its sequence at one.

The producer owns no detector, OCR, tracker, validator, or database logic.

### Scheduling and event-driven execution

- `LatestOnlyPriorityQueue` keeps at most one live job per camera/track key.
- Re-submission replaces stale work; a global generation prevents the ABA bug
  where an obsolete heap node could become live again after key reuse.
- Capacity pressure can evict lower-priority work. A fairness penalty prevents
  one hot camera from monopolizing consecutive scheduling turns.
- Queue age is bounded. Expired frames are counted and never inferred.
- Idle cameras run only the motion/ROI gate at an adaptive stride. They submit
  no detector or OCR work until motion crosses the configured threshold.

### Per-vehicle episode state

Each camera owns a cheap tracker, while each tracked plate/vehicle owns an
independent `TrackEpisode`:

```text
IDLE -> ACTIVE -> TRACKING -> PLATE_FOUND -> COLLECTING
     -> OCR -> VALIDATED -> DONE
```

- The tracker associates multiple simultaneous detections without neural state
  and predicts boxes between selected detector frames.
- DONE is held for the same track/episode; it cannot trigger repeated OCR.
- Track removal forces one final OCR attempt from the best collected candidates.
  An invalid terminal attempt closes the episode instead of leaving the camera
  permanently stuck in COLLECTING.
- A new producer epoch resets stale sequence/tracker state. If the previous
  stream was active, one detector job is scheduled immediately so a restart does
  not create a long blind window.

### Candidate quality and OCR

`BestPlateFrameSelector` retains a bounded, temporally diverse candidate set.
Its score combines:

- Laplacian sharpness;
- centered exposure and clipped-pixel penalty;
- contrast;
- plate area relative to the main frame;
- detector confidence;
- directional edge/motion-blur proxy.

Normally the best two or three candidates are sent to the shared OCR worker.
Expired or priority-evicted OCR tasks are surfaced back to the runtime so their
episodes return to COLLECTING or terminate after exit; no track can remain stuck
in OCR with an empty queue.
The temporal voter:

- counts support only from distinct source sequences;
- rejects observations below confidence/quality floors;
- normalizes Persian/Arabic digits and character forms;
- requires structural Iranian-plate validation;
- accepts a single crop only at stricter confidence and quality thresholds.

The bundled CTC label contract emits `ا`; the validator also accepts the word
`الف` and canonicalizes both to `ا` before voting. Diplomatic `d/s` output is
canonicalized to uppercase. Bidi/zero-width controls are removed. The validator
is structural and deliberately does not guess ambiguous OCR characters.

Duplicate suppression is exact-match and bounded. It uses sliding same-camera
and short overlapping-camera windows, handles delayed timestamps without moving
last-seen state backwards, and never performs fuzzy suppression that could hide
a different vehicle.

## Adaptive load shedding

`AdaptiveLoadController` observes:

- host CPU percentage;
- detector latency EMA;
- OCR latency;
- detector queue depth/capacity;
- recent stale-frame drop rate;
- active and total camera count.

Escalation is immediate and recovery is hysteretic. Policies progressively:

- increase detector and idle strides;
- use more tracker-predicted frames between detections;
- reduce active and idle target FPS;
- shorten permitted queue age;
- reduce OCR candidates only when quality permits.

Returning to NORMAL restores the configured cadence and up to three OCR
candidates. Camera count alone does not create AI pressure; active work,
latency, queueing, and CPU do.

`producer_cadence_policy(camera_id, source_fps)` is wired directly as the
producer's `cadence_provider`. The producer applies its detector/tracking
cadence without sleeping or buffering old frames, and tagged adaptive packets
bypass the runtime's fallback modulo stride so throttling is not applied twice.
If the policy callback fails or is unbound, admission fails open for accuracy.

```python
producer = DualStreamRTSPProducer(
    stream_config,
    engine.submit_frame,
    cadence_provider=engine.producer_cadence_policy,
)
```

## Telemetry

Runtime telemetry includes frames received, motion evaluations/wakeups, active
and idle cameras, detector/OCR counts and mean latency, queue depth,
replacements/expired stale frames, emitted events, duplicate suppressions, and
the current load policy.

RTSP producer telemetry separately records decoder backend, hardware
accelerator, connection attempts/reconnects, decoded frames, main-frame pairing
drops, sink rejections, and errors.

## Benchmark harness

The independent CLI never changes the production engine:

```bash
python tools/benchmark_engine_v2.py performance \
  --matrix standard \
  --paced \
  --include-32 \
  --output-dir artifacts/engine-v2
```

`--matrix standard` runs both:

1. fixed-active `1/4/8/16[/32]` for the incremental cost of idle cameras;
2. all-active `1/4/8/16[/32]` for busy-site scaling.

The report records CPU, process CPU, RAM and metric provenance, decode
utilization and provenance, detector/OCR inference rate, queue average/max,
latest replacements, expired/dropped frames, average/P95 latency, event rate,
and active/idle camera counts. Per-camera rows and idle/busy scaling deltas are
included.

The built-in synthetic adapter validates scheduling and report accounting only.
It is always marked non-production. A real callable adapter must run inside the
measured process and provide verifiable recording/model SHA-256 identities,
execution-provider metadata, decoder lifecycle instrumentation, and resource
scope before the report can classify it as production evidence. Even then, the
harness reports evidence and never switches engines.

`--paced` schedules producer ticks against the real monotonic clock. Production
evidence is rejected unless pacing, verified input/model hashes, in-process
resource scope, complete stream lifecycle hooks, and numeric measured decode
utilization with provenance are all present.

The initial 1/4/8/16/32 host smoke and shared-model microbenchmark are recorded
in `docs/benchmarks/engine_v2_initial_host_2026-08-11/README.md`. They are
explicitly synthetic/non-production: idle cameras performed zero detector/OCR
calls, while motion-gate CPU and per-camera state memory remained non-zero.

### Accuracy comparison

```bash
python tools/benchmark_engine_v2.py compare-accuracy \
  --manifest /path/to/verified-manifest.json \
  --v1-callable package.v1_adapter:predict \
  --v2-callable package.v2_adapter:predict \
  --output-dir artifacts/engine-v2-accuracy
```

The runner evaluates V1 and V2 independently against the same operator-verified
manifest and constructs an inference-only request for each adapter. Complete
promotion evidence must cover:

- clear plate;
- night;
- overexposure;
- motion blur;
- angled plate;
- multiple vehicles;
- fast vehicle;
- partial/dirty plate.

The default complete-coverage policy requires at least one verified negative
sample. Multiple-vehicle coverage includes an `expected_events` sample with at
least two events. These completeness rules are separate from strict evidence,
which is the inference-only provenance and byte-integrity contract: local
content-addressed media with required media type, SHA-256 and byte-size
identities. The runner re-hashes every input before V1, between engines, and
after V2.

Manifest sample IDs must be opaque and final media paths are SHA-256-named.
Adapters receive only an opaque request identity and allowlisted inference input;
category, labels, label scope/provenance, notes and arbitrary per-sample adapter
configuration stay scorer-side. This boundary prevents accidental leakage but
is not an operating-system sandbox, so evidence adapters must be trusted and
reviewed local code.

Any `known_positives` sample requires the explicit
`--allow-partial-coverage` opt-out and makes the accuracy promotion gate
non-evaluable. Built-in comparisons also reject frame truncation, unequal V1/V2
frame steps and a V1-only ROI so both engines consume the same effective input.
The gate additionally requires both adapters to prove detector/OCR SHA-256 and
runtime/provider identity and to declare symmetric effective-input options;
missing reproducibility metadata remains fail-closed.

Reports separate V1 and V2 predictions and measure exact event-set accuracy,
event recall/precision, false accepts, false-positive events, duplicate events,
character error rate, and latency overall and per category.

## Current evidence and remaining limitations

The deterministic suite proves contracts and failure handling, not real ANPR
accuracy or real camera capacity. Promotion is still blocked by:

- no repository recording set covering all required accuracy categories;
- only one same-input real-media comparison, with three known-positive labels
  but no exhaustive event inventory, negative sample, or seven other required
  categories; therefore its precision, false-positive, duplicate and exact-set
  metrics are deliberately unavailable and its promotion gate is fail-closed;
- no real RTSP/hardware-decode benchmark on the target Intel systems;
- no long-duration day/night soak test;
- no repeated/process-isolated benchmark with statistical confidence bounds;
- latest-arrival main/sub pairing rather than a nearest-PTS frame buffer;
- continuous main-stream decode cost, which must be measured on real hardware;
- axis-aligned boxes and scale mapping; strongly angled/letterboxed/cropped
  installations may require explicit transform calibration or quadrilateral
  rectification;
- static provider priority rather than a startup micro-benchmark across every
  installed accelerator;
- feature-flag/dashboard/database integration is intentionally not enabled.

## Production compatibility rule

Do not import V2 into the legacy camera worker and do not change the production
engine selector until all of the following are true:

1. V1 and V2 have run on the same immutable, verified media set.
2. V2 has no unacceptable overall or per-category accuracy regression.
3. Real fixed-active and all-active camera matrices demonstrate better resource
   behavior and bounded P95 latency on target hardware.
4. Idle-camera additions do not create linear detector/OCR work.
5. Day/night real-camera soak tests pass with bounded queues and duplicate rate.
6. Activation remains reversible behind an explicit feature flag.

Until then, V1 remains the production engine and V2 remains a separate Draft PR
evaluation path.
