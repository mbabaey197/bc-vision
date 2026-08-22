# BC Vision fixed Windows benchmark workstation

This workstation exists to make ANPR performance changes measurable and
repeatable.  It is deliberately separate from developer laptops and production
installations.  The same machine, power settings, benchmark video, model files,
Python environment and workflow inputs must be used when comparing two builds.

The repository already owns the production 1/3/6-camera replay runner in
`app.ai.capacity_baseline`.  The dedicated workflow
`.github/workflows/anpr-capacity-benchmark.yml` runs that real production path
on a self-hosted Windows machine and optionally compares the result against an
accepted fixed-host baseline with `tools/compare_capacity_baseline.py`.

## Required runner identity

Register a GitHub Actions self-hosted runner for this repository and give it
these labels:

```text
self-hosted
windows
x64
bcvision-benchmark
```

The custom `bcvision-benchmark` label is intentional.  The benchmark workflow
must never fall back to a GitHub-hosted VM because CPU measurements from a
changing host are not comparable evidence.

Use one dedicated physical machine or a permanently pinned VM.  Do not run
unrelated CPU-heavy work during a measurement.  Keep the Windows power plan,
logical CPU topology and virtualization settings fixed.  A Windows patch may
change the platform string and produces a warning; changing CPU topology makes
the comparison fail closed.

## Python benchmark environment

Create one stable Python 3.13 virtual environment outside the checkout so every
run uses the same interpreter location.  For example:

```powershell
py -3.13 -m venv C:\BCVisionBenchmark\.venv
C:\BCVisionBenchmark\.venv\Scripts\python.exe -m pip install --upgrade pip
C:\BCVisionBenchmark\.venv\Scripts\python.exe -m pip install -r requirements-ai-lock.txt
C:\BCVisionBenchmark\.venv\Scripts\python.exe -m pip install -r requirements-test.txt
```

After dependency-lock changes, intentionally refresh this environment before
accepting a new baseline.  Do not silently update packages between baseline and
candidate measurements.

## Local immutable inputs

The benchmark video and labelled passage evidence stay on the controlled
workstation; they are not committed to the public repository.  Use one fixed
video file and preserve its bytes.  The capacity report records its SHA-256 and
the comparator refuses to compare a different file.

Configure these machine-level environment variables, then restart the Actions
runner service so it inherits them:

```powershell
[Environment]::SetEnvironmentVariable(
  "BCVISION_BENCH_PYTHON",
  "C:\BCVisionBenchmark\.venv\Scripts\python.exe",
  "Machine"
)
[Environment]::SetEnvironmentVariable(
  "BCVISION_BENCH_VIDEO",
  "C:\BCVisionBenchmark\evidence\capacity.mp4",
  "Machine"
)
[Environment]::SetEnvironmentVariable(
  "BCVISION_BENCH_MODEL_ROOT",
  "C:\ProgramData\BCVision\data\models",
  "Machine"
)
```

Optional independent passage evidence can be configured as:

```powershell
[Environment]::SetEnvironmentVariable(
  "BCVISION_BENCH_PASSAGE_EVIDENCE",
  "C:\BCVisionBenchmark\evidence\passages.json",
  "Machine"
)
```

The passage evidence must satisfy the existing fail-closed production passage
contract.  Capacity replay event counts are diagnostics and never substitute
for independently labelled readable, miss, wrong-read, false-accept,
duplicate, day/night, distance, angle and image-quality evidence.

## Production model files

By default the workflow reads the standard installed model tree:

```text
C:\ProgramData\BCVision\data\models\plate\plate_yolo11n.onnx
C:\ProgramData\BCVision\data\models\plate\plate_yolov8n.onnx
C:\ProgramData\BCVision\data\models\plate\plate_yolo_fallback.onnx
C:\ProgramData\BCVision\data\models\hezar\crnn_fa_v2.onnx
C:\ProgramData\BCVision\data\models\crnn\ocr_crnn.onnx
C:\ProgramData\BCVision\data\models\cnn\ocr_cnn.onnx
```

`app.ai.model_manager` still verifies the expected model identities.  The
workflow fails before measurement if any required file is missing.

## Establish the first accepted baseline

1. Use a known-good released build and the matching repository revision.
2. Dispatch **ANPR dedicated capacity benchmark** with `mode=measure`.
3. Run with `viewers_per_camera=0` for the headless production cost baseline.
4. Inspect the JSON artifact and the workflow summary.  All 1/3/6-camera runs
   must be valid, model-ready and free of persistence backpressure.
5. Copy the accepted JSON to a stable local path such as:

```text
C:\BCVisionBenchmark\baselines\accepted-headless.json
```

6. Configure that exact path on the machine:

```powershell
[Environment]::SetEnvironmentVariable(
  "BCVISION_BENCH_BASELINE",
  "C:\BCVisionBenchmark\baselines\accepted-headless.json",
  "Machine"
)
```

7. Restart the self-hosted runner service.

Do not automatically overwrite the accepted baseline.  Replacing it is a
review decision because silently moving a baseline can hide a regression.

## Compare a candidate

Dispatch the same workflow with `mode=compare` and exactly the same detector,
FPS, stream width, JPEG quality and viewer mode as the accepted baseline.  The
comparison fails closed when:

- the source video SHA-256 differs;
- the fixed host CPU topology or architecture differs;
- either report is not valid/comparable;
- host CPU regresses beyond the configured relative threshold;
- decode or inference latency regresses beyond the configured threshold;
- per-camera inference FPS drops beyond the configured threshold;
- application frame coalescing increases beyond the configured absolute rate;
- emitted and persisted event counts diverge;
- a headless run performs any preview JPEG encoding;
- an existing independently verified passage-accuracy claim disappears or its
  exact-accuracy confidence lower bound falls beyond the allowed budget.

The defaults are deliberately conservative rather than marketing thresholds:
10% relative CPU/latency/FPS tolerance, 0.05 absolute coalescing-rate increase,
and 0.005 absolute accuracy-confidence-lower-bound drop.  Tighten these after
the fixed workstation has enough repeated-run variance data.

## Active-preview benchmark

A second accepted baseline may be kept for `viewers_per_camera=1`.  Never
compare a viewer-active result against a headless baseline; the comparator
requires matching settings.  This makes the cost of preview JPEG encoding
visible instead of mixing it into ANPR inference cost.

## Remote control

A remote-control integration may be attached to this dedicated workstation so
an authorized operator can inspect files, run commands and manage benchmark
processes without using a personal laptop.  Keep the GitHub self-hosted runner
as the reproducible execution path and the repository workflow as the source
of truth for measurements.
