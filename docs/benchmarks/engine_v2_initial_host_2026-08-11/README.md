# Engine V2 initial host smoke benchmark — 2026-08-11

This is a paced, reproducible **non-production smoke benchmark** of the real
shared Engine V2 ONNX detector/OCR sessions with deterministic synthetic pixel
frames. It does not contain RTSP/video decode or accuracy evidence and cannot
justify replacing V1.

## Host and model path

- Linux 6.18.35, 9 logical Intel Xeon Platinum 8370C CPUs, 21 GiB RAM;
- ONNX Runtime 1.28.0, `CPUExecutionProvider` selected for both shared models;
- OpenVINO was not installed and no `/dev/dri` device was exposed;
- exactly two model sessions for the service and zero sessions per camera;
- detector input 320×320 from a 640×360 synthetic detector stream;
- main/crop stream 1280×720;
- five detector and two OCR warm-up calls before measurement;
- each scenario was paced for 3 seconds at 5 ticks/s with producer burst 2.

Model identities, provider fallback logs, and full configuration are in
`runtime_metadata.json`. Raw rows are in `engine_v2_performance.json` and CSV.

## Fixed-active idle-camera matrix

One camera is configured active in every row. `Idle gate CPU` is the normalized
host CPU measured specifically inside idle callbacks; idle cameras performed no
detector or OCR inference.

| Cameras | Active / idle | Total CPU % | RAM MiB | Detector calls | Idle detector / OCR | Idle gate CPU % |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 / 0 | 13.81 | 189.62 | 9 | 0 / 0 | 0.000 |
| 4 | 1 / 3 | 24.07 | 192.24 | 15 | 0 / 0 | 0.204 |
| 8 | 1 / 7 | 22.63 | 195.86 | 15 | 0 / 0 | 0.603 |
| 16 | 1 / 15 | 27.69 | 202.88 | 15 | 0 / 0 | 1.206 |
| 32 | 1 / 31 | 9.92 | 216.89 | 4 | 0 / 0 | 2.388 |

The important architectural result is zero incremental detector/OCR work from
idle cameras. Their motion-gate state is not free: 31 idle cameras added about
27.27 MiB lifetime peak RSS and 2.39% normalized host CPU in the measured idle
callbacks. Total CPU and total detector counts are not monotonic because the
resource controller changed active-camera cadence during this single short run;
they must not be treated as a clean idle-cost regression curve.

## All-active stress matrix

| Cameras | CPU % | RAM MiB | Detector/s | Queue max | Stale drops | Avg / P95 latency ms | Jain fairness |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 29.16 | 216.89 | 5.00 | 1 | 15 | 32.81 / 92.32 | 1.000 |
| 4 | 19.48 | 216.89 | 4.00 | 4 | 88 | 24.69 / 108.64 | 1.000 |
| 8 | 20.76 | 216.89 | 5.66 | 8 | 176 | 17.02 / 94.06 | 0.992 |
| 16 | 43.06 | 216.89 | 10.67 | 16 | 357 | 36.95 / 208.93 | 0.986 |
| 32 | 62.12 | 216.89 | 22.06 | 32 | 731 | 40.28 / 199.75 | 0.801 |

The 32-camera row reached every configured active camera, but its fairness and
drop counts show pressure. This is evidence that adaptive admission and stale
replacement are operating, not a claim that this host supports 32 production
cameras. Latency covers processed survivors only.

## Shared model microbenchmark

The separate 100-call synthetic model microbenchmark measured preprocessing,
shared inference, and postprocessing:

| Model | Inference/s | Average ms | P95 ms | Normalized host CPU % | Max RSS MiB |
|---|---:|---:|---:|---:|---:|
| YOLO plate detector | 31.41 | 31.84 | 49.81 | 78.97 | 167.92 |
| CTC plate OCR | 77.08 | 12.97 | 35.36 | 62.59 | 167.92 |

The end-to-end matrix reports zero OCR/s and zero events because the synthetic
pixels deliberately contain no detectable plate. The microbenchmark exercises
OCR capacity but says nothing about recognition accuracy.

## Evidence limits

- `production_evidence=false` and `production_decision_allowed=false`;
- decode utilization is unavailable and stream lifecycle is not instrumented;
- RSS uses process lifetime `maxrss`, so later scenario RAM rows are not
  independent peaks;
- this is one short ordered run, not repeated median/CI capacity evidence;
- there is no real main/sub RTSP timing, Intel iGPU/QSV utilization, or network
  reconnect cost;
- no verified eight-category media set was available, so V1/V2 accuracy was not
  executed and no production promotion decision is possible.
