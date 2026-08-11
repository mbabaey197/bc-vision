# BC Vision ANPR Engine V2

Engine V2 is a clean-room runtime path that does not reuse the camera-centric execution model of the existing ANPR engine.

## Design goals

- Scale camera count without loading detector/OCR sessions per camera.
- Keep expensive AI asleep while a fixed lane is idle.
- Discard stale video work instead of accumulating latency.
- Detect on a low-resolution stream and crop/OCR from the main stream.
- Share detector and OCR instances across all cameras.
- Treat one passing vehicle/plate episode as one job, not one job per frame.
- Make load shedding and future CPU/iGPU/NPU routing possible at one central scheduler.

## Data path

```text
RTSP readers (producers only)
        |
        +--> main stream (evidence/crop)
        |
        `--> detector stream (low resolution)
                    |
             AdaptiveMotionGate
                    |
          LatestOnlyPriorityQueue
                    |
          Shared Plate Detector
                    |
       coordinate map to main frame
                    |
          Plate Quality Selector
                    |
              Shared OCR
                    |
               Event sink
```

## What is implemented in the first slice

`app/engine_v2/` contains an independent core with:

- adaptive background/motion wake-up;
- bounded newest-frame-wins priority queue;
- per-camera state without per-camera model sessions;
- shared detector and OCR protocol boundaries;
- detector-substream to main-stream coordinate mapping;
- best-crop quality scoring using sharpness, exposure and contrast;
- DONE cooldown to suppress repeated OCR for the same passing episode;
- deterministic unit tests using fake detector/OCR implementations.

## What is intentionally not wired yet

The first slice is not enabled in production and does not change the old engine. Before activation it still needs:

1. a real ONNX/OpenVINO shared detector adapter;
2. a real Iranian OCR adapter and canonical validator;
3. RTSP dual-stream producer integration;
4. a real multi-object tracker/state machine for simultaneous vehicles;
5. temporal OCR voting across multiple best candidates;
6. adaptive load controller based on queue age, CPU load and inference latency;
7. benchmark harness against 1/4/8/16+ camera recordings;
8. feature flag and dashboard observability;
9. long-duration real-camera validation day/night.

## Compatibility rule

Engine V2 must remain behind a separate feature flag until it beats the existing production path on both accuracy and resource use. No V2 change may alter the legacy engine's runtime behavior, model files or database schema during the evaluation phase.
