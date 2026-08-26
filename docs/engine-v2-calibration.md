# Engine V2 TCAM calibration

This calibration path is evaluation-only. It never enables Engine V2, changes
the production ANPR route, or writes a runtime configuration automatically.

## Evidence layers

Calibration deliberately separates two evidence sources:

1. **IR-LPR static evidence** measures OCR exact-match accuracy, character
   error rate, confidence reliability, beam alternatives, crop quality, and
   useful plate-size ranges. The official dataset contains 20,967 annotated
   car images and 27,745 labelled plate images with separate train,
   validation, and test downloads.
2. **BC Vision temporal camera evidence** measures tracking continuity,
   independent-frame timing, audit behavior, finalization delay, false events,
   and OCR calls per track. It must contain real multi-frame day and night
   tracks plus exhaustively reviewed negative tracks.

IR-LPR alone cannot validate tracking or temporal voting because its records
are independent images. It also has no authoritative day/night field. An
operator may assign a profile only after selecting a verified subset; the
collector records that assignment and keeps the final gate closed until real
temporal and negative evidence is merged.

Primary references:

- Dataset repository: <https://github.com/mut-deep/IR-LPR>
- Paper: <https://arxiv.org/abs/2209.04680>

## Collect an IR-LPR OCR trace

Download the official `License Plate` train, validation, and test sets and keep
their original XML/JPG pairs. The reader accepts the original Pascal-VOC style
objects, reconstructs the eight-character label from character boxes, maps the
official validation/test rows to holdout, hashes every image and XML file, and
rejects malformed or unsupported labels by default.

```bash
python tools/calibrate_engine_v2_tcam.py collect-ir-lpr \
  --dataset-root /data/IR-LPR/license-plate \
  --ocr-runtime hezar \
  --ocr-model /models/crnn_fa_v2.onnx \
  --backend onnxruntime \
  --device CPU \
  --profile day \
  --output /data/calibration/ir-lpr-day.json \
  --static-report-output /data/calibration/ir-lpr-day-report.json
```

`hezar` is the default because Hezar v2 is the current production-first OCR
reader. The collector imports its exact mirrored `32x384`, blank-index-zero
contract from `app.ai.onnx_hezar` and fails closed unless the supplied ONNX
file matches the size and SHA-256 pinned by the production model manager.

Run a newer fixed-slot CCT only as a separate Shadow comparison. Its signed
manifest selects the exact model revision and preprocessing profile, including
the dual-view `stretch-letterbox-geomean-v1` profile when present:

```bash
python tools/calibrate_engine_v2_tcam.py collect-ir-lpr \
  --dataset-root /data/IR-LPR/license-plate \
  --ocr-runtime cct \
  --ocr-model /models/cct-candidate.onnx \
  --ocr-manifest /models/active-models.json \
  --backend onnxruntime \
  --device CPU \
  --profile day \
  --output /data/calibration/ir-lpr-cct-shadow.json \
  --static-report-output /data/calibration/ir-lpr-cct-shadow-report.json
```

Do not merge traces from different OCR revisions into one threshold search.
Compare their static reports, then collect real temporal day/night traces with
the selected revision. The generic `ctc` runtime remains available only for
an explicitly described legacy/custom CTC graph; it is not the Hezar v2
production contract.

This command uses ground-truth plate crops. It does not evaluate the detector,
and the report records that limitation explicitly. The static report includes
exact-match accuracy, CER, Brier score, expected calibration error, ten
confidence bins, and coverage/error sweeps at thresholds from 0.50 through
0.999. These values diagnose whether an OCR confidence such as `0.75` is
trustworthy; they are not sufficient to make `0.75` an event threshold.

The loader searches nested official XML directories and sibling image
directories, maps `train` to calibration train and `validation`/`test` to
holdout, derives dimensions from the paired image when the official XML omits
`size`, rejects ambiguous split paths, and fails on unknown annotation labels.
Use `--skip-invalid` only for an exploratory report; every skipped annotation
is retained in provenance.

An existing static trace can be analyzed separately:

```bash
python tools/calibrate_engine_v2_tcam.py analyze-static \
  --dataset /data/calibration/ir-lpr-day.json \
  --output /data/calibration/ir-lpr-day-report.json
```

## Merge and search

Static and temporal traces use the same closed schema. Merge them only after
the camera trace has exhaustive positive and negative labels:

```bash
python tools/calibrate_engine_v2_tcam.py merge \
  --input /data/calibration/ir-lpr-day.json \
  --input /data/calibration/camera-day-night.json \
  --dataset-id bcvision-tcam-calibration-v1 \
  --output /data/calibration/combined.json

python tools/calibrate_engine_v2_tcam.py search \
  --dataset /data/calibration/combined.json \
  --grid tests/fixtures/engine_v2_tcam_calibration_grid.json \
  --output /data/calibration/report.json
```

The complete synthetic schema example at
`tests/fixtures/engine_v2_tcam_calibration_trace.json` is only a contract and
CLI smoke-test; it is explicitly not calibration evidence.

Candidate policies are selected using only `train`. The chosen policy is then
evaluated once on `holdout`. Day and night profiles are calibrated separately.
The default gate requires 99% exact accuracy and recall, 99.5% precision, at
most 0.1% false accepts/wrong events, at most 1% mean CER, and at least 50 train
plus 50 holdout tracks in each profile. Missing night, negatives, or holdout
data produces a report with `valid=false`; it never silently relaxes a gate.

The optimizer ranks accurate candidates first, then lower CER, fewer OCR calls
per track, and lower finalization latency. It reports the selected
`TemporalFusionConfig` for each profile but does not apply it.

Neither the IR-LPR images nor annotations are copied into this repository. Keep
the downloaded dataset external and review its upstream license/terms before
redistribution.

## Runtime profile selection

An evaluated profile can be supplied explicitly:

```python
config = EngineV2Config(
    track_temporal_fusion_enabled=True,
    temporal_fusion_profiles={
        "day": calibrated_day,
        "night": calibrated_night,
    },
    default_temporal_fusion_profile="day",
)
```

The camera producer sets `FramePacket.metadata["illumination_profile"]` to
`day` or `night`. The selected name is fixed for that tracker episode and is
written to event metadata as `calibration_profile`.
