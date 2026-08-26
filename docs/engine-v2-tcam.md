# Engine V2 track-centric temporal ANPR

This slice adds an opt-in recognition policy to the independent Engine V2
runtime. It keeps OCR evidence on the tracker episode instead of treating each
crop as an isolated final answer.

## Runtime flow

1. The detector creates or updates a persistent Track ID.
2. The first crop above the quality floor is sent to OCR immediately.
3. Full-string votes and per-slot character evidence are accumulated together.
4. A reading at the provisional threshold is retained but is not emitted.
5. The track locks only through either:
   - one strict, high-confidence, high-quality express read; or
   - independent multi-frame agreement with slot confidence, margin, and
     support checks.
6. A locked track emits at most one event. Expensive OCR stops, while the cheap
   tracker continues to own the vehicle until its normal exit/removal.

Correlated adjacent frames receive reduced weight and do not count as
independent confirmation. A re-read is scheduled only for a concrete reason:
first usable crop, unresolved/conflicting slots, material quality improvement,
plate-area growth, periodic refresh, provisional confirmation, or final
pre-exit evidence.

## Enable for an A/B run

```python
from app.engine_v2 import EngineV2Config, TemporalFusionConfig

config = EngineV2Config(
    track_temporal_fusion_enabled=True,
    temporal_fusion=TemporalFusionConfig(
        provisional_confidence=0.75,
        lock_confidence=0.86,
        express_lock_confidence=0.93,
        max_ocr_attempts=4,
    ),
)
```

All thresholds are policy inputs and must be calibrated on held-out traffic
footage. In particular, `0.75` is a provisional threshold, not an event
acceptance threshold.

## Event and telemetry contract

Fused events add the following metadata:

- `recognition_phase`
- `fusion_reason`
- `independent_observations`
- `full_sequence_support`
- `slot_confidences`
- `slot_margins`
- `ocr_schedule_reason`
- `ocr_attempts`

Runtime telemetry also reports fusion track counts, provisional/locked counts,
and total fusion OCR attempts. These fields support comparison of recall, CER,
OCR calls per track, latency, and CPU before the feature flag is widened.
