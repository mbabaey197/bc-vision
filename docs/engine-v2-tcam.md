# Engine V2 track-centric temporal ANPR

This slice adds an opt-in recognition policy to the independent Engine V2
runtime. It keeps OCR evidence on the tracker episode instead of treating each
crop as an isolated final answer.

## Runtime flow

1. The detector creates or updates a persistent Track ID.
2. The first crop above the quality floor is sent to OCR immediately.
3. Full-string votes and per-slot character evidence are accumulated together.
4. A reading at the provisional threshold is retained but is not emitted.
5. The track enters a reversible soft lock only through either:
   - one strict, high-confidence, high-quality express read; or
   - independent multi-frame agreement with slot confidence, margin, and
     support checks.
6. Routine OCR stops during the soft lock. At most one audit OCR is allowed
   when the crop materially improves or the track is about to leave.
7. The track is finalized after the audit, a short real-time hold, or track
   exit. Only this finalized state can emit, and it emits at most one event.

Correlated observations are separated by timestamps rather than assumed frame
rate; adjacent frames receive reduced weight and do not count as independent
confirmation. A re-read is scheduled only for a concrete reason:
first usable crop, unresolved/conflicting slots, material quality improvement,
plate-area growth, periodic refresh, provisional confirmation, or final
pre-exit evidence. Absolute plate width/height gates also prevent a tiny,
distant detection from taking the express path.

The CTC adapter retains greedy decoding by default for compatibility. Setting
`beam_width > 1` publishes Top-K candidates; `constrain_iranian_layout=True`
prunes prefixes that cannot match two digits, a letter, and five digits. The
fusion layer consumes those candidates as alternatives from the same source
frame, so they add evidence without pretending to be independent frames.

## Enable for an A/B run

```python
from app.engine_v2 import EngineV2Config, TemporalFusionConfig

config = EngineV2Config(
    track_temporal_fusion_enabled=True,
    temporal_fusion=TemporalFusionConfig(
        provisional_confidence=0.75,
        lock_confidence=0.86,
        express_lock_confidence=0.93,
        independent_time_gap_seconds=0.08,
        soft_lock_hold_seconds=0.12,
        min_plate_width_px=80,
        min_plate_height_px=20,
        max_ocr_attempts=4,
    ),
)
```

All thresholds are policy inputs and must be calibrated on held-out traffic
footage. In particular, `0.75` is a provisional threshold, not an event
acceptance threshold. The fail-closed IR-LPR plus day/night camera workflow is
documented in [Engine V2 TCAM calibration](engine-v2-calibration.md).

## Event and telemetry contract

Fused events add the following metadata:

- `recognition_phase`
- `fusion_reason`
- `soft_lock_reason`
- `finalization_reason`
- `audit_attempts`
- `calibration_profile`
- `independent_observations`
- `full_sequence_support`
- `slot_confidences`
- `slot_margins`
- `ocr_schedule_reason`
- `ocr_attempts`

Runtime telemetry also reports fusion track counts, provisional/soft-locked/
finalized counts, and total fusion OCR attempts. These fields support
comparison of recall, CER, OCR calls per track, latency, and CPU before the
feature flag is widened.
