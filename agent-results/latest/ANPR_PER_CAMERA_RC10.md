# BC Vision ANPR — per-camera two-thread engine

Version: `2.2.0-rc10`

## Implemented

- Hard maximum of two native compute threads for each camera inference.
- CPU-aware concurrent camera capacity with two logical processors reserved.
- Separate bounded Ultralytics model instances and locks per runtime slot.
- Independent newest-frame scheduling for each camera.
- Top-five semantic character hypotheses instead of destructive global NMS.
- Per-position, multi-frame probability consensus for close OCR alternatives.
- Safe fusion of incomplete five-to-seven-character observations.
- Shared-lock protection for the large generic EasyOCR fallback.
- Graceful ANPR startup when feedback-table migration is not yet available.
- Reviewable best-effort output when complete OCR evidence exists but strict
  consensus confidence is insufficient.
- Visible partial output with question marks for five-to-seven-character reads;
  only captures with no character evidence remain `ناخوانا`.
- Immediate exact feedback reuse plus repeated, operator-confirmed
  character-confusion re-ranking of real OCR alternatives.
- Affine, forward/backward-validated live overlay tracking with template
  matching fallback; plate boxes remain green and move on every display frame.

## Verification

- Targeted feedback, pipeline, overlay, database and live-worker tests:
  `47 passed`.
- Full source regression: `114 passed, 1 skipped`.
- Python compileall: passed.
- Whitespace validation: passed.
- Synthetic moving/scaling plate sequence: `18/18` frames tracked, zero misses,
  mean/minimum IoU `1.000`.
- Incremental schema change: `plate_events.review_status`.
- Existing database, users, settings, license, media and models are unchanged.
- Real uploaded-video acceptance remains pending because `01.mp4` is not
  present in this checkout.

## Pending release gates

- Real-model Windows runtime inference.
- Multi-camera Windows CPU measurement.
- Portable build and GUI-subsystem verification.
- Full installer and one-click updater build.
- Clean install and RC9-to-RC10 update with persistence checks.
