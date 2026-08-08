# BC Vision 2.2.0-rc25

RC25 fixes the silent uploaded-video ANPR failure reproduced after RC24.

## Detection recovery

- Uploaded videos always use the full frame instead of inheriting a camera ROI.
- Motion detection is an accelerator only; it cannot suppress uploaded-video inference.
- Upload processing stays active through stationary frames and end-of-file draining.
- Camera LPR state no longer prevents an uploaded file from being analyzed.
- Dashboard status always shows received, processed, detected and emitted counts.
- Frame-dispatch and AI status-channel failures are retained and displayed.

## Data quality improvements

- Plate keys use versioned, idempotent Unicode normalization.
- Existing event, feedback and watch-list keys are safely re-canonicalized.
- Dataset preparation blocks perceptual near-duplicates across splits.
- Confirmed annotation-free images can be included as bounded detector negatives.

## Deliberate exclusions

The reviewed v6.7 source package is not used as a production engine. It contains
no plate weights or positive model fixture, defaults to a generic detector in a
common configuration, and introduces unverified runtime downloads. RC25 keeps
the existing offline, hash-verified ONNX model chain and consensus tracker.

## Validation

- Source regression covers upload scheduling, full-frame inference, dispatch
  error visibility, Unicode migration and dataset split hygiene.
- The packaged RC24 model chain was exercised on the supplied 1920x1080 video.
  It returned the known plate `55ط63974` exactly on a positive frame.
