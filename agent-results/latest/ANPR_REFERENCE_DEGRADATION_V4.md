# BC Vision — real-reference degradation calibration v4

Date: 2026-07-30

## Classification

The newly supplied rear-plate photograph is not a glyph or label reference.
The plate region is substantially overexposed and defocused, with broad
retroreflective bloom and weak character edges. No character was inferred,
transcribed or added to a training manifest.

The photograph remains outside Git, Train, Validation, Test and Golden data.
Only anonymous image-formation characteristics were used to define the
synthetic `overexposed_defocus` condition profile.

## Profile

`rear-plate-overexposed-defocus-v1` models:

- bounded optical defocus;
- broad highlight bloom from the procedural plate itself;
- controlled exposure gain, contrast loss and black lift;
- optional mild perspective and low sensor noise; and
- a five-percent hard tail where blur and exposure cannot both be extreme.

Directional motion blur, rolling shutter and plate curvature are excluded
because the photograph does not provide reliable evidence for them.

The profile preserves deterministic labels from the synthetic source string.
An edge-retention gate blends a limited amount of the clean procedural sample
back when the degradation destroys too much recoverable structure. It never
changes or guesses a label.

## Scope

This profile applies only to future synthetic dataset builds. It does not
retroactively alter the completed 30,000-image corpus or its trained ONNX
model. Retraining remains blocked until the full reference set and licensed
plate-faithful glyph pack are ready. The active model remains Shadow-only.
