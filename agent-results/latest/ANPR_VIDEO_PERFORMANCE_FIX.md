# ANPR Video Accuracy and Performance Fix

Date: 2026-07-26
Status: implementation in progress on `agent/anpr-fast-accurate-video`

## Confirmed root causes

- The verified `best.pt` model contains plate and character classes.
- Class `30` is the full plate, but the old detector treated character boxes
  as complete plates.
- On the 1920×1080 reference video, the old sampled detector produced 436
  candidates across 109 frames and took 99.756 seconds.
- Every false candidate could trigger six EasyOCR preprocessing/inference
  passes plus vehicle analysis.
- A successful YOLO frame with zero plates incorrectly triggered the broad
  OpenCV fallback, creating more false candidates and OCR work.

## Implemented

- Filter full-frame YOLO inference to class `30`.
- Run the same model on the plate crop for dedicated character recognition.
- Select the highest-confidence eight-character subsequence that matches the
  Iranian plate layout and drops country-strip noise.
- Use 640px localization and 416px crop recognition defaults.
- Cap verified-model plate candidates to four per frame.
- Process crops sequentially to bound peak RAM.
- Skip generic EasyOCR after dedicated character recognition was attempted.
- Use OpenCV fallback only when YOLO is unavailable or fails, not on a valid
  zero-result frame.
- Defer vehicle color/type analysis until a stable event is actually saved.

## Current verification

- Three real reference crops: dedicated character inference completed in
  0.328 seconds total at 416px before template selection.
- Targeted detector/pipeline/video/live tests: 20 passed.
- Full source regression suite: 33 passed, 1 skipped.
- Compile-all and whitespace validation: passed.

## Pending before merge

- Clean full-video run after unrelated stale benchmark processes leave the
  constrained test sandbox.
- Exact comparison with the plate strings reported by the other ANPR product.
- Windows AI runner result for the final commit.
