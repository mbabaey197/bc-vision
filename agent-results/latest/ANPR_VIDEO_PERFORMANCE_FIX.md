# ANPR Video Accuracy and Performance Fix

Date: 2026-07-27
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
- Serialize full-frame and character inference on the shared Ultralytics
  instance. Concurrent live workers previously could return an empty result
  for one of two simultaneous calls because predictor state is mutable.
- Add conditional mild-motion deblurring followed by one dedicated-AI reread.
- Keep the original read unless the restored read agrees or improves by a
  decisive confidence margin; conflicting plausible reads become unreadable.

## Current verification

- Three real reference crops: dedicated character inference completed in
  0.328 seconds total at 416px before template selection.
- Targeted detector/recovery/pipeline/video/live tests: 28 passed.
- Full source regression suite: 40 passed, 1 skipped.
- Compile-all and whitespace validation: passed.

## Multi-camera and mild-blur verification

- Two real video frames that produce one and two plate candidates were read
  both sequentially and through two simultaneous worker threads.
- Sequential and concurrent signatures matched exactly after serialization:
  candidate counts `[1, 2]` in both modes, including bounding boxes and text.
- Three real plate crops were given controlled mild horizontal motion blur.
  The raw model became empty or changed one digit; the recovery+reread path
  returned the same stable model text seen before blur in all three samples.
  This is a robustness check, not a ground-truth accuracy claim.
- A corrupted/recompressed local copy of the reference video was usable only
  through frame 225 and does not match the trusted fixture SHA-256. On that
  prefix, tuned recovery attempted nine rereads and produced the same 14
  stable event signatures as the recovery-disabled control.
- Prefix control: 28.410 seconds, 163.966 CPU seconds, 5.77 average cores,
  1,254.2 MB peak RSS.
- Prefix with recovery: 29.098 seconds, 168.607 CPU seconds, 5.80 average
  cores, 1,245.8 MB peak RSS. The measured wall-time overhead was about 2.4
  percent.

The prefix benchmark cannot replace the clean full-fixture result below.

## Clean full-video verification

The complete 546-frame, 68.25-second, 1920x1080 four-camera reference video
was processed from a clean runtime with the verified 119,237,050-byte model
(`SHA-256 258104262D3A16A6BC613938CC1DD0198DA8A7DDEAB4843197666CB9CE0DB756`).

- 109 frames were selected with `frame_step=5`.
- The unbounded runtime completed in 63.614 seconds while using about 7.76 CPU
  cores and emitted 29 stable plate events.
- A four-thread limit preserved all 29 events and reduced CPU use to about
  4.01 cores, but required 89.814 seconds.
- A six-thread limit preserved all 29 events, completed in 69.638 seconds,
  used about 5.92 CPU cores, and peaked at about 1.32 GB RSS.
- The six-thread result is the selected default because it reduces CPU use by
  about 24 percent while remaining within about 1.4 seconds of the video
  duration on this CPU-only test host.
- Ultralytics resets its Torch thread pool during model construction, so the
  loader now reapplies the CPU budget after construction.

The video is a four-camera mosaic with many vehicles, so the three previously
listed plate strings are not a complete ground-truth set. Exact character
accuracy remains pending a direct export from the comparison ANPR product;
BC Vision does not replace or guess disputed characters.

## Pending before merge

- Exact comparison with a direct plate export from the other ANPR product.
- Windows AI runner result for the final commit.
