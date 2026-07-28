# BC Vision ANPR modernization and Persian datetime — RC12

Version: `2.2.0-rc12`

## Implemented

- Replaced the 119 MB PyTorch/Ultralytics production detector with verified
  lightweight primary and fallback YOLOv8 ONNX plate detectors.
- Kept the whole-plate CRNN as the primary reader and added a verified Iranian
  character CNN fallback.
- Removed EasyOCR and Tesseract from the production, dependency and PyInstaller
  paths.
- Added two-pass ByteTrack-style association and constant-velocity Kalman box
  prediction while retaining Optical Flow, template recovery and multi-frame
  consensus.
- Captured confirmed corrections as immutable SHA-256-labelled local samples
  with deterministic train/validation splits.
- Added bounded background CRNN training, validation against the active model
  and explicit administrator approval before atomic model promotion.
- Displayed Persian digits for visible dates and times throughout the
  application while preserving ASCII database, filename, API and control
  formats.
- Preserved existing databases, users, settings, licenses, media and persistent
  vendor/custom model storage during update and uninstall.

## Verified locally

- Full regression: `130 passed, 1 skipped`.
- Focused modernization/localization coverage: `51 passed`.
- Compile, whitespace and secret-pattern checks: passed.
- Primary detector and CRNN real ONNX path on ten reference crops: `10/10`.
- CNN fallback produced valid eight-character output only on four reliably
  segmented crops and abstained on the remaining six instead of guessing.
- All active ONNX sessions reported the two-thread ceiling.

## Verified on Windows

- Corrected PR validation run `30359483088`: passed.
- Main validation run `30359720847`: passed.
- Windows release-candidate run `30359478238`: passed.
- Source regression and real four-model ONNX load/inference: passed.
- Portable application build and GUI-subsystem/windowless-host check: passed.
- Setup and one-click Update builds: passed.
- Clean install and installed offline ANPR self-test: passed.
- In-place update and updated offline ANPR self-test: passed.
- SQLite setting and persistent model markers survived update: passed.
- Standard uninstall preserved persistent database and model data: passed.
- Release bundle creation, upload and SHA-256 verification: passed.

The first PR AI job (`30359204989`) failed before inference because removing
EasyOCR also removed its transitive OpenCV dependency. RC12 now pins
`opencv-python-headless==5.0.0.93` directly; the corrected Windows job passed.

## Publication

- Application PR: `#21`
- Release-workflow guard PR: `#22`
- Application merge/tag commit:
  `273ebf43b7f12b20fd46e68f30ebcfab3784c113`
- Verified candidate artifact: `8688568010`
- Candidate artifact digest:
  `46ed7cff917a2f74baf985b38b0639bd09a0aa2c6e987596ded1707a1d0b0258`
- Complete release run: `30361052399`
- Release tag: `v2.2.0-rc12`

Published SHA-256:

- Setup:
  `acdfcdeda80b38501af2d8fd7c15b009599b2e3da8e04e915f053f710028c387`
- Update:
  `dd1830fbf1dd8eb4aea0147b37f46fae080c50f28cb9037579552aacfc75fd25`
- Source:
  `d622fda8a34176b3d3651dfe518e8a95520c40e5db8273b66187d5a3fd11159e`

## Remaining field validation

The Windows and reference-image gates prove packaging, loading and execution;
they do not prove final accuracy on the customer's real `01.mp4` or camera
feeds. That comparison remains the next field test, and ambiguous plates must
continue to be reported as unreadable rather than guessed.
