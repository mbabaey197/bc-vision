# BC Vision ANPR models

BC Vision stores AI model files in the persistent data directory instead of the
application directory. Installer and one-click updates therefore do not delete
models, settings, events, snapshots, or the database.

## Iranian plate and character detector

- Model repository: `makhresearch/persian-license-plate-detector`
- Source: `https://huggingface.co/makhresearch/persian-license-plate-detector`
- File: `best.pt`
- Expected size: `119237050` bytes
- SHA-256: `258104262D3A16A6BC613938CC1DD0198DA8A7DDEAB4843197666CB9CE0DB756`
- License declared by the model repository: MIT

This is a 32-class model. Class `30` is the full plate; the remaining
documented classes represent Persian plate characters. BC Vision must first
filter the full-frame result to class `30`, then run character recognition on
the resulting plate crop. Treating every class as a full plate produces false
candidates and excessive OCR load.

The application downloads this model only over HTTPS. It is moved into the
active model path only after both the exact size and SHA-256 digest match.

## EasyOCR Persian recognition models

- `arabic.pth`
  - SHA-256: `2A9AFD42C374DEB98AED0B53C9B77D75E1D00D4E0501F3B0276C54190C89B1A8`
- `craft_mlt_25k.pth`
  - SHA-256: `4A5EFBFB48B4081100544E75E1E2B57F8DE3D84F213004B14B85FD4B3748DB17`

EasyOCR downloads are also verified before the models are considered ready.

## Persistent Windows paths

By default, models are stored below:

```text
C:\ProgramData\BCVision\data\models\plate\best.pt
C:\ProgramData\BCVision\data\models\easyocr\arabic.pth
C:\ProgramData\BCVision\data\models\easyocr\craft_mlt_25k.pth
```

Environment variables can override the locations:

```text
BCVISION_PLATE_MODEL
BCVISION_EASYOCR_MODEL_DIR
BCVISION_MODEL_SOURCE_DIR
BCVISION_EASYOCR_SOURCE_DIR
BCVISION_DETECTOR_IMAGE_SIZE
BCVISION_CHARACTER_IMAGE_SIZE
```

Default inference sizes are `640` for full-frame plate localization and `416`
for the cropped character pass. These defaults were selected from the real
1920×1080 reference video to reduce CPU time while retaining plate detections.
The character crops are processed sequentially to bound peak RAM.

## Operational limitation

No ANPR implementation can guarantee correct recognition when the plate has no
recoverable pixels, is fully hidden, is outside the frame, or is severely
overexposed. BC Vision reduces failure rates through YOLO localization,
perspective correction, multiple exposure variants, Persian OCR, positional
repair, and multi-frame voting. Camera placement, shutter speed, focus,
resolution, and lighting remain part of overall system accuracy.
