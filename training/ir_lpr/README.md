# IR-LPR research profile

BC Vision can use the official IR-LPR data only for an isolated,
non-distributable Shadow comparison.

Official source: https://github.com/mut-deep/IR-LPR

Expected source sets:

| Task | Train | Validation | Test |
| --- | --- | --- | --- |
| Detector | `car_img-train.zip` | `car_img-val.zip` | `car_img-test.zip` |
| OCR | `plate_img-train.zip` | `plate_img-val.zip` | `plate_img-test.zip` |

`tools/prepare_ir_lpr_dataset.py` accepts the six ZIP files directly or their
extracted directories. It requires the explicit
`--accept-gpl-3.0-research-only` switch, reconstructs only valid standard
eight-slot Iranian plates, removes image and plate-identity leakage, and
creates:

- `ocr/train`, `ocr/val`, `ocr/test` for FastPlateOCR CCT;
- `detector/train`, `detector/validation`, `detector/test` for PP-YOLOE-R;
- machine-readable provenance and exclusion reports.

The IR-LPR VOC labels are axis-aligned. Detector polygons produced from those
boxes therefore have zero rotation; real BC Vision quadrilateral annotations
are still required to fine-tune plate angle and perspective handling.

Never place archives, extracted data, generated models or training runs in
the Git repository. They belong under ignored `training-data/` and
`training-runs/` directories.
