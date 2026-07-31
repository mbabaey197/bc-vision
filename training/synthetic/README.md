# BC Vision synthetic Iranian plate data

`tools/generate_cct_synthetic_dataset.py` creates a commercially clean,
procedural OCR bootstrap dataset. It never downloads, copies or transforms
pixels from IR-LPR, the supplied reference photographs or another third-party
plate dataset.

The white private-plate renderer is implemented in
`tools/iran_plate_renderer.py`. Layout profile
`iran-national-photo-reference-v1` uses fixed character cells, a measured
national-band/region split, the Iranian flag, `I.R. IRAN`, the small `ایران`
heading and a mild embossed rim. The measurements came from visual QA only;
no real plate crop or real registration identity is stored in the generator.

## Output contract

- RGB images are written at the production CCT input size: `128x64`.
- Labels use the canonical eight-slot form: `12ب34567`.
- Every plate identity belongs to only one of `train`, `val` or the optional
  held-out `test` split.
- `annotations.csv` remains directly readable by
  `tools/train_fastplate_cct.py`.
- `samples.jsonl` records the deterministic seed, visual conditions,
  difficulty, simulated source width and quality measurements for every
  image.
- `dataset-license.json` records the generator, font hash/license, split
  counts, class distribution, renderer/layout version and provenance
  boundaries.
- Schema 3 also records the exact `iran_plate_renderer.py` SHA-256. Training
  fails closed if the generating renderer differs from the checked-in
  renderer, so geometry changes cannot be silently mixed into one corpus.

The synthetic validation split is for checkpoint selection only. It is not a
Golden or production test set and cannot replace independent, operator-labelled
real camera footage. A model trained only on this dataset is distributable as
a company-owned pilot artifact, but its candidate metadata must keep
`activation_allowed=false` until an independent real-camera gate passes.

## Condition profiles

| Profile | Simulated effects |
|---|---|
| `clean` | Small print and exposure variation |
| `daylight` | Brightness, contrast and colour temperature |
| `night` | Low exposure, vignetting and sensor noise |
| `motion_blur` | Directional blur suitable for moving vehicles |
| `perspective` | Four-corner projective distortion |
| `headlight_glare` | Local bloom and partial overexposure |
| `rain` | Slanted rain streaks and mild optical blur |
| `dirt` | Limited translucent mud/dirt spots |
| `low_resolution` | Downsampling to a simulated width of 68–155 pixels |
| `mixed_hard` | Three to five combined hard conditions |

The generator gives extra post-coverage weight to `ژ، ث، ا، ف، ک، گ، D، S`,
which were absent or weak in the earlier research training. All configured
letters are still covered before oversampling begins. Canonical `ژ` is
rendered as the blue accessibility symbol and canonical `ا` as the visual word
`الف`, while their training labels remain one OCR slot.

## Pilot generation

Linux:

```bash
python tools/generate_cct_synthetic_dataset.py \
  --output training-data/bcvision-synthetic-pilot \
  --train-plates 2400 \
  --validation-plates 300 \
  --test-plates 300 \
  --views-per-plate 10 \
  --font /usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf \
  --font-license DejaVu-font-license \
  --seed 20260730
```

Windows must use a font whose commercial training rights are known and must
state that license explicitly:

```powershell
python tools\generate_cct_synthetic_dataset.py `
  --output training-data\bcvision-synthetic-pilot `
  --train-plates 2400 `
  --validation-plates 300 `
  --test-plates 300 `
  --views-per-plate 10 `
  --font C:\BCVision\assets\fonts\ApprovedPlateFont.ttf `
  --font-license bcvision-company-owned `
  --seed 20260730
```

An unknown or unapproved font license fails closed. Do not relabel a Windows
system font as company-owned.

A plate-specific primary font may omit Latin glyphs used in the national band
or special plates. An independently approved fallback can be declared without
silently changing provenance:

```powershell
python tools\generate_cct_synthetic_dataset.py `
  --output training-data\bcvision-synthetic-reference-v1 `
  --font C:\BCVision\assets\fonts\ApprovedPlateFont.ttf `
  --font-license bcvision-company-owned `
  --fallback-font C:\BCVision\assets\fonts\DejaVuSans-Bold.ttf `
  --fallback-font-license DejaVu-font-license
```

Both fallback arguments are mandatory as a pair, and the manifest records the
name, SHA-256 and license of both fonts. BC Vision does not bundle or approve
the unverified Traffic/Titr fonts or raster glyphs found in public plate
generator repositories; those assets remain research references only until
their rights holder grants explicit training and model-distribution rights.

For visual QA without importing a reference photo into training, use
`tools/compare_plate_renderer.py` with explicit four-corner coordinates and
either a dummy canonical identity or an operator-confirmed label. Its output
is EXIF-free and must remain outside dataset split directories.

## Scale-up gate

The first pilot is 30,000 total images: 24,000 training, 3,000 validation and
3,000 held-out synthetic test images. Generate the planned 100,000–300,000
image corpus only after:

1. visually reviewing every profile and special plate style;
2. training a pilot model and comparing per-profile accuracy;
3. confirming that synthetic fine-tuning improves, rather than regresses,
   independent real day/night camera results; and
4. replacing the generic DejaVu bootstrap font with an approved,
   plate-appropriate font or a set of independently licensed glyph sources.
