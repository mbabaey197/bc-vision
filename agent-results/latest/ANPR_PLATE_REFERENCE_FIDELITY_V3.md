# BC Vision — Iranian plate reference fidelity v3

Date: 2026-07-30

## Outcome

The white private-plate bootstrap renderer now uses a photographed-reference
layout instead of drawing three proportional text strings. Every visible
digit and letter has its own fixed cell, while the national band and two-digit
region panel have independent measured geometry.

The supplied real-world photographs were used only for visual measurement and
QA. No reference pixels, registration identity, file name, EXIF metadata or
location data were copied into the repository, generated dataset or training
labels. An unreadable or unconfirmed real plate is never guessed.

## Measured private-plate profile

The `iran-national-photo-reference-v1` profile records:

- a `420x90` base plate rendered from a `2x` calibration canvas;
- a national blue band occupying about 9–10.5% of plate width;
- the city/region separator at about 80.4% of plate width;
- fixed two-digit prefix, single-letter, three-digit serial and two-digit
  region cells;
- separate `ایران`, Iranian flag and `I.R. IRAN` elements;
- a restrained raised border and embossed glyph edge; and
- small seeded positional/colour variation before camera degradation.

Non-private colour families retain the existing procedural layout under the
separate `legacy-procedural-v2` profile. They must be calibrated from lawful
references before being described as photograph-matched.

## Font boundary

The generic DejaVu bootstrap font is no longer presented as plate-faithful.
Public Traffic/Titr font files and raster glyph sets were examined only as
private research references because their underlying training and
model-distribution rights have not been verified. None of those font assets
was added to the repository or production package.

The renderer accepts a plate-specific primary font plus a separately approved
fallback font. Both files require an allowlisted license declaration and a
SHA-256 record. Exact production glyph fidelity remains blocked until BC
Vision has a company-owned or expressly licensed complete glyph pack,
including all configured Persian and special symbols.

## Reproducibility and safety

Synthetic manifest schema 3 now records:

- renderer filename and exact renderer SHA-256;
- private and special layout-profile identifiers;
- primary and fallback font name, SHA-256 and license;
- base and model-input dimensions; and
- explicit assertions that no real plate pixels or Golden data were used.

Training preflight rejects a missing, malformed or stale renderer fingerprint,
an unexpected layout mapping, unapproved font provenance, real/Golden media
references and any train/validation/test identity overlap.

The comparison utility accepts only a dummy identity or an explicitly
operator-confirmed label. It writes an EXIF-free side-by-side QA PNG and is not
a dataset importer.

## Verification

Focused renderer, dataset-contract and security tests:

`60 passed`

Full repository regression:

`329 passed, 1 skipped`

This change improves the synthetic plate shape but does not establish
real-camera OCR accuracy. The existing model remains Shadow-only with
`activation_allowed=false`; training is not restarted solely from an
unapproved font preview.
