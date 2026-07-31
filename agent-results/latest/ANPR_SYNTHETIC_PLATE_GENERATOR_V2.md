# BC Vision — synthetic Iranian plate generator v2

Date: 2026-07-30

## Outcome

The existing bootstrap generator was expanded into a deterministic,
license-clean synthetic OCR pipeline. The implementation is ready for a pilot
dataset, but no claim is made that synthetic data alone improves the production
model. RC12 remains active and the IR-LPR-derived model remains research-only.

## Implemented

- exact CCT output contract: RGB `128x64`, eight canonical label slots;
- Persian digit glyph rendering with standard, public, government, military,
  diplomatic and service colour families;
- visual accessibility-symbol rendering for canonical `ژ` and word `الف`
  rendering for canonical `ا`, each preserving its single OCR label slot;
- ten balanced condition profiles: clean, daylight, night, motion blur,
  perspective, headlight glare, rain, dirt, low resolution and mixed hard;
- low-resolution hard samples simulate source widths from 68 through 155
  pixels, targeting the measured weakness below 156 pixels;
- extra post-coverage sampling weight for `ژ، ث، ا، ف، ک، گ، D، S`;
- deterministic per-sample seeds independent of generation order;
- byte-reproducible JPEG output for the same seed and inputs;
- per-image CSV and JSONL metadata including condition, difficulty, simulated
  width, quality score, plate style and exact seed;
- font-license allowlist and SHA-256 provenance;
- explicit manifest assertions that no third-party plate pixels were used;
- train/validation plate identities remain disjoint;
- quality rescue prevents an extremely degraded sample from silently becoming
  an unusable training label.

## Pilot verification

The local visual pilot used 30 train identities, 10 validation identities and
two views per identity.

| Check | Result |
|---|---:|
| Train images | 60 |
| Validation images | 20 |
| Train/validation identity overlap | 0 |
| Profiles in train | 10/10 |
| Images per train profile | 6 |
| Output shape | `128x64x3` |
| Train mean quality score | 0.8597 |
| Validation mean quality score | 0.8803 |
| Quality rescues | 0 |
| Real/IR-LPR pixels used | 0 |

Visual inspection confirmed readable Persian digits, stable canonical layout,
distinct low-light/motion/glare/low-resolution effects and special colour
families. DejaVu Sans remains a permissively licensed bootstrap font, not a
claim of exact Iranian plate-font fidelity.

## Automated verification

Generator test file: `17 passed`.

Broader CCT dataset/split/IR-LPR-isolation subset: `26 passed`.

The tests cover:

- identity isolation;
- all condition profiles and exact CCT image shape;
- byte reproducibility;
- every configured letter before oversampling;
- elevated sampling of known weak classes;
- invalid font-license rejection;
- invalid condition rejection;
- special class colour-family routing;
- commercial company-crop and Golden-data isolation gates; and
- pretrained OCR-head exclusion.

## 30K pilot completed

The follow-up pilot generated 24,000 training, 3,000 validation and 3,000
held-out test images with 3,000 unique plate identities and zero cross-split
identity overlap. Training, per-profile synthetic evaluation and the fixed
real-video rejection gate are recorded in
`agent-results/latest/ANPR_SYNTHETIC_30K_TRAINING_V1.md`.

Do not generate the final 100,000–300,000 corpus until font fidelity and pilot
transfer to operator-labelled real imagery are improved.
