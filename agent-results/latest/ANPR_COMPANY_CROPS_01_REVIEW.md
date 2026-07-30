# BC Vision — company-owned crop batch 01 review

Date: 2026-07-30

## Source contract

- Private archive: `01.zip`
- SHA-256:
  `83307A71EF3428739C20C286FB62DF0759C0572F803B77361D0CDCC95FFC5628`
- Images: 505 JPEG plate crops
- Exact duplicate files: 0
- Source license: `operator-confirmed-company-owned`
- Ownership attested: yes
- Evidence:
  `user-attestation-chat-2026-07-30-company-owned-01`
- Git/public distribution: prohibited for the source images

These are cropped plate images, not full vehicle frames. They can be used for
OCR review and fine-tuning after exact operator confirmation, but they do not
provide detector boxes, vehicle context or multi-frame tracking evidence.

## Quality triage

- good: 204
- careful review: 217
- hard/evaluation candidate: 84

The buckets prioritize review; they are not ground-truth labels. Ambiguous
characters remain unreadable and are not inferred from adjacent images or a
model guess.

## Offline review tool

`tools/build_plate_label_review.py` creates one self-contained RTL HTML page:

- all images embedded locally;
- no automatic network upload;
- four explicit Iranian plate fields;
- Confirmed, Unreadable, Excluded and Pending states;
- browser-local progress plus JSON/CSV export;
- import of a prior JSON export for continuation;
- filters for state and quality bucket;
- model suggestions visibly marked
  `untrusted-shadow-suggestion`.

The 30K synthetic CCT-XS model produced a draft hypothesis for every crop.
Visual spot checks show substantial domain-transfer error, so these
hypotheses are never treated as labels and are not bulk-confirmed.

## Training boundary

At review-tool creation time, no batch-01 crop had entered Train, Validation
or Golden. A returned operator review must pass all of the following before
preparation:

1. exact canonical `2 digits + letter + 3 digits + 2 digits` label;
2. explicit Confirmed state;
3. source-image SHA-256 match;
4. one split per confirmed plate identity;
5. zero Train/Validation identity overlap;
6. no unreadable or excluded row;
7. no independent Golden crop reused for training.

## Verification

- Python compile for generator and test file: passed
- focused generator smoke test: passed
- JavaScript syntax check on the generated 505-image page: passed
- generated image count: 505
- generated bucket counts: 204 / 217 / 84
- generated review HTML size: about 5.6 MiB
- generated review HTML SHA-256:
  `16EB126375DAB1F155288BB873493AEC8B1963729F3F8B6E2718ED54089B91BB`

The first partial operator export was subsequently imported and evaluated
under the rules above. Its aggregate training result is recorded separately
in `agent-results/latest/ANPR_COMPANY_CROPS_01_FINE_TUNE.md`. Source crops,
operator labels and generated private datasets remain outside Git.

The focused importer, review, CCT security-contract and dataset regression
suite now passes: `64 passed`.
