# BC Vision RC15 — PP-YOLOE-R and raw-guess observation

Date: 2026-07-29

## Implemented

- signed `bcvision-rc15` detector contract for `ppyoloe-r-onnx`;
- official PaddleDetection three-input preprocessing;
- official `B x N x 8` rotated-box and `B x C x N` score decoding;
- rotated NMS and mandatory perspective rectification before OCR;
- licensed rotated-COCO preparation for real plates and hard negatives;
- Baseline/Candidate-Shadow separation in full-frame video tests;
- visible raw plate guess, confidence, model revision and rejection reason;
- rejected guesses excluded from definitive consensus;
- operator-labelled exact accuracy and character-error measurement by model.

## Operator-assisted decision

RC12 remains the active customer-visible engine. RC15 does not contain trained
PP-YOLOE-R or CCT field weights. An unsigned, modified, incomplete or unknown
model bundle fails closed. A complete multi-frame guess may now enter the
event list as `auto-confirmed`, but it remains experimental and carries an
AI-only confirmation source. It cannot become a training label until an
operator confirms or corrects it. Strict `confirmed-ai` consensus remains a
separate state.

## Remaining release gate

The detector and OCR must be trained on licensed company footage that is
separate from Golden evaluation videos. The signed pair must then beat the
RC12 exact full-plate result on a larger labelled Golden Dataset without
latency or scenario-slice regression before `next` can be activated.

## Source validation

- regression: `181 passed, 1 skipped`;
- Python compile: passed;
- the skip is the existing Windows-only real-model integration test.
