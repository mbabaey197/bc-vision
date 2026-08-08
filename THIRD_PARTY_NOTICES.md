# Third-party notices

## IR-LPR research dataset

BC Vision includes an offline importer for the separately downloaded IR-LPR
dataset:

- Project: https://github.com/mut-deep/IR-LPR
- Paper: https://arxiv.org/abs/2209.04680
- Repository license: GNU General Public License v3.0
- Repository license Git blob:
  `f288702d2fa16d3cdf0035b15a9fcbc552cd88e7`

No IR-LPR image, annotation, archive or trained weight is committed or bundled.
Any candidate trained with IR-LPR is marked non-distributable,
`research-shadow-only` and cannot become the active BC Vision engine. This
notice is a software safety policy, not a legal opinion about whether model
weights are derivative works.

## PaddleDetection PP-YOLOE-R

BC Vision's rotated-detector configuration and ONNX preprocessing/output
contract are adapted from PaddleDetection release 2.9:

- Project: https://github.com/PaddlePaddle/PaddleDetection
- PP-YOLOE-R configuration:
  `configs/rotate/ppyoloe_r/`
- Official ONNX example:
  `configs/rotate/tools/onnx_infer.py`
- License: Apache License 2.0
- Copyright: PaddlePaddle Authors

The Apache License 2.0 text is available at:
https://www.apache.org/licenses/LICENSE-2.0

No PaddleDetection trained plate weight is bundled. The BC Vision detector
must be trained on separately licensed Iranian camera data.

## YOLO11n license-plate detector

BC Vision can download and execute this fine-tuned single-class YOLO11n ONNX
model:

- Model: https://huggingface.co/morsetechlab/yolov11-license-plate-detection
- Pinned revision: `0f8dc030388b3660418ac7d8c37d3a40148064c1`
- File: `license-plate-finetune-v1n.onnx`
- SHA-256:
  `693133a1db97a3ba1e90068986f80afb72c3fcddb681e57181a89a9a3dc351d6`
- Size: `10481682` bytes
- Declared upstream license: GNU Affero General Public License v3.0

The model card says its Roboflow-derived data has train/test contamination and
reported metrics may be inflated. BC Vision therefore treats those metrics as
unverified and requires independent field/Golden evaluation. The AGPL-3.0
license text is available at:
https://www.gnu.org/licenses/agpl-3.0.html

## Hezar Persian license-plate CRNN v2

BC Vision's build utility downloads a fixed Hezar model revision, verifies its
source files and exports a deterministic ONNX graph:

- Model: https://huggingface.co/hezarai/crnn-fa-license-plate-recognition-v2
- Pinned revision: `0c48a86abe5bfb140ceeb160c028701028d236b9`
- Source `model.pt` SHA-256:
  `c20ad7be2b1fe383da6f22cbc7bdf8a9a37119f0b20235d736faa59b731f6620`
- Exported `crnn_fa_v2.onnx` SHA-256:
  `57cb02bc10bdebd14be2ac50cd7c25d657bcdee6efe77a37a561b832206b0c8`
- Exported size: `37146355` bytes
- Hezar Python library: https://github.com/hezarai/hezar
- Hezar library license: Apache License 2.0

The model repository does not currently display a separate license declaration
for the trained weight. The model and its ONNX export must remain subject to
the owner's distribution/licensing review. This notice records provenance and
is not a legal opinion.

## Platrix Iranian ANPR models

BC Vision retains these Platrix ONNX assets as detector/OCR fallbacks:

- Model: https://huggingface.co/Dibachain/Platrix
- Reference project: https://github.com/AliAkrami1375/Platrix
- Retired `plate_yolo.onnx` SHA-256 (provenance only; no longer active):
  `a54e475c402e6036bb5c70f1a6ff75179e76098a5c8039bb5d148c0b6421f5c6`
- `plate_yolo_fallback.onnx` SHA-256:
  `a6974fcb0a79755c270d50f1ebefd4d96d765c879a29051a19aac00dfda8b5af`
- `ocr_crnn.onnx` SHA-256:
  `45f8c45f29eb1ee91f6274cb8d9c328da1a2050ea7d8596bae61f4a6b9f9fb1e`
- `ocr_cnn.onnx` SHA-256:
  `7d573c51cc855a8e080f1f88597477f4fb5a2b9cafa1bb125bd6038e441f5bca`

MIT License

Copyright (c) 2026 AliAkrami1375

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## FastPlateOCR training architecture

BC Vision's offline CCT training configs and export workflow are adapted from:

- Project: https://github.com/ankandrew/fast-plate-ocr
- Version: `1.1.0`
- Upstream commit: `9ce7a5b64a939aa421c243b331d42e6bc25ffd44`

FastPlateOCR, its CCT model builder and released training configs are MIT
licensed. Keras/TensorFlow training dependencies and trained weights are not
loaded by the normal BC Vision camera runtime.

MIT License

Copyright (c) 2024 ankandrew

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
