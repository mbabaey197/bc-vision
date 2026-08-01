# BC Vision ANPR Engine Contract

این سند قرارداد موتور فعال RC19 است و از اجرای ناخواسته مدل یا Fallback متفاوت در Build جلوگیری می‌کند.

## مسیر فعال (Baseline)

| نقش | موتور | وضعیت |
|---|---|---|
| Detector اصلی | YOLOv8 ONNX، `plate_yolo.onnx` | فعال |
| Detector جایگزین | YOLOv8 ONNX، `plate_yolo_fallback.onnx` | فقط هنگام ضعف/عدم دسترسی مسیر اصلی |
| OCR سریع | Dedicated character detector | انتخاب اول |
| OCR نجات | CRNN ONNX، `ocr_crnn.onnx` | فقط برای خوانش مفقود یا ضعیف |
| OCR نهایی سبک | Character CNN ONNX، `ocr_cnn.onnx` | فقط روی Crop معتبر و در صورت نیاز |
| تصمیم نهایی | Multi-frame consensus + review policy | فعال |

CRNN و CNN برای هر Crop بدون شرط با هم اجرا نمی‌شوند. `app/ai/pipeline.py` فقط در شرایط Rescue، OCR دوم را فعال می‌کند تا مصرف CPU کنترل شود.

## موتور نسل بعد

`app/ai/next_engine.py` به‌صورت پیش‌فرض جای Baseline را نمی‌گیرد. فعال‌شدن آن نیازمند همه موارد زیر است:

1. Manifest امضاشده Ed25519
2. Hash و اندازه معتبر هر مدل
3. Runtime و Input contract معتبر
4. اجرای Shadow و ارزیابی
5. فعال‌سازی صریح اپراتور

مدل Research-only فقط در Shadow قابل اجراست و نباید در خروجی قابل‌توزیع قرار گیرد.

## مدل‌های Baseline

| مدل | اندازه | SHA-256 |
|---|---:|---|
| `plate_yolo.onnx` | 12,608,775 | `A54E475C402E6036BB5C70F1A6FF75179E76098A5C8039BB5D148C0B6421F5C6` |
| `plate_yolo_fallback.onnx` | 12,265,080 | `A6974FCB0A79755C270D50F1EBEFD4D96D765C879A29051A19AAC00DFDA8B5AF` |
| `ocr_crnn.onnx` | 10,452,525 | `45F8C45F29EB1EE91F6274CB8D9C328DA1A2050EA7D8596BAE61F4A6B9F9FB1E` |
| `ocr_cnn.onnx` | 2,226,402 | `7D573C51CC855A8E080F1F88597477F4FB5A2B9CAFA1BB125BD6038E441F5BCA` |

منبع اجرایی این مقادیر `app/ai/model_manager.py` است. برنامه هیچ مدل دانلودشده‌ای را بدون تطبیق هم‌زمان اندازه و SHA-256 فعال نمی‌کند.

## آستانه‌ها

- Confidence دوربین و Frame step از تنظیم هر دوربین خوانده می‌شود.
- Thresholdهای موتور نسل بعد فقط از Manifest امضاشده پذیرفته می‌شوند.
- خروجی دارای اختلاف OCR یا شواهد ناکافی با `needs_review` ذخیره می‌شود.
- تأیید اپراتور منبع آموزش است؛ حدس خام به‌تنهایی حقیقت آموزشی محسوب نمی‌شود.

هر تغییری در Runtime، Input size، Threshold، Hash یا سیاست Fallback باید همراه با تست و به‌روزرسانی این سند انجام شود.
