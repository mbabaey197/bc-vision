"""Render self-contained RTL HTML rows from CCT Golden benchmark JSON."""
from __future__ import annotations

import argparse
import base64
from html import escape
import json
import mimetypes
from pathlib import Path


def _image_data_uri(path: Path) -> str:
    if not path.is_file():
        return ""
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def _normal_plate(value) -> str:
    return "".join(
        character
        for character in str(value or "")
        if character.isalnum() or "\u0600" <= character <= "\u06ff"
    ).upper()


def _result_section(label: str, result: dict) -> str:
    truth = {
        _normal_plate(value)
        for value in (
            list(result.get("matched_truth", []))
            + list(result.get("missed_truth", []))
        )
    }
    artifact_dir = Path(str(result.get("artifact_dir") or "."))
    rows = []
    for index, event in enumerate(result.get("emitted", []), start=1):
        crop_path = artifact_dir / str(event.get("crop_path") or "")
        image_uri = _image_data_uri(crop_path)
        image = (
            f"<img src='{image_uri}' alt='برش پلاک ردیف {index}'>"
            if image_uri
            else "<span class='missing'>تصویر ذخیره نشده</span>"
        )
        normalized = _normal_plate(
            event.get("plate_norm") or event.get("plate")
        )
        if normalized in truth:
            status = "<span class='status match'>مطابق Golden</span>"
        elif normalized:
            status = (
                "<span class='status review'>خارج از Golden؛ "
                "نیازمند برچسب اپراتور</span>"
            )
        else:
            status = "<span class='status unreadable'>ناخوانا</span>"
        confidence = round(float(event.get("confidence") or 0.0) * 100, 1)
        ocr_confidence = round(
            float(event.get("ocr_confidence") or 0.0) * 100,
            1,
        )
        second = round(float(event.get("video_second") or 0.0), 3)
        rows.append(
            "<tr>"
            f"<td>{index}</td>"
            f"<td class='crop'>{image}</td>"
            f"<td><strong dir='ltr'>{escape(str(event.get('plate') or 'ناخوانا'))}</strong>"
            f"{status}</td>"
            f"<td>{confidence}%</td>"
            f"<td>{ocr_confidence}%</td>"
            f"<td>{second}s</td>"
            f"<td>{escape(str(event.get('track_id') or '—'))}</td>"
            "</tr>"
        )
    if not rows:
        rows.append(
            "<tr><td colspan='7' class='empty'>هیچ خروجی ثبت نشد.</td></tr>"
        )
    matched = int(result.get("matched_truth_count") or 0)
    truth_count = int(result.get("truth_count") or 0)
    return (
        f"<section><h2>{escape(label)}</h2>"
        "<div class='metrics'>"
        f"<div><small>فریم پردازش‌شده</small><b>{int(result.get('processed_frames') or 0)}</b></div>"
        f"<div><small>تشخیص اولیه</small><b>{int(result.get('detections') or 0)}</b></div>"
        f"<div><small>تطابق Golden</small><b>{matched}/{truth_count}</b></div>"
        f"<div><small>خروجی Track</small><b>{int(result.get('emitted_count') or 0)}</b></div>"
        f"<div><small>زمان CPU</small><b>{float(result.get('elapsed_seconds') or 0.0):.3f}s</b></div>"
        "</div>"
        "<div class='table-wrap'><table><thead><tr>"
        "<th>ردیف</th><th>تصویر پلاک</th><th>متن تشخیص‌داده‌شده</th>"
        "<th>اطمینان کل</th><th>اطمینان OCR</th><th>زمان</th><th>Track</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div></section>"
    )


def render_report(results: list[tuple[str, dict]]) -> str:
    first = results[0][1] if results else {}
    truth = (
        list(first.get("matched_truth", []))
        + list(first.get("missed_truth", []))
    )
    truth_html = "، ".join(escape(value) for value in truth) or "ثبت نشده"
    sections = "".join(
        _result_section(label, result)
        for label, result in results
    )
    return f"""<!doctype html>
<html lang="fa" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>گزارش تصویری تست پلاک‌خوان BC Vision RC14</title>
<style>
body{{margin:0;background:#eef3f8;color:#17212b;font-family:Tahoma,Arial,sans-serif}}
main{{max-width:1380px;margin:auto;padding:28px}}
h1{{margin:0 0 10px}}h2{{margin-top:0}}
.lead{{color:#536170;line-height:1.9;margin-bottom:24px}}
section{{background:#fff;border:1px solid #dce5ee;border-radius:18px;padding:20px;margin:18px 0;box-shadow:0 8px 24px #17334d12}}
.metrics{{display:grid;grid-template-columns:repeat(5,minmax(120px,1fr));gap:10px;margin:14px 0 18px}}
.metrics div{{background:#f5f8fb;border-radius:12px;padding:12px}}
.metrics small{{display:block;color:#667787;margin-bottom:6px}}.metrics b{{font-size:20px}}
.table-wrap{{overflow:auto}}table{{width:100%;border-collapse:collapse;min-width:940px}}
th,td{{padding:10px;border-bottom:1px solid #e5ebf1;text-align:right;vertical-align:middle}}
th{{position:sticky;top:0;background:#eaf1f7;color:#455767;z-index:1}}
tbody tr:hover{{background:#f7fafc}}.crop img{{width:220px;height:78px;object-fit:contain;background:#101820;border-radius:8px}}
.missing{{display:inline-flex;width:220px;height:78px;align-items:center;justify-content:center;border:1px dashed #9aabb9;border-radius:8px;color:#687887}}
.status{{display:block;width:max-content;margin-top:7px;padding:4px 8px;border-radius:99px;font-size:12px}}
.match{{background:#dff7e9;color:#087747}}.review{{background:#fff1d6;color:#8c5b00}}.unreadable{{background:#fde5e5;color:#a42828}}
.empty{{text-align:center;padding:28px;color:#687887}}
@media(max-width:800px){{main{{padding:14px}}.metrics{{grid-template-columns:1fr 1fr}}}}
</style>
</head>
<body><main>
<h1>گزارش تصویری تست پلاک‌خوان BC Vision RC14</h1>
<p class="lead">ویدیوی مرجع: <code>{escape(str(first.get('video') or '—'))}</code><br>
SHA‑256: <code>{escape(str(first.get('video_sha256') or '—'))}</code><br>
پلاک‌های Golden ثبت‌شده: <strong dir="ltr">{truth_html}</strong><br>
هر ردیف دقیقاً تصویر Crop ورودی OCR و متن خروجی همان Track را کنار هم نشان می‌دهد.
رشته‌های خارج از Golden تا زمان برچسب‌گذاری اپراتور «تأییدنشده» هستند، نه نتیجه صحیح.</p>
{sections}
</main></body></html>"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        help="LABEL=path/to/result.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    results = []
    for item in args.result:
        label, separator, raw_path = item.partition("=")
        if not separator or not label.strip() or not raw_path.strip():
            raise ValueError("--result must use LABEL=path")
        path = Path(raw_path).resolve()
        results.append(
            (
                label.strip(),
                json.loads(path.read_text(encoding="utf-8")),
            )
        )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_report(results), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
