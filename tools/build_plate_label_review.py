"""Build a private, self-contained operator review page for plate crops.

The generated HTML embeds the source images so an operator can label them
offline. Model output is always kept as untrusted draft evidence and never
becomes a confirmed training label without an explicit operator action.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_LETTERS = "ابپتثجدزژسشصطعفقکگلمنوهیDS"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _quality_metrics(image: Image.Image) -> dict:
    grayscale = image.convert("L")
    grayscale.thumbnail((320, 96))
    values = np.asarray(grayscale, dtype=np.float32)
    brightness = float(values.mean())
    contrast = float(values.std())
    horizontal = (
        float(np.abs(np.diff(values, axis=1)).mean())
        if values.shape[1] > 1
        else 0.0
    )
    vertical = (
        float(np.abs(np.diff(values, axis=0)).mean())
        if values.shape[0] > 1
        else 0.0
    )
    edge_strength = (horizontal + vertical) / 2.0
    width, height = image.size
    resolution = min(1.0, width / 180.0, height / 45.0)
    exposure = max(0.0, 1.0 - abs(brightness - 135.0) / 150.0)
    contrast_score = min(1.0, contrast / 58.0)
    edge_score = min(1.0, edge_strength / 24.0)
    score = (
        0.40 * resolution
        + 0.25 * edge_score
        + 0.20 * contrast_score
        + 0.15 * exposure
    )
    return {
        "quality_score": round(float(score), 6),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "edge_strength": round(edge_strength, 2),
    }


def _mime_type(path: Path) -> str:
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }[path.suffix.lower()]


def _load_suggestions(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Suggestions JSON must contain a list")
    suggestions = {}
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Suggestions JSON contains a non-object item")
        filename = str(item.get("file_name", "")).strip()
        proposed = str(item.get("proposed_plate", "")).strip()
        if not filename or not proposed:
            raise ValueError("Suggestion is missing file_name or proposed_plate")
        suggestions[filename] = {
            "plate": proposed,
            "confidence": round(float(item.get("confidence", 0.0)), 6),
            "min_position_confidence": round(
                float(item.get("min_position_confidence", 0.0)),
                6,
            ),
            "min_position_margin": round(
                float(item.get("min_position_margin", 0.0)),
                6,
            ),
            "layout_conflict": bool(item.get("layout_conflict", False)),
            "status": "untrusted-shadow-suggestion",
        }
    return suggestions


def collect_samples(
    image_directory: Path,
    *,
    suggestions_path: Path | None = None,
    good_count: int = 0,
    review_count: int = 0,
) -> list[dict]:
    image_directory = image_directory.resolve()
    paths = sorted(
        path
        for path in image_directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise ValueError("No supported plate crop images were found")
    if good_count < 0 or review_count < 0:
        raise ValueError("Quality bucket counts cannot be negative")
    if good_count + review_count > len(paths):
        raise ValueError("Quality bucket counts exceed the image count")
    suggestions = _load_suggestions(suggestions_path)
    samples = []
    for index, path in enumerate(paths, 1):
        raw = path.read_bytes()
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            metrics = _quality_metrics(image)
        samples.append({
            "id": index,
            "file_name": path.name,
            "sha256": hashlib.sha256(raw).hexdigest().upper(),
            "width": width,
            "height": height,
            **metrics,
            "quality_bucket": "",
            "suggestion": suggestions.get(path.name),
            "image_data": (
                f"data:{_mime_type(path)};base64,"
                + base64.b64encode(raw).decode("ascii")
            ),
        })
    ranked = sorted(
        range(len(samples)),
        key=lambda item: (
            samples[item]["quality_score"],
            samples[item]["width"] * samples[item]["height"],
            samples[item]["file_name"],
        ),
        reverse=True,
    )
    for rank, sample_index in enumerate(ranked):
        if rank < good_count:
            bucket = "good"
        elif rank < good_count + review_count:
            bucket = "review"
        else:
            bucket = "hard"
        samples[sample_index]["quality_bucket"] = bucket
    return samples


def _json_for_script(value) -> str:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        .replace("<", "\\u003C")
        .replace(">", "\\u003E")
        .replace("&", "\\u0026")
    )


def build_review_page(
    image_directory: Path,
    output: Path,
    *,
    source_archive: Path | None = None,
    suggestions_path: Path | None = None,
    good_count: int = 0,
    review_count: int = 0,
    ownership_evidence: str = "",
    title: str = "بازبینی برچسب پلاک‌های BC Vision",
) -> dict:
    samples = collect_samples(
        image_directory,
        suggestions_path=suggestions_path,
        good_count=good_count,
        review_count=review_count,
    )
    archive_sha256 = _sha256(source_archive) if source_archive else ""
    source_identity_material = (
        archive_sha256
        or "".join(sample["sha256"] for sample in samples)
    )
    source_id = hashlib.sha256(
        source_identity_material.encode("ascii")
    ).hexdigest().upper()[:20]
    generated_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "schema": 1,
        "source_id": source_id,
        "source_archive_sha256": archive_sha256,
        "image_count": len(samples),
        "generated_at": generated_at,
        "training_source": "operator-confirmed-only",
        "source_license": "operator-confirmed-company-owned",
        "ownership_attested": bool(ownership_evidence.strip()),
        "distribution_allowed": bool(ownership_evidence.strip()),
        "license_evidence": ownership_evidence.strip(),
        "model_suggestions_are_labels": False,
        "quality_buckets": {
            name: sum(
                sample["quality_bucket"] == name for sample in samples
            )
            for name in ("good", "review", "hard")
        },
        "mean_quality_score": round(
            mean(sample["quality_score"] for sample in samples),
            6,
        ),
    }
    page = PAGE_TEMPLATE
    page = page.replace("__TITLE__", html.escape(title, quote=True))
    page = page.replace("__METADATA__", _json_for_script(metadata))
    page = page.replace("__SAMPLES__", _json_for_script(samples))
    page = page.replace("__LETTERS__", _json_for_script(ALLOWED_LETTERS))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    return metadata


PAGE_TEMPLATE = r"""<!doctype html>
<html lang="fa" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--navy:#07162e;--blue:#087cf0;--cyan:#18b9e8;--surface:#fff;
--bg:#edf3fa;--text:#152238;--muted:#65738a;--line:#d9e3ef;--ok:#168458;
--warn:#c77900;--bad:#c63838;--shadow:0 12px 34px rgba(27,55,90,.12)}
*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#e8f1fb,#f8fbff);
color:var(--text);font-family:Tahoma,"Segoe UI",sans-serif;min-height:100vh}
button,input,select{font:inherit}.shell{max-width:1380px;margin:auto;padding:18px}
.header{display:flex;align-items:center;gap:14px;background:linear-gradient(135deg,var(--navy),#0c315c);
color:#fff;border-radius:18px;padding:16px 20px;box-shadow:var(--shadow);flex-wrap:wrap}
.brand{font-weight:900;font-size:20px}.header small{opacity:.72}.header .spacer{flex:1}
.progress{min-width:260px}.progress-line{height:8px;background:#ffffff2b;border-radius:9px;overflow:hidden;margin-top:7px}
.progress-line span{display:block;height:100%;background:linear-gradient(90deg,var(--cyan),#78e9ff)}
.notice{margin:14px 0;padding:11px 14px;border:1px solid #ffd694;background:#fff7e7;
color:#7d5100;border-radius:12px;font-weight:700}.toolbar,.card{background:rgba(255,255,255,.94);
border:1px solid var(--line);border-radius:16px;box-shadow:var(--shadow)}
.toolbar{display:flex;gap:9px;align-items:center;flex-wrap:wrap;padding:12px;margin-bottom:14px}
button,.file-button{border:0;border-radius:10px;padding:9px 14px;background:linear-gradient(135deg,var(--blue),#075dc5);
color:#fff;cursor:pointer;font-weight:800;text-decoration:none;display:inline-flex;align-items:center;justify-content:center}
button.secondary,.file-button.secondary{background:#65738a}button.ok{background:var(--ok)}
button.warn{background:var(--warn)}button.bad{background:var(--bad)}
button.ghost{background:#e8f1fb;color:#23446b}button:disabled{opacity:.42;cursor:not-allowed}
select,input[type=search],input[type=text]{border:1px solid var(--line);background:#fff;color:var(--text);
border-radius:10px;padding:9px 11px;outline:0}input:focus,select:focus{border-color:var(--blue);box-shadow:0 0 0 3px #087cf020}
.layout{display:grid;grid-template-columns:minmax(0,1.55fr) minmax(350px,.85fr);gap:14px}
.card{padding:16px}.viewer{min-height:540px;display:flex;flex-direction:column}
.image-stage{flex:1;min-height:400px;display:grid;place-items:center;overflow:auto;
background:radial-gradient(circle at center,#20334b,#07111f);border-radius:14px;padding:24px}
.image-stage img{max-width:96%;max-height:62vh;image-rendering:auto;transform:scale(var(--zoom,1));
transform-origin:center;transition:transform .15s;filter:drop-shadow(0 6px 14px #0008)}
.image-meta{display:flex;gap:13px;flex-wrap:wrap;color:var(--muted);margin:12px 2px 0;font-size:13px}
.badge{display:inline-flex;padding:4px 9px;border-radius:99px;font-weight:800;font-size:12px}
.badge.good{background:#e2f6ec;color:#0d6d46}.badge.review{background:#fff0d3;color:#936000}
.badge.hard{background:#ffe5e7;color:#9d2731}.badge.confirmed{background:#e2f6ec;color:#0d6d46}
.badge.unreadable{background:#fff0d3;color:#936000}.badge.excluded{background:#e8ecf2;color:#516077}
.badge.pending{background:#e4f1ff;color:#1268b0}.form-title{display:flex;justify-content:space-between;gap:10px;align-items:center}
.plate-grid{direction:ltr;display:grid;grid-template-columns:76px 76px 105px 76px;gap:9px;align-items:end;margin:18px 0}
.field label{display:block;color:var(--muted);font-size:12px;font-weight:800;margin-bottom:5px;text-align:center}
.field input,.field select{width:100%;height:52px;text-align:center;font-size:23px;font-weight:900;margin:0}
.region{border-right:3px solid var(--navy)!important}.actions{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.actions .wide{grid-column:1/-1}.suggestion{border:1px dashed #efb859;background:#fff8e9;border-radius:12px;padding:11px;margin:15px 0}
.suggestion b{direction:ltr;display:inline-block;font-size:19px}.suggestion .danger-note{color:#9d4f00;font-size:12px;line-height:1.7}
.help{font-size:12px;color:var(--muted);line-height:1.8;margin-top:14px;border-top:1px solid var(--line);padding-top:12px}
.nav{display:flex;gap:8px;margin-top:14px}.nav button{flex:1}.empty{display:grid;place-items:center;min-height:420px;color:var(--muted);text-align:center}
.stats{display:flex;gap:9px;flex-wrap:wrap}.stat{background:#fff2;border:1px solid #fff3;border-radius:10px;padding:7px 10px}
#toast{position:fixed;bottom:22px;left:22px;background:#102945;color:#fff;padding:11px 16px;border-radius:11px;
box-shadow:var(--shadow);opacity:0;transform:translateY(12px);pointer-events:none;transition:.2s;z-index:9}
#toast.show{opacity:1;transform:none}.file-button input{display:none}
@media(max-width:900px){.layout{grid-template-columns:1fr}.viewer{min-height:auto}.image-stage{min-height:310px}.plate-grid{grid-template-columns:1fr 1fr}.shell{padding:9px}}
</style>
</head>
<body>
<div class="shell">
  <header class="header">
    <div><div class="brand">BC Vision — تأیید اپراتوری پلاک</div><small>آفلاین، خصوصی و بدون ارسال خودکار تصویر</small></div>
    <div class="spacer"></div>
    <div class="stats">
      <span class="stat">تأیید: <b id="confirmedCount">۰</b></span>
      <span class="stat">ناخوانا: <b id="unreadableCount">۰</b></span>
      <span class="stat">باقی‌مانده: <b id="pendingCount">۰</b></span>
    </div>
    <div class="progress"><b id="progressText"></b><div class="progress-line"><span id="progressBar"></span></div></div>
  </header>
  <div class="notice">شمارهٔ داخل بخش «پیشنهاد آزمایشی» خروجی مدل Shadow است و ممکن است کاملاً غلط باشد. فقط آنچه با چشم و با قطعیت می‌بینید تأیید کنید؛ موارد مبهم را «ناخوانا» بزنید.</div>
  <div class="toolbar">
    <select id="filter">
      <option value="all">همه تصاویر</option><option value="pending">فقط بررسی‌نشده</option>
      <option value="confirmed">تأییدشده</option><option value="unreadable">ناخوانا</option>
      <option value="excluded">حذف‌شده</option><option value="good">کیفیت خوب</option>
      <option value="review">نیازمند دقت</option><option value="hard">آزمون سخت</option>
    </select>
    <input id="search" type="search" placeholder="جستجوی نام فایل یا پلاک">
    <button id="nextPending" class="ghost">بعدیِ بررسی‌نشده</button>
    <button id="exportJson">خروجی JSON</button>
    <button id="exportCsv" class="secondary">خروجی CSV</button>
    <label class="file-button secondary">ورود فایل ادامه کار<input id="importJson" type="file" accept=".json,application/json"></label>
  </div>
  <main class="layout">
    <section class="card viewer">
      <div id="imageStage" class="image-stage"><img id="plateImage" alt="تصویر برش پلاک"></div>
      <div class="image-meta">
        <span id="fileName"></span><span id="dimensions"></span><span id="qualityBadge"></span>
        <label>بزرگ‌نمایی <input id="zoom" type="range" min="1" max="5" step=".25" value="2"></label>
      </div>
    </section>
    <section id="reviewCard" class="card">
      <div class="form-title"><h2 style="margin:0">ثبت نتیجه</h2><span id="statusBadge"></span></div>
      <div class="plate-grid">
        <div class="field"><label>دو رقم اول</label><input id="prefix" type="text" inputmode="numeric" maxlength="2" autocomplete="off"></div>
        <div class="field"><label>حرف</label><select id="letter"></select></div>
        <div class="field"><label>سه رقم میانی</label><input id="serial" type="text" inputmode="numeric" maxlength="3" autocomplete="off"></div>
        <div class="field"><label>کد ایران</label><input id="region" class="region" type="text" inputmode="numeric" maxlength="2" autocomplete="off"></div>
      </div>
      <div id="suggestion" class="suggestion"></div>
      <div class="actions">
        <button id="confirm" class="ok wide">تأیید قطعی و رفتن به بعدی</button>
        <button id="unreadable" class="warn">ناخوانا</button>
        <button id="exclude" class="bad">حذف از دیتاست</button>
        <button id="clear" class="ghost wide">پاک‌کردن نتیجه و بازگشت به بررسی‌نشده</button>
      </div>
      <div class="help">میانبرها: <b>Enter</b> تأیید، <b>U</b> ناخوانا، <b>X</b> حذف، کلیدهای جهت برای قبلی/بعدی. پیشرفت در همین مرورگر ذخیره می‌شود. برای انتقال به سیستم دیگر، خروجی JSON بگیرید.</div>
      <div class="nav"><button id="previous" class="secondary">قبلی</button><button id="next" class="secondary">بعدی</button></div>
    </section>
  </main>
</div>
<div id="toast"></div>
<script>
"use strict";
const META=__METADATA__;
const SAMPLES=__SAMPLES__;
const LETTERS=Array.from(__LETTERS__);
const DIGITS={"۰":"0","۱":"1","۲":"2","۳":"3","۴":"4","۵":"5","۶":"6","۷":"7","۸":"8","۹":"9","٠":"0","١":"1","٢":"2","٣":"3","٤":"4","٥":"5","٦":"6","٧":"7","٨":"8","٩":"9"};
const fa=n=>String(n).replace(/[0-9]/g,d=>"۰۱۲۳۴۵۶۷۸۹"[d]);
const normalizeDigits=value=>Array.from(String(value||"")).map(c=>DIGITS[c]??c).join("").replace(/\D/g,"");
const storageKey="bcvision-plate-review-"+META.source_id;
let reviews=JSON.parse(localStorage.getItem(storageKey)||"{}");
let visible=[],position=0;
const $=id=>document.getElementById(id);
const fields=["prefix","letter","serial","region"];
function toast(text){$("toast").textContent=text;$("toast").classList.add("show");setTimeout(()=>$("toast").classList.remove("show"),1600)}
function save(){localStorage.setItem(storageKey,JSON.stringify(reviews));updateStats()}
function statusOf(sample){return reviews[sample.sha256]?.status||"pending"}
function plateOf(sample){return reviews[sample.sha256]?.plate||""}
function splitPlate(plate){return plate?.length===8?{prefix:plate.slice(0,2),letter:plate[2],serial:plate.slice(3,6),region:plate.slice(6,8)}:{prefix:"",letter:"",serial:"",region:""}}
function current(){return visible[position]||null}
function applyFilter(){
  const filter=$("filter").value,term=$("search").value.trim().toLowerCase();
  visible=SAMPLES.filter(sample=>{
    const status=statusOf(sample),plate=plateOf(sample).toLowerCase();
    const match=!term||sample.file_name.toLowerCase().includes(term)||plate.includes(normalizeDigits(term));
    if(!match)return false;
    if(filter==="all")return true;
    if(["pending","confirmed","unreadable","excluded"].includes(filter))return status===filter;
    return sample.quality_bucket===filter;
  });
  position=Math.min(position,Math.max(0,visible.length-1));render();
}
function bucketLabel(bucket){return {good:"کیفیت خوب",review:"نیازمند دقت",hard:"آزمون سخت"}[bucket]||bucket}
function statusLabel(status){return {pending:"بررسی‌نشده",confirmed:"تأییدشده",unreadable:"ناخوانا",excluded:"حذف‌شده"}[status]||status}
function setFields(plate){const parts=splitPlate(plate);for(const key of fields)$(key).value=parts[key]}
function render(){
  const sample=current();
  $("reviewCard").style.display=sample?"block":"none";
  $("imageStage").classList.toggle("empty",!sample);
  if(!sample){$("imageStage").innerHTML="<div><b>موردی در این فیلتر نیست.</b><br>فیلتر را تغییر دهید.</div>";return}
  if(!$("plateImage"))$("imageStage").innerHTML='<img id="plateImage" alt="تصویر برش پلاک">';
  $("plateImage").src=sample.image_data;
  $("plateImage").style.setProperty("--zoom",$("zoom").value);
  $("fileName").textContent=`فایل: ${sample.file_name}`;
  $("dimensions").textContent=`ابعاد: ${fa(sample.width)}×${fa(sample.height)}`;
  $("qualityBadge").innerHTML=`<span class="badge ${sample.quality_bucket}">${bucketLabel(sample.quality_bucket)}</span>`;
  const status=statusOf(sample);
  $("statusBadge").innerHTML=`<span class="badge ${status}">${statusLabel(status)}</span>`;
  setFields(plateOf(sample));
  const suggestion=sample.suggestion;
  $("suggestion").innerHTML=suggestion?
    `<div>پیشنهاد آزمایشی مدل: <b>${suggestion.plate}</b> ـ اطمینان محاسباتی ${fa(Math.round(suggestion.confidence*100))}٪</div>
     <div class="danger-note">این مقدار برچسب نیست و روی تصاویر واقعی این بسته خطای مدل زیاد است.</div>
     <button id="draftSuggestion" class="ghost" style="margin-top:8px">کپی فقط به‌عنوان پیش‌نویس</button>`:
    `<span class="danger-note">پیشنهاد مدلی برای این تصویر وجود ندارد.</span>`;
  if(suggestion)$("draftSuggestion").onclick=()=>{setFields(suggestion.plate);toast("فقط به‌عنوان پیش‌نویس کپی شد")};
  $("previous").disabled=position===0;$("next").disabled=position>=visible.length-1;
}
function enteredPlate(){
  const prefix=normalizeDigits($("prefix").value),serial=normalizeDigits($("serial").value),region=normalizeDigits($("region").value);
  const letter=$("letter").value;
  if(!/^\d{2}$/.test(prefix)||!LETTERS.includes(letter)||!/^\d{3}$/.test(serial)||!/^\d{2}$/.test(region))return "";
  return prefix+letter+serial+region;
}
function record(status,plate=""){
  const sample=current();if(!sample)return;
  reviews[sample.sha256]={status,plate,file_name:sample.file_name,reviewed_at:new Date().toISOString()};
  save();nextPending(false);render()
}
function confirm(){
  const plate=enteredPlate();
  if(!plate){toast("شماره پلاک کامل و معتبر نیست");return}
  record("confirmed",plate)
}
function nextPending(showMessage=true){
  if(!visible.length)return;
  for(let offset=1;offset<=visible.length;offset++){
    const index=(position+offset)%visible.length;
    if(statusOf(visible[index])==="pending"){position=index;render();return}
  }
  if(position<visible.length-1){position++;render();return}
  if(showMessage)toast("مورد بررسی‌نشده‌ای در این فیلتر نمانده است")
}
function updateStats(){
  const counts={confirmed:0,unreadable:0,excluded:0,pending:0};
  SAMPLES.forEach(sample=>counts[statusOf(sample)]++);
  const done=SAMPLES.length-counts.pending;
  $("confirmedCount").textContent=fa(counts.confirmed);$("unreadableCount").textContent=fa(counts.unreadable);
  $("pendingCount").textContent=fa(counts.pending);$("progressText").textContent=`${fa(done)} از ${fa(SAMPLES.length)}`;
  $("progressBar").style.width=`${done/SAMPLES.length*100}%`
}
function download(name,text,type){const blob=new Blob([text],{type});const link=document.createElement("a");link.href=URL.createObjectURL(blob);link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000)}
function exportRows(){return SAMPLES.map(sample=>({file_name:sample.file_name,sha256:sample.sha256,status:statusOf(sample),plate:plateOf(sample),quality_bucket:sample.quality_bucket,quality_score:sample.quality_score,reviewed_at:reviews[sample.sha256]?.reviewed_at||"",model_suggestion:sample.suggestion?.plate||"",model_confidence:sample.suggestion?.confidence||0}))}
function exportJson(){
  const payload={schema:1,export_kind:"bcvision-operator-plate-review",exported_at:new Date().toISOString(),source:META,records:exportRows()};
  download(`BCVision_Label_Review_${META.source_id}.json`,JSON.stringify(payload,null,2),"application/json;charset=utf-8");toast("خروجی JSON ساخته شد")
}
function csvCell(value){const text=String(value??"");return `"${text.replaceAll('"','""')}"`}
function exportCsv(){
  const keys=["file_name","sha256","status","plate","quality_bucket","quality_score","reviewed_at","model_suggestion","model_confidence"];
  const rows=[keys.join(","),...exportRows().map(row=>keys.map(key=>csvCell(row[key])).join(","))];
  download(`BCVision_Label_Review_${META.source_id}.csv`,"\ufeff"+rows.join("\r\n"),"text/csv;charset=utf-8");toast("خروجی CSV ساخته شد")
}
async function importJson(file){
  try{
    const payload=JSON.parse(await file.text());
    if(payload?.export_kind!=="bcvision-operator-plate-review"||payload?.source?.source_id!==META.source_id||!Array.isArray(payload.records))throw new Error();
    const allowed=new Set(["pending","confirmed","unreadable","excluded"]),known=new Set(SAMPLES.map(sample=>sample.sha256));
    const restored={};
    for(const row of payload.records){
      if(!known.has(row.sha256)||!allowed.has(row.status))continue;
      if(row.status==="confirmed"&&!/^\d{2}[ابپتثجدزژسشصطعفقکگلمنوهیDS]\d{5}$/.test(row.plate))continue;
      if(row.status!=="pending")restored[row.sha256]={status:row.status,plate:row.plate||"",file_name:row.file_name,reviewed_at:row.reviewed_at||new Date().toISOString()}
    }
    reviews=restored;save();applyFilter();toast("پیشرفت قبلی بازیابی شد")
  }catch{toast("فایل ادامهٔ کار معتبر یا مربوط به این بسته نیست")}
}
for(const letter of ["",...LETTERS]){const option=document.createElement("option");option.value=letter;option.textContent=letter||"—";$("letter").append(option)}
$("filter").onchange=()=>{position=0;applyFilter()};$("search").oninput=()=>{position=0;applyFilter()};
$("previous").onclick=()=>{if(position>0){position--;render()}};$("next").onclick=()=>{if(position<visible.length-1){position++;render()}};
$("nextPending").onclick=()=>nextPending();$("confirm").onclick=confirm;$("unreadable").onclick=()=>record("unreadable");
$("exclude").onclick=()=>record("excluded");$("clear").onclick=()=>record("pending");$("exportJson").onclick=exportJson;$("exportCsv").onclick=exportCsv;
$("importJson").onchange=event=>event.target.files[0]&&importJson(event.target.files[0]);
$("zoom").oninput=()=>{$("plateImage")?.style.setProperty("--zoom",$("zoom").value)};
for(const key of ["prefix","serial","region"])$(key).oninput=event=>event.target.value=normalizeDigits(event.target.value).slice(0,event.target.maxLength);
document.addEventListener("keydown",event=>{
  if(event.target.matches("input,select")){if(event.key==="Enter"){event.preventDefault();confirm()}return}
  if(event.key==="Enter")confirm();else if(event.key.toLowerCase()==="u")record("unreadable");
  else if(event.key.toLowerCase()==="x")record("excluded");else if(event.key==="ArrowLeft"&&position<visible.length-1){position++;render()}
  else if(event.key==="ArrowRight"&&position>0){position--;render()}
});
updateStats();applyFilter();
</script>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a self-contained offline plate-label review page",
    )
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--suggestions", type=Path)
    parser.add_argument("--good-count", type=int, default=0)
    parser.add_argument("--review-count", type=int, default=0)
    parser.add_argument("--ownership-evidence", default="")
    parser.add_argument("--title", default="بازبینی برچسب پلاک‌های BC Vision")
    args = parser.parse_args(argv)
    metadata = build_review_page(
        args.images,
        args.output,
        source_archive=args.source_archive,
        suggestions_path=args.suggestions,
        good_count=args.good_count,
        review_count=args.review_count,
        ownership_evidence=args.ownership_evidence,
        title=args.title,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
