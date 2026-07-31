from __future__ import annotations
import argparse,base64,csv,hashlib,html,json,os,re,shutil,time,zipfile
from datetime import datetime,timezone
from pathlib import Path
import cv2,onnxruntime as ort
from app.ai.evaluation import character_distance
from app.ai.onnx_cct import (CCT_DEFAULT_ALPHABET,CCT_FUSION_GEOMETRIC_MEAN,CCT_FUSION_IDENTITY,CCT_PREPROCESS_DUAL_VIEW,CCT_PREPROCESS_LEGACY,infer_cct_session)
from tools.prepare_ir_lpr_dataset import _annotation_rows
SHA='AD8D77D69CD0C914CB0CB3E0AC4E18709C446F78625A440D8F2D7AD2FB669482'
def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  while b:=f.read(1048576):h.update(b)
 return h.hexdigest().upper()
def extract(z,d):
 d.mkdir(parents=True,exist_ok=True);r=d.resolve()
 with zipfile.ZipFile(z) as a:
  for m in a.infolist():
   t=(d/m.filename).resolve()
   if r not in t.parents and t!=r:raise ValueError('unsafe zip')
   if m.is_dir():t.mkdir(parents=True,exist_ok=True)
   else:
    t.parent.mkdir(parents=True,exist_ok=True)
    with a.open(m) as s,open(t,'wb') as o:shutil.copyfileobj(s,o)
def locate_model(temp):
 roots=[]
 programdata=Path(os.environ.get('ProgramData','C:/ProgramData'))
 roots += [programdata/'BCVision', Path('C:/Users')]
 for key in ('RUNNER_WORKSPACE','RUNNER_TEMP','USERPROFILE'):
  if os.environ.get(key): roots.append(Path(os.environ[key]))
 names=('rc15-cct-xs-ir-lpr-stage4.onnx','bcvision-cct-xs.onnx')
 direct=[]
 for root in roots:
  if not root.exists():continue
  for name in names:
   direct += list(root.rglob(name))
 for candidate in direct:
  try:
   if candidate.is_file() and digest(candidate)==SHA:return candidate
  except OSError:pass
 for root in roots:
  if not root.exists():continue
  try:
   candidates=root.rglob('*.onnx')
   for candidate in candidates:
    try:
     if candidate.is_file() and 2200000<=candidate.stat().st_size<=2700000 and digest(candidate)==SHA:return candidate
    except OSError:pass
  except OSError:pass
 for root in roots:
  if not root.exists():continue
  try:
   installers=root.rglob('BCVision_RC15_Experimental_Model_Installer.cmd')
   for installer in installers:
    try:
     text=installer.read_text(encoding='utf-8',errors='ignore')
     m=re.search(r'(?ms)^::BCVISION_PAYLOAD_BEGIN:ocr\r?\n(.*?)^::BCVISION_PAYLOAD_END:ocr\r?$',text)
     if not m:continue
     candidate=temp/'rc15-cct-xs-ir-lpr-stage4.onnx';candidate.write_bytes(base64.b64decode(re.sub(r'\s','',m.group(1))))
     if digest(candidate)==SHA:return candidate
    except (OSError,ValueError):pass
  except OSError:pass
 raise FileNotFoundError('verified RC15 model not found')
def ci_inputs(temp):
 import gdown
 temp.mkdir(parents=True,exist_ok=True)
 model=locate_model(temp)
 validation=temp/'plate_image_with_dummy-validation.zip';test=temp/'plate_image_with_dummy-test.zip'
 if not validation.is_file():gdown.download(id='1yZYdSNYBPXOoT_QySlH2RVTQS0AySA1z',output=str(validation),quiet=False)
 if not test.is_file():gdown.download(id='1HzZ5vgP5XsmbFCE-n8F70qlRRUtABIq6',output=str(test),quiet=False)
 for archive in (validation,test):
  if not archive.is_file():raise FileNotFoundError(archive)
  with zipfile.ZipFile(archive) as z:
   bad=z.testzip()
   if bad:raise ValueError(f'corrupt zip member: {bad}')
 return model,validation,test
def spec(profile):
 dual=profile==CCT_PREPROCESS_DUAL_VIEW
 return dict(input_width=128,input_height=64,input_layout='nhwc',input_dtype='uint8',image_color_mode='rgb',keep_aspect_ratio=False,interpolation='linear',padding_color=[114,114,114],alphabet=CCT_DEFAULT_ALPHABET,max_plate_slots=8,beam_width=16,top_k=5,preprocess_profile=profile,fusion_method=CCT_FUSION_GEOMETRIC_MEAN if dual else CCT_FUSION_IDENTITY,min_confidence=.58,min_position_confidence=.50 if dual else .42,min_position_margin=.08 if dual else .06,min_hypothesis_margin=.03 if dual else .025,min_view_agreement=.75 if dual else 0)
def img64(im):
 ok,b=cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,86]);return 'data:image/jpeg;base64,'+base64.b64encode(b).decode() if ok else ''
def finish(x):
 n=x['n'];a=x['accepted'];return dict(samples=n,raw_exact_matches=x['exact'],raw_exact_accuracy=x['exact']/n,raw_character_accuracy=x['chars']/(8*n),raw_mean_character_error=x['dist']/n,accepted_samples=a,accepted_exact_matches=x['acc_exact'],accepted_precision=x['acc_exact']/a if a else 0,rejection_rate=(n-a)/n,mean_latency_ms=x['ms']/n)
def bench(name,root,session,input_name):
 rows=[r for r in _annotation_rows(root) if r['plate_text']]
 old,new=spec(CCT_PREPROCESS_LEGACY),spec(CCT_PREPROCESS_DUAL_VIEW)
 counts=[dict(n=0,exact=0,chars=0,dist=0,accepted=0,acc_exact=0,ms=0) for _ in range(2)]
 cmp=dict(improved=0,regressed=0,unchanged_correct=0,unchanged_wrong=0,accepted_became_correct=0,accepted_became_wrong=0)
 records=[];visual=[];limits={'improved':12,'regressed':8,'unchanged_correct':3,'unchanged_wrong':3};seen={k:0 for k in limits}
 for i,r in enumerate(rows,1):
  im=cv2.imread(str(r['image']),cv2.IMREAD_COLOR);exp=r['plate_text'];out=[]
  for j,cfg in enumerate((old,new)):
   t=time.perf_counter();res=infer_cct_session(session,input_name,im,cfg);ms=(time.perf_counter()-t)*1000
   raw=res['hypotheses'][0]['plate_norm'] if res.get('hypotheses') else ''
   c=counts[j];c['n']+=1;c['exact']+=raw==exp;c['chars']+=sum(a==b for a,b in zip(raw,exp));c['dist']+=character_distance(raw,exp);c['accepted']+=bool(res['accepted']);c['acc_exact']+=bool(res['accepted'] and res['plate_norm']==exp);c['ms']+=ms
   out.append((raw,res,ms))
  a,b=out;cat='improved' if a[0]!=exp and b[0]==exp else 'regressed' if a[0]==exp and b[0]!=exp else 'unchanged_correct' if a[0]==exp else 'unchanged_wrong';cmp[cat]+=1
  cmp['accepted_became_correct']+=bool((not a[1]['accepted'] or a[1]['plate_norm']!=exp) and b[1]['accepted'] and b[1]['plate_norm']==exp)
  cmp['accepted_became_wrong']+=bool(a[1]['accepted'] and a[1]['plate_norm']==exp and b[1]['accepted'] and b[1]['plate_norm']!=exp)
  rec=dict(split=name,index=i,expected=exp,legacy_raw=a[0],dual_raw=b[0],legacy_accepted=bool(a[1]['accepted']),dual_accepted=bool(b[1]['accepted']),legacy_confidence=float(a[1].get('confidence',0)),dual_confidence=float(b[1].get('confidence',0)),legacy_latency_ms=a[2],dual_latency_ms=b[2],category=cat)
  records.append(rec)
  if seen[cat]<limits[cat]:visual.append({**rec,'image':img64(im)});seen[cat]+=1
  if i%500==0 or i==len(rows):print(f'{name}: {i}/{len(rows)}',flush=True)
 return dict(legacy=finish(counts[0]),dual_view=finish(counts[1]),comparison=cmp),records,visual
def pct(x):return f'{100*x:.2f}%'
def pp(a,b):return f'{100*(a-b):+.2f} pp'
def report(data,vis):
 sections=[];fa={'improved':'بهبود یافته','regressed':'پسرفت','unchanged_correct':'هر دو درست','unchanged_wrong':'هر دو نادرست'}
 for split in ('validation','test'):
  x=data['splits'][split];o=x['legacy'];n=x['dual_view'];c=x['comparison'];rows=[]
  for r in sorted(vis[split],key=lambda q:({'improved':0,'regressed':1,'unchanged_correct':2,'unchanged_wrong':3}[q['category']],q['index'])):
   rows.append(f"<tr class='{r['category']}'><td>{r['index']}</td><td><img src='{r['image']}'></td><td dir=ltr><b>{html.escape(r['expected'])}</b></td><td dir=ltr>{html.escape(r['legacy_raw'])}<small>{pct(r['legacy_confidence'])} | {'پذیرفته' if r['legacy_accepted'] else 'رد'}</small></td><td dir=ltr>{html.escape(r['dual_raw'])}<small>{pct(r['dual_confidence'])} | {'پذیرفته' if r['dual_accepted'] else 'رد'}</small></td><td>{fa[r['category']]}</td></tr>")
  sections.append(f"<section><h2>{split.title()} — {o['samples']:,} نمونه</h2><div class=metrics><div>دقت کامل قدیم<b>{pct(o['raw_exact_accuracy'])}</b></div><div>دقت کامل جدید<b>{pct(n['raw_exact_accuracy'])}</b><small>{pp(n['raw_exact_accuracy'],o['raw_exact_accuracy'])}</small></div><div>دقت کاراکتر جدید<b>{pct(n['raw_character_accuracy'])}</b><small>{pp(n['raw_character_accuracy'],o['raw_character_accuracy'])}</small></div><div>Precision ثبت جدید<b>{pct(n['accepted_precision'])}</b></div><div>رد خروجی جدید<b>{pct(n['rejection_rate'])}</b></div><div>Latency جدید<b>{n['mean_latency_ms']:.2f} ms</b><small>{n['mean_latency_ms']-o['mean_latency_ms']:+.2f} ms</small></div></div><p>اصلاح‌شده: <b>{c['improved']:,}</b> | پسرفت: <b>{c['regressed']:,}</b> | هر دو درست: {c['unchanged_correct']:,} | هر دو نادرست: {c['unchanged_wrong']:,}</p><table><thead><tr><th>#</th><th>تصویر</th><th>واقعی</th><th>قدیم</th><th>جدید</th><th>نتیجه</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>")
 t=data['splits']['test'];return f"""<!doctype html><html lang=fa dir=rtl><head><meta charset=utf-8><title>گزارش بازآزمایی BC Vision</title><style>body{{margin:0;background:#edf2f7;color:#17212b;font:14px Tahoma,Arial;line-height:1.7}}main{{max-width:1400px;margin:auto;padding:24px}}header,section{{background:white;border:1px solid #dbe5ee;border-radius:16px;padding:20px;margin:16px 0}}header{{background:#163b5c;color:white}}.verdict{{display:inline-block;background:#dff7e9;color:#087747;padding:7px 12px;border-radius:99px;font-weight:bold}}.metrics{{display:grid;grid-template-columns:repeat(6,1fr);gap:8px}}.metrics div{{background:#f4f7fa;padding:10px;border-radius:10px}}.metrics b,.metrics small{{display:block}}.metrics b{{font-size:18px}}table{{width:100%;border-collapse:collapse}}th,td{{padding:8px;border-bottom:1px solid #e5ebf1;text-align:right}}th{{background:#eaf1f7}}td img{{width:235px;height:82px;object-fit:contain;background:#101820;border-radius:6px}}td small{{display:block;color:#607080}}tr.improved{{background:#f0fbf5}}tr.regressed{{background:#fff2f2}}code{{direction:ltr}}@media(max-width:900px){{.metrics{{grid-template-columns:1fr 1fr}}}}</style></head><body><main><header><h1>گزارش تصویری بازآزمایی پلاک‌خوان BC Vision</h1><p>مقایسه مستقیم روش قدیم Stretch و روش جدید Dual-view روی همان فایل‌های Validation و Test، با مدل ثابت RC15 Stage-4.</p><span class=verdict>{html.escape(data['verdict'])}</span><p><code>{data['model']['sha256']}</code></p></header><section><h2>جمع‌بندی Test</h2><p>دقت کامل: <b>{pct(t['legacy']['raw_exact_accuracy'])}</b> ← <b>{pct(t['dual_view']['raw_exact_accuracy'])}</b> ({pp(t['dual_view']['raw_exact_accuracy'],t['legacy']['raw_exact_accuracy'])}). دقت کاراکتری جدید: <b>{pct(t['dual_view']['raw_character_accuracy'])}</b>. روش جدید {t['comparison']['improved']:,} نمونه را اصلاح و در {t['comparison']['regressed']:,} نمونه پسرفت ایجاد کرده است.</p></section>{''.join(sections)}<section><h2>ملاحظه</h2><p>این A/B روی فایل خام یکسان انجام شده؛ بنابراین اختلاف دو روش معتبر است. تعداد نمونه ممکن است با گزارش فیلترشدهٔ مستقل قبلی متفاوت باشد. IR-LPR و این مدل همچنان فقط در حالت تحقیقاتی/Shadow استفاده می‌شوند.</p></section></main></body></html>"""
def main():
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path);p.add_argument('--validation-zip',type=Path);p.add_argument('--test-zip',type=Path);p.add_argument('--ci',action='store_true');p.add_argument('--output',type=Path,required=True);p.add_argument('--source-commit',default='unknown');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 if a.ci:
  temp=Path(os.environ.get('RUNNER_TEMP',str(a.output.parent)))/'bcvision-ir-lpr-inputs';a.model,a.validation_zip,a.test_zip=ci_inputs(temp)
 if not all((a.model,a.validation_zip,a.test_zip)):raise ValueError('model and both archives are required')
 if digest(a.model)!=SHA:raise ValueError('unexpected model hash')
 ext=a.output/'_raw';extract(a.validation_zip,ext/'validation');extract(a.test_zip,ext/'test');s=ort.InferenceSession(str(a.model),providers=['CPUExecutionProvider']);inp=s.get_inputs()[0]
 if s.get_outputs()[0].shape[-1]!=len(CCT_DEFAULT_ALPHABET):raise ValueError('alphabet mismatch')
 splits={};allrows=[];vis={}
 for name in ('validation','test'):
  splits[name],rows,vis[name]=bench(name,ext/name,s,inp.name);allrows+=rows
 d=splits['test']['dual_view']['raw_exact_accuracy']-splits['test']['legacy']['raw_exact_accuracy'];verdict='تغییر جدید روی فایل‌های تست بهتر شده است' if d>.001 else 'تغییر جدید روی فایل‌های تست پسرفت داشته است' if d<-.001 else 'دقت کامل تقریباً ثابت مانده است'
 data=dict(schema=1,generated_at=datetime.now(timezone.utc).isoformat(),source_commit=a.source_commit,model=dict(sha256=SHA,size_bytes=a.model.stat().st_size,input_shape=list(inp.shape),output_shape=list(s.get_outputs()[0].shape)),archives=dict(validation=dict(sha256=digest(a.validation_zip),size_bytes=a.validation_zip.stat().st_size),test=dict(sha256=digest(a.test_zip),size_bytes=a.test_zip.stat().st_size)),splits=splits,verdict=verdict)
 (a.output/'BCVision_IR_LPR_Retest_2026-07-31.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
 with open(a.output/'BCVision_IR_LPR_Retest_All_Predictions.csv','w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
 (a.output/'BCVision_IR_LPR_Retest_Visual_Report.html').write_text(report(data,vis),encoding='utf-8');shutil.rmtree(ext,ignore_errors=True);print(json.dumps(data,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
