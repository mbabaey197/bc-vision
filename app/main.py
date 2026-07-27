from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, StreamingResponse, FileResponse
from app.config import (APP_NAME, COMPANY_NAME, APP_VERSION, DB_PATH, BACKUP_DIR,
    DATA_DIR, STORAGE_CONFIG_PATH, SNAPSHOT_DIR, PLATE_DIR, VIDEO_DIR)
from app.database import (
    backup_database as create_database_backup,
    connect,
    get_setting,
    init_db,
    set_settings_for_database,
    set_setting,
)
from app.security import COOKIE_NAME, create_token, read_token, verify_password, hash_password
from app.streams import manager, CV_OK
from app.license import status as license_status, install_license, activate_online, deactivate_local, machine_id
from html import escape
import time, csv, shutil, os, json, secrets
from datetime import datetime, timedelta
from urllib.parse import quote
from pathlib import Path

try:
    import psutil
except Exception:
    psutil = None

init_db()
app = FastAPI(title=f"{APP_NAME} | {COMPANY_NAME}", docs_url="/api/docs", redoc_url=None)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault(
        "Permissions-Policy",
        "camera=(), microphone=(), geolocation=()",
    )
    return response


@app.get("/api/health")
def health():
    return {
        "service": "bc-vision",
        "status": "ok",
        "version": APP_VERSION,
    }


def _gregorian_to_jalali(gy:int, gm:int, gd:int):
    gdm=[0,31,59,90,120,151,181,212,243,273,304,334]
    gy2=gy+1 if gm>2 else gy
    days=355666+365*gy+((gy2+3)//4)-((gy2+99)//100)+((gy2+399)//400)+gd+gdm[gm-1]
    jy=-1595+33*(days//12053); days%=12053
    jy+=4*(days//1461); days%=1461
    if days>365:
        jy+=(days-1)//365; days=(days-1)%365
    if days<186:
        jm=1+days//31; jd=1+days%31
    else:
        jm=7+(days-186)//30; jd=1+(days-186)%30
    return jy,jm,jd

def _parse_dt(value):
    if not value: return None
    try: return datetime.fromisoformat(str(value).replace('Z','+00:00'))
    except Exception:
        for fmt in ('%Y-%m-%d %H:%M:%S','%Y-%m-%d'):
            try: return datetime.strptime(str(value),fmt)
            except Exception: pass
    return None

def jalali_datetime(value, with_seconds=True):
    dt=_parse_dt(value)
    if not dt: return '—'
    jy,jm,jd=_gregorian_to_jalali(dt.year,dt.month,dt.day)
    t=dt.strftime('%H:%M:%S' if with_seconds else '%H:%M')
    return f'{jy:04d}/{jm:02d}/{jd:02d} - {t}'

def jalali_date(value):
    dt=_parse_dt(value)
    if not dt:return ''
    jy,jm,jd=_gregorian_to_jalali(dt.year,dt.month,dt.day)
    return f'{jy:04d}/{jm:02d}/{jd:02d}'

def normalize_plate(text):
    if not text:return ''
    trans=str.maketrans('۰۱۲۳۴۵۶۷۸۹','0123456789')
    return ''.join(ch for ch in str(text).translate(trans).upper() if ch.isalnum() or '\u0600'<=ch<='\u06ff')

def event_status_badge(status):
    labels={'allowed':'مجاز','blocked':'غیرمجاز','vip':'VIP','unknown':'ثبت‌نشده'}
    classes={'allowed':'ok','blocked':'bad','vip':'vip','unknown':'muted'}
    st=status if status in labels else 'unknown'
    return f"<span class='status-pill {classes[st]}'>{labels[st]}</span>"

CSS = """<style>
:root{--bc-navy:#071b3f;--bc-navy2:#0b2e63;--bc-blue:#087cf0;--bc-cyan:#11a8f7;--bc-bg:#f3f6fb;--bc-surface:#fff;--bc-surface2:#f7f9fc;--bc-border:#e1e8f0;--bc-text:#172033;--bc-muted:#69778d;--bc-radius:16px;--bc-shadow:0 10px 30px rgba(10,38,78,.08);--sidebar:258px}
[data-theme=dark]{--bc-bg:#0b1220;--bc-surface:#111b2d;--bc-surface2:#162238;--bc-border:#25344d;--bc-text:#e8eef8;--bc-muted:#9aabc3;--bc-shadow:0 12px 34px rgba(0,0,0,.25)}
*{box-sizing:border-box}html{direction:rtl;scroll-behavior:smooth}body{margin:0;background:var(--bc-bg);color:var(--bc-text);font-family:Tahoma,"Segoe UI",Arial,sans-serif;font-size:14px;line-height:1.75;text-rendering:optimizeLegibility;-webkit-font-smoothing:antialiased;transition:background .2s,color .2s}a{color:inherit}
.app-shell{min-height:100vh}.sidebar{position:fixed;top:0;right:0;bottom:0;width:var(--sidebar);background:linear-gradient(175deg,var(--bc-navy),#06152f 75%);color:#dcecff;z-index:1100;padding:18px 13px;display:flex;flex-direction:column;box-shadow:-6px 0 24px rgba(0,0,0,.12);transition:width .22s,transform .22s}.sidebar.collapsed{width:84px}.brand-row{display:flex;align-items:center;gap:11px;padding:4px 7px 20px;border-bottom:1px solid rgba(255,255,255,.1);text-decoration:none;color:#fff;overflow:hidden}.brand-mark{min-width:45px;width:45px;height:45px;border-radius:14px;background:linear-gradient(145deg,var(--bc-cyan),var(--bc-blue));display:grid;place-items:center;font-weight:900;font-size:19px;box-shadow:0 7px 20px rgba(17,168,247,.28)}.brand-copy{white-space:nowrap}.brand-copy b{display:block;font-size:17px;line-height:1.2}.brand-copy small{opacity:.65;font-size:10px}.nav-menu{margin-top:16px;display:flex;flex-direction:column;gap:5px}.nav-menu a{display:flex;align-items:center;gap:12px;padding:11px 13px;border-radius:11px;text-decoration:none;color:#cfe2fb;white-space:nowrap;overflow:hidden;transition:.18s}.nav-menu a:hover,.nav-menu a.active{background:linear-gradient(100deg,rgba(17,168,247,.23),rgba(8,124,240,.08));color:#fff}.nav-icon{min-width:25px;width:25px;text-align:center;font-size:19px}.sidebar-foot{margin-top:auto}.sidebar-foot a{display:flex;align-items:center;gap:12px;padding:11px 13px;border-radius:11px;text-decoration:none;color:#cfe2fb}.sidebar.collapsed .brand-copy,.sidebar.collapsed .nav-label{opacity:0;width:0;overflow:hidden}.sidebar-toggle{position:absolute;left:-15px;top:78px;width:30px;height:30px;border-radius:50%;border:2px solid var(--bc-bg);background:var(--bc-blue);color:#fff;cursor:pointer;display:grid;place-items:center;padding:0;box-shadow:0 5px 15px rgba(0,0,0,.2)}
.main{margin-right:var(--sidebar);min-height:100vh;transition:margin-right .22s}.main.collapsed{margin-right:84px}.topbar{height:70px;background:color-mix(in srgb,var(--bc-surface) 92%,transparent);backdrop-filter:blur(12px);border-bottom:1px solid var(--bc-border);display:flex;align-items:center;gap:12px;padding:0 24px;position:sticky;top:0;z-index:900}.top-title{font-size:18px;font-weight:900;color:var(--bc-navy);margin-left:auto}.resource-strip{display:flex;align-items:center;gap:7px;direction:ltr}.resource-chip{display:flex;align-items:center;gap:5px;min-width:66px;padding:5px 8px;border:1px solid var(--bc-border);background:var(--bc-surface);border-radius:10px;font-size:12px;font-weight:800}.resource-dot{width:8px;height:8px;border-radius:50%;background:#22a06b;box-shadow:0 0 0 3px rgba(34,160,107,.12)}.resource-chip.warn .resource-dot{background:#e5a11a}.resource-chip.danger .resource-dot{background:#d64545}.storage-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.drive-card{border:1px solid var(--bc-border);background:var(--bc-surface2);border-radius:13px;padding:14px}.storage-progress{height:10px;background:var(--bc-border);border-radius:8px;overflow:hidden;margin:9px 0}.storage-progress span{display:block;height:100%;background:linear-gradient(90deg,var(--bc-blue),var(--bc-cyan));border-radius:8px}[data-theme=dark] .top-title{color:#eaf2ff}.top-action{width:40px;height:40px;border-radius:11px;border:1px solid var(--bc-border);background:var(--bc-surface);color:var(--bc-text);display:grid;place-items:center;cursor:pointer;box-shadow:none;padding:0}.user-chip{display:flex;align-items:center;gap:9px;border:1px solid var(--bc-border);background:var(--bc-surface);padding:6px 10px;border-radius:12px}.avatar{width:29px;height:29px;border-radius:9px;background:linear-gradient(135deg,var(--bc-blue),var(--bc-cyan));color:#fff;display:grid;place-items:center;font-weight:900}.wrap{max-width:1550px;margin:auto;padding:25px}.page-title{font-weight:900;font-size:28px;margin:0;color:var(--bc-navy)}[data-theme=dark] .page-title,[data-theme=dark] h1{color:#edf4ff}h1{font-size:27px;font-weight:900;color:var(--bc-navy)}h3{font-size:18px;font-weight:800}.page-sub{color:var(--bc-muted);margin:2px 0 0}
.card{background:var(--bc-surface);border:1px solid var(--bc-border);border-radius:var(--bc-radius);padding:20px;box-shadow:var(--bc-shadow);margin-bottom:17px}.login{max-width:430px;margin:7vh auto;padding:30px}.login .brand{text-align:center;font-size:28px;margin-bottom:3px}.login .muted{text-align:center}.brand{font-size:25px;font-weight:900;color:var(--bc-navy)}.muted{color:var(--bc-muted)}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px}.stats-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:15px;margin-bottom:17px}.stat-card{position:relative;overflow:hidden}.stat-card:after{content:"";position:absolute;width:90px;height:90px;border-radius:50%;left:-25px;top:-28px;background:rgba(8,124,240,.07)}.stat-head{display:flex;align-items:center;justify-content:space-between}.stat-icon{width:43px;height:43px;border-radius:13px;background:rgba(8,124,240,.1);color:var(--bc-blue);display:grid;place-items:center;font-size:21px}.stat{font-size:31px;font-weight:900;color:var(--bc-text);margin-top:5px;line-height:1.2}.trend{font-size:12px;color:var(--bc-muted)}
label{display:block;font-weight:700;color:var(--bc-text);margin-bottom:3px}input,select,textarea,button{font-family:inherit;font-size:14px}input:not([type=checkbox]),select,textarea{width:100%;padding:10px 12px;border:1px solid var(--bc-border);border-radius:10px;margin:5px 0 13px;background:var(--bc-surface);color:var(--bc-text);outline:0;transition:.18s}input:focus,select:focus,textarea:focus{border-color:var(--bc-blue);box-shadow:0 0 0 3px rgba(8,124,240,.13)}button,.btn{display:inline-flex;align-items:center;justify-content:center;gap:6px;border:0;background:linear-gradient(135deg,var(--bc-blue),#075dc5);color:#fff!important;padding:9px 16px;border-radius:9px;text-decoration:none;cursor:pointer;font-weight:700;box-shadow:0 4px 12px rgba(8,124,240,.18);transition:.18s}button:hover,.btn:hover{transform:translateY(-1px);filter:brightness(1.04)}.secondary{background:#65738a!important;box-shadow:none}.danger{background:#c63838!important;box-shadow:none}.ok{color:#168458}.bad{color:#d34747}.replay-layout{display:grid;grid-template-columns:minmax(0,1.7fr) minmax(300px,.8fr);gap:18px}.video-panel video{width:100%;max-height:70vh;background:#05080d;border-radius:14px}.replay-controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-top:12px}.replay-controls button{padding:8px 12px}.event-meta{display:grid;grid-template-columns:1fr 1fr;gap:10px}.meta-item{padding:10px;border:1px solid var(--bc-border);border-radius:10px;background:var(--bc-surface2)}.meta-item small{display:block;color:var(--bc-muted)}.detail-images{display:grid;grid-template-columns:1fr 1fr;gap:10px}.detail-images img{width:100%;height:165px;object-fit:contain;background:#0b1220;border-radius:11px}.time-badge{font-size:20px;font-weight:900;color:var(--bc-blue)}@media(max-width:900px){.replay-layout{grid-template-columns:1fr}.event-meta{grid-template-columns:1fr}}.alert{padding:12px 15px;border-radius:10px;background:#fff1f2;color:#9b1c2d;border:1px solid #ffd5da;margin-bottom:14px}.toast-box{position:fixed;left:22px;bottom:22px;z-index:2000;min-width:270px;background:var(--bc-surface);border:1px solid var(--bc-border);border-right:4px solid #168458;border-radius:12px;padding:12px 15px;box-shadow:var(--bc-shadow);animation:toastin .3s ease}.toast-box.hide{opacity:0;transform:translateY(12px);transition:.35s}@keyframes toastin{from{opacity:0;transform:translateY(15px)}}
.table-wrap{overflow:auto}table{width:100%;border-collapse:separate;border-spacing:0;min-width:700px}th,td{padding:12px 11px;border-bottom:1px solid var(--bc-border);text-align:right;vertical-align:middle}th{font-size:13px;color:var(--bc-muted);background:var(--bc-surface2);font-weight:800}tr:last-child td{border-bottom:0}tbody tr:hover td{background:var(--bc-surface2)}.code{direction:ltr;text-align:left;font-family:Consolas,"Courier New",monospace}.live-grid{display:grid;grid-template-columns:repeat(var(--cols),minmax(0,1fr));gap:14px}.camera-tile{background:#101820;border-radius:14px;overflow:hidden;position:relative;min-height:180px;box-shadow:var(--bc-shadow)}.camera-tile img{display:block;width:100%;aspect-ratio:16/9;object-fit:cover;background:#101820}.camera-head{display:flex;justify-content:space-between;align-items:center;color:#fff;padding:9px 12px;background:#162631}.badge{font-size:11px;padding:4px 9px;border-radius:20px;background:#657180}.badge.online{background:#168458}.toolbar{display:flex;gap:9px;align-items:center;flex-wrap:wrap;margin-bottom:17px}.toolbar h1{margin-left:auto;margin-bottom:0}.toolbar select{width:auto;margin:0}.two-col{display:grid;grid-template-columns:1fr 1fr;gap:16px}.system-bars{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.meter-label{display:flex;justify-content:space-between;margin-bottom:7px}.meter{height:8px;border-radius:8px;background:var(--bc-border);overflow:hidden}.meter span{display:block;height:100%;background:linear-gradient(90deg,var(--bc-blue),var(--bc-cyan));border-radius:8px;transition:width .4s}.empty-state{text-align:center;padding:38px 20px}.grid-switch{display:flex;background:var(--bc-surface);border:1px solid var(--bc-border);border-radius:10px;padding:3px;gap:3px}.grid-switch button{box-shadow:none;background:transparent;color:var(--bc-muted)!important;padding:6px 10px}.grid-switch button.active{background:var(--bc-blue);color:#fff!important}.mobile-menu{display:none}
@media(max-width:1150px){.stats-grid{grid-template-columns:repeat(2,1fr)}.system-bars{grid-template-columns:1fr}}
@media(max-width:900px){.resource-chip span.label{display:none}.resource-chip{min-width:auto}}
@media(max-width:760px){.sidebar{transform:translateX(110%);width:258px}.sidebar.mobile-open{transform:translateX(0)}.sidebar-toggle{display:none}.main,.main.collapsed{margin-right:0}.mobile-menu{display:grid}.topbar{padding:0 12px}.user-chip span:last-child{display:none}.wrap{padding:17px 12px}.stats-grid{grid-template-columns:1fr 1fr;gap:10px}.card{padding:15px}.live-grid{grid-template-columns:1fr!important}.two-col,.storage-grid{grid-template-columns:1fr}.toolbar h1{width:100%;font-size:23px}.login{margin:4vh 12px}}
@media(max-width:440px){.stats-grid{grid-template-columns:1fr}.top-title{font-size:15px}}
.thumb{width:110px;height:62px;object-fit:cover;border-radius:9px;border:1px solid var(--bc-border);background:#eef2f7;cursor:pointer}.plate-thumb{width:130px;height:48px}.status-pill{display:inline-block;padding:3px 9px;border-radius:999px;font-size:12px;font-weight:900}.status-pill.ok{background:#e5f7ef;color:#147a50}.status-pill.bad{background:#ffe8e8;color:#b42318}.status-pill.vip{background:#fff3cd;color:#8a6100}.event-blocked{background:rgba(214,69,69,.07)}.event-vip{background:rgba(229,161,26,.08)}.filter-grid{display:grid;grid-template-columns:repeat(5,minmax(140px,1fr));gap:10px;align-items:end}.modal-img{position:fixed;inset:0;background:rgba(0,0,0,.78);z-index:5000;display:none;place-items:center;padding:30px}.modal-img.open{display:grid}.modal-img img{max-width:95vw;max-height:90vh;border-radius:14px}.modal-img button{position:absolute;top:20px;left:20px}@media(max-width:900px){.filter-grid{grid-template-columns:1fr 1fr}}
.login-page{min-height:100vh;display:grid;grid-template-columns:minmax(320px,520px) 1fr;background:linear-gradient(135deg,#071b3f 0%,#0b2e63 52%,#087cf0 100%);direction:ltr;overflow:hidden}.login-panel{direction:rtl;background:var(--bc-surface);padding:clamp(26px,5vw,68px);display:flex;align-items:center;justify-content:center;box-shadow:20px 0 60px rgba(0,0,0,.18);z-index:2}.login-box{width:100%;max-width:410px}.login-logo{display:flex;align-items:center;gap:13px;margin-bottom:34px}.login-logo .brand-mark{width:58px;height:58px;min-width:58px;font-size:22px}.login-logo h1{margin:0;font-size:28px}.login-logo p{margin:0;color:var(--bc-muted)}.login-title{font-size:25px;font-weight:900;margin:0 0 5px}.login-subtitle{color:var(--bc-muted);margin:0 0 26px}.password-wrap{position:relative}.password-wrap input{padding-left:48px}.password-toggle{position:absolute;left:7px;top:10px;width:36px;height:36px;background:transparent!important;color:var(--bc-muted)!important;box-shadow:none;padding:0}.password-toggle:hover{transform:none;background:var(--bc-surface2)!important}.login-submit{width:100%;height:46px;font-size:15px;margin-top:5px}.login-help{display:flex;justify-content:space-between;gap:12px;margin-top:17px;font-size:12px;color:var(--bc-muted)}.login-visual{direction:rtl;color:#fff;display:flex;align-items:center;justify-content:center;padding:60px;position:relative}.login-visual:before,.login-visual:after{content:'';position:absolute;border-radius:50%;background:rgba(255,255,255,.08)}.login-visual:before{width:420px;height:420px;left:-130px;top:-170px}.login-visual:after{width:300px;height:300px;right:8%;bottom:-140px}.login-hero{max-width:670px;position:relative;z-index:1}.login-hero h2{font-size:clamp(32px,4vw,54px);font-weight:900;line-height:1.35;margin:0 0 16px}.login-hero p{font-size:17px;opacity:.82;max-width:570px}.login-features{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin-top:34px}.login-feature{padding:17px;border:1px solid rgba(255,255,255,.16);background:rgba(255,255,255,.08);backdrop-filter:blur(10px);border-radius:15px}.login-feature b{display:block;font-size:15px;margin-bottom:3px}.login-feature span{font-size:12px;opacity:.75}.login-version{position:absolute;bottom:24px;left:30px;opacity:.62;font-size:12px}@media(max-width:900px){.login-page{grid-template-columns:1fr}.login-visual{display:none}.login-panel{min-height:100vh;padding:24px}.login-help{flex-direction:column}}
.anpr-status{display:block;padding:7px 12px;color:#c8d5df;background:#0c141a;font-size:11px;line-height:1.7;border-top:1px solid #263945}.anpr-status.bad{color:#ffb4ab;background:#301716}
</style>"""

BOOTSTRAP = "<link href='https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.rtl.min.css' rel='stylesheet'>"

NAV_ITEMS = [
    ('/dashboard','⌂','داشبورد و نمایش زنده'),('/cameras','▣','دوربین‌ها'),('/events','▤','ترددها و گزارش‌ها'),
    ('/license','◆','لایسنس'),('/users','👥','کاربران'),('/audit','☷','لاگ فعالیت‌ها'),('/settings','⚙','تنظیمات')
]

def page(title, body, username=None, request=None):
    shell_start=shell_end=''
    if username:
        path = request.url.path if request else ''
        links=''.join(f"<a href='{href}' class='{'active' if (path==href or (href!='/dashboard' and path.startswith(href))) else ''}'><span class='nav-icon'>{icon}</span><span class='nav-label'>{label}</span></a>" for href,icon,label in NAV_ITEMS)
        shell_start=f"""<div class='app-shell'><aside class='sidebar' id='sidebar'><a class='brand-row' href='/dashboard'><span class='brand-mark'>BC</span><span class='brand-copy'><b>BC Vision</b><small>{COMPANY_NAME}</small></span></a><button class='sidebar-toggle' id='sidebarToggle' title='جمع کردن منو'>‹</button><nav class='nav-menu'>{links}</nav><div class='sidebar-foot'><a href='/logout'><span class='nav-icon'>⇥</span><span class='nav-label'>خروج از حساب</span></a></div></aside><main class='main' id='main'><header class='topbar'><button class='top-action mobile-menu' id='mobileMenu'>☰</button><div class='top-title'>{escape(title)}</div><div class='resource-strip'><div class='resource-chip' id='head-cpu'><span class='resource-dot'></span><span class='label'>CPU</span><span id='head-cpu-value'>—</span></div><div class='resource-chip' id='head-ram'><span class='resource-dot'></span><span class='label'>RAM</span><span id='head-ram-value'>—</span></div><div class='resource-chip' id='head-disk'><span class='resource-dot'></span><span class='label'>DISK</span><span id='head-disk-value'>—</span></div></div><button class='top-action' id='themeToggle' title='حالت تاریک'>◐</button><div class='user-chip'><span class='avatar'>{escape(username[:1].upper())}</span><span>{escape(username)}</span></div></header>"""
        shell_end='</main></div>'
    common_js="""<script>
(function(){
 const root=document.documentElement, saved=localStorage.getItem('bc-theme')||'light';root.dataset.theme=saved;
 const st=document.getElementById('sidebar'),mn=document.getElementById('main');
 if(localStorage.getItem('bc-sidebar')==='collapsed'){st?.classList.add('collapsed');mn?.classList.add('collapsed')}
 document.getElementById('sidebarToggle')?.addEventListener('click',()=>{st.classList.toggle('collapsed');mn.classList.toggle('collapsed');localStorage.setItem('bc-sidebar',st.classList.contains('collapsed')?'collapsed':'open')});
 document.getElementById('mobileMenu')?.addEventListener('click',()=>st.classList.toggle('mobile-open'));
 document.getElementById('themeToggle')?.addEventListener('click',()=>{root.dataset.theme=root.dataset.theme==='dark'?'light':'dark';localStorage.setItem('bc-theme',root.dataset.theme)});
 const toast=document.querySelector('.toast-box');if(toast)setTimeout(()=>{toast.classList.add('hide');setTimeout(()=>toast.remove(),400)},3200);
 async function updateHeaderResources(){try{const r=await fetch('/api/system/status');if(!r.ok)return;const x=await r.json();for(const k of ['cpu','ram','disk']){const v=Math.round(x[k]||0), el=document.getElementById('head-'+k), val=document.getElementById('head-'+k+'-value');if(val)val.textContent=v+'%';if(el){el.classList.toggle('warn',v>=80&&v<90);el.classList.toggle('danger',v>=90)}}}catch(e){}}
 updateHeaderResources();setInterval(updateHeaderResources,5000);
})();
</script>"""
    return HTMLResponse(f"<!doctype html><html lang='fa' dir='rtl'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><meta name='theme-color' content='#071b3f'><title>{escape(title)} | BC Vision</title>{BOOTSTRAP}{CSS}</head><body>{shell_start}{body}{shell_end}{common_js}</body></html>")

def user(request): return read_token(request)
def auth(request):
    username=read_token(request)
    if not username:return None
    with connect() as con:
        row=con.execute('SELECT * FROM users WHERE username=? AND is_active=1',(username,)).fetchone()
    return row['username'] if row else None

def current_user(request):
    username=auth(request)
    if not username:return None
    with connect() as con:return con.execute('SELECT * FROM users WHERE username=?',(username,)).fetchone()

def require_admin(request):
    u=current_user(request)
    return bool(u and (u['role']=='admin' or u['is_admin']))

ROLE_PERMISSIONS = {
    'admin': {'system.manage', 'camera.manage', 'license.manage', 'watchlist.manage', 'video.process'},
    'system': {'system.manage', 'camera.manage', 'license.manage', 'watchlist.manage', 'video.process'},
    'operator': {'watchlist.manage', 'video.process'},
    'guard': set(),
}

def has_permission(request, permission):
    u=current_user(request)
    if not u:
        return False
    role='admin' if u['is_admin'] else (u['role'] or 'guard')
    return permission in ROLE_PERMISSIONS.get(role,set())

def access_denied(message='شما اجازه انجام این عملیات را ندارید.'):
    return HTMLResponse(
        f"<!doctype html><html lang='fa' dir='rtl'><head><meta charset='utf-8'>"
        f"<title>عدم دسترسی | BC Vision</title>{CSS}</head><body>"
        f"<div class='wrap'><div class='card alert'>{escape(message)}</div>"
        "<a class='btn secondary' href='/dashboard'>بازگشت به داشبورد</a></div></body></html>",
        status_code=403,
    )

def audit(request, action, details=''):
    username=auth(request) or 'anonymous'
    ip=request.client.host if request and request.client else ''
    with connect() as con:con.execute('INSERT INTO audit_logs(username,action,details,ip_address) VALUES(?,?,?,?)',(username,action,details,ip))

def camera_rows(enabled_only=False):
    with connect() as con:
        q="SELECT * FROM cameras" + (" WHERE enabled=1" if enabled_only else "") + " ORDER BY sort_order,id"
        return con.execute(q).fetchall()

@app.get('/')
def root(request:Request): return RedirectResponse('/dashboard' if user(request) else '/login',302)
@app.get('/login')
def login_form(request:Request,error:str='',next:str='/dashboard',logged_out:int=0):
    if user(request): return RedirectResponse('/dashboard',302)
    safe_next=next if next.startswith('/') and not next.startswith('//') else '/dashboard'
    alert="<div class='alert'>نام کاربری یا رمز عبور صحیح نیست.</div>" if error else ''
    notice="<div class='alert' style='background:#eaf8f1;color:#146b45;border-color:#bdebd5'>با موفقیت از حساب خارج شدید.</div>" if logged_out else ''
    body=f"""<div class='login-page'>
    <section class='login-panel'><div class='login-box'>
      <div class='login-logo'><span class='brand-mark'>BC</span><div><h1>BC Vision</h1><p>{escape(COMPANY_NAME)}</p></div></div>
      <h2 class='login-title'>ورود به سامانه</h2><p class='login-subtitle'>برای مدیریت دوربین‌ها و گزارش تردد وارد حساب خود شوید.</p>
      {alert}{notice}
      <form method='post' action='/login' autocomplete='on'>
        <input type='hidden' name='next' value='{escape(safe_next)}'>
        <label for='username'>نام کاربری</label><input id='username' name='username' autocomplete='username' autofocus required placeholder='نام کاربری را وارد کنید'>
        <label for='password'>رمز عبور</label><div class='password-wrap'><input id='password' type='password' name='password' autocomplete='current-password' required placeholder='رمز عبور را وارد کنید'><button type='button' class='password-toggle' id='passwordToggle' aria-label='نمایش رمز'>◉</button></div>
        <button class='login-submit' type='submit'>ورود به BC Vision</button>
      </form>
      <div class='login-help'><span>ورود اولیه: <b>admin</b> / <b>123456</b></span><span>پس از ورود رمز را تغییر دهید.</span></div>
    </div></section>
    <section class='login-visual'><div class='login-hero'><h2>مدیریت هوشمند<br>نظارت و تردد خودرو</h2><p>مشاهده زنده دوربین‌ها، پلاک‌خوانی، جست‌وجوی رویدادها و گزارش‌گیری در یک محیط یکپارچه.</p><div class='login-features'><div class='login-feature'><b>نمایش زنده</b><span>مدیریت هم‌زمان چند دوربین</span></div><div class='login-feature'><b>پلاک‌خوان هوشمند</b><span>ثبت و جست‌وجوی سریع ترددها</span></div><div class='login-feature'><b>گزارش‌های دقیق</b><span>فیلتر بر اساس دوربین، رنگ و نوع خودرو</span></div><div class='login-feature'><b>امنیت حساب</b><span>نشست رمزنگاری‌شده و خروج امن</span></div></div></div><span class='login-version'>نسخه {APP_VERSION}</span></section>
    </div><script>document.getElementById('passwordToggle').addEventListener('click',function(){{const p=document.getElementById('password');p.type=p.type==='password'?'text':'password';this.textContent=p.type==='password'?'◉':'⊘';}});</script>"""
    return page('ورود',body)
@app.post('/login')
def login(request:Request,username:str=Form(...),password:str=Form(...),next:str=Form('/dashboard')):
    username=username.strip()
    safe_next=next if next.startswith('/') and not next.startswith('//') else '/dashboard'
    with connect() as con:
        u=con.execute('SELECT * FROM users WHERE username=?',(username,)).fetchone()
        now=datetime.now()
        locked=False
        if u and u['locked_until']:
            try: locked=datetime.fromisoformat(u['locked_until'])>now
            except Exception: locked=False
        if not u or not u['is_active'] or locked or not verify_password(password,u['password_hash']):
            if u and not locked:
                attempts=int(u['failed_attempts'] or 0)+1
                lock_until=(now+timedelta(minutes=15)).isoformat(timespec='seconds') if attempts>=5 else None
                con.execute('UPDATE users SET failed_attempts=?,locked_until=? WHERE id=?',(0 if lock_until else attempts,lock_until,u['id']))
            con.execute('INSERT INTO audit_logs(username,action,details,ip_address) VALUES(?,?,?,?)',(username or 'unknown','login_failed','ورود ناموفق',request.client.host if request.client else ''))
            time.sleep(0.35)
            return RedirectResponse(f'/login?error=1&next={quote(safe_next)}',303)
        con.execute('UPDATE users SET failed_attempts=0,locked_until=NULL,last_login=CURRENT_TIMESTAMP WHERE id=?',(u['id'],))
        con.execute('INSERT INTO audit_logs(username,action,details,ip_address) VALUES(?,?,?,?)',(username,'login','ورود موفق',request.client.host if request.client else ''))
    r=RedirectResponse(safe_next,303)
    r.set_cookie(COOKIE_NAME,create_token(u['username']),httponly=True,samesite='lax',secure=False,max_age=43200,path='/')
    return r
@app.get('/logout')
def logout(request:Request):
    if auth(request):audit(request,'logout','خروج از حساب')
    r=RedirectResponse('/login?logged_out=1',302);r.delete_cookie(COOKIE_NAME,path='/');return r

@app.get('/dashboard')
def dashboard(request:Request):
    u=auth(request)
    if not u:return RedirectResponse('/login',302)
    cams=camera_rows(True); cols=max(1,min(4,int(get_setting('dashboard_grid','2'))))
    with connect() as con:
        total=con.execute('SELECT COUNT(*) c FROM plate_events').fetchone()['c']
        today=con.execute("SELECT COUNT(*) c FROM plate_events WHERE date(created_at)=date('now','localtime')").fetchone()['c']
        alerts=con.execute("SELECT COUNT(*) c FROM plate_events WHERE confidence < 0.70 AND date(created_at)=date('now','localtime')").fetchone()['c']
        recent=con.execute("SELECT plate_text,camera_name,confidence,created_at FROM plate_events ORDER BY id DESC LIMIT 6").fetchall()
    lic=license_status()
    tiles=''.join(f"<div class='camera-tile'><div class='camera-head'><span>{escape(c['name'])}</span><span class='badge' id='st-{c['id']}'>در حال اتصال</span></div><img loading='lazy' src='/live/{c['id']}?t={int(time.time())}' alt='{escape(c['name'])}'><span class='anpr-status' id='anpr-{c['id']}'>پلاک‌خوان: در انتظار اولین فریم</span><div class='camera-head'><small>{escape(c['location'] or 'بدون موقعیت')}</small><a style='color:#bdefff' href='/cameras/{c['id']}/snapshot'>گرفتن عکس</a></div></div>" for c in cams)
    if not tiles: tiles="<div class='card empty-state'><h3>هنوز دوربینی فعال نیست</h3><p class='muted'>برای شروع، دوربین واقعی خود را اضافه کنید.</p><a class='btn' href='/cameras/new'>افزودن اولین دوربین</a></div>"
    ids=','.join(str(c['id']) for c in cams)
    recent_rows=''.join(f"<tr><td><b>{escape(r['plate_text'] or '—')}</b></td><td>{escape(r['camera_name'] or '—')}</td><td>{int((r['confidence'] or 0)*100)}٪</td><td>{jalali_datetime(r['created_at'],False)}</td></tr>" for r in recent) or "<tr><td colspan='4'>هنوز پلاکی ثبت نشده است.</td></tr>"
    js=f"""<script>
const ids=[{ids}];
async function cameraStatus(){{for(const id of ids){{try{{let r=await fetch('/api/cameras/'+id+'/status');let s=await r.json();let e=document.getElementById('st-'+id),a=document.getElementById('anpr-'+id),n=v=>Number(v||0).toLocaleString('fa-IR');e.textContent=s.online?'آنلاین':'آفلاین';e.className='badge '+(s.online?'online':'');const p=s.anpr||{{}},m=p.models||{{}};if(!m.ready){{a.textContent='پلاک‌خوان آماده نیست: مدل تشخیص یا OCR نصب نشده است';a.className='anpr-status bad'}}else if(p.last_error){{a.textContent='خطای پلاک‌خوان: '+p.last_error;a.className='anpr-status bad'}}else{{a.textContent='پردازش: '+n(p.processed_frames)+' فریم | تشخیص: '+n(p.detected_candidates)+' | ثبت: '+n(p.emitted_events);a.className='anpr-status'}}}}catch(e){{}}}}}}
function setGrid(n){{document.getElementById('liveGrid').style.setProperty('--cols',n);document.querySelectorAll('.grid-switch button').forEach(b=>b.classList.toggle('active',Number(b.dataset.n)===n));localStorage.setItem('bc-grid',n)}}
const savedGrid=Number(localStorage.getItem('bc-grid')||{cols});setGrid(savedGrid);cameraStatus();setInterval(cameraStatus,4000);
</script>"""
    valid_class='ok' if lic['valid'] else 'bad'
    body=f"""<div class='wrap'><div class='toolbar'><div style='margin-left:auto'><h1 class='page-title'>داشبورد</h1><p class='page-sub'>نمای کلی وضعیت سامانه و تصاویر زنده</p></div><a class='btn' href='/cameras/new'>＋ افزودن دوربین</a></div>
    <div class='stats-grid'>
      <div class='card stat-card'><div class='stat-head'><span class='muted'>دوربین‌های فعال</span><span class='stat-icon'>📷</span></div><div class='stat'>{len(cams)}</div><div class='trend'>از ظرفیت {lic['camera_limit']} دوربین</div></div>
      <div class='card stat-card'><div class='stat-head'><span class='muted'>تردد امروز</span><span class='stat-icon'>🚘</span></div><div class='stat'>{today}</div><div class='trend'>ثبت‌شده از ابتدای امروز</div></div>
      <div class='card stat-card'><div class='stat-head'><span class='muted'>هشدارهای امروز</span><span class='stat-icon'>⚠</span></div><div class='stat'>{alerts}</div><div class='trend'>پلاک‌های با اطمینان پایین</div></div>
      <div class='card stat-card'><div class='stat-head'><span class='muted'>وضعیت لایسنس</span><span class='stat-icon'>◆</span></div><div class='{valid_class}' style='font-size:20px;font-weight:900;margin-top:10px'>{escape(lic['plan'])}</div><div class='trend'>{escape(lic['message'])}</div></div>
    </div>
    <div class='card'><div class='toolbar'><div style='margin-left:auto'><h3 style='margin:0'>نمایش زنده</h3><span class='muted'>تصاویر دوربین‌های فعال</span></div><div class='grid-switch'><button data-n='1' onclick='setGrid(1)'>۱</button><button data-n='2' onclick='setGrid(2)'>۴</button><button data-n='3' onclick='setGrid(3)'>۹</button><button data-n='4' onclick='setGrid(4)'>۱۶</button></div><button class='secondary' onclick='document.documentElement.requestFullscreen?.()'>تمام‌صفحه</button></div><div class='live-grid' id='liveGrid' style='--cols:{cols}'>{tiles}</div></div>
    <div class='two-col'><div class='card'><h3>آخرین ترددها</h3><div class='table-wrap'><table><tr><th>پلاک</th><th>دوربین</th><th>اطمینان</th><th>زمان</th></tr>{recent_rows}</table></div><div style='margin-top:12px'><a class='btn secondary' href='/events'>مشاهده همه گزارش‌ها</a></div></div>
    <div class='card'><h3>وضعیت سامانه</h3><p class='muted'>موتور تصویر: <b class='{'ok' if CV_OK else 'bad'}'>{'آماده' if CV_OK else 'غیرفعال'}</b></p><p class='muted'>کل ترددهای ثبت‌شده: <b>{total}</b></p><p class='muted'>مسیر ذخیره‌سازی: <b class='code'>{escape(str(DATA_DIR))}</b></p><a class='btn secondary' href='/settings#storage'>مدیریت ذخیره‌سازی</a></div></div>{js}</div>"""
    return page('داشبورد',body,u,request)

@app.get('/live/{camera_id}')
def live(camera_id:int,request:Request):
    if not auth(request): return RedirectResponse('/login',302)
    with connect() as con:c=con.execute('SELECT * FROM cameras WHERE id=? AND enabled=1',(camera_id,)).fetchone()
    if not c:return JSONResponse({'error':'camera not found'},404)
    s=manager.get(c['id'],c['rtsp_url'],c['name'],int(get_setting('stream_width','640')),int(get_setting('live_fps','5')),int(get_setting('jpeg_quality','70')))
    return StreamingResponse(s.frames(),media_type='multipart/x-mixed-replace; boundary=frame',headers={'Cache-Control':'no-store'})

@app.get('/api/cameras/{camera_id}/status')
def cam_status(camera_id:int,request:Request):
    if not auth(request):return JSONResponse({'error':'unauthorized'},401)
    return JSONResponse(manager.status(camera_id))

@app.get('/cameras')
def cameras(request:Request,msg:str=''):
    u=auth(request)
    if not u:return RedirectResponse('/login',302)
    if not has_permission(request,'camera.manage'):return access_denied()
    rows=camera_rows(); trs=''.join(f"<tr><td>{c['id']}</td><td>{escape(c['name'])}</td><td>{escape(c['location'])}</td><td>{'فعال' if c['enabled'] else 'غیرفعال'}</td><td>{'ویدئوی آپلودی' if str(c['rtsp_url']).startswith('video://') else ('آزمایشی' if c['is_demo'] else 'RTSP')}</td><td><a class='btn' href='/cameras/{c['id']}/edit'>ویرایش</a> <form style='display:inline' method='post' action='/cameras/{c['id']}/delete' onsubmit=\"return confirm('حذف شود؟')\"><button class='danger'>حذف</button></form></td></tr>" for c in rows) or "<tr><td colspan='6'>دوربینی ثبت نشده است.</td></tr>"
    notice="<div class='card ok'>عملیات انجام شد.</div>" if msg else ''
    return page('دوربین‌ها',f"""<div class='wrap'><div class='toolbar'><h1 style='margin-left:auto'>مدیریت دوربین‌ها</h1><a class='btn' href='/cameras/new'>افزودن دوربین</a></div>{notice}<div class='card'><div class='table-wrap'><table><tr><th>ID</th><th>نام</th><th>موقعیت</th><th>وضعیت</th><th>نوع</th><th>عملیات</th></tr>{trs}</table></div></div><div class='card'><h2>🎞️ نمایش ویدئو به‌صورت دوربین زنده</h2><p class='muted'>پس از پایان آپلود، ویدئو به‌عنوان یک دوربین مجازی در داشبورد پخش می‌شود و پلاک‌خوان در پس‌زمینه روی آن کار می‌کند.</p><form id='videoUploadForm' action='/cameras/video-upload' method='post' enctype='multipart/form-data'><label>تنظیمات کدام دوربین استفاده شود؟</label><select name='camera_id'>{''.join(f"<option value='{c['id']}'>{escape(c['name'])}</option>" for c in rows if not str(c['rtsp_url']).startswith('video://'))}</select><br><label>فایل ویدئو</label><input type='file' name='video' accept='.mp4,.avi,.mkv,.mov' required><div id='uploadState' class='muted' style='display:none;margin:10px 0'>در حال آپلود: <b id='uploadPercent'>۰٪</b><progress id='uploadProgress' value='0' max='100' style='width:100%'></progress></div><br><button id='uploadButton'>آپلود و نمایش در پخش زنده</button></form></div></div>
<script>
const uploadForm=document.getElementById('videoUploadForm');
uploadForm?.addEventListener('submit',event=>{{
 event.preventDefault();
 const button=document.getElementById('uploadButton'),state=document.getElementById('uploadState'),bar=document.getElementById('uploadProgress'),percent=document.getElementById('uploadPercent');
 button.disabled=true;button.textContent='در حال آپلود…';state.style.display='block';
 const xhr=new XMLHttpRequest();xhr.open('POST',uploadForm.action);xhr.setRequestHeader('X-Requested-With','XMLHttpRequest');
 xhr.upload.onprogress=e=>{{if(e.lengthComputable){{const p=Math.round(e.loaded/e.total*100);bar.value=p;percent.textContent=p.toLocaleString('fa-IR')+'٪'}}}};
 xhr.onload=()=>{{let result={{}};try{{result=JSON.parse(xhr.responseText)}}catch(e){{}}if(xhr.status>=200&&xhr.status<300&&result.ok){{location.href=result.redirect||'/dashboard'}}else{{alert(result.error||'آپلود ویدئو انجام نشد.');button.disabled=false;button.textContent='آپلود و نمایش در پخش زنده'}}}};
 xhr.onerror=()=>{{alert('ارتباط هنگام آپلود قطع شد.');button.disabled=false;button.textContent='تلاش دوباره'}};
 xhr.send(new FormData(uploadForm));
}});
</script>""",u,request)

def cam_form(c=None):
    c=dict(c) if c else {'id':'','name':'','rtsp_url':'','location':'','enabled':1,'is_demo':0,'sort_order':0,'lpr_enabled':1,'lpr_confidence':60,'frame_step':5,'duplicate_seconds':30,'roi_x':0,'roi_y':0,'roi_w':100,'roi_h':100,'line_y':50}
    action=f"/cameras/{c['id']}/edit" if c['id'] else '/cameras/new'
    return f"""<div class='wrap'><h1>{'ویرایش' if c['id'] else 'افزودن'} دوربین</h1><div class='card'><form method='post' action='{action}'><div class='two-col'><div><label>نام دوربین</label><input name='name' value='{escape(str(c['name']))}' required></div><div><label>موقعیت</label><input name='location' value='{escape(str(c['location']))}'></div></div><label>آدرس RTSP</label><input class='code' name='rtsp_url' value='{escape(str(c['rtsp_url']))}' placeholder='rtsp://user:pass@192.168.1.10:554/...'><div class='two-col'><div><label>ترتیب نمایش</label><input type='number' name='sort_order' value='{c['sort_order']}'></div><div><label>نوع تصویر</label><select name='is_demo'><option value='0' {'selected' if not c['is_demo'] else ''}>دوربین RTSP</option><option value='1' {'selected' if c['is_demo'] else ''}>دوربین آزمایشی</option></select></div></div><label><input style='width:auto' type='checkbox' name='enabled' value='1' {'checked' if c['enabled'] else ''}> فعال باشد</label><hr><h3>تنظیمات پلاک‌خوان این دوربین</h3><label><input style='width:auto' type='checkbox' name='lpr_enabled' value='1' {'checked' if c.get('lpr_enabled',1) else ''}> پلاک‌خوان فعال باشد</label><div class='two-col'><div><label>حداقل اطمینان تشخیص (درصد)</label><input type='number' min='1' max='99' name='lpr_confidence' value='{c.get('lpr_confidence',60)}'></div><div><label>پردازش هر چند فریم</label><input type='number' min='1' max='60' name='frame_step' value='{c.get('frame_step',5)}'></div><div><label>زمان حذف پلاک تکراری (ثانیه)</label><input type='number' min='0' max='3600' step='0.5' name='duplicate_seconds' value='{c.get('duplicate_seconds',30)}'></div><div><label>خط عبور عمودی (درصد ارتفاع تصویر)</label><input type='number' min='0' max='100' name='line_y' value='{c.get('line_y',50)}'></div></div><h3>منطقه تشخیص ROI برحسب درصد تصویر</h3><p class='muted'>برای بررسی کل تصویر: X=0، Y=0، عرض=100 و ارتفاع=100 قرار دهید.</p><div class='storage-grid'><div><label>X شروع</label><input type='number' min='0' max='99' name='roi_x' value='{c.get('roi_x',0)}'></div><div><label>Y شروع</label><input type='number' min='0' max='99' name='roi_y' value='{c.get('roi_y',0)}'></div><div><label>عرض ROI</label><input type='number' min='1' max='100' name='roi_w' value='{c.get('roi_w',100)}'></div><div><label>ارتفاع ROI</label><input type='number' min='1' max='100' name='roi_h' value='{c.get('roi_h',100)}'></div></div><button>ذخیره</button> <a class='btn secondary' href='/cameras'>انصراف</a></form></div></div>"""
@app.get('/cameras/new')
def new_cam_form(request:Request):
    u=auth(request)
    if not u:return RedirectResponse('/login',302)
    if not has_permission(request,'camera.manage'):return access_denied()
    return page('افزودن دوربین',cam_form(),u,request)
@app.post('/cameras/new')
def new_cam(request:Request,name:str=Form(...),rtsp_url:str=Form(''),location:str=Form(''),enabled:str|None=Form(None),is_demo:int=Form(0),sort_order:int=Form(0),lpr_enabled:str|None=Form(None),lpr_confidence:int=Form(60),frame_step:int=Form(5),duplicate_seconds:float=Form(30),roi_x:int=Form(0),roi_y:int=Form(0),roi_w:int=Form(100),roi_h:int=Form(100),line_y:int=Form(50)):
    if not auth(request):return RedirectResponse('/login',302)
    if not has_permission(request,'camera.manage'):return access_denied()
    lic=license_status()
    with connect() as con:
        count=con.execute('SELECT COUNT(*) c FROM cameras').fetchone()['c']
    if count >= lic['camera_limit']:
        return page('محدودیت لایسنس',f"<div class='wrap'><div class='card alert'>حداکثر تعداد دوربین در پلن {escape(lic['plan'])} برابر {lic['camera_limit']} است. برای افزایش ظرفیت، لایسنس را ارتقا دهید.</div><a class='btn' href='/license'>مدیریت لایسنس</a></div>",auth(request),request)
    url='demo://camera' if is_demo else rtsp_url.strip()
    with connect() as con:con.execute('INSERT INTO cameras(name,rtsp_url,location,enabled,is_demo,sort_order,lpr_enabled,lpr_confidence,frame_step,duplicate_seconds,roi_x,roi_y,roi_w,roi_h,line_y) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',(name.strip(),url,location.strip(),1 if enabled else 0,is_demo,sort_order,1 if lpr_enabled else 0,max(1,min(99,lpr_confidence)),max(1,min(60,frame_step)),max(0,min(3600,duplicate_seconds)),max(0,min(99,roi_x)),max(0,min(99,roi_y)),max(1,min(100-roi_x,roi_w)),max(1,min(100-roi_y,roi_h)),max(0,min(100,line_y))))
    return RedirectResponse('/cameras?msg=1',303)
@app.get('/cameras/{camera_id}/edit')
def edit_cam_form(camera_id:int,request:Request):
    u=auth(request)
    if not u:return RedirectResponse('/login',302)
    if not has_permission(request,'camera.manage'):return access_denied()
    with connect() as con:c=con.execute('SELECT * FROM cameras WHERE id=?',(camera_id,)).fetchone()
    if not c:return RedirectResponse('/cameras',302)
    return page('ویرایش دوربین',cam_form(c),u,request)
@app.post('/cameras/{camera_id}/edit')
def edit_cam(camera_id:int,request:Request,name:str=Form(...),rtsp_url:str=Form(''),location:str=Form(''),enabled:str|None=Form(None),is_demo:int=Form(0),sort_order:int=Form(0),lpr_enabled:str|None=Form(None),lpr_confidence:int=Form(60),frame_step:int=Form(5),duplicate_seconds:float=Form(30),roi_x:int=Form(0),roi_y:int=Form(0),roi_w:int=Form(100),roi_h:int=Form(100),line_y:int=Form(50)):
    if not auth(request):return RedirectResponse('/login',302)
    if not has_permission(request,'camera.manage'):return access_denied()
    url='demo://camera' if is_demo else rtsp_url.strip()
    with connect() as con:con.execute('UPDATE cameras SET name=?,rtsp_url=?,location=?,enabled=?,is_demo=?,sort_order=?,lpr_enabled=?,lpr_confidence=?,frame_step=?,duplicate_seconds=?,roi_x=?,roi_y=?,roi_w=?,roi_h=?,line_y=? WHERE id=?',(name.strip(),url,location.strip(),1 if enabled else 0,is_demo,sort_order,1 if lpr_enabled else 0,max(1,min(99,lpr_confidence)),max(1,min(60,frame_step)),max(0,min(3600,duplicate_seconds)),max(0,min(99,roi_x)),max(0,min(99,roi_y)),max(1,min(100-roi_x,roi_w)),max(1,min(100-roi_y,roi_h)),max(0,min(100,line_y)),camera_id))
    manager.remove(camera_id);return RedirectResponse('/cameras?msg=1',303)
@app.post('/cameras/{camera_id}/delete')
def delete_cam(camera_id:int,request:Request):
    if not auth(request):return RedirectResponse('/login',302)
    if not has_permission(request,'camera.manage'):return access_denied()
    with connect() as con:con.execute('DELETE FROM cameras WHERE id=?',(camera_id,))
    manager.remove(camera_id);return RedirectResponse('/cameras?msg=1',303)

@app.get('/media')
def media(request:Request,path:str=''):
    if not auth(request): return RedirectResponse('/login',302)
    try:
        target=Path(path).resolve()
        allowed=[Path(get_setting('snapshot_path',str(SNAPSHOT_DIR))).resolve(),Path(get_setting('plate_path',str(PLATE_DIR))).resolve(),Path(get_setting('video_path',str(VIDEO_DIR))).resolve()]
        if not target.is_file() or not any(target.is_relative_to(root) for root in allowed): return JSONResponse({'error':'not found'},404)
        return FileResponse(target)
    except Exception:return JSONResponse({'error':'not found'},404)

@app.get('/events')
def events(request:Request,q:str='',camera:str='',status:str='',vehicle_type:str='',vehicle_color:str='',date_from:str='',date_to:str=''):
    u=auth(request)
    if not u:return RedirectResponse('/login',302)
    where=[];params=[]
    if q: where.append('e.plate_text LIKE ?');params.append(f'%{q}%')
    if camera: where.append('e.camera_name=?');params.append(camera)
    if status: where.append("COALESCE(w.status,'unknown')=?");params.append(status)
    if vehicle_type: where.append('e.vehicle_type=?');params.append(vehicle_type)
    if vehicle_color: where.append('e.vehicle_color=?');params.append(vehicle_color)
    sql="""SELECT e.*,COALESCE(w.status,'unknown') watch_status,w.owner_name,w.vehicle_model,w.vehicle_color
             FROM plate_events e LEFT JOIN plate_watchlist w ON w.plate_norm=REPLACE(REPLACE(UPPER(e.plate_text),'-',''),' ','')"""
    if where: sql+=' WHERE '+' AND '.join(where)
    sql+=' ORDER BY e.id DESC LIMIT 500'
    with connect() as con:
        rows=con.execute(sql,params).fetchall()
        cameras=[r['camera_name'] for r in con.execute("SELECT DISTINCT camera_name FROM plate_events WHERE camera_name IS NOT NULL AND camera_name<>'' ORDER BY camera_name").fetchall()]
        vehicle_types=[r['vehicle_type'] for r in con.execute("SELECT DISTINCT vehicle_type FROM plate_events WHERE vehicle_type IS NOT NULL AND vehicle_type<>'' ORDER BY vehicle_type").fetchall()]
        vehicle_colors=[r['vehicle_color'] for r in con.execute("SELECT DISTINCT vehicle_color FROM plate_events WHERE vehicle_color IS NOT NULL AND vehicle_color<>'' ORDER BY vehicle_color").fetchall()]
    def in_jdate(row):
        jd=jalali_date(row['created_at'])
        return (not date_from or jd>=date_from) and (not date_to or jd<=date_to)
    rows=[r for r in rows if in_jdate(r)]
    trs=[]
    for r in rows:
        st=r['watch_status'] or 'unknown'; cls='event-blocked' if st=='blocked' else ('event-vip' if st=='vip' else '')
        vehicle=(f"<img class='thumb' onclick=\"showImage(this.src)\" src='/media?path={quote(r['image_path'])}'>" if r['image_path'] and Path(r['image_path']).exists() else '—')
        plateimg=(f"<img class='thumb plate-thumb' onclick=\"showImage(this.src)\" src='/media?path={quote(r['plate_image_path'])}'>" if r['plate_image_path'] and Path(r['plate_image_path']).exists() else '—')
        owner=escape(' / '.join(x for x in [r['owner_name'],r['vehicle_model'],r['vehicle_color']] if x) or '—')
        trs.append(f"<tr class='{cls}'><td>{r['id']}</td><td>{vehicle}</td><td>{plateimg}</td><td><b>{escape(r['plate_text'] or '—')}</b><br>{event_status_badge(st)}</td><td>{owner}</td><td>{escape(r['vehicle_type'] or 'نامشخص')}<br><span class='muted'>{escape(r['vehicle_color'] or 'نامشخص')}</span></td><td>{int((r['confidence'] or 0)*100)}٪</td><td>{escape(r['camera_name'] or '—')}</td><td>{jalali_datetime(r['created_at'])}</td><td><a class='btn' href='/events/{r['id']}'>جزئیات و پخش</a></td></tr>")
    trs=''.join(trs) or "<tr><td colspan='10'>رکوردی با این فیلتر پیدا نشد.</td></tr>"
    cam_opts=''.join(f"<option {'selected' if camera==c else ''}>{escape(c)}</option>" for c in cameras)
    type_opts=''.join(f"<option {'selected' if vehicle_type==v else ''}>{escape(v)}</option>" for v in vehicle_types)
    color_opts=''.join(f"<option {'selected' if vehicle_color==v else ''}>{escape(v)}</option>" for v in vehicle_colors)
    status_opts=''.join(f"<option value='{v}' {'selected' if status==v else ''}>{l}</option>" for v,l in [('allowed','مجاز'),('blocked','غیرمجاز'),('vip','VIP'),('unknown','ثبت‌نشده')])
    body=f"""<div class='wrap'><div class='toolbar'><h1 style='margin-left:auto'>گزارش ترددها</h1><a class='btn' href='/watchlist'>مدیریت پلاک‌ها</a><a class='btn secondary' href='/events/export.csv'>خروجی CSV</a></div>
    <div class='card'><form class='filter-grid'><div><label>پلاک</label><input name='q' value='{escape(q)}'></div><div><label>دوربین</label><select name='camera'><option value=''>همه</option>{cam_opts}</select></div><div><label>وضعیت</label><select name='status'><option value=''>همه</option>{status_opts}</select></div><div><label>نوع خودرو</label><select name='vehicle_type'><option value=''>همه</option>{type_opts}</select></div><div><label>رنگ خودرو</label><select name='vehicle_color'><option value=''>همه</option>{color_opts}</select></div><div><label>از تاریخ شمسی</label><input name='date_from' value='{escape(date_from)}' placeholder='۱۴۰۵/۰۵/۰۲'></div><div><label>تا تاریخ شمسی</label><input name='date_to' value='{escape(date_to)}' placeholder='۱۴۰۵/۰۵/۰۳'></div><div><button>اعمال فیلتر</button></div></form></div>
    <div class='card'><div class='table-wrap'><table><tr><th>ردیف</th><th>تصویر خودرو</th><th>تصویر پلاک</th><th>پلاک/وضعیت</th><th>مالک/خودرو</th><th>تشخیص خودرو</th><th>اطمینان</th><th>دوربین</th><th>تاریخ و ساعت شمسی</th><th>عملیات</th></tr>{trs}</table></div></div></div>
    <div id='imgModal' class='modal-img' onclick='this.classList.remove("open")'><button>بستن</button><img id='modalImage'></div><script>function showImage(src){{document.getElementById('modalImage').src=src;document.getElementById('imgModal').classList.add('open')}}</script>"""
    return page('ترددها',body,u,request)

@app.get('/events/{event_id:int}')
def event_detail(event_id:int, request:Request):
    u=auth(request)
    if not u:return RedirectResponse('/login',302)
    with connect() as con:
        r=con.execute("""SELECT e.*,COALESCE(w.status,'unknown') watch_status,w.owner_name,w.phone,w.vehicle_model,w.vehicle_color,w.notes
            FROM plate_events e LEFT JOIN plate_watchlist w ON w.plate_norm=REPLACE(REPLACE(UPPER(e.plate_text),'-',''),' ','') WHERE e.id=?""",(event_id,)).fetchone()
    if not r:return page('تردد پیدا نشد',"<div class='wrap'><div class='card'><h1>تردد پیدا نشد</h1><a class='btn' href='/events'>بازگشت</a></div></div>",u,request)
    st=r['watch_status'] or 'unknown'
    second=float(r['video_second'] or 0)
    video_ok=bool(r['video_path'] and Path(r['video_path']).is_file())
    image_ok=bool(r['image_path'] and Path(r['image_path']).is_file())
    plate_ok=bool(r['plate_image_path'] and Path(r['plate_image_path']).is_file())
    video=(f"<video id='eventVideo' controls preload='metadata' src='/media?path={quote(r['video_path'])}'></video><div class='replay-controls'><button type='button' onclick='jumpToEvent()'>رفتن به لحظه عبور</button><button type='button' class='secondary' onclick='stepFrame(-1)'>فریم قبل</button><button type='button' class='secondary' onclick='stepFrame(1)'>فریم بعد</button><button type='button' class='secondary' onclick='setSpeed(.25)'>¼×</button><button type='button' class='secondary' onclick='setSpeed(.5)'>½×</button><button type='button' class='secondary' onclick='setSpeed(1)'>1×</button><button type='button' class='secondary' onclick='setSpeed(2)'>2×</button><span id='speedLabel' class='muted'>سرعت: 1×</span></div>" if video_ok else "<div class='alert'>فایل ویدئوی این تردد موجود نیست یا طبق تنظیمات نگهداری حذف شده است.</div>")
    vehicle=(f"<img onclick='showImage(this.src)' src='/media?path={quote(r['image_path'])}'>" if image_ok else "<div class='muted'>تصویر خودرو موجود نیست</div>")
    plate=(f"<img onclick='showImage(this.src)' src='/media?path={quote(r['plate_image_path'])}'>" if plate_ok else "<div class='muted'>تصویر پلاک موجود نیست</div>")
    owner=' / '.join(x for x in [r['owner_name'],r['vehicle_model'],r['vehicle_color']] if x) or 'ثبت نشده'
    body=f"""<div class='wrap'><div class='toolbar'><h1 style='margin-left:auto'>جزئیات تردد شماره {r['id']}</h1><a class='btn secondary' href='/events'>بازگشت به ترددها</a></div>
    <div class='replay-layout'><div class='card video-panel'><h3>پخش ویدئو از لحظه عبور</h3><div class='time-badge'>زمان ثبت در ویدئو: {second:.2f} ثانیه</div>{video}</div>
    <div><div class='card'><h3>اطلاعات تردد</h3><div class='event-meta'><div class='meta-item'><small>پلاک</small><b>{escape(r['plate_text'] or '—')}</b></div><div class='meta-item'><small>وضعیت</small>{event_status_badge(st)}</div><div class='meta-item'><small>دوربین</small>{escape(r['camera_name'] or '—')}</div><div class='meta-item'><small>اطمینان</small>{(r['confidence'] or 0)*100:.1f}٪</div><div class='meta-item'><small>تاریخ و ساعت شمسی</small>{jalali_datetime(r['created_at'])}</div><div class='meta-item'><small>روش تشخیص</small>{escape(r['detector_method'] or '—')}</div><div class='meta-item'><small>نوع خودرو</small>{escape(r['vehicle_type'] or 'نامشخص')}</div><div class='meta-item'><small>رنگ خودرو</small>{escape(r['vehicle_color'] or 'نامشخص')}</div><div class='meta-item'><small>اطمینان تشخیص خودرو</small>{(r['vehicle_confidence'] or 0)*100:.1f}٪</div><div class='meta-item'><small>مالک / خودرو</small>{escape(owner)}</div><div class='meta-item'><small>شماره تماس</small>{escape(r['phone'] or '—')}</div></div></div>
    <div class='card'><h3>تصاویر ثبت‌شده</h3><div class='detail-images'><div>{vehicle}<small>تصویر خودرو</small></div><div>{plate}<small>تصویر پلاک</small></div></div></div></div></div></div>
    <div id='imgModal' class='modal-img' onclick='this.classList.remove("open")'><button>بستن</button><img id='modalImage'></div>
    <script>const eventSecond={second:.3f};const v=document.getElementById('eventVideo');function jumpToEvent(){{if(!v)return;v.currentTime=Math.max(0,eventSecond-.5);v.play().catch(()=>{{}})}}function stepFrame(dir){{if(!v)return;v.pause();v.currentTime=Math.max(0,v.currentTime+dir/25)}}function setSpeed(rate){{if(!v)return;v.playbackRate=rate;document.getElementById('speedLabel').textContent='سرعت: '+rate+'×'}}function showImage(src){{document.getElementById('modalImage').src=src;document.getElementById('imgModal').classList.add('open')}}if(v){{v.addEventListener('loadedmetadata',jumpToEvent,{{once:true}})}}</script>"""
    return page('جزئیات تردد',body,u,request)

@app.get('/watchlist')
def watchlist(request:Request,msg:str=''):
    u=auth(request)
    if not u:return RedirectResponse('/login',302)
    if not has_permission(request,'watchlist.manage'):return access_denied()
    with connect() as con:rows=con.execute('SELECT * FROM plate_watchlist ORDER BY id DESC').fetchall()
    trs=''.join(f"<tr><td>{escape(r['plate_text'])}</td><td>{event_status_badge(r['status'])}</td><td>{escape(r['owner_name'] or '—')}</td><td>{escape(r['phone'] or '—')}</td><td>{escape(r['vehicle_model'] or '—')}</td><td>{escape(r['vehicle_color'] or '—')}</td><td>{jalali_datetime(r['created_at'])}</td><td><form method='post' action='/watchlist/{r['id']}/delete' onsubmit=\"return confirm('حذف شود؟')\"><button class='danger'>حذف</button></form></td></tr>" for r in rows) or "<tr><td colspan='8'>هنوز پلاکی تعریف نشده است.</td></tr>"
    notice="<div class='card ok'>پلاک ذخیره شد.</div>" if msg else ''
    body=f"""<div class='wrap'><div class='toolbar'><h1 style='margin-left:auto'>مدیریت پلاک‌ها</h1><a class='btn secondary' href='/events'>بازگشت به ترددها</a></div>{notice}<div class='card'><h3>افزودن پلاک</h3><form method='post' class='grid'><div><label>شماره پلاک</label><input name='plate_text' required></div><div><label>وضعیت</label><select name='status'><option value='allowed'>مجاز</option><option value='blocked'>غیرمجاز</option><option value='vip'>VIP</option></select></div><div><label>نام مالک</label><input name='owner_name'></div><div><label>شماره موبایل</label><input name='phone'></div><div><label>مدل خودرو</label><input name='vehicle_model'></div><div><label>رنگ خودرو</label><input name='vehicle_color'></div><div style='grid-column:1/-1'><label>توضیحات</label><input name='notes'></div><div><button>ذخیره پلاک</button></div></form></div><div class='card'><div class='table-wrap'><table><tr><th>پلاک</th><th>وضعیت</th><th>مالک</th><th>موبایل</th><th>خودرو</th><th>رنگ</th><th>تاریخ ثبت شمسی</th><th>عملیات</th></tr>{trs}</table></div></div></div>"""
    return page('مدیریت پلاک‌ها',body,u,request)

@app.post('/watchlist')
def add_watchlist(request:Request,plate_text:str=Form(...),status:str=Form('allowed'),owner_name:str=Form(''),phone:str=Form(''),vehicle_model:str=Form(''),vehicle_color:str=Form(''),notes:str=Form('')):
    if not auth(request):return RedirectResponse('/login',302)
    if not has_permission(request,'watchlist.manage'):return access_denied()
    status=status if status in ('allowed','blocked','vip') else 'allowed'
    with connect() as con:
        con.execute("INSERT INTO plate_watchlist(plate_text,plate_norm,status,owner_name,phone,vehicle_model,vehicle_color,notes) VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(plate_norm) DO UPDATE SET plate_text=excluded.plate_text,status=excluded.status,owner_name=excluded.owner_name,phone=excluded.phone,vehicle_model=excluded.vehicle_model,vehicle_color=excluded.vehicle_color,notes=excluded.notes",(plate_text.strip(),normalize_plate(plate_text),status,owner_name.strip(),phone.strip(),vehicle_model.strip(),vehicle_color.strip(),notes.strip()))
    return RedirectResponse('/watchlist?msg=1',303)

@app.post('/watchlist/{item_id}/delete')
def delete_watchlist(item_id:int,request:Request):
    if not auth(request):return RedirectResponse('/login',302)
    if not has_permission(request,'watchlist.manage'):return access_denied()
    with connect() as con:con.execute('DELETE FROM plate_watchlist WHERE id=?',(item_id,))
    return RedirectResponse('/watchlist',303)


def _safe_int(value, default=0, minimum=0, maximum=100000):
    try: return max(minimum, min(maximum, int(value)))
    except Exception: return default

def _path_usage(path_value):
    try:
        p=Path(path_value).expanduser()
        if not p.is_dir(): raise ValueError('مسیر ذخیره‌سازی در دسترس نیست.')
        usage=shutil.disk_usage(p)
        return {'ok':True,'path':str(p),'total':usage.total,'used':usage.used,'free':usage.free,'percent':round(usage.used/usage.total*100,1) if usage.total else 0}
    except Exception as e:
        return {'ok':False,'path':str(path_value),'error':str(e),'total':0,'used':0,'free':0,'percent':0}

def _fmt_bytes(n):
    n=float(n)
    for unit in ['B','KB','MB','GB','TB']:
        if n < 1024 or unit=='TB': return f"{n:.1f} {unit}"
        n/=1024

def _csv_cell(value):
    text='' if value is None else str(value)
    if text.startswith(('=', '+', '-', '@', '\t', '\r')):
        return "'" + text
    return text

def _storage_paths(storage_root, snapshot_path, plate_path, video_path, backup_path):
    raw=[storage_root,snapshot_path,plate_path,video_path,backup_path]
    if any(not str(value).strip() for value in raw):
        raise ValueError('همه مسیرها باید وارد شوند.')
    paths=[Path(str(value).strip()).expanduser().resolve() for value in raw]
    root=paths[0]
    if root == Path(root.anchor):
        raise ValueError('پوشه ریشه درایو یا سیستم‌عامل قابل انتخاب نیست.')
    for child in paths[1:]:
        if child == root or not child.is_relative_to(root):
            raise ValueError('مسیر تصاویر، پلاک‌ها، ویدئوها و پشتیبان باید داخل مسیر اصلی باشد.')
    if len(set(paths[1:])) != 4:
        raise ValueError('برای هر نوع اطلاعات یک زیرپوشه جداگانه انتخاب کنید.')
    return paths

def _configured_storage_child(setting_key, default):
    paths=_storage_paths(
        get_setting('storage_root',str(DATA_DIR)),
        get_setting('snapshot_path',str(SNAPSHOT_DIR)),
        get_setting('plate_path',str(PLATE_DIR)),
        get_setting('video_path',str(VIDEO_DIR)),
        get_setting('backup_path',str(BACKUP_DIR)),
    )
    index={'snapshot_path':1,'plate_path':2,'video_path':3,'backup_path':4}[setting_key]
    return paths[index]

VIDEO_EXTENSIONS={'.mp4','.avi','.mkv','.mov','.m4v'}
MAX_VIDEO_UPLOAD_BYTES=2*1024*1024*1024

def _video_suffix(filename):
    safe_name=Path(str(filename or '').replace('\\','/')).name
    suffix=Path(safe_name).suffix.lower()
    return suffix if suffix in VIDEO_EXTENSIONS else ''

async def _save_video_upload(video, save_dir, suffix):
    save_dir=Path(save_dir).resolve()
    save_dir.mkdir(parents=True,exist_ok=True)
    target=save_dir / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(8)}{suffix}"
    size=0
    try:
        with target.open('xb') as f:
            while chunk:=await video.read(1024*1024):
                size+=len(chunk)
                if size>MAX_VIDEO_UPLOAD_BYTES:
                    raise ValueError('حجم ویدئو بیشتر از ۲ گیگابایت است.')
                f.write(chunk)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    return target

def _cleanup_old_files(folder, days, storage_root):
    if days <= 0: return 0
    removed=0; cutoff=time.time()-days*86400
    root=Path(storage_root).resolve()
    p=Path(folder).resolve()
    if p == root or not p.is_relative_to(root): return 0
    if not p.exists(): return 0
    for f in p.rglob('*'):
        try:
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink(); removed+=1
        except Exception: pass
    return removed

def run_retention_cleanup():
    removed=0
    root=get_setting('storage_root',str(DATA_DIR))
    try:
        paths=_storage_paths(
            root,
            get_setting('snapshot_path',str(SNAPSHOT_DIR)),
            get_setting('plate_path',str(PLATE_DIR)),
            get_setting('video_path',str(VIDEO_DIR)),
            get_setting('backup_path',str(BACKUP_DIR)),
        )
    except ValueError:
        return 0
    removed += _cleanup_old_files(paths[1], _safe_int(get_setting('retention_snapshots_days','90'),90), paths[0])
    removed += _cleanup_old_files(paths[2], _safe_int(get_setting('retention_plates_days','90'),90), paths[0])
    removed += _cleanup_old_files(paths[3], _safe_int(get_setting('retention_videos_days','7'),7), paths[0])
    event_days=_safe_int(get_setting('retention_events_days','0'),0)
    if event_days>0:
        with connect() as con:
            con.execute("DELETE FROM plate_events WHERE created_at < datetime('now', ?)",(f'-{event_days} days',))
    return removed


ROLE_LABELS={'admin':'مدیر کل','system':'مدیر سیستم','operator':'اپراتور','guard':'نگهبان'}

@app.get('/users')
def users_page(request:Request,msg:str='',error:str=''):
    u=auth(request)
    if not u:return RedirectResponse('/login',302)
    if not require_admin(request):return page('عدم دسترسی',"<div class='wrap'><div class='card alert'>فقط مدیر کل به مدیریت کاربران دسترسی دارد.</div></div>",u,request)
    with connect() as con:rows=con.execute('SELECT * FROM users ORDER BY id').fetchall()
    trs=''.join(f"<tr><td>{r['id']}</td><td><b>{escape(r['username'])}</b><br><span class='muted'>{escape(r['display_name'])}</span></td><td>{ROLE_LABELS.get(r['role'],r['role'])}</td><td>{'فعال' if r['is_active'] else 'غیرفعال'}</td><td>{jalali_datetime(r['last_login']) if r['last_login'] else '—'}</td><td><a class='btn secondary' href='/users/{r['id']}/edit'>ویرایش</a> <form style='display:inline' method='post' action='/users/{r['id']}/toggle'><button>{'غیرفعال‌سازی' if r['is_active'] else 'فعال‌سازی'}</button></form></td></tr>" for r in rows)
    note=("<div class='card ok'>عملیات با موفقیت انجام شد.</div>" if msg else '')+(f"<div class='card alert'>{escape(error)}</div>" if error else '')
    form="""<div class='card'><h3>افزودن کاربر</h3><form method='post' action='/users'><div class='two-col'><div><label>نام کاربری</label><input name='username' required></div><div><label>نام نمایشی</label><input name='display_name' required></div><div><label>رمز عبور</label><input type='password' name='password' minlength='6' required></div><div><label>نقش</label><select name='role'><option value='operator'>اپراتور</option><option value='guard'>نگهبان</option><option value='system'>مدیر سیستم</option><option value='admin'>مدیر کل</option></select></div></div><button>ایجاد کاربر</button></form></div>"""
    return page('مدیریت کاربران',f"<div class='wrap'><div class='toolbar'><h1 style='margin-left:auto'>مدیریت کاربران</h1><a class='btn secondary' href='/audit'>لاگ فعالیت‌ها</a></div>{note}{form}<div class='card'><div class='table-wrap'><table><tr><th>ID</th><th>کاربر</th><th>نقش</th><th>وضعیت</th><th>آخرین ورود</th><th>عملیات</th></tr>{trs}</table></div></div></div>",u,request)

@app.post('/users')
def create_user_route(request:Request,username:str=Form(...),display_name:str=Form(...),password:str=Form(...),role:str=Form('operator')):
    if not require_admin(request):return RedirectResponse('/dashboard',303)
    if role not in ROLE_LABELS or len(password)<6:return RedirectResponse('/users?error='+quote('اطلاعات کاربر معتبر نیست'),303)
    try:
        with connect() as con:con.execute('INSERT INTO users(username,password_hash,display_name,is_admin,role,is_active) VALUES(?,?,?,?,?,1)',(username.strip(),hash_password(password),display_name.strip(),1 if role=='admin' else 0,role))
        audit(request,'user_create',f'ایجاد کاربر {username} با نقش {role}')
        return RedirectResponse('/users?msg=1',303)
    except Exception:return RedirectResponse('/users?error='+quote('نام کاربری تکراری است'),303)

@app.get('/users/{user_id}/edit')
def edit_user_form(user_id:int,request:Request):
    u=auth(request)
    if not u:return RedirectResponse('/login',302)
    if not require_admin(request):return RedirectResponse('/dashboard',303)
    with connect() as con:r=con.execute('SELECT * FROM users WHERE id=?',(user_id,)).fetchone()
    if not r:return RedirectResponse('/users',303)
    opts=''.join(f"<option value='{k}' {'selected' if r['role']==k else ''}>{v}</option>" for k,v in ROLE_LABELS.items())
    body=f"""<div class='wrap'><div class='card'><h1>ویرایش کاربر</h1><form method='post'><label>نام نمایشی</label><input name='display_name' value='{escape(r['display_name'])}' required><label>نقش</label><select name='role'>{opts}</select><label>رمز جدید (اختیاری)</label><input type='password' name='password' minlength='6'><button>ذخیره</button> <a class='btn secondary' href='/users'>بازگشت</a></form></div></div>"""
    return page('ویرایش کاربر',body,u,request)

@app.post('/users/{user_id}/edit')
def edit_user_route(user_id:int,request:Request,display_name:str=Form(...),role:str=Form(...),password:str=Form('')):
    if not require_admin(request):return RedirectResponse('/dashboard',303)
    with connect() as con:
        if password:con.execute('UPDATE users SET display_name=?,role=?,is_admin=?,password_hash=? WHERE id=?',(display_name.strip(),role,1 if role=='admin' else 0,hash_password(password),user_id))
        else:con.execute('UPDATE users SET display_name=?,role=?,is_admin=? WHERE id=?',(display_name.strip(),role,1 if role=='admin' else 0,user_id))
    audit(request,'user_update',f'ویرایش کاربر شماره {user_id}')
    return RedirectResponse('/users?msg=1',303)

@app.post('/users/{user_id}/toggle')
def toggle_user(user_id:int,request:Request):
    cu=current_user(request)
    if not cu or not require_admin(request):return RedirectResponse('/dashboard',303)
    if cu['id']==user_id:return RedirectResponse('/users?error='+quote('نمی‌توانید حساب خودتان را غیرفعال کنید'),303)
    with connect() as con:con.execute('UPDATE users SET is_active=CASE is_active WHEN 1 THEN 0 ELSE 1 END WHERE id=?',(user_id,))
    audit(request,'user_toggle',f'تغییر وضعیت کاربر شماره {user_id}')
    return RedirectResponse('/users?msg=1',303)

@app.get('/audit')
def audit_page(request:Request,q:str='',action:str=''):
    u=auth(request)
    if not u:return RedirectResponse('/login',302)
    if not require_admin(request):return page('عدم دسترسی',"<div class='wrap'><div class='card alert'>فقط مدیر کل به لاگ‌ها دسترسی دارد.</div></div>",u,request)
    where=[];params=[]
    if q:where.append('(username LIKE ? OR details LIKE ?)');params += [f'%{q}%',f'%{q}%']
    if action:where.append('action=?');params.append(action)
    sql='SELECT * FROM audit_logs'+((' WHERE '+' AND '.join(where)) if where else '')+' ORDER BY id DESC LIMIT 1000'
    with connect() as con:
        rows=con.execute(sql,params).fetchall(); actions=[x['action'] for x in con.execute('SELECT DISTINCT action FROM audit_logs ORDER BY action').fetchall()]
    trs=''.join(f"<tr><td>{r['id']}</td><td>{escape(r['username'] or '—')}</td><td><span class='status-pill ok'>{escape(r['action'])}</span></td><td>{escape(r['details'] or '—')}</td><td>{escape(r['ip_address'] or '—')}</td><td>{jalali_datetime(r['created_at'])}</td></tr>" for r in rows) or "<tr><td colspan='6'>لاگی ثبت نشده است.</td></tr>"
    opts=''.join(f"<option {'selected' if action==a else ''}>{escape(a)}</option>" for a in actions)
    body=f"""<div class='wrap'><div class='toolbar'><h1 style='margin-left:auto'>لاگ فعالیت کاربران</h1><a class='btn' href='/users'>مدیریت کاربران</a></div><div class='card'><form class='filter-grid'><div><label>جستجو</label><input name='q' value='{escape(q)}'></div><div><label>عملیات</label><select name='action'><option value=''>همه</option>{opts}</select></div><div><button>اعمال فیلتر</button></div></form></div><div class='card'><div class='table-wrap'><table><tr><th>ID</th><th>کاربر</th><th>عملیات</th><th>جزئیات</th><th>IP</th><th>زمان</th></tr>{trs}</table></div></div></div>"""
    return page('لاگ فعالیت‌ها',body,u,request)

@app.get('/settings')
def settings(request:Request,saved:int=0,restart:int=0,error:str=''):
    u=auth(request)
    if not u:return RedirectResponse('/login',302)
    if not has_permission(request,'system.manage'):return access_denied()
    msg="<div class='card ok'>تنظیمات ذخیره شد.</div>" if saved else ''
    if restart: msg += "<div class='alert' style='background:#fff8e5;color:#815b00;border-color:#ffe3a3'>مسیر اصلی ذخیره‌سازی تغییر کرد. برای استفاده کامل از دیتابیس در مسیر جدید، برنامه را یک‌بار ببندید و دوباره اجرا کنید.</div>"
    if error: msg += f"<div class='alert'>{escape(error)}</div>"
    root=get_setting('storage_root',str(DATA_DIR)); snap=get_setting('snapshot_path',str(Path(root)/'snapshots')); plates=get_setting('plate_path',str(Path(root)/'plates')); videos=get_setting('video_path',str(Path(root)/'videos')); backups=get_setting('backup_path',str(Path(root)/'backups'))
    usage=_path_usage(root)
    usage_html=(f"<div class='drive-card'><div class='meter-label'><b>{escape(usage['path'])}</b><span>{usage['percent']}٪</span></div><div class='storage-progress'><span style='width:{usage['percent']}%'></span></div><span class='muted'>آزاد: {_fmt_bytes(usage['free'])} از {_fmt_bytes(usage['total'])}</span></div>" if usage['ok'] else f"<div class='alert'>مسیر ذخیره‌سازی در دسترس نیست: {escape(usage.get('error',''))}</div>")
    checked=lambda k: 'checked' if get_setting(k,'0')=='1' else ''
    selected=lambda k,v: 'selected' if get_setting(k,'')==v else ''
    body=f"""<div class='wrap'><h1>تنظیمات</h1>{msg}
    <div class='card'><h3>نمایش زنده</h3><form method='post' action='/settings/display'><div class='two-col'><div><label>تعداد ستون نمایش زنده</label><select name='dashboard_grid'>{''.join(f'<option value={x} '+('selected' if get_setting('dashboard_grid','2')==str(x) else '')+f'>{x} ستون</option>' for x in [1,2,3,4])}</select></div><div><label>تعداد فریم نمایش در ثانیه</label><input type='number' min='1' max='15' name='live_fps' value='{get_setting('live_fps','5')}'></div><div><label>عرض تصویر لایو</label><select name='stream_width'>{''.join(f'<option value={x} '+('selected' if get_setting('stream_width','640')==str(x) else '')+f'>{x}px</option>' for x in [480,640,960,1280])}</select></div><div><label>کیفیت JPEG</label><input type='number' min='30' max='95' name='jpeg_quality' value='{get_setting('jpeg_quality','70')}'></div></div><label>رمز جدید مدیر (اختیاری)</label><input type='password' name='new_password'><button>ذخیره تنظیمات نمایش</button></form></div>
    <div class='card' id='storage'><h3>ذخیره‌سازی</h3><p class='muted'>درایو یا پوشه اصلی و مسیر جداگانه هر نوع اطلاعات را انتخاب کنید.</p>{usage_html}<form method='post' action='/settings/storage' style='margin-top:18px'><label>مسیر اصلی ذخیره‌سازی</label><input class='code' name='storage_root' value='{escape(root)}' placeholder='D:\\BCVisionData'><div class='storage-grid'><div><label>تصاویر خودرو</label><input class='code' name='snapshot_path' value='{escape(snap)}'></div><div><label>تصاویر پلاک</label><input class='code' name='plate_path' value='{escape(plates)}'></div><div><label>ویدئوها</label><input class='code' name='video_path' value='{escape(videos)}'></div><div><label>نسخه‌های پشتیبان</label><input class='code' name='backup_path' value='{escape(backups)}'></div></div>
    <div class='two-col'><label><input style='width:auto' type='checkbox' name='save_snapshots' value='1' {checked('save_snapshots')}> ذخیره تصویر خودرو</label><label><input style='width:auto' type='checkbox' name='save_plate_images' value='1' {checked('save_plate_images')}> ذخیره تصویر پلاک</label><label><input style='width:auto' type='checkbox' name='save_videos' value='1' {checked('save_videos')}> ذخیره ویدئو</label><div><label>حداکثر فضای مجاز (GB؛ صفر یعنی نامحدود)</label><input type='number' min='0' name='max_storage_gb' value='{get_setting('max_storage_gb','0')}'></div></div>
    <label>وقتی فضا پر شد</label><select name='storage_full_action'><option value='delete_oldest' {selected('storage_full_action','delete_oldest')}>حذف قدیمی‌ترین اطلاعات</option><option value='stop' {selected('storage_full_action','stop')}>توقف ذخیره‌سازی</option><option value='alert' {selected('storage_full_action','alert')}>فقط نمایش هشدار</option></select>
    <h3>مدت نگهداری</h3><div class='storage-grid'><div><label>تصاویر خودرو (روز)</label><input type='number' min='0' name='retention_snapshots_days' value='{get_setting('retention_snapshots_days','90')}'></div><div><label>تصاویر پلاک (روز)</label><input type='number' min='0' name='retention_plates_days' value='{get_setting('retention_plates_days','90')}'></div><div><label>ویدئوها (روز)</label><input type='number' min='0' name='retention_videos_days' value='{get_setting('retention_videos_days','7')}'></div><div><label>رویدادها (روز؛ صفر یعنی نامحدود)</label><input type='number' min='0' name='retention_events_days' value='{get_setting('retention_events_days','0')}'></div></div><button>ذخیره تنظیمات ذخیره‌سازی</button></form></div>
    <div class='card'><h3>🧠 تنظیمات هوش مصنوعی</h3>
<form method='post' action='/settings/ai'>
<label>روش پردازش AI</label>
<select name='ai_accelerator'>
<option value='auto'>Auto (پیشنهادی)</option>
<option value='cpu'>CPU Only</option>
<option value='gpu'>NVIDIA GPU (CUDA)</option>
</select>
<label>حالت پردازش</label>
<select name='ai_quality'>
<option value='fast'>سرعت بالا</option>
<option value='balanced'>متعادل</option>
<option value='accuracy'>بیشترین دقت</option>
</select>
<label>حداقل اطمینان تشخیص (%)</label>
<input type='number' min='1' max='99' name='ai_confidence' value='{get_setting("ai_confidence","85")}'>
<label>تعداد فریم تأیید</label>
<input type='number' min='1' max='20' name='ai_frames' value='{get_setting("ai_frames","5")}'>
<button>ذخیره تنظیمات AI</button>
</form>
</div>
<div class='card'><form method='post' action='/backup'><button class='secondary'>دریافت نسخه پشتیبان دیتابیس</button></form></div><div class='card'><b>وضعیت موتور تصویر:</b> {'آماده' if CV_OK else 'OpenCV بارگذاری نشده است'}</div></div>"""
    return page('تنظیمات',body,u,request)

@app.post('/settings/display')
def save_display_settings(request:Request,dashboard_grid:int=Form(2),live_fps:int=Form(5),stream_width:int=Form(640),jpeg_quality:int=Form(70),new_password:str=Form('')):
    u=auth(request)
    if not u:return RedirectResponse('/login',302)
    if not has_permission(request,'system.manage'):return access_denied()
    set_setting('dashboard_grid',max(1,min(4,dashboard_grid)));set_setting('live_fps',max(1,min(15,live_fps)));set_setting('stream_width',stream_width);set_setting('jpeg_quality',max(30,min(95,jpeg_quality)))
    if new_password.strip():
        with connect() as con:con.execute('UPDATE users SET password_hash=? WHERE username=?',(hash_password(new_password.strip()),u))
    for cid in list(manager.streams): manager.remove(cid)
    return RedirectResponse('/settings?saved=1',303)

@app.post('/settings/storage')
def save_storage_settings(request:Request,storage_root:str=Form(...),snapshot_path:str=Form(...),plate_path:str=Form(...),video_path:str=Form(...),backup_path:str=Form(...),save_snapshots:str|None=Form(None),save_plate_images:str|None=Form(None),save_videos:str|None=Form(None),max_storage_gb:int=Form(0),storage_full_action:str=Form('delete_oldest'),retention_snapshots_days:int=Form(90),retention_plates_days:int=Form(90),retention_videos_days:int=Form(7),retention_events_days:int=Form(0)):
    if not auth(request):return RedirectResponse('/login',302)
    if not has_permission(request,'system.manage'):return access_denied()
    try:
        paths=_storage_paths(storage_root,snapshot_path,plate_path,video_path,backup_path)
        for x in paths: x.mkdir(parents=True,exist_ok=True)
        old_root=Path(get_setting('storage_root',str(DATA_DIR))).resolve(); new_root=paths[0]; restart=old_root!=new_root
        values={'storage_root':new_root,'snapshot_path':paths[1],'plate_path':paths[2],'video_path':paths[3],'backup_path':paths[4],'save_snapshots':'1' if save_snapshots else '0','save_plate_images':'1' if save_plate_images else '0','save_videos':'1' if save_videos else '0','max_storage_gb':max(0,max_storage_gb),'storage_full_action':storage_full_action if storage_full_action in {'delete_oldest','stop','alert'} else 'delete_oldest','retention_snapshots_days':max(0,retention_snapshots_days),'retention_plates_days':max(0,retention_plates_days),'retention_videos_days':max(0,retention_videos_days),'retention_events_days':max(0,retention_events_days)}
        if restart:
            database_target=new_root/'bcvision.db'
            persistent_names=['bcvision.db','.secret','license.json','.trial.json']
            if any((new_root/name).exists() for name in persistent_names):
                raise ValueError('مسیر جدید از قبل دارای اطلاعات BC Vision است؛ یک پوشه خالی انتخاب کنید.')
            create_database_backup(database_target)
            set_settings_for_database(database_target,values)
            for src_name in persistent_names[1:]:
                src=DATA_DIR/src_name; dst=new_root/src_name
                if src.exists(): shutil.copy2(src,dst)
            config_temp=STORAGE_CONFIG_PATH.with_suffix('.tmp')
            config_temp.write_text(json.dumps({'storage_root':str(new_root)},ensure_ascii=False,indent=2),encoding='utf-8')
            config_temp.replace(STORAGE_CONFIG_PATH)
        else:
            for k,v in values.items(): set_setting(k,v)
            run_retention_cleanup()
        return RedirectResponse('/settings?saved=1'+('&restart=1' if restart else '')+'#storage',303)
    except Exception as e:
        return RedirectResponse('/settings?error='+quote(str(e))+'#storage',303)

@app.get('/api/storage/status')
def api_storage_status(request:Request):
    if not auth(request): return JSONResponse({'error':'unauthorized'},status_code=401)
    root=get_setting('storage_root',str(DATA_DIR)); result=_path_usage(root)
    result.update({'max_storage_gb':_safe_int(get_setting('max_storage_gb','0')),'action':get_setting('storage_full_action','delete_oldest')})
    return JSONResponse(result)

@app.get('/api/system/status')
def api_system_status(request:Request):
    if not auth(request): return JSONResponse({'error':'unauthorized'},status_code=401)
    try:
        if psutil:
            cpu=psutil.cpu_percent(interval=None); ram=psutil.virtual_memory().percent; disk=psutil.disk_usage(get_setting('storage_root',str(DATA_DIR))).percent
        else:
            disk=shutil.disk_usage(get_setting('storage_root',str(DATA_DIR))); cpu=0; ram=0; disk=(disk.used/disk.total*100) if disk.total else 0
        return JSONResponse({'cpu':round(cpu,1),'ram':round(ram,1),'disk':round(disk,1)})
    except Exception:
        return JSONResponse({'cpu':0,'ram':0,'disk':0})

@app.get('/health')
def legacy_health():
    return JSONResponse({
        'service':'bc-vision',
        'status':'ok',
        'version':APP_VERSION,
        'opencv':CV_OK,
    })


@app.get('/license')
def license_page(request:Request,ok:int=0,error:str='',message:str=''):
    u=auth(request)
    if not u:return RedirectResponse('/login',302)
    if not has_permission(request,'license.manage'):return access_denied()
    s=license_status(); badge="ok" if s['valid'] else "bad"
    notice=(f"<div class='card ok'>{escape(message or 'لایسنس فعال شد.')}</div>" if ok else (f"<div class='alert'>{escape(error)}</div>" if error else ""))
    labels={'trial':'آزمایشی','basic':'پایه','professional':'حرفه‌ای','enterprise':'سازمانی'}
    feature_labels={'anpr':'پلاک‌خوان','events':'رویدادها','reports':'گزارش‌ها','vehicle_ai':'هوش خودرو','watchlist':'فهرست مراقبت','api':'API','gate':'کنترل راهبند','multi_site':'چند شعبه','priority_support':'پشتیبانی ویژه'}
    chips=''.join(f"<span class='status-pill ok'>{escape(feature_labels.get(x,x))}</span> " for x in s.get('features',[]))
    body=f"""<div class='wrap'><h1>مدیریت لایسنس</h1><p class='page-sub'>فعال‌سازی امن آنلاین یا آفلاین BC Vision</p>{notice}
    <div class='grid'><div class='card'><span class='muted'>وضعیت</span><div class='{badge}' style='font-size:20px;margin-top:10px'>{escape(s['message'])}</div></div>
    <div class='card'><span class='muted'>پلن</span><div class='stat'>{escape(labels.get(s['plan'],s['plan']))}</div></div>
    <div class='card'><span class='muted'>ظرفیت دوربین</span><div class='stat'>{s['camera_limit']}</div></div>
    <div class='card'><span class='muted'>تاریخ انقضا</span><div style='font-weight:900;font-size:18px'>{escape(str(s['expires_at']))}</div></div></div>
    <div class='card'><b>امکانات فعال</b><div style='margin-top:12px'>{chips or '—'}</div></div>
    <div class='card'><b>شناسه دستگاه</b><input class='code' readonly value='{machine_id()}' onclick='this.select()'><p class='muted'>برای صدور لایسنس آفلاین، این شناسه را برای مدیر فروش ارسال کنید.</p></div>
    <div class='grid'>
      <div class='card'><h3>فعال‌سازی آنلاین</h3><form method='post' action='/license/online'><label>آدرس سرور لایسنس</label><input name='server_url' placeholder='https://license.example.com' required><label>کد فعال‌سازی</label><input name='activation_code' style='direction:ltr' required><button>فعال‌سازی آنلاین</button></form></div>
      <div class='card'><h3>فعال‌سازی آفلاین</h3><form method='post' action='/license/offline'><label>محتوای فایل license.json</label><textarea name='license_text' rows='8' style='width:100%;direction:ltr;border:1px solid var(--bc-border);border-radius:9px;padding:10px;background:var(--bc-surface);color:var(--bc-text)' required></textarea><br><br><button>فعال‌سازی آفلاین</button></form></div>
    </div>
    <div class='card'><form method='post' action='/license/deactivate' onsubmit="return confirm('لایسنس از این دستگاه حذف شود؟')"><button class='danger'>حذف لایسنس از دستگاه</button></form></div></div>"""
    return page('لایسنس',body,u,request)

@app.post('/license')
@app.post('/license/offline')
def activate_license(request:Request,license_text:str=Form(...)):
    if not auth(request):return RedirectResponse('/login',302)
    if not has_permission(request,'license.manage'):return access_denied()
    ok,msg=install_license(license_text)
    return RedirectResponse('/license?ok=1&message='+quote(msg) if ok else '/license?error='+quote(msg),303)

@app.post('/license/online')
def activate_license_online(request:Request,server_url:str=Form(...),activation_code:str=Form(...)):
    if not auth(request):return RedirectResponse('/login',302)
    if not has_permission(request,'license.manage'):return access_denied()
    ok,msg=activate_online(server_url,activation_code)
    return RedirectResponse('/license?ok=1&message='+quote(msg) if ok else '/license?error='+quote(msg),303)

@app.post('/license/deactivate')
def deactivate_license(request:Request):
    if not auth(request):return RedirectResponse('/login',302)
    if not has_permission(request,'license.manage'):return access_denied()
    ok,msg=deactivate_local()
    return RedirectResponse('/license?ok=1&message='+quote(msg) if ok else '/license?error='+quote(msg),303)

@app.get('/events/export.csv')
def export_events(request:Request):
    if not auth(request):return RedirectResponse('/login',302)
    out=Path(get_setting('backup_path',str(BACKUP_DIR)));out.mkdir(parents=True,exist_ok=True);out=out / f"events-{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    with connect() as con, out.open('w',newline='',encoding='utf-8-sig') as f:
        rows=con.execute('SELECT id,plate_text,confidence,camera_name,vehicle_type,vehicle_color,vehicle_confidence,created_at FROM plate_events ORDER BY id DESC').fetchall()
        w=csv.writer(f);w.writerow(['ردیف','پلاک','اطمینان پلاک','دوربین','نوع خودرو','رنگ خودرو','اطمینان خودرو','تاریخ و ساعت شمسی'])
        w.writerows([
            (
                r['id'],
                _csv_cell(r['plate_text']),
                r['confidence'],
                _csv_cell(r['camera_name']),
                _csv_cell(r['vehicle_type']),
                _csv_cell(r['vehicle_color']),
                r['vehicle_confidence'],
                jalali_datetime(r['created_at']),
            )
            for r in rows
        ])
    return FileResponse(out,media_type='text/csv',filename=out.name)

@app.post('/settings/ai')
def save_ai_settings(request:Request, ai_accelerator:str=Form('auto'), ai_quality:str=Form('balanced'), ai_confidence:int=Form(85), ai_frames:int=Form(5)):
    u=auth(request)
    if not u:return RedirectResponse('/login',302)
    if not has_permission(request,'system.manage'):return access_denied()
    set_setting('ai_accelerator', ai_accelerator)
    set_setting('ai_quality', ai_quality)
    set_setting('ai_confidence', max(1,min(99,ai_confidence)))
    set_setting('ai_frames', max(1,min(20,ai_frames)))
    return RedirectResponse('/settings?saved=1',302)

@app.post('/backup')
def backup_database(request:Request):
    if not auth(request):return RedirectResponse('/login',302)
    if not has_permission(request,'system.manage'):return access_denied()
    out=_configured_storage_child('backup_path',BACKUP_DIR);out.mkdir(parents=True,exist_ok=True);out=out / f"bcvision-{datetime.now().strftime('%Y%m%d-%H%M%S')}.db"
    create_database_backup(out)
    return FileResponse(out,media_type='application/octet-stream',filename=out.name)


@app.post('/cameras/video-upload', response_class=HTMLResponse)
async def cameras_video_upload(request: Request, camera_id: int = Form(...), video: UploadFile = File(...)):
    u=auth(request)
    if not u: return RedirectResponse('/login',302)
    if not has_permission(request,'video.process'):return access_denied()
    wants_json=request.headers.get('x-requested-with')=='XMLHttpRequest'

    def upload_error(message, status_code=400):
        if wants_json:
            return JSONResponse({'ok':False,'error':str(message)},status_code)
        return page('خطای ویدئو',f"<div class='wrap'><div class='alert'>{escape(str(message))}</div><a class='btn' href='/cameras'>بازگشت</a></div>",u,request)

    suffix=_video_suffix(video.filename)
    if not suffix:
        return upload_error('فرمت فایل پشتیبانی نمی‌شود.')
    with connect() as con:
        source_camera=con.execute(
            "SELECT * FROM cameras WHERE id=? AND rtsp_url NOT LIKE 'video://%'",
            (camera_id,),
        ).fetchone()
    if not source_camera:
        return upload_error('دوربین انتخاب‌شده پیدا نشد.')
    if not source_camera['lpr_enabled']:
        return upload_error('پلاک‌خوان دوربین انتخاب‌شده غیرفعال است.')
    try:
        target=await _save_video_upload(video,_configured_storage_child('video_path',VIDEO_DIR),suffix)
    except ValueError as e:
        return upload_error(e)

    try:
        from app.ai.video_test import VideoTester
        tester=VideoTester(target)
        info=tester.info()
        tester.close()
        old_stream_ids=[]
        with connect() as con:
            old_stream_ids=[
                int(row['id']) for row in con.execute(
                    "SELECT id FROM cameras WHERE rtsp_url LIKE 'video://%'"
                ).fetchall()
            ]
            con.execute("DELETE FROM cameras WHERE rtsp_url LIKE 'video://%'")
            display_name=(Path(video.filename or target.name).stem or 'ویدئو')[:80]
            cursor=con.execute(
                "INSERT INTO cameras("
                "name,rtsp_url,location,enabled,is_demo,sort_order,"
                "lpr_enabled,lpr_confidence,frame_step,duplicate_seconds,"
                "roi_x,roi_y,roi_w,roi_h,line_y"
                ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    f"ویدئو: {display_name}",
                    f"video://{target}",
                    f"فایل آپلودی با تنظیمات {source_camera['name']}",
                    1,
                    1,
                    -1,
                    1,
                    source_camera['lpr_confidence'],
                    source_camera['frame_step'],
                    source_camera['duplicate_seconds'],
                    source_camera['roi_x'],
                    source_camera['roi_y'],
                    source_camera['roi_w'],
                    source_camera['roi_h'],
                    source_camera['line_y'],
                ),
            )
            virtual_camera_id=int(cursor.lastrowid)
        for old_id in old_stream_ids:
            manager.remove(old_id)
        manager.get(
            virtual_camera_id,
            f"video://{target}",
            f"ویدئو: {display_name}",
            int(get_setting('stream_width','640')),
            int(get_setting('live_fps','5')),
            int(get_setting('jpeg_quality','70')),
        )
        if wants_json:
            return JSONResponse({
                'ok':True,
                'camera_id':virtual_camera_id,
                'video':{
                    'frames':info['frames'],
                    'fps':info['fps'],
                    'duration':info['duration'],
                },
                'redirect':'/dashboard?video=1',
            })
        return RedirectResponse('/dashboard?video=1',303)
    except Exception as e:
        target.unlink(missing_ok=True)
        return upload_error(f'خطا در آماده‌سازی ویدئو: {e}',500)

# ---------- Video AI Test Upload ----------
@app.get('/ai/video-test', response_class=HTMLResponse)
def ai_video_test_page(request: Request):
    u=auth(request)
    if not u:
        return RedirectResponse('/login',302)
    if not has_permission(request,'video.process'):return access_denied()
    return page('تست ویدئو',"""
    <div style="direction:rtl;font-family:Tahoma;padding:30px">
    <h2>🧠 تست پلاک‌خوان با فایل ویدئو</h2>
    <form action="/ai/video-test/upload" method="post" enctype="multipart/form-data">
      <input type="file" name="video" accept=".mp4,.avi,.mkv,.mov" />
      <button type="submit">شروع تست AI</button>
    </form>
    </div>
    """,u,request)

@app.post('/ai/video-test/upload', response_class=HTMLResponse)
async def ai_video_test_upload(request: Request, video: UploadFile = File(...)):
    u=auth(request)
    if not u:
        return RedirectResponse('/login',302)
    if not has_permission(request,'video.process'):return access_denied()
    suffix=_video_suffix(video.filename)
    if not suffix:
        return page('خطای ویدئو',"<div class='wrap'><div class='alert'>فرمت فایل پشتیبانی نمی‌شود.</div></div>",u,request)
    try:
        target=await _save_video_upload(video,_configured_storage_child('video_path',VIDEO_DIR),suffix)
    except ValueError as e:
        return page('خطای ویدئو',f"<div class='wrap'><div class='alert'>{escape(str(e))}</div></div>",u,request)
    try:
        from app.ai.video_test import VideoTester
        tester = VideoTester(target)
        info = tester.info()
        tester.close()
        return page('نتیجه آماده‌سازی ویدئو',f"""
        <div style="direction:rtl;font-family:Tahoma;padding:30px">
        <h2>نتیجه آماده‌سازی ویدئو</h2>
        <p>فایل: {escape(video.filename or '')}</p>
        <p>فریم: {info['frames']}</p>
        <p>FPS: {info['fps']}</p>
        <p>رزولوشن: {info['width']}x{info['height']}</p>
        <p>مرحله بعد: اتصال همین ورودی به موتور تشخیص پلاک</p>
        </div>
        """,u,request)
    except Exception as e:
        return page('خطای ویدئو',f"<div class='wrap'><div class='alert'>خطا: {escape(str(e))}</div></div>",u,request)
