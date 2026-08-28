from app.cpu_budget import configure_process_cpu_budget

configure_process_cpu_budget()

from fastapi import Depends, FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, StreamingResponse, FileResponse
from app.config import (APP_NAME, COMPANY_NAME, APP_VERSION, DB_PATH, BACKUP_DIR,
    DATA_DIR, STORAGE_CONFIG_PATH, STORAGE_MIGRATION_MARKER_NAME,
    SNAPSHOT_DIR, PLATE_DIR, VIDEO_DIR, ensure_storage_migration_marker)
from app.database import (
    backup_database as create_database_backup,
    connect,
    get_setting,
    init_db,
    set_settings_for_database,
    set_setting,
)
from app.security import (
    COOKIE_NAME,
    create_token,
    hash_password,
    read_session,
    read_session_details,
    session_fingerprint,
    verify_password,
)
from app.streams import manager, CV_OK
from app.csv_export import csv_safe_cell, iter_event_csv
from app.file_identity import descriptor_file_identity, path_file_identity
from app.storage_policy import (
    StoragePolicyError,
    WriterPreferredGate,
    begin_media_write,
    delete_older_than,
    enforce_storage_limit,
    fsync_parent_directory,
    invalidate_storage_cache,
    pin_media_paths,
    require_media_writes_quiescent,
    storage_status,
    validate_storage_layout,
)
from app.upload_limits import VideoUploadBodyLimitMiddleware
from app.update_package import (
    MAX_UPDATE_ZIP_BYTES,
    UpdatePackageError,
    exit_after_update_launch,
    launch_staged_update,
    stage_update_zip,
    validate_update_target,
)
from html import escape
import asyncio, time, shutil, os, json, secrets, stat, tempfile, threading
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from functools import wraps
from urllib.parse import parse_qsl, quote, urlencode, urlsplit
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from app.ai.plate_rules import (
    iran_plate_parts,
    normalize_plate as canonical_normalize_plate,
    persian_digits,
    split_iran_plate,
)
from app.ai.feedback import invalidate_feedback_cache, validate_correction
from app.ai.evaluation import character_distance, feedback_quality_summary
from app.ai.training import (
    apply_candidate,
    capture_feedback_sample,
    close_pending_feedback_source_pins,
    evaluate_candidate_on_golden,
    latest_training_status,
    reconcile_feedback_sample_files,
    recover_pending_feedback_samples,
    refresh_pending_feedback_source_pins,
    start_training,
)
from app.async_jobs import (
    SubprocessJobTimeout,
    run_module_job_subprocess,
    run_to_thread_quiescent,
)

try:
    import psutil
except Exception:
    psutil = None


_ANPR_V3_DETECTOR_MIGRATION = "migration_anpr_v3_yolo11n_default_v1"
_ANPR_V2_SAFE_SHADOW_MIGRATION = "migration_anpr_v2_safe_shadow_v1"
APP_RELEASE_LABEL = (
    "RC" + APP_VERSION.rsplit("-rc", 1)[1]
    if "-rc" in APP_VERSION
    else APP_VERSION
)
_STORAGE_MUTATION_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_STORAGE_MUTATING_GET_PATHS = frozenset({
    "/logout",
})
_STORAGE_MUTATION_GATE = WriterPreferredGate()
_STORAGE_RESTART_REQUIRED = threading.Event()
_PASSWORD_CHANGE_ALLOWED_PATHS = frozenset({
    "/login",
    "/logout",
    "/setup",
    "/settings",
    "/settings/display",
    "/api/system/status",
    "/health",
})
_VIDEO_UPLOAD_PATHS = frozenset({
    "/cameras/video-upload",
    "/ai/video-test/upload",
})
_UPDATE_UPLOAD_PATHS = frozenset({"/settings/update"})
MAX_VIDEO_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024
MAX_VIDEO_MULTIPART_OVERHEAD_BYTES = 1024 * 1024
VIDEO_TEST_JOB_TIMEOUT_SECONDS = 30 * 60
VIDEO_TEST_MEDIA_RESULT_BYTES = 48 * 1024 * 1024
VIDEO_TEST_PROCESS_RESULT_BYTES = 64 * 1024 * 1024
_VIDEO_CAMERA_HANDOFF_LOCK = threading.Lock()
_VIDEO_TEST_PROCESS_SLOT = threading.BoundedSemaphore(1)
_UPDATE_STAGE_ROOT = DATA_DIR / "updates"
_UPDATE_UPLOAD_LOCK = threading.Lock()


def _max_declared_video_upload_bytes() -> int:
    # Content-Length covers the multipart envelope as well as the file. Keep
    # the allowance proportional for tests/small deployments while capping it
    # at 1 MiB for the production 2 GiB file limit.
    overhead = min(
        MAX_VIDEO_MULTIPART_OVERHEAD_BYTES,
        max(0, MAX_VIDEO_UPLOAD_BYTES // 1024),
    )
    return MAX_VIDEO_UPLOAD_BYTES + overhead


async def _acquire_video_test_process_slot() -> None:
    """Wait cancellably without leaking a semaphore waiter thread."""

    while not _VIDEO_TEST_PROCESS_SLOT.acquire(blocking=False):
        await asyncio.sleep(0.05)


def _uses_storage_mutation_gate(request: Request) -> bool:
    method = request.method.upper()
    if method in _STORAGE_MUTATION_METHODS:
        return True
    if method != "GET":
        return False
    path = request.url.path
    return (
        path in _STORAGE_MUTATING_GET_PATHS
        or path.startswith("/live/")
    )


def _uses_exclusive_storage_gate(request: Request) -> bool:
    return (
        request.method.upper() == "POST"
        and request.url.path == "/settings/storage"
    )


def _storage_restart_response() -> JSONResponse:
    return JSONResponse(
        {
            "error": "storage-restart-required",
            "detail": (
                "Storage migration completed; restart BC Vision "
                "before making further changes."
            ),
        },
        status_code=503,
    )


class _StorageRestartRequired(RuntimeError):
    pass


async def _storage_mutation_dependency(request: Request):
    """Quiesce mutations after FastAPI has parsed request bodies.

    FastAPI reads form/multipart bodies before solving dependencies. Keeping
    the reader/writer gate here prevents slow clients from holding a shared or
    exclusive slot while they are still transmitting a request.
    """

    if not _uses_storage_mutation_gate(request):
        return

    exclusive = _uses_exclusive_storage_gate(request)
    # An unauthenticated storage-settings request cannot mutate anything and
    # must not be allowed to queue a writer ticket that stalls real users.
    if exclusive and (
        not auth(request)
        or not has_permission(request, 'system.manage')
    ):
        return

    gate_mode = None
    writer_ticket = None
    try:
        if exclusive:
            writer_ticket = _STORAGE_MUTATION_GATE.queue_exclusive()
            try:
                while not _STORAGE_MUTATION_GATE.try_acquire_exclusive(
                    writer_ticket
                ):
                    await asyncio.sleep(0.01)
            except BaseException:
                _STORAGE_MUTATION_GATE.cancel_exclusive(writer_ticket)
                writer_ticket = None
                raise
            gate_mode = "exclusive"
        else:
            while not _STORAGE_MUTATION_GATE.try_acquire_shared():
                await asyncio.sleep(0.01)
            gate_mode = "shared"
        request.state.storage_gate_mode = gate_mode
        request.state.storage_writer_ticket = writer_ticket
        if _STORAGE_RESTART_REQUIRED.is_set():
            raise _StorageRestartRequired
    except BaseException:
        # Once ownership is published on request.state the outer middleware
        # releases it after the endpoint returns (but before a streaming body
        # is consumed). Failures before publication are cleaned up here.
        if gate_mode is None and writer_ticket is not None:
            _STORAGE_MUTATION_GATE.cancel_exclusive(writer_ticket)
        raise


def _release_request_storage_gate(request: Request) -> None:
    gate_mode = getattr(request.state, "storage_gate_mode", None)
    writer_ticket = getattr(
        request.state,
        "storage_writer_ticket",
        None,
    )
    request.state.storage_gate_mode = None
    request.state.storage_writer_ticket = None
    if gate_mode == "shared":
        _STORAGE_MUTATION_GATE.release_shared()
    elif gate_mode == "exclusive":
        _STORAGE_MUTATION_GATE.release_exclusive(writer_ticket)


def _password_change_required(request: Request) -> bool:
    session = read_session(request)
    if not session:
        return False
    username, session_version = session
    fingerprint = session_fingerprint(request)
    if not fingerprint:
        return False
    with connect() as con:
        row = con.execute(
            "SELECT must_change_password,session_version FROM users "
            "WHERE username=? AND is_active=1 AND session_version=? "
            "AND NOT EXISTS(SELECT 1 FROM revoked_sessions "
            "WHERE token_hash=? AND expires_at>=?)",
            (
                username,session_version,fingerprint,int(time.time()),
            ),
        ).fetchone()
    return bool(
        row
        and int(row["session_version"] or 0) == session_version
        and int(row["must_change_password"] or 0)
    )


def _session_revoked(connection, request: Request) -> bool:
    fingerprint = session_fingerprint(request)
    if not fingerprint:
        return True
    return connection.execute(
        "SELECT 1 FROM revoked_sessions WHERE token_hash=? "
        "AND expires_at>=?",
        (fingerprint, int(time.time())),
    ).fetchone() is not None


def _migrate_anpr_v3_detector_selection():
    """Select YOLO11n once without overriding later operator choices."""
    if get_setting(_ANPR_V3_DETECTOR_MIGRATION, "") == "1":
        return False
    set_setting("anpr_detector_model", "yolo11n")
    set_setting(_ANPR_V3_DETECTOR_MIGRATION, "1")
    return True


def _migrate_anpr_v2_to_safe_shadow():
    """Demote every installed V2 primary setting exactly once."""
    if get_setting(_ANPR_V2_SAFE_SHADOW_MIGRATION, "") == "1":
        return False
    set_setting("anpr_engine_v2_shadow", "0")
    set_setting(_ANPR_V2_SAFE_SHADOW_MIGRATION, "1")
    return True


init_db()
_migrate_anpr_v3_detector_selection()
_migrate_anpr_v2_to_safe_shadow()


@asynccontextmanager
async def _application_lifespan(_app):
    from app.ai.live_worker import start_live_anpr_worker

    worker_started=False
    try:
        # Reconcile and pin every pending truth source before any retention,
        # quota enforcement, live worker, or camera decoder can mutate media.
        await asyncio.to_thread(reconcile_feedback_sample_files)
        await asyncio.to_thread(refresh_pending_feedback_source_pins)
        await asyncio.to_thread(recover_pending_feedback_samples, 256)
        # Quota and age checks are no longer tied to saving the settings form.
        await asyncio.to_thread(run_retention_cleanup)
        start_live_anpr_worker()
        worker_started=True
        manager.start_enabled_cameras()
        yield
    finally:
        from app.ai.live_worker import shutdown_live_anpr_worker

        manager.stop_all()
        if worker_started:
            shutdown_live_anpr_worker(retry_timeout=5.0)
        close_pending_feedback_source_pins()


app = FastAPI(
    title=f"{APP_NAME} | {COMPANY_NAME}",
    docs_url="/api/docs",
    redoc_url=None,
    lifespan=_application_lifespan,
    dependencies=[Depends(_storage_mutation_dependency)],
)


@app.exception_handler(_StorageRestartRequired)
async def _storage_restart_required_handler(_request, _exception):
    return _storage_restart_response()

try:
    APP_LOCAL_TIMEZONE = ZoneInfo("Asia/Tehran")
except ZoneInfoNotFoundError:
    # Windows may not expose an IANA timezone database. Iran currently uses a
    # fixed UTC+03:30 offset, so local time remains correct without tzdata.
    APP_LOCAL_TIMEZONE = timezone(timedelta(hours=3, minutes=30))


@app.middleware("http")
async def security_headers(request: Request, call_next):
    # Reject an oversized body before waiting behind a storage migration. This
    # avoids buffering/spooling a known-invalid multipart upload while the
    # exclusive gate is occupied.
    content_length = request.headers.get("content-length", "").strip()
    if request.url.path in _VIDEO_UPLOAD_PATHS and content_length:
        try:
            declared_size = int(content_length)
        except ValueError:
            declared_size = -1
        if declared_size < 0:
            response = JSONResponse(
                {"error": "invalid-content-length"},
                status_code=400,
            )
        elif declared_size > _max_declared_video_upload_bytes():
            response = JSONResponse(
                {"error": "video-upload-too-large"},
                status_code=413,
            )
        else:
            response = None
    else:
        response = None

    if response is None and request.url.path in _VIDEO_UPLOAD_PATHS:
        if not auth(request):
            response = RedirectResponse('/login', 302)
        elif not has_permission(request, 'video.process'):
            response = access_denied()
    if response is None and request.url.path in _UPDATE_UPLOAD_PATHS:
        if not auth(request):
            response = RedirectResponse('/login', 302)
        elif not has_permission(request, 'system.manage'):
            response = access_denied()
    if response is None and request.url.path == '/settings/storage':
        if not auth(request):
            response = RedirectResponse('/login', 302)
        elif not has_permission(request, 'system.manage'):
            response = access_denied()

    try:
        if response is not None:
            pass
        elif (
            request.url.path not in _PASSWORD_CHANGE_ALLOWED_PATHS
            and _password_change_required(request)
        ):
            if (
                request.url.path.startswith("/api/")
                or request.url.path.startswith("/live/")
            ):
                response = JSONResponse(
                    {
                        "error": "password-change-required",
                        "detail": "Change the bootstrap password first.",
                    },
                    status_code=403,
                )
            else:
                response = RedirectResponse(
                    "/settings?password_required=1",
                    status_code=303,
                )
        else:
            response = await call_next(request)
    finally:
        _release_request_storage_gate(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault(
        "Permissions-Policy",
        "camera=(), microphone=(), geolocation=()",
    )
    return response


# Install this after the function-based middleware so raw request bytes are
# capped before a chunked upload can acquire even a shared storage-gate slot.
app.add_middleware(
    VideoUploadBodyLimitMiddleware,
    max_body_bytes=_max_declared_video_upload_bytes,
    paths=_VIDEO_UPLOAD_PATHS | _UPDATE_UPLOAD_PATHS,
    path_body_limits={
        "/settings/update": MAX_UPDATE_ZIP_BYTES + 1024 * 1024,
    },
)


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

def _as_local_datetime(value):
    dt=_parse_dt(value)
    if not dt:return None
    if dt.tzinfo is None:
        dt=dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(APP_LOCAL_TIMEZONE)

def _local_to_utc_naive(value):
    if value is None:return None
    local=value.replace(tzinfo=APP_LOCAL_TIMEZONE)
    return local.astimezone(timezone.utc).replace(tzinfo=None)

def _utc_now_text():
    return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S.%f')

def _local_day_utc_bounds(value=None):
    local_now=value or datetime.now(APP_LOCAL_TIMEZONE)
    start=local_now.replace(hour=0,minute=0,second=0,microsecond=0)
    return (
        start.astimezone(timezone.utc).replace(tzinfo=None),
        (start+timedelta(days=1)).astimezone(timezone.utc).replace(
            tzinfo=None
        ),
    )

def jalali_datetime(value, with_seconds=True):
    dt=_as_local_datetime(value)
    if not dt: return '—'
    jy,jm,jd=_gregorian_to_jalali(dt.year,dt.month,dt.day)
    t=dt.strftime('%H:%M:%S' if with_seconds else '%H:%M')
    return persian_digits(f'{jy:04d}/{jm:02d}/{jd:02d} - {t}')

def jalali_date(value):
    dt=_as_local_datetime(value)
    if not dt:return ''
    jy,jm,jd=_gregorian_to_jalali(dt.year,dt.month,dt.day)
    return persian_digits(f'{jy:04d}/{jm:02d}/{jd:02d}')

def normalize_plate(text):
    return canonical_normalize_plate(text)


_ALL_DIGITS = str.maketrans(
    "۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩",
    "01234567890123456789",
)


def _jalali_to_gregorian(jy, jm, jd):
    jy = int(jy) + 1595
    jm = int(jm)
    jd = int(jd)
    if not 1 <= jm <= 12 or jd < 1:
        raise ValueError("تاریخ شمسی معتبر نیست.")
    month_days = 31 if jm <= 6 else 30
    if jm == 12:
        month_days = 30
    if jd > month_days:
        raise ValueError("تاریخ شمسی معتبر نیست.")
    days = (
        -355668
        + (365 * jy)
        + ((jy // 33) * 8)
        + (((jy % 33) + 3) // 4)
        + jd
    )
    days += (jm - 1) * 31 if jm < 7 else ((jm - 7) * 30) + 186
    gy = 400 * (days // 146097)
    days %= 146097
    if days > 36524:
        days -= 1
        gy += 100 * (days // 36524)
        days %= 36524
        if days >= 365:
            days += 1
    gy += 4 * (days // 1461)
    days %= 1461
    if days > 365:
        gy += (days - 1) // 365
        days = (days - 1) % 365
    gd = days + 1
    leap = gy % 4 == 0 and (gy % 100 != 0 or gy % 400 == 0)
    gregorian_month_days = [
        0, 31, 29 if leap else 28, 31, 30, 31, 30,
        31, 31, 30, 31, 30, 31,
    ]
    gm = 1
    while gm <= 12 and gd > gregorian_month_days[gm]:
        gd -= gregorian_month_days[gm]
        gm += 1
    if gm > 12:
        raise ValueError("تاریخ شمسی معتبر نیست.")
    converted = datetime(gy, gm, gd)
    if _gregorian_to_jalali(gy, gm, gd) != (
        int(jy) - 1595,
        int(jm),
        int(jd),
    ):
        raise ValueError("تاریخ شمسی معتبر نیست.")
    return converted


def _parse_jalali_date(value):
    text = str(value or "").strip().translate(_ALL_DIGITS)
    if not text:
        return None
    parts = text.replace("-", "/").replace(".", "/").split("/")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise ValueError("تاریخ شمسی را مانند ۱۴۰۵/۰۵/۰۸ وارد کنید.")
    jy, jm, jd = (int(part) for part in parts)
    if not 1200 <= jy <= 1600:
        raise ValueError("سال شمسی معتبر نیست.")
    return _jalali_to_gregorian(jy, jm, jd)


def _parse_time(value):
    text = str(value or "").strip().translate(_ALL_DIGITS)
    if not text:
        return None
    try:
        return datetime.strptime(text, "%H:%M").time()
    except ValueError as exc:
        raise ValueError("ساعت را مانند ۱۴:۳۰ وارد کنید.") from exc


def _plate_region(text):
    parts = split_iran_plate(text)
    return parts["region"] if parts else ""


def _page_url(path, params, page_key, value):
    query = {
        key: str(item)
        for key, item in params.items()
        if item not in (None, "")
    }
    query[page_key] = str(value)
    return f"{path}?{urlencode(query)}"


def _dashboard_query_int(value, default=0, minimum=0, maximum=2**63-1):
    try:
        parsed = int(str(value or "").strip())
    except (TypeError, ValueError):
        parsed = int(default)
    return max(int(minimum), min(int(maximum), parsed))


def _dashboard_url(
    *,
    video=0,
    events_camera=0,
    events_after=0,
    events_snapshot=0,
    events_page=1,
    corrected=0,
):
    """Build the only dashboard return URL accepted from correction forms."""

    params = {}
    if _dashboard_query_int(video, maximum=1):
        params["video"] = "1"
    camera = _dashboard_query_int(events_camera)
    after = _dashboard_query_int(events_after)
    snapshot = _dashboard_query_int(events_snapshot)
    page_number = _dashboard_query_int(
        events_page,
        default=1,
        minimum=1,
        maximum=1000000,
    )
    if camera:
        params["events_camera"] = str(camera)
    if after:
        params["events_after"] = str(after)
    if snapshot:
        params["events_snapshot"] = str(snapshot)
    if page_number > 1:
        params["events_page"] = str(page_number)
    if _dashboard_query_int(corrected, maximum=1):
        params["corrected"] = "1"
    return "/dashboard" + (f"?{urlencode(params)}" if params else "")


def _safe_dashboard_return_to(value, *, corrected=False):
    """Accept a local dashboard URL and rebuild it from allow-listed fields."""

    raw = str(value or "/dashboard").strip()
    try:
        parsed = urlsplit(raw)
    except ValueError:
        parsed = urlsplit("/dashboard")
    if (
        parsed.scheme
        or parsed.netloc
        or parsed.fragment
        or parsed.path != "/dashboard"
    ):
        parsed = urlsplit("/dashboard")
    query = dict(parse_qsl(parsed.query, keep_blank_values=False))
    return _dashboard_url(
        video=query.get("video", 0),
        events_camera=query.get("events_camera", 0),
        events_after=query.get("events_after", 0),
        events_snapshot=query.get("events_snapshot", 0),
        events_page=query.get("events_page", 1),
        corrected=1 if corrected else 0,
    )


def _dashboard_navigation_url(request):
    if request is None or request.url.path != "/dashboard":
        return "/dashboard"
    query = request.query_params
    return _dashboard_url(
        video=query.get("video", 0),
        events_camera=query.get("events_camera", 0),
        events_after=query.get("events_after", 0),
    )


def pagination_html(
    path,
    current,
    total_pages,
    total_items,
    params,
    page_key,
    page_size,
):
    total_pages = max(1, int(total_pages))
    current = max(1, min(total_pages, int(current)))
    total_items = max(0, int(total_items))
    if total_items:
        start = (current - 1) * page_size + 1
        end = min(total_items, current * page_size)
        summary = (
            f"نمایش {persian_digits(start)} تا {persian_digits(end)} "
            f"از {persian_digits(total_items)} رکورد"
        )
    else:
        summary = "بدون رکورد"
    visible = {1, total_pages}
    visible.update(
        range(max(1, current - 2), min(total_pages, current + 2) + 1)
    )
    page_links = []
    previous = None
    for number in sorted(visible):
        if previous is not None and number - previous > 1:
            page_links.append("<span class='page-gap'>…</span>")
        cls = "page-number active" if number == current else "page-number"
        page_links.append(
            f"<a class='{cls}' href='"
            f"{escape(_page_url(path, params, page_key, number))}'>"
            f"{persian_digits(number)}</a>"
        )
        previous = number
    previous_link = (
        f"<a class='page-nav' href='"
        f"{escape(_page_url(path, params, page_key, current - 1))}'>"
        "قبلی</a>"
        if current > 1
        else "<span class='page-nav disabled'>قبلی</span>"
    )
    next_link = (
        f"<a class='page-nav' href='"
        f"{escape(_page_url(path, params, page_key, current + 1))}'>"
        "بعدی</a>"
        if current < total_pages
        else "<span class='page-nav disabled'>بعدی</span>"
    )
    return (
        "<div class='pagination'>"
        f"<div class='pagination-summary'>{summary} — صفحه "
        f"{persian_digits(current)} از {persian_digits(total_pages)}</div>"
        f"<div class='pagination-controls'>{previous_link}"
        f"{''.join(page_links)}{next_link}</div></div>"
    )

def event_status_badge(status):
    labels={'allowed':'مجاز','blocked':'غیرمجاز','vip':'VIP','unknown':'ثبت‌نشده'}
    classes={'allowed':'ok','blocked':'bad','vip':'vip','unknown':'muted'}
    st=status if status in labels else 'unknown'
    return f"<span class='status-pill {classes[st]}'>{labels[st]}</span>"


def iran_plate_html(text, compact=False):
    parts = iran_plate_parts(text)
    if parts is None:
        value = persian_digits(text) if text else "ناخوانا"
        return f"<span class='plate-unreadable'>{escape(value)}</span>"
    size = " compact" if compact else ""
    return (
        f"<span class='iran-plate{size}' dir='ltr'>"
        f"<span class='plate-blue'>🇮🇷<small>I.R.</small></span>"
        f"<span class='plate-main'><b>{escape(parts['prefix'])}</b>"
        f"<b>{escape(parts['letter'])}</b>"
        f"<b>{escape(parts['serial'])}</b></span>"
        f"<span class='plate-iran'><small>ایران</small>"
        f"<b>{escape(parts['region'])}</b></span>"
        f"</span>"
    )


def anpr_confirmation_badge(status):
    value = str(status or "confirmed-ai")
    return {
        "auto-confirmed": (
            "<span class='read-badge auto-confirmed'>"
            "تأیید خودکار مدل؛ قابل اصلاح</span>"
        ),
        "suggested": (
            "<span class='read-badge suggested'>"
            "خوانش احتمالی؛ اصلاح کنید</span>"
        ),
        "unreadable": (
            "<span class='read-badge unreadable'>واقعاً ناخوانا</span>"
        ),
        "confirmed": (
            "<span class='read-badge confirmed'>"
            "تأیید اپراتور و ثبت برای آموزش</span>"
        ),
        "confirmed-ai": (
            "<span class='read-badge confirmed-ai'>"
            "خوانش قطعی چندفریمی</span>"
        ),
    }.get(value, "")


def dashboard_event_row(row, return_to="/dashboard"):
    image_path = row["image_path"] or ""
    plate_path = row["plate_image_path"] or ""
    vehicle = (
        f"<a href='/events/{row['id']}'><img class='recent-vehicle-thumb' "
        f"src='/media?path={quote(image_path)}' alt='تصویر خودرو'></a>"
        if image_path and Path(image_path).is_file()
        else "<span class='recent-media-missing'>بدون تصویر خودرو</span>"
    )
    plate_image = (
        f"<a href='/events/{row['id']}'><img class='thumb plate-thumb' "
        f"src='/media?path={quote(plate_path)}' alt='تصویر پلاک'></a>"
        if plate_path and Path(plate_path).is_file()
        else "<span class='recent-media-missing' "
        "style='width:130px;height:48px'>بدون تصویر پلاک</span>"
    )
    review_status = (
        row["review_status"]
        if "review_status" in row.keys()
        else "confirmed-ai"
    )
    review_badge = anpr_confirmation_badge(review_status)
    city = (
        str(row["city"] or "")
        if "city" in row.keys()
        else ""
    )
    media_status = (
        str(row["media_status"] or "")
        if "media_status" in row.keys()
        else ""
    )
    media_warning = (
        "<span class='read-badge unreadable'>خطای ذخیره تصویر</span>"
        if media_status in {"error", "partial"}
        else ""
    )
    city_html = (
        f"<br><small class='muted'>{escape(city)}</small>"
        if city
        else ""
    )
    current_plate = str(row["plate_text"] or "")
    correction_value = (
        f" value='{escape(current_plate)}'"
        if current_plate not in {"", "ناخوانا", "در حال بررسی"}
        else ""
    )
    safe_return_to = _safe_dashboard_return_to(return_to)
    return (
        f"<tr><td>{vehicle}</td>"
        f"<td><div class='recent-plate-result'>{plate_image}"
        f"<div>{iran_plate_html(row['plate_text'], True)}"
        f"{review_badge}{media_warning}</div></div></td>"
        f"<td>{escape(row['camera_name'] or '—')}{city_html}</td>"
        f"<td>{persian_digits(int((row['confidence'] or 0) * 100))}٪</td>"
        f"<td>{persian_digits(jalali_datetime(row['created_at'], False))}</td>"
        f"<td><form class='correction-form' method='post' "
        f"action='/events/{row['id']}/correct'>"
        f"<input type='hidden' name='return_to' "
        f"value='{escape(safe_return_to, quote=True)}'>"
        f"<input name='corrected_plate' required maxlength='20' "
        f"{correction_value} "
        f"placeholder='مثال: ۱۲ ب ۳۴۵ ایران ۶۷'>"
        f"<button>تأیید/اصلاح و آموزش</button></form></td></tr>"
    )


CSS = """<style>
:root{--bc-navy:#071b3f;--bc-navy2:#0b2e63;--bc-blue:#087cf0;--bc-cyan:#11a8f7;--bc-bg:#f3f6fb;--bc-surface:#fff;--bc-surface2:#f7f9fc;--bc-border:#e1e8f0;--bc-text:#172033;--bc-muted:#69778d;--bc-radius:16px;--bc-shadow:0 10px 30px rgba(10,38,78,.08);--sidebar:258px}
[data-theme=dark]{--bc-bg:#0b1220;--bc-surface:#111b2d;--bc-surface2:#162238;--bc-border:#25344d;--bc-text:#e8eef8;--bc-muted:#9aabc3;--bc-shadow:0 12px 34px rgba(0,0,0,.25)}
*{box-sizing:border-box}html{direction:rtl;scroll-behavior:smooth}body{margin:0;background:var(--bc-bg);color:var(--bc-text);font-family:Tahoma,"Segoe UI",Arial,sans-serif;font-size:14px;line-height:1.75;text-rendering:optimizeLegibility;-webkit-font-smoothing:antialiased;transition:background .2s,color .2s}a{color:inherit}
.app-shell{min-height:100vh}.sidebar{position:fixed;top:0;right:0;bottom:0;width:var(--sidebar);background:linear-gradient(175deg,var(--bc-navy),#06152f 75%);color:#dcecff;z-index:1100;padding:18px 13px;display:flex;flex-direction:column;box-shadow:-6px 0 24px rgba(0,0,0,.12);transition:width .22s,transform .22s}.sidebar.collapsed{width:84px}.brand-row{display:flex;align-items:center;gap:11px;padding:4px 7px 20px;border-bottom:1px solid rgba(255,255,255,.1);text-decoration:none;color:#fff;overflow:hidden}.brand-mark{min-width:45px;width:45px;height:45px;border-radius:14px;background:linear-gradient(145deg,var(--bc-cyan),var(--bc-blue));display:grid;place-items:center;font-weight:900;font-size:19px;box-shadow:0 7px 20px rgba(17,168,247,.28)}.brand-copy{white-space:nowrap}.brand-copy b{display:block;font-size:17px;line-height:1.2}.brand-copy small{opacity:.65;font-size:10px}.nav-menu{margin-top:16px;display:flex;flex-direction:column;gap:5px}.nav-menu a{display:flex;align-items:center;gap:12px;padding:11px 13px;border-radius:11px;text-decoration:none;color:#cfe2fb;white-space:nowrap;overflow:hidden;transition:.18s}.nav-menu a:hover,.nav-menu a.active{background:linear-gradient(100deg,rgba(17,168,247,.23),rgba(8,124,240,.08));color:#fff}.nav-icon{min-width:25px;width:25px;text-align:center;font-size:19px}.sidebar-foot{margin-top:auto}.sidebar-foot a{display:flex;align-items:center;gap:12px;padding:11px 13px;border-radius:11px;text-decoration:none;color:#cfe2fb}.sidebar.collapsed .brand-copy,.sidebar.collapsed .nav-label{opacity:0;width:0;overflow:hidden}.sidebar-toggle{position:absolute;left:-15px;top:78px;width:30px;height:30px;border-radius:50%;border:2px solid var(--bc-bg);background:var(--bc-blue);color:#fff;cursor:pointer;display:grid;place-items:center;padding:0;box-shadow:0 5px 15px rgba(0,0,0,.2)}
.main{margin-right:var(--sidebar);min-height:100vh;transition:margin-right .22s}.main.collapsed{margin-right:84px}.topbar{height:70px;background:color-mix(in srgb,var(--bc-surface) 92%,transparent);backdrop-filter:blur(12px);border-bottom:1px solid var(--bc-border);display:flex;align-items:center;gap:12px;padding:0 24px;position:sticky;top:0;z-index:900}.top-title{font-size:18px;font-weight:900;color:var(--bc-navy);margin-left:auto}.version-chip{direction:ltr;white-space:nowrap;padding:5px 9px;border:1px solid #b8d8ff;background:#eaf4ff;color:#075ca8;border-radius:999px;font-size:11px;font-weight:900}[data-theme=dark] .version-chip{border-color:#235d96;background:#102d4d;color:#8cc8ff}.resource-strip{display:flex;align-items:center;gap:7px;direction:ltr}.resource-chip{display:flex;align-items:center;gap:5px;min-width:66px;padding:5px 8px;border:1px solid var(--bc-border);background:var(--bc-surface);border-radius:10px;font-size:12px;font-weight:800}.resource-dot{width:8px;height:8px;border-radius:50%;background:#22a06b;box-shadow:0 0 0 3px rgba(34,160,107,.12)}.resource-chip.warn .resource-dot{background:#e5a11a}.resource-chip.danger .resource-dot{background:#d64545}.storage-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.drive-card{border:1px solid var(--bc-border);background:var(--bc-surface2);border-radius:13px;padding:14px}.storage-progress{height:10px;background:var(--bc-border);border-radius:8px;overflow:hidden;margin:9px 0}.storage-progress span{display:block;height:100%;background:linear-gradient(90deg,var(--bc-blue),var(--bc-cyan));border-radius:8px}[data-theme=dark] .top-title{color:#eaf2ff}.top-action{width:40px;height:40px;border-radius:11px;border:1px solid var(--bc-border);background:var(--bc-surface);color:var(--bc-text);display:grid;place-items:center;cursor:pointer;box-shadow:none;padding:0}.user-chip{display:flex;align-items:center;gap:9px;border:1px solid var(--bc-border);background:var(--bc-surface);padding:6px 10px;border-radius:12px}.avatar{width:29px;height:29px;border-radius:9px;background:linear-gradient(135deg,var(--bc-blue),var(--bc-cyan));color:#fff;display:grid;place-items:center;font-weight:900}.wrap{max-width:1550px;margin:auto;padding:25px}.page-title{font-weight:900;font-size:28px;margin:0;color:var(--bc-navy)}[data-theme=dark] .page-title,[data-theme=dark] h1{color:#edf4ff}h1{font-size:27px;font-weight:900;color:var(--bc-navy)}h3{font-size:18px;font-weight:800}.page-sub{color:var(--bc-muted);margin:2px 0 0}
.card{background:var(--bc-surface);border:1px solid var(--bc-border);border-radius:var(--bc-radius);padding:20px;box-shadow:var(--bc-shadow);margin-bottom:17px}.login{max-width:430px;margin:7vh auto;padding:30px}.login .brand{text-align:center;font-size:28px;margin-bottom:3px}.login .muted{text-align:center}.brand{font-size:25px;font-weight:900;color:var(--bc-navy)}.muted{color:var(--bc-muted)}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px}.stats-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:15px;margin-bottom:17px}.stat-card{position:relative;overflow:hidden}.stat-card:after{content:"";position:absolute;width:90px;height:90px;border-radius:50%;left:-25px;top:-28px;background:rgba(8,124,240,.07)}.stat-head{display:flex;align-items:center;justify-content:space-between}.stat-icon{width:43px;height:43px;border-radius:13px;background:rgba(8,124,240,.1);color:var(--bc-blue);display:grid;place-items:center;font-size:21px}.stat{font-size:31px;font-weight:900;color:var(--bc-text);margin-top:5px;line-height:1.2}.trend{font-size:12px;color:var(--bc-muted)}
label{display:block;font-weight:700;color:var(--bc-text);margin-bottom:3px}input,select,textarea,button{font-family:inherit;font-size:14px}input:not([type=checkbox]),select,textarea{width:100%;padding:10px 12px;border:1px solid var(--bc-border);border-radius:10px;margin:5px 0 13px;background:var(--bc-surface);color:var(--bc-text);outline:0;transition:.18s}input:focus,select:focus,textarea:focus{border-color:var(--bc-blue);box-shadow:0 0 0 3px rgba(8,124,240,.13)}button,.btn{display:inline-flex;align-items:center;justify-content:center;gap:6px;border:0;background:linear-gradient(135deg,var(--bc-blue),#075dc5);color:#fff!important;padding:9px 16px;border-radius:9px;text-decoration:none;cursor:pointer;font-weight:700;box-shadow:0 4px 12px rgba(8,124,240,.18);transition:.18s}button:hover,.btn:hover{transform:translateY(-1px);filter:brightness(1.04)}.secondary{background:#65738a!important;box-shadow:none}.danger{background:#c63838!important;box-shadow:none}.ok{color:#168458}.bad{color:#d34747}.replay-layout{display:grid;grid-template-columns:minmax(0,1.7fr) minmax(300px,.8fr);gap:18px}.video-panel video{width:100%;max-height:70vh;background:#05080d;border-radius:14px}.replay-controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-top:12px}.replay-controls button{padding:8px 12px}.event-meta{display:grid;grid-template-columns:1fr 1fr;gap:10px}.meta-item{padding:10px;border:1px solid var(--bc-border);border-radius:10px;background:var(--bc-surface2)}.meta-item small{display:block;color:var(--bc-muted)}.detail-images{display:grid;grid-template-columns:1fr 1fr;gap:10px}.detail-images img{width:100%;height:165px;object-fit:contain;background:#0b1220;border-radius:11px}.time-badge{font-size:20px;font-weight:900;color:var(--bc-blue)}@media(max-width:900px){.replay-layout{grid-template-columns:1fr}.event-meta{grid-template-columns:1fr}}.alert{padding:12px 15px;border-radius:10px;background:#fff1f2;color:#9b1c2d;border:1px solid #ffd5da;margin-bottom:14px}.toast-box{position:fixed;left:22px;bottom:22px;z-index:2000;min-width:270px;background:var(--bc-surface);border:1px solid var(--bc-border);border-right:4px solid #168458;border-radius:12px;padding:12px 15px;box-shadow:var(--bc-shadow);animation:toastin .3s ease}.toast-box.hide{opacity:0;transform:translateY(12px);transition:.35s}@keyframes toastin{from{opacity:0;transform:translateY(15px)}}
.table-wrap{overflow:auto}table{width:100%;border-collapse:separate;border-spacing:0;min-width:700px}th,td{padding:12px 11px;border-bottom:1px solid var(--bc-border);text-align:right;vertical-align:middle}th{font-size:13px;color:var(--bc-muted);background:var(--bc-surface2);font-weight:800}tr:last-child td{border-bottom:0}tbody tr:hover td{background:var(--bc-surface2)}.pagination{display:flex;align-items:center;justify-content:space-between;gap:14px;flex-wrap:wrap;margin-top:14px;padding-top:13px;border-top:1px solid var(--bc-border)}.pagination-summary{color:var(--bc-muted);font-weight:700}.pagination-controls{display:flex;align-items:center;gap:5px;direction:rtl}.page-nav,.page-number{min-width:37px;height:37px;padding:5px 10px;border:1px solid var(--bc-border);background:var(--bc-surface);border-radius:9px;text-decoration:none;display:grid;place-items:center;font-weight:800;color:var(--bc-text)}.page-nav{min-width:64px}.page-number.active{background:var(--bc-blue);border-color:var(--bc-blue);color:#fff}.page-nav.disabled{opacity:.42;cursor:not-allowed}.page-gap{padding:0 4px;color:var(--bc-muted)}.new-events-notice{display:none;margin:10px 0;padding:9px 12px;border-radius:10px;background:#e8f4ff;color:#075dc5;font-weight:800}.new-events-notice.show{display:block}.code{direction:ltr;text-align:left;font-family:Consolas,"Courier New",monospace}.live-grid{display:grid;grid-template-columns:repeat(var(--cols),minmax(0,520px));gap:14px;justify-content:end}.camera-tile{background:#101820;border-radius:14px;overflow:hidden;position:relative;min-height:160px;box-shadow:var(--bc-shadow)}.camera-tile img{display:block;width:100%;aspect-ratio:16/9;object-fit:cover;background:#101820}.camera-head{display:flex;justify-content:space-between;align-items:center;color:#fff;padding:8px 11px;background:#162631}.badge{font-size:11px;padding:4px 9px;border-radius:20px;background:#657180}.badge.online{background:#168458}.toolbar{display:flex;gap:9px;align-items:center;flex-wrap:wrap;margin-bottom:17px}.toolbar h1{margin-left:auto;margin-bottom:0}.toolbar select{width:auto;margin:0}.dashboard-wrap{padding-top:8px}.dashboard-summary{display:flex;align-items:center;min-height:32px;overflow-x:auto;background:var(--bc-surface);border:1px solid var(--bc-border);border-radius:9px;margin-bottom:8px;padding:0 6px;box-shadow:0 3px 12px rgba(10,38,78,.05)}.dashboard-summary-item{display:flex;align-items:center;justify-content:center;gap:6px;flex:1 0 auto;min-width:120px;padding:4px 12px;border-left:1px solid var(--bc-border);white-space:nowrap;color:var(--bc-muted);font-size:11px;font-weight:700;line-height:1.2}.dashboard-summary-item:last-child{border-left:0}.dashboard-summary-item b{color:var(--bc-text);font-size:15px;line-height:1;font-weight:900}.dashboard-layout{display:grid;grid-template-columns:minmax(360px,.82fr) minmax(0,1.58fr);gap:18px;direction:ltr;align-items:start}.dashboard-camera-column{grid-column:1;direction:rtl;min-width:0}.dashboard-main-column{grid-column:2;direction:rtl;min-width:0}.dashboard-events-card{padding:12px 14px;margin-bottom:0}.dashboard-events-head{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-bottom:8px}.dashboard-events-title{display:flex;align-items:center;gap:9px;flex-wrap:wrap}.dashboard-events-title h3{margin:0}.dashboard-count{display:inline-flex;align-items:center;padding:3px 9px;border-radius:999px;background:rgba(8,124,240,.1);color:var(--bc-blue);font-size:12px;font-weight:900}.dashboard-clear{padding:5px 9px!important;font-size:12px!important;box-shadow:none!important}.two-col{display:grid;grid-template-columns:1fr 1fr;gap:16px}.system-bars{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.meter-label{display:flex;justify-content:space-between;margin-bottom:7px}.meter{height:8px;border-radius:8px;background:var(--bc-border);overflow:hidden}.meter span{display:block;height:100%;background:linear-gradient(90deg,var(--bc-blue),var(--bc-cyan));border-radius:8px;transition:width .4s}.empty-state{text-align:center;padding:38px 20px}.grid-switch{display:flex;background:var(--bc-surface);border:1px solid var(--bc-border);border-radius:10px;padding:3px;gap:3px}.grid-switch button{box-shadow:none;background:transparent;color:var(--bc-muted)!important;padding:6px 10px}.grid-switch button.active{background:var(--bc-blue);color:#fff!important}.mobile-menu{display:none}
@media(max-width:1180px){.dashboard-layout{grid-template-columns:1fr;direction:rtl}.dashboard-camera-column,.dashboard-main-column{grid-column:1}.dashboard-main-column{order:1}.dashboard-camera-column{order:2}}
@media(max-width:1150px){.stats-grid{grid-template-columns:repeat(2,1fr)}.system-bars{grid-template-columns:1fr}}
@media(max-width:900px){.resource-chip span.label{display:none}.resource-chip{min-width:auto}}
@media(max-width:760px){.sidebar{transform:translateX(110%);width:258px}.sidebar.mobile-open{transform:translateX(0)}.sidebar-toggle{display:none}.main,.main.collapsed{margin-right:0}.mobile-menu{display:grid}.topbar{padding:0 12px}.user-chip span:last-child,.resource-strip{display:none}.version-chip{font-size:10px;padding:4px 7px}.wrap{padding:17px 12px}.stats-grid{grid-template-columns:1fr 1fr;gap:10px}.card{padding:15px}.live-grid{grid-template-columns:1fr!important}.two-col,.storage-grid{grid-template-columns:1fr}.toolbar h1{width:100%;font-size:23px}.login{margin:4vh 12px}}
@media(max-width:440px){.stats-grid{grid-template-columns:1fr}.top-title{font-size:15px}}
.thumb{width:110px;height:62px;object-fit:cover;border-radius:9px;border:1px solid var(--bc-border);background:#eef2f7;cursor:pointer}.plate-thumb{width:130px;height:48px}.recent-plate-result{display:flex;align-items:center;gap:10px;min-width:275px}.recent-plate-result .plate-thumb{flex:0 0 auto}.recent-vehicle-thumb{width:126px;height:72px;object-fit:cover;border-radius:10px;border:1px solid var(--bc-border);background:#eef2f7}.recent-media-missing{display:inline-flex;width:126px;height:72px;align-items:center;justify-content:center;border:1px dashed var(--bc-border);border-radius:10px;color:var(--bc-muted);font-size:12px}.status-pill{display:inline-block;padding:3px 9px;border-radius:999px;font-size:12px;font-weight:900}.status-pill.ok{background:#e5f7ef;color:#147a50}.status-pill.bad{background:#ffe8e8;color:#b42318}.status-pill.vip{background:#fff3cd;color:#8a6100}.event-blocked{background:rgba(214,69,69,.07)}.event-vip{background:rgba(229,161,26,.08)}.filter-grid{display:grid;grid-template-columns:repeat(5,minmax(140px,1fr));gap:10px;align-items:end}.modal-img{position:fixed;inset:0;background:rgba(0,0,0,.78);z-index:5000;display:none;place-items:center;padding:30px}.modal-img.open{display:grid}.modal-img img{max-width:95vw;max-height:90vh;border-radius:14px}.modal-img button{position:absolute;top:20px;left:20px}@media(max-width:900px){.filter-grid{grid-template-columns:1fr 1fr}}
.login-page{min-height:100vh;display:grid;grid-template-columns:minmax(320px,520px) 1fr;background:linear-gradient(135deg,#071b3f 0%,#0b2e63 52%,#087cf0 100%);direction:ltr;overflow:hidden}.login-panel{direction:rtl;background:var(--bc-surface);padding:clamp(26px,5vw,68px);display:flex;align-items:center;justify-content:center;box-shadow:20px 0 60px rgba(0,0,0,.18);z-index:2}.login-box{width:100%;max-width:410px}.login-logo{display:flex;align-items:center;gap:13px;margin-bottom:34px}.login-logo .brand-mark{width:58px;height:58px;min-width:58px;font-size:22px}.login-logo h1{margin:0;font-size:28px}.login-logo p{margin:0;color:var(--bc-muted)}.login-title{font-size:25px;font-weight:900;margin:0 0 5px}.login-subtitle{color:var(--bc-muted);margin:0 0 26px}.password-wrap{position:relative}.password-wrap input{padding-left:48px}.password-toggle{position:absolute;left:7px;top:10px;width:36px;height:36px;background:transparent!important;color:var(--bc-muted)!important;box-shadow:none;padding:0}.password-toggle:hover{transform:none;background:var(--bc-surface2)!important}.login-submit{width:100%;height:46px;font-size:15px;margin-top:5px}.login-help{display:flex;justify-content:space-between;gap:12px;margin-top:17px;font-size:12px;color:var(--bc-muted)}.login-visual{direction:rtl;color:#fff;display:flex;align-items:center;justify-content:center;padding:60px;position:relative}.login-visual:before,.login-visual:after{content:'';position:absolute;border-radius:50%;background:rgba(255,255,255,.08)}.login-visual:before{width:420px;height:420px;left:-130px;top:-170px}.login-visual:after{width:300px;height:300px;right:8%;bottom:-140px}.login-hero{max-width:670px;position:relative;z-index:1}.login-hero h2{font-size:clamp(32px,4vw,54px);font-weight:900;line-height:1.35;margin:0 0 16px}.login-hero p{font-size:17px;opacity:.82;max-width:570px}.login-features{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin-top:34px}.login-feature{padding:17px;border:1px solid rgba(255,255,255,.16);background:rgba(255,255,255,.08);backdrop-filter:blur(10px);border-radius:15px}.login-feature b{display:block;font-size:15px;margin-bottom:3px}.login-feature span{font-size:12px;opacity:.75}.login-version{position:absolute;bottom:24px;left:30px;opacity:.62;font-size:12px}@media(max-width:900px){.login-page{grid-template-columns:1fr}.login-visual{display:none}.login-panel{min-height:100vh;padding:24px}.login-help{flex-direction:column}}
.anpr-status{display:block;padding:7px 12px;color:#c8d5df;background:#0c141a;font-size:11px;line-height:1.7;border-top:1px solid #263945}.anpr-status.bad{color:#ffb4ab;background:#301716}.playback-controls{display:flex;gap:7px;padding:8px 11px;background:#0c141a;border-top:1px solid #263945}.playback-controls button{padding:6px 12px;font-size:12px;box-shadow:none}.playback-controls button.active{background:#16a36b}
.iran-plate{display:inline-flex;direction:ltr;align-items:stretch;height:54px;min-width:250px;border:2px solid #15191f;border-radius:7px;overflow:hidden;background:#fff;color:#111;font-family:Tahoma,"Segoe UI",sans-serif;box-shadow:0 2px 8px rgba(0,0,0,.14)}.iran-plate.compact{height:42px;min-width:205px}.plate-blue{width:32px;background:#0868b7;color:#fff;display:flex;flex-direction:column;align-items:center;justify-content:center;font-size:12px;line-height:1}.plate-blue small{font-size:7px;margin-top:3px}.plate-main{display:flex;align-items:center;justify-content:space-evenly;gap:8px;flex:1;padding:0 9px;font-size:21px}.compact .plate-main{font-size:17px;gap:6px;padding:0 7px}.plate-iran{width:54px;border-left:2px solid #15191f;display:flex;flex-direction:column;align-items:center;justify-content:center;line-height:1}.plate-iran small{font-size:9px}.plate-iran b{font-size:17px;margin-top:4px}.compact .plate-iran{width:46px}.compact .plate-iran b{font-size:14px}.plate-unreadable{display:inline-block;padding:6px 10px;border-radius:7px;background:#fff1c7;color:#714f00;font-weight:800}.read-badge{display:block;width:max-content;margin-top:5px;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:800}.read-badge.suggested{background:#fff1c7;color:#714f00}.read-badge.unreadable{background:#ffe8e8;color:#a12a2a}.read-badge.confirmed{background:#e5f7ef;color:#147a50}.read-badge.confirmed-ai{background:#e7f5ff;color:#0969a9}.read-badge.auto-confirmed{background:#e9f7ed;color:#226b35;border:1px solid #b9e2c4}.correction-form{display:flex;gap:7px;align-items:center;min-width:265px}.correction-form input:not([type=checkbox]){margin:0;min-width:170px;padding:7px 9px}.correction-form button{padding:7px 10px;white-space:nowrap}
</style>"""

BOOTSTRAP = "<link href='https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.rtl.min.css' rel='stylesheet'>"

NAV_ITEMS = [
    ('/dashboard','⌂','داشبورد و نمایش زنده'),('/cameras','▣','دوربین‌ها'),('/events','▤','ترددها و گزارش‌ها'),
    ('/users','👥','کاربران'),('/audit','☷','لاگ فعالیت‌ها'),('/settings','⚙','تنظیمات')
]

def page(title, body, username=None, request=None):
    shell_start=shell_end=''
    if username:
        path = request.url.path if request else ''
        dashboard_href=_dashboard_navigation_url(request)
        links=''.join(
            f"<a href='{escape(dashboard_href if href=='/dashboard' else href, quote=True)}' "
            f"class='{'active' if (path==href or (href!='/dashboard' and path.startswith(href))) else ''}'>"
            f"<span class='nav-icon'>{icon}</span>"
            f"<span class='nav-label'>{label}</span></a>"
            for href,icon,label in NAV_ITEMS
        )
        shell_start=f"""<div class='app-shell'><aside class='sidebar' id='sidebar'><a class='brand-row' href='{escape(dashboard_href, quote=True)}'><span class='brand-mark'>BC</span><span class='brand-copy'><b>BC Vision</b><small>{COMPANY_NAME}</small></span></a><button class='sidebar-toggle' id='sidebarToggle' title='جمع کردن منو'>‹</button><nav class='nav-menu'>{links}</nav><div class='sidebar-foot'><a href='/logout'><span class='nav-icon'>⇥</span><span class='nav-label'>خروج از حساب</span></a></div></aside><main class='main' id='main'><header class='topbar'><button class='top-action mobile-menu' id='mobileMenu'>☰</button><div class='top-title'>{escape(title)}</div><span class='version-chip' title='نسخه نصب‌شده'>{escape(APP_RELEASE_LABEL)}</span><div class='resource-strip'><div class='resource-chip' id='head-cpu'><span class='resource-dot'></span><span class='label'>CPU</span><span id='head-cpu-value'>—</span></div><div class='resource-chip' id='head-ram'><span class='resource-dot'></span><span class='label'>RAM</span><span id='head-ram-value'>—</span></div><div class='resource-chip' id='head-disk'><span class='resource-dot'></span><span class='label'>DISK</span><span id='head-disk-value'>—</span></div></div><button class='top-action' id='themeToggle' title='حالت تاریک'>◐</button><div class='user-chip'><span class='avatar'>{escape(username[:1].upper())}</span><span>{escape(username)}</span></div></header>"""
        shell_end='</main></div>'
    common_js="""<script>
window.faDigits=function(value){return String(value).replace(/[0-9]/g,d=>'۰۱۲۳۴۵۶۷۸۹'[Number(d)]).replace('.', '٫')};
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

def _validated_user(request):
    session=read_session(request)
    if not session:return None
    username,session_version=session
    fingerprint=session_fingerprint(request)
    if not fingerprint:return None
    with connect() as con:
        row=con.execute(
            'SELECT * FROM users WHERE username=? AND is_active=1 '
            'AND session_version=? AND NOT EXISTS('
            'SELECT 1 FROM revoked_sessions WHERE token_hash=? '
            'AND expires_at>=?)',
            (
                username,session_version,fingerprint,int(time.time()),
            ),
        ).fetchone()
    if not row:return None
    return row

def user(request): return auth(request)
def auth(request):
    row=_validated_user(request)
    if not row:return None
    return row['username']

def current_user(request):
    return _validated_user(request)

def require_admin(request):
    u=current_user(request)
    return bool(u and (u['role']=='admin' or u['is_admin']))

ROLE_PERMISSIONS = {
    'admin': {'system.manage', 'camera.manage', 'watchlist.manage', 'video.process'},
    'system': {'system.manage', 'camera.manage', 'watchlist.manage', 'video.process'},
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


def _administrator_setup_required():
    with connect() as con:
        return con.execute("SELECT 1 FROM users LIMIT 1").fetchone() is None


def _legacy_login_help():
    with connect() as con:
        row = con.execute(
            "SELECT must_change_password FROM users "
            "WHERE username='admin' AND is_active=1"
        ).fetchone()
    if not row or not int(row["must_change_password"] or 0):
        return ""
    return (
        "<div class='login-help'><span>حساب مدیر از نسخه قدیمی شناسایی شد."
        "</span><span>پس از ورود با رمز فعلی، تعویض رمز اجباری است."
        "</span></div>"
    )

def camera_rows(enabled_only=False):
    with connect() as con:
        q="SELECT * FROM cameras" + (" WHERE enabled=1" if enabled_only else "") + " ORDER BY sort_order,id"
        return con.execute(q).fetchall()

@app.get('/')
def root(request:Request):
    if user(request):
        return RedirectResponse('/dashboard',302)
    return RedirectResponse(
        '/setup' if _administrator_setup_required() else '/login',
        302,
    )


@app.get('/setup')
def setup_form(request:Request,error:str=''):
    if not _administrator_setup_required():
        return RedirectResponse('/login',302)
    messages={
        'weak':'رمز مدیر باید حداقل ۱۲ کاراکتر و اختصاصی باشد.',
        'mismatch':'تکرار رمز با رمز انتخابی یکسان نیست.',
        'name':'نام نمایشی مدیر معتبر نیست.',
    }
    alert=(
        f"<div class='alert'>{escape(messages.get(error,'اطلاعات راه‌اندازی معتبر نیست.'))}</div>"
        if error else ''
    )
    body=f"""<div class='login-page'>
    <section class='login-panel'><div class='login-box'>
      <div class='login-logo'><span class='brand-mark'>BC</span><div><h1>BC Vision</h1><p>{escape(COMPANY_NAME)}</p></div></div>
      <h2 class='login-title'>راه‌اندازی امن مدیر</h2><p class='login-subtitle'>در اولین اجرا، رمز اختصاصی مدیر را خودتان تعیین کنید. هیچ رمز پیش‌فرضی وجود ندارد.</p>
      {alert}
      <form method='post' action='/setup' autocomplete='off'>
        <label for='display_name'>نام نمایشی</label><input id='display_name' name='display_name' value='مدیر سیستم' maxlength='80' required autofocus>
        <label for='password'>رمز مدیر (حداقل ۱۲ کاراکتر)</label><div class='password-wrap'><input id='password' type='password' name='password' minlength='12' autocomplete='new-password' required><button type='button' class='password-toggle' id='passwordToggle' aria-label='نمایش رمز'>◉</button></div>
        <label for='password_confirm'>تکرار رمز</label><input id='password_confirm' type='password' name='password_confirm' minlength='12' autocomplete='new-password' required>
        <button class='login-submit' type='submit'>ایجاد مدیر و ادامه</button>
      </form>
    </div></section>
    <section class='login-visual'><div class='login-hero'><h2>شروع امن<br>بدون رمز عمومی</h2><p>حساب مدیر فقط یک‌بار و روی همین دستگاه ساخته می‌شود. رمز انتخابی را در محل امن نگهداری کنید.</p></div><span class='login-version'>نسخه {APP_VERSION}</span></section>
    </div><script>document.getElementById('passwordToggle').addEventListener('click',function(){{const p=document.getElementById('password');p.type=p.type==='password'?'text':'password';this.textContent=p.type==='password'?'◉':'⊘';}});</script>"""
    return page('راه‌اندازی مدیر',body)


@app.post('/setup')
def setup_admin(
    request:Request,
    display_name:str=Form(...),
    password:str=Form(...),
    password_confirm:str=Form(...),
):
    display_name=display_name.strip()
    if not display_name or len(display_name)>80:
        return RedirectResponse('/setup?error=name',303)
    if password!=password_confirm:
        return RedirectResponse('/setup?error=mismatch',303)
    if (
        len(password)<12
        or password!=password.strip()
        or password.casefold() in {'admin','bcvision'}
    ):
        return RedirectResponse('/setup?error=weak',303)
    password_hash=hash_password(password)
    with connect() as con:
        con.execute('BEGIN IMMEDIATE')
        if con.execute('SELECT 1 FROM users LIMIT 1').fetchone() is not None:
            return RedirectResponse('/login',303)
        con.execute(
            'INSERT INTO users('
            'username,password_hash,display_name,is_admin,role,is_active,'
            'must_change_password,session_version'
            ") VALUES('admin',?,?,1,'admin',1,0,0)",
            (password_hash,display_name),
        )
        con.execute(
            'INSERT INTO audit_logs(username,action,details,ip_address) '
            'VALUES(?,?,?,?)',
            (
                'admin','bootstrap_admin_created',
                'ایجاد امن مدیر در اولین اجرا',
                request.client.host if request.client else '',
            ),
        )
    return RedirectResponse('/login?created=1',303)


@app.get('/login')
def login_form(request:Request,error:str='',next:str='/dashboard',logged_out:int=0,created:int=0):
    if _administrator_setup_required():
        return RedirectResponse('/setup',302)
    if user(request): return RedirectResponse('/dashboard',302)
    safe_next=next if next.startswith('/') and not next.startswith('//') else '/dashboard'
    alert="<div class='alert'>نام کاربری یا رمز عبور صحیح نیست.</div>" if error else ''
    notice="<div class='alert' style='background:#eaf8f1;color:#146b45;border-color:#bdebd5'>با موفقیت از حساب خارج شدید.</div>" if logged_out else ''
    if created:
        notice="<div class='alert' style='background:#eaf8f1;color:#146b45;border-color:#bdebd5'>حساب مدیر ساخته شد؛ اکنون وارد شوید.</div>"
    login_help=_legacy_login_help()
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
      {login_help}
    </div></section>
    <section class='login-visual'><div class='login-hero'><h2>مدیریت هوشمند<br>نظارت و تردد خودرو</h2><p>مشاهده زنده دوربین‌ها، پلاک‌خوانی، جست‌وجوی رویدادها و گزارش‌گیری در یک محیط یکپارچه.</p><div class='login-features'><div class='login-feature'><b>نمایش زنده</b><span>مدیریت هم‌زمان چند دوربین</span></div><div class='login-feature'><b>پلاک‌خوان هوشمند</b><span>ثبت و جست‌وجوی سریع ترددها</span></div><div class='login-feature'><b>گزارش‌های دقیق</b><span>فیلتر بر اساس دوربین، رنگ و نوع خودرو</span></div><div class='login-feature'><b>امنیت حساب</b><span>نشست رمزنگاری‌شده و خروج امن</span></div></div></div><span class='login-version'>نسخه {APP_VERSION}</span></section>
    </div><script>document.getElementById('passwordToggle').addEventListener('click',function(){{const p=document.getElementById('password');p.type=p.type==='password'?'text':'password';this.textContent=p.type==='password'?'◉':'⊘';}});</script>"""
    return page('ورود',body)
@app.post('/login')
def login(request:Request,username:str=Form(...),password:str=Form(...),next:str=Form('/dashboard')):
    if _administrator_setup_required():
        return RedirectResponse('/setup',303)
    username=username.strip()
    safe_next=next if next.startswith('/') and not next.startswith('//') else '/dashboard'
    login_succeeded=False
    with connect() as con:
        u=con.execute('SELECT * FROM users WHERE username=?',(username,)).fetchone()
        now=datetime.now()
        now_text=now.isoformat(timespec='seconds')
        locked=False
        if u and u['locked_until']:
            try: locked=datetime.fromisoformat(u['locked_until'])>now
            except Exception: locked=False
        credentials_valid=bool(
            u
            and u['is_active']
            and not locked
            and verify_password(password,u['password_hash'])
        )
        if credentials_valid:
            # The compare-and-update prevents a password reset, deactivation,
            # generation bump or concurrent lockout between verification and
            # issuing a session for the stale row.
            result=con.execute(
                'UPDATE users SET failed_attempts=0,locked_until=NULL,'
                'last_login=CURRENT_TIMESTAMP WHERE id=? AND is_active=1 '
                'AND password_hash=? AND session_version=? '
                'AND (locked_until IS NULL OR locked_until<=?)',
                (
                    u['id'],u['password_hash'],
                    int(u['session_version'] or 0),now_text,
                ),
            )
            login_succeeded=result.rowcount==1
        if not login_succeeded:
            if u:
                lock_until=(now+timedelta(minutes=15)).isoformat(timespec='seconds')
                # SQLite evaluates this expression after acquiring its write
                # lock. Parallel failures therefore increment the latest
                # committed value instead of overwriting one another.
                con.execute(
                    'UPDATE users SET '
                    'locked_until=CASE WHEN COALESCE(failed_attempts,0)+1>=5 '
                    'THEN ? ELSE locked_until END,'
                    'failed_attempts=CASE '
                    'WHEN COALESCE(failed_attempts,0)+1>=5 THEN 0 '
                    'ELSE COALESCE(failed_attempts,0)+1 END '
                    'WHERE id=? AND is_active=1 '
                    'AND (locked_until IS NULL OR locked_until<=?)',
                    (lock_until,u['id'],now_text),
                )
            con.execute('INSERT INTO audit_logs(username,action,details,ip_address) VALUES(?,?,?,?)',(username or 'unknown','login_failed','ورود ناموفق',request.client.host if request.client else ''))
        else:
            con.execute(
                'DELETE FROM revoked_sessions WHERE expires_at<?',
                (int(time.time()),),
            )
            con.execute('INSERT INTO audit_logs(username,action,details,ip_address) VALUES(?,?,?,?)',(username,'login','ورود موفق',request.client.host if request.client else ''))
    if not login_succeeded:
        # Keep deliberate timing equalisation outside the DB transaction so
        # it cannot serialize otherwise parallel login attempts.
        time.sleep(0.35)
        return RedirectResponse(f'/login?error=1&next={quote(safe_next)}',303)
    if int(u['must_change_password'] or 0):
        safe_next='/settings?password_required=1'
    r=RedirectResponse(safe_next,303)
    r.set_cookie(
        COOKIE_NAME,
        create_token(
            u['username'],
            session_version=int(u['session_version'] or 0),
        ),
        httponly=True,
        samesite='lax',
        secure=False,
        max_age=43200,
        path='/',
    )
    return r
@app.get('/logout')
def logout(request:Request):
    username=auth(request)
    if username:
        audit(request,'logout','خروج از حساب')
        details=read_session_details(request)
        fingerprint=session_fingerprint(request)
        if details and fingerprint:
            with connect() as con:
                con.execute(
                    'INSERT OR REPLACE INTO revoked_sessions('
                    'token_hash,expires_at,revoked_at) '
                    'VALUES(?,?,CURRENT_TIMESTAMP)',
                    (fingerprint,details[2]),
                )
    r=RedirectResponse('/login?logged_out=1',302);r.delete_cookie(COOKIE_NAME,path='/');return r

@app.get('/dashboard')
def dashboard(
    request:Request,
    video:int=0,
    events_page:int=1,
    events_snapshot:int=0,
    events_after:int=0,
    events_camera:int=0,
):
    u=auth(request)
    if not u:return RedirectResponse('/login',302)
    cams=camera_rows(True); cols=max(1,min(4,int(get_setting('dashboard_grid','2'))))
    video=1 if int(video or 0) else 0
    events_after=max(0,int(events_after or 0))
    events_camera=max(0,int(events_camera or 0))
    event_page_size=max(
        6,
        min(50, _safe_int(get_setting('dashboard_event_rows','12'), 12)),
    )
    today_start,today_end=_local_day_utc_bounds()
    today_params=(
        today_start.strftime('%Y-%m-%d %H:%M:%S'),
        today_end.strftime('%Y-%m-%d %H:%M:%S'),
    )
    with connect() as con:
        today=con.execute(
            "SELECT COUNT(*) c FROM plate_events "
            "WHERE created_at>=? AND created_at<?",
            today_params,
        ).fetchone()['c']
        alerts=con.execute(
            "SELECT COUNT(*) c FROM plate_events "
            "WHERE confidence < 0.70 AND created_at>=? AND created_at<?",
            today_params,
        ).fetchone()['c']
        archive_total_events=int(con.execute(
            "SELECT COUNT(*) FROM plate_events"
        ).fetchone()[0])
        archive_latest_event_id=int(con.execute(
            "SELECT COALESCE(MAX(id),0) FROM plate_events"
        ).fetchone()[0])
        events_after=min(events_after,archive_latest_event_id)
        scope_conditions=["id>?"]
        scope_params=[events_after]
        if events_camera:
            scope_conditions.append("camera_id=?")
            scope_params.append(events_camera)
        scope_where=" WHERE "+" AND ".join(scope_conditions)
        latest_event_id=int(con.execute(
            "SELECT COALESCE(MAX(id),?) FROM plate_events"+scope_where,
            (events_after,*scope_params),
        ).fetchone()[0])
        latest_event_updated=str(con.execute(
            "SELECT COALESCE(MAX(updated_at),'') "
            "FROM plate_events"+scope_where,
            tuple(scope_params),
        ).fetchone()[0] or '')
        snapshot=max(events_after,int(events_snapshot or latest_event_id))
        if snapshot > latest_event_id:
            snapshot=latest_event_id
        page_conditions=[*scope_conditions,"id<=?"]
        page_params=[*scope_params,snapshot]
        page_where=" WHERE "+" AND ".join(page_conditions)
        visible_total_events=int(con.execute(
            "SELECT COUNT(*) FROM plate_events"+page_where,
            tuple(page_params),
        ).fetchone()[0])
        total_event_pages=max(
            1,
            (
                visible_total_events + event_page_size - 1
            ) // event_page_size,
        )
        events_page=max(1,min(total_event_pages,int(events_page or 1)))
        recent=con.execute(
            "SELECT id,plate_text,camera_name,confidence,created_at,"
            "image_path,plate_image_path,review_status,city,media_status "
            "FROM plate_events"+page_where+
            " ORDER BY id DESC LIMIT ? OFFSET ?",
            (
                *page_params,
                event_page_size,
                (events_page-1)*event_page_size,
            ),
        ).fetchall()
    def camera_tile(c):
        camera_id=int(c['id'])
        is_video=str(c['rtsp_url']).startswith('video://')
        camera_place=' / '.join(
            value for value in (c['city'],c['location']) if value
        ) or 'بدون موقعیت'
        controls=(
            f"<div class='playback-controls'>"
            f"<button type='button' id='play-{camera_id}' "
            f"onclick=\"videoPlayback({camera_id},'play')\">▶ پخش</button>"
            f"<button type='button' class='secondary' id='pause-{camera_id}' "
            f"onclick=\"videoPlayback({camera_id},'pause')\">⏸ توقف</button>"
            f"</div>"
            if is_video else ""
        )
        return (
            f"<div class='camera-tile'><div class='camera-head'>"
            f"<span>{escape(c['name'])}</span><span class='badge' "
            f"id='st-{camera_id}'>در حال اتصال</span></div>"
            f"<img loading='lazy' src='/live/{camera_id}?t={int(time.time())}' "
            f"alt='{escape(c['name'])}'><span class='anpr-status' "
            f"id='anpr-{camera_id}'>پلاک‌خوان: در انتظار اولین فریم</span>"
            f"{controls}<div class='camera-head'><small>"
            f"{escape(camera_place)}</small>"
            f"<a style='color:#bdefff' href='/cameras/{camera_id}/snapshot'>"
            f"گرفتن عکس</a></div></div>"
        )
    tiles=''.join(camera_tile(c) for c in cams)
    if not tiles: tiles="<div class='card empty-state'><h3>هنوز دوربینی فعال نیست</h3><p class='muted'>برای شروع، دوربین واقعی خود را اضافه کنید.</p><a class='btn' href='/cameras/new'>افزودن اولین دوربین</a></div>"
    ids=','.join(str(c['id']) for c in cams)
    dashboard_return_to=_dashboard_url(
        video=video,
        events_camera=events_camera,
        events_after=events_after,
        events_snapshot=snapshot,
        events_page=events_page,
    )
    recent_rows=''.join(
        dashboard_event_row(r, dashboard_return_to) for r in recent
    ) or (
        "<tr><td colspan='6'>در این نوبت هنوز پلاکی ثبت نشده است."
        "</td></tr>"
    )
    pagination_params={'events_snapshot':snapshot}
    if video:
        pagination_params['video']=1
    if events_after:
        pagination_params['events_after']=events_after
    if events_camera:
        pagination_params['events_camera']=events_camera
    recent_pagination=pagination_html(
        '/dashboard',
        events_page,
        total_event_pages,
        visible_total_events,
        pagination_params,
        'events_page',
        event_page_size,
    )
    notice_params={}
    if video:
        notice_params['video']=1
    if events_after:
        notice_params['events_after']=events_after
    if events_camera:
        notice_params['events_camera']=events_camera
    new_events_url='/dashboard'
    if notice_params:
        new_events_url+=f"?{urlencode(notice_params)}"
    count_label=(
        'پلاک‌های این ویدئو'
        if events_camera
        else 'پلاک‌های این نمایش'
    )
    js=f"""<script>
const ids=[{ids}];
async function cameraStatus(){{for(const id of ids){{try{{let r=await fetch('/api/cameras/'+id+'/status');let s=await r.json();let e=document.getElementById('st-'+id),a=document.getElementById('anpr-'+id),n=v=>Number(v||0).toLocaleString('fa-IR');e.textContent=s.ended?'پایان ویدئو':(s.paused?'متوقف':(s.online?'آنلاین':'آفلاین'));e.className='badge '+(s.online?'online':'');const play=document.getElementById('play-'+id),pause=document.getElementById('pause-'+id);if(play)play.classList.toggle('active',!s.paused);if(pause)pause.classList.toggle('active',!!s.paused);const p=s.anpr||{{}},m=p.models||{{}},sh=p.shadow||{{}},engine=m.selected_detector==='yolox'?'YOLOX اختصاصی':(m.selected_detector==='yolov8n'?'YOLOv8n':'YOLO11n'),v2=sh.enabled?(' | V2 '+(sh.detector_variant||'')+': '+(sh.ready?'آماده':'در حال آماده‌سازی')+'، خوانش '+n(sh.events)+'، توافق '+n(sh.agreements)+'، اختلاف '+n(sh.disagreements)+(sh.last_error?'، خطا':'')):'';if(p.last_error){{a.textContent='خطای '+engine+': '+p.last_error;a.className='anpr-status bad'}}else if(s.anpr_marker_error){{a.textContent='خطای تکمیل پردازش ویدئو: '+s.anpr_marker_error;a.className='anpr-status bad'}}else if(s.anpr_preview_only){{a.textContent=s.anpr_completed?'بازپخش فقط نمایشی است؛ پلاک‌خوان این ویدئو یک‌بار کامل شده است':'پردازش قبلی ناتمام بود؛ برای جلوگیری از ثبت تکراری، پخش فقط نمایشی است';a.className='anpr-status'}}else if(m.preparation_error){{a.textContent='خطای آماده‌سازی مدل: '+m.preparation_error;a.className='anpr-status bad'}}else if(!m.ready){{a.textContent=engine+' آماده نیست: مدل تشخیص یا OCR نصب نشده است';a.className='anpr-status bad'}}else{{const idle=p.idle_mode?' | حالت کم‌مصرف':'';a.textContent=engine+' | پردازش: '+n(p.processed_frames)+' فریم | تشخیص: '+n(p.detected_candidates)+' | ثبت: '+n(p.emitted_events)+idle+v2;a.className='anpr-status'}}}}catch(e){{}}}}}}
async function videoPlayback(id,action){{try{{const r=await fetch('/api/cameras/'+id+'/playback',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{action}})}});if(!r.ok)throw new Error();await cameraStatus()}}catch(e){{alert('تغییر وضعیت پخش انجام نشد.')}}}}
let latestEventId={latest_event_id};
let latestEventUpdated={json.dumps(latest_event_updated)};
const dashboardEventsPage={events_page};
const dashboardVideo={video};
const dashboardEventsAfter={events_after};
const dashboardEventsCamera={events_camera};
function dashboardScopeQuery(){{
 const query=new URLSearchParams();
 if(dashboardVideo)query.set('video','1');
 if(dashboardEventsAfter)query.set('events_after',String(dashboardEventsAfter));
 if(dashboardEventsCamera)query.set('events_camera',String(dashboardEventsCamera));
 return query;
}}
function clearDashboardReadings(){{
 const query=new URLSearchParams();
 if(dashboardVideo)query.set('video','1');
 query.set('events_after',String(latestEventId));
 if(dashboardEventsCamera)query.set('events_camera',String(dashboardEventsCamera));
 window.location.assign('/dashboard?'+query.toString());
}}
async function refreshRecentEvents(){{
 try{{
  const query=dashboardScopeQuery();
  query.set('after',String(latestEventId));
  query.set('after_updated',latestEventUpdated);
  const url='/api/dashboard/recent-events?'+query.toString();
  const r=await fetch(url,{{cache:'no-store'}});
  if(!r.ok)return;
  const data=await r.json();
  if(Number.isFinite(Number(data.visible_count))){{
   document.getElementById('dashboardPlateCount').textContent=Number(data.visible_count).toLocaleString('fa-IR');
  }}
  if(data.rows_html&&(
    data.latest_id>latestEventId||
    (data.latest_updated&&data.latest_updated>latestEventUpdated)
  )){{
   const focused=document.activeElement?.closest?.('#recentEventsBody');
   if(dashboardEventsPage!==1){{
    document.getElementById('newEventsNotice').classList.add('show');
   }}else if(focused){{
    document.getElementById('newEventsNotice').classList.add('show');
    return;
   }}else{{
    document.getElementById('recentEventsBody').innerHTML=data.rows_html;
    if(data.pagination_html)document.getElementById('recentEventsPagination').innerHTML=data.pagination_html;
   }}
   latestEventId=data.latest_id;
   latestEventUpdated=data.latest_updated||latestEventUpdated;
  }}
 }}catch(e){{}}
}}
function setGrid(n){{document.getElementById('liveGrid').style.setProperty('--cols',n);document.querySelectorAll('.grid-switch button').forEach(b=>b.classList.toggle('active',Number(b.dataset.n)===n));localStorage.setItem('bc-grid',n)}}
const savedGrid=Number(localStorage.getItem('bc-grid')||{cols});setGrid(savedGrid);cameraStatus();setInterval(cameraStatus,4000);setInterval(refreshRecentEvents,1500);
</script>"""
    body=f"""<div class='wrap dashboard-wrap'>
    <div id='dashboardCounterStrip' class='dashboard-summary' aria-label='خلاصه آمار داشبورد'>
      <div class='dashboard-summary-item'><span>دوربین فعال</span><b>{persian_digits(len(cams))}</b></div>
      <div class='dashboard-summary-item'><span>تردد امروز</span><b>{persian_digits(today)}</b></div>
      <div class='dashboard-summary-item'><span>هشدار امروز</span><b>{persian_digits(alerts)}</b></div>
      <div class='dashboard-summary-item'><span>کل ترددها</span><b>{persian_digits(archive_total_events)}</b></div>
    </div>
    <div class='dashboard-layout'>
    <section class='dashboard-camera-column' id='dashboardCameraColumn'>
    <div class='card'><div class='toolbar'><h3 style='margin:0;margin-left:auto'>نمایش زنده</h3><div class='grid-switch'><button data-n='1' onclick='setGrid(1)'>۱</button><button data-n='2' onclick='setGrid(2)'>۴</button><button data-n='3' onclick='setGrid(3)'>۹</button><button data-n='4' onclick='setGrid(4)'>۱۶</button></div><button class='secondary' onclick='document.documentElement.requestFullscreen?.()'>تمام‌صفحه</button></div><div class='live-grid' id='liveGrid' style='--cols:{cols}'>{tiles}</div></div>
    </section>
    <section class='dashboard-main-column' id='dashboardMainColumn'>
    <div id='dashboardEventsCard' class='card dashboard-events-card'><div class='dashboard-events-head'><div class='dashboard-events-title'><h3>آخرین تشخیص‌های پلاک و خودرو</h3><span class='dashboard-count'>{count_label}: <b id='dashboardPlateCount'>{persian_digits(visible_total_events)}</b></span></div><button id='dashboardClearButton' type='button' class='secondary dashboard-clear' onclick='clearDashboardReadings()' title='فقط فهرست این نوبت پاک می‌شود؛ آرشیو و تصاویر حذف نمی‌شوند.'>پاک‌کردن نمایش</button></div><a id='newEventsNotice' class='new-events-notice' href='{escape(new_events_url)}'>رویداد جدید ثبت شد — نمایش صفحهٔ اول</a><div class='table-wrap'><table><thead><tr><th>تصویر خودرو</th><th>تصویر پلاک / پلاک خوانده‌شده</th><th>دوربین / شهر</th><th>اطمینان</th><th>زمان</th><th>تأیید، اصلاح و آموزش</th></tr></thead><tbody id='recentEventsBody'>{recent_rows}</tbody></table></div><div id='recentEventsPagination'>{recent_pagination}</div></div>
    </section></div>{js}</div>"""
    return page('داشبورد',body,u,request)

@app.get('/api/dashboard/recent-events')
def dashboard_recent_events(
    request:Request,
    after:int=0,
    after_updated:str='',
    video:int=0,
    events_after:int=0,
    events_camera:int=0,
):
    if not auth(request):
        return JSONResponse({'error':'unauthorized'},401)
    events_after=max(0,int(events_after or 0))
    events_camera=max(0,int(events_camera or 0))
    event_page_size=max(
        6,
        min(50, _safe_int(get_setting('dashboard_event_rows','12'), 12)),
    )
    with connect() as con:
        archive_latest_id=int(con.execute(
            "SELECT COALESCE(MAX(id),0) FROM plate_events"
        ).fetchone()[0])
        events_after=min(events_after,archive_latest_id)
        scope_conditions=["id>?"]
        scope_params=[events_after]
        if events_camera:
            scope_conditions.append("camera_id=?")
            scope_params.append(events_camera)
        scope_where=" WHERE "+" AND ".join(scope_conditions)
        latest_id=int(con.execute(
            "SELECT COALESCE(MAX(id),?) FROM plate_events"+scope_where,
            (events_after,*scope_params),
        ).fetchone()[0])
        latest_updated=str(con.execute(
            "SELECT COALESCE(MAX(updated_at),'') "
            "FROM plate_events"+scope_where,
            tuple(scope_params),
        ).fetchone()[0] or '')
        unchanged=(
            latest_id <= max(0,int(after))
            and (
                not after_updated
                or latest_updated <= str(after_updated)
            )
        )
        if unchanged:
            return JSONResponse({'latest_id':latest_id,'rows_html':''})
        visible_count=int(con.execute(
            "SELECT COUNT(*) FROM plate_events"+scope_where,
            tuple(scope_params),
        ).fetchone()[0])
        recent=con.execute(
            "SELECT id,plate_text,camera_name,confidence,created_at,"
            "image_path,plate_image_path,review_status,city,media_status "
            "FROM plate_events"+scope_where+
            " ORDER BY id DESC LIMIT ?",
            (*scope_params,event_page_size),
        ).fetchall()
    dashboard_return_to=_dashboard_url(
        video=video,
        events_camera=events_camera,
        events_after=events_after,
        events_snapshot=latest_id,
    )
    rows=''.join(
        dashboard_event_row(r, dashboard_return_to) for r in recent
    ) or "<tr><td colspan='6'>هنوز پلاکی ثبت نشده است.</td></tr>"
    total_pages=max(
        1,
        (visible_count+event_page_size-1)//event_page_size,
    )
    pagination_params={'events_snapshot':latest_id}
    if int(video or 0):
        pagination_params['video']=1
    if events_after:
        pagination_params['events_after']=events_after
    if events_camera:
        pagination_params['events_camera']=events_camera
    pager=pagination_html(
        '/dashboard',
        1,
        total_pages,
        visible_count,
        pagination_params,
        'events_page',
        event_page_size,
    )
    return JSONResponse({
        'latest_id':latest_id,
        'latest_updated':latest_updated,
        'rows_html':rows,
        'pagination_html':pager,
        'visible_count':visible_count,
    })

@app.get('/live/{camera_id}')
def live(camera_id:int,request:Request):
    if not auth(request): return RedirectResponse('/login',302)
    with connect() as con:c=con.execute('SELECT * FROM cameras WHERE id=? AND enabled=1',(camera_id,)).fetchone()
    if not c:return JSONResponse({'error':'camera not found'},404)
    s=manager.get(
        c['id'],c['rtsp_url'],c['name'],
        int(get_setting('stream_width','640')),
        int(get_setting('live_fps','5')),
        int(get_setting('jpeg_quality','70')),
        bool(c['video_anpr_started']),
        bool(c['video_anpr_completed']),
    )
    return StreamingResponse(s.frames(),media_type='multipart/x-mixed-replace; boundary=frame',headers={'Cache-Control':'no-store'})

@app.get('/api/cameras/{camera_id}/status')
def cam_status(camera_id:int,request:Request):
    if not auth(request):return JSONResponse({'error':'unauthorized'},401)
    return JSONResponse(manager.status(camera_id))

@app.post('/api/cameras/{camera_id}/playback')
async def camera_playback(camera_id:int,request:Request):
    if not auth(request):
        return JSONResponse({'error':'unauthorized'},401)
    if not has_permission(request,'video.process'):
        return access_denied()
    try:
        payload=await request.json()
    except Exception:
        payload={}
    action=str(payload.get('action','')).strip().lower()
    if action not in {'play','pause'}:
        return JSONResponse({'error':'invalid action'},400)
    if not manager.set_playback(camera_id,action):
        return JSONResponse(
            {'error':'uploaded video stream is not active'},
            404,
        )
    audit(request,'video_playback',f'camera={camera_id}; action={action}')
    return JSONResponse({'ok':True,'action':action})

@app.get('/cameras')
def cameras(request:Request,msg:str='',error:str=''):
    u=auth(request)
    if not u:return RedirectResponse('/login',302)
    if not has_permission(request,'camera.manage'):return access_denied()
    rows=camera_rows(); trs=''.join(f"<tr><td>{c['id']}</td><td>{escape(c['name'])}</td><td>{escape(c['city'] or '—')}</td><td>{escape(c['location'])}</td><td><span class='status-pill {'ok' if c['enabled'] else ''}'>{'فعال' if c['enabled'] else 'غیرفعال'}</span></td><td>{'ویدئوی آپلودی' if str(c['rtsp_url']).startswith('video://') else ('آزمایشی' if c['is_demo'] else 'RTSP')}</td><td><form style='display:inline' method='post' action='/cameras/{c['id']}/toggle'><button class='{'danger' if c['enabled'] else 'secondary'}'>{'غیرفعال‌کردن' if c['enabled'] else 'فعال‌کردن'}</button></form> <a class='btn' href='/cameras/{c['id']}/edit'>ویرایش</a> <form style='display:inline' method='post' action='/cameras/{c['id']}/delete' onsubmit=\"return confirm('حذف شود؟')\"><button class='danger'>حذف</button></form></td></tr>" for c in rows) or "<tr><td colspan='7'>دوربینی ثبت نشده است.</td></tr>"
    notice="<div class='card ok'>عملیات انجام شد.</div>" if msg else ''
    if error: notice += f"<div class='alert'>{escape(error)}</div>"
    return page('دوربین‌ها',f"""<div class='wrap'><div class='toolbar'><h1 style='margin-left:auto'>مدیریت دوربین‌ها</h1><a class='btn' href='/cameras/new'>افزودن دوربین</a></div>{notice}<div class='card'><div class='table-wrap'><table><tr><th>ID</th><th>نام</th><th>شهر</th><th>موقعیت</th><th>وضعیت</th><th>نوع</th><th>عملیات</th></tr>{trs}</table></div></div><div class='card'><h2>🎞️ افزودن فایل تست به دوربین‌ها</h2><p class='muted'>هر فایل به‌عنوان یک دوربین مجازی جداگانه ذخیره و در داشبورد نمایش داده می‌شود.</p><form id='videoUploadForm' action='/cameras/video-upload' method='post' enctype='multipart/form-data'><label>پروفایل تنظیمات پلاک‌خوان</label><select name='camera_id'><option value='0'>تنظیمات پیش‌فرض</option>{''.join(f"<option value='{c['id']}'>{escape(c['name'])}</option>" for c in rows if not str(c['rtsp_url']).startswith('video://'))}</select><br><label>فایل ویدئو</label><input type='file' name='video' accept='.mp4,.avi,.mkv,.mov' required><div id='uploadState' class='muted' style='display:none;margin:10px 0'>در حال آپلود: <b id='uploadPercent'>۰٪</b><progress id='uploadProgress' value='0' max='100' style='width:100%'></progress></div><br><button id='uploadButton'>افزودن فایل تست</button></form></div></div>
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
    c=dict(c) if c else {'id':'','name':'','rtsp_url':'','location':'','city':'','enabled':1,'is_demo':0,'sort_order':0,'lpr_enabled':1,'lpr_confidence':60,'frame_step':5,'duplicate_seconds':30,'roi_x':0,'roi_y':0,'roi_w':100,'roi_h':100,'line_y':50}
    action=f"/cameras/{c['id']}/edit" if c['id'] else '/cameras/new'
    return f"""<div class='wrap'><h1>{'ویرایش' if c['id'] else 'افزودن'} دوربین</h1><div class='card'><form method='post' action='{action}'><div class='two-col'><div><label>نام دوربین</label><input name='name' value='{escape(str(c['name']))}' required></div><div><label>شهر</label><input name='city' value='{escape(str(c.get('city','')))}' placeholder='مثال: کرج'></div><div><label>موقعیت دقیق</label><input name='location' value='{escape(str(c['location']))}' placeholder='مثال: ورودی پارکینگ'></div></div><label>آدرس RTSP</label><input class='code' name='rtsp_url' value='{escape(str(c['rtsp_url']))}' placeholder='rtsp://user:pass@192.168.1.10:554/...'><div class='two-col'><div><label>ترتیب نمایش</label><input type='number' name='sort_order' value='{c['sort_order']}'></div><div><label>نوع تصویر</label><select name='is_demo'><option value='0' {'selected' if not c['is_demo'] else ''}>دوربین RTSP</option><option value='1' {'selected' if c['is_demo'] else ''}>دوربین آزمایشی</option></select></div></div><label><input style='width:auto' type='checkbox' name='enabled' value='1' {'checked' if c['enabled'] else ''}> فعال باشد</label><hr><h3>تنظیمات پلاک‌خوان این دوربین</h3><label><input style='width:auto' type='checkbox' name='lpr_enabled' value='1' {'checked' if c.get('lpr_enabled',1) else ''}> پلاک‌خوان فعال باشد</label><div class='two-col'><div><label>حداقل اطمینان تشخیص (درصد)</label><input type='number' min='1' max='99' name='lpr_confidence' value='{c.get('lpr_confidence',60)}'></div><div><label>پردازش هر چند فریم</label><input type='number' min='1' max='60' name='frame_step' value='{c.get('frame_step',5)}'></div><div><label>زمان حذف پلاک تکراری (ثانیه)</label><input type='number' min='0' max='3600' step='0.5' name='duplicate_seconds' value='{c.get('duplicate_seconds',30)}'></div><div><label>خط عبور عمودی (درصد ارتفاع تصویر)</label><input type='number' min='0' max='100' name='line_y' value='{c.get('line_y',50)}'></div></div><h3>منطقه تشخیص ROI برحسب درصد تصویر</h3><p class='muted'>برای بررسی کل تصویر: X=0، Y=0، عرض=100 و ارتفاع=100 قرار دهید.</p><div class='storage-grid'><div><label>X شروع</label><input type='number' min='0' max='99' name='roi_x' value='{c.get('roi_x',0)}'></div><div><label>Y شروع</label><input type='number' min='0' max='99' name='roi_y' value='{c.get('roi_y',0)}'></div><div><label>عرض ROI</label><input type='number' min='1' max='100' name='roi_w' value='{c.get('roi_w',100)}'></div><div><label>ارتفاع ROI</label><input type='number' min='1' max='100' name='roi_h' value='{c.get('roi_h',100)}'></div></div><button>ذخیره</button> <a class='btn secondary' href='/cameras'>انصراف</a></form></div></div>"""
@app.get('/cameras/new')
def new_cam_form(request:Request):
    u=auth(request)
    if not u:return RedirectResponse('/login',302)
    if not has_permission(request,'camera.manage'):return access_denied()
    return page('افزودن دوربین',cam_form(),u,request)
@app.post('/cameras/new')
def new_cam(request:Request,name:str=Form(...),rtsp_url:str=Form(''),location:str=Form(''),city:str=Form(''),enabled:str|None=Form(None),is_demo:int=Form(0),sort_order:int=Form(0),lpr_enabled:str|None=Form(None),lpr_confidence:int=Form(60),frame_step:int=Form(5),duplicate_seconds:float=Form(30),roi_x:int=Form(0),roi_y:int=Form(0),roi_w:int=Form(100),roi_h:int=Form(100),line_y:int=Form(50)):
    if not auth(request):return RedirectResponse('/login',302)
    if not has_permission(request,'camera.manage'):return access_denied()
    url='demo://camera' if is_demo else rtsp_url.strip()
    if url.lower().startswith('video://'):
        return RedirectResponse(
            '/cameras?error='+quote(
                'آدرس video:// فقط از مسیر آپلود ویدئو قابل ایجاد است.'
            ),
            303,
        )
    with connect() as con:con.execute('INSERT INTO cameras(name,rtsp_url,location,city,enabled,is_demo,sort_order,lpr_enabled,lpr_confidence,frame_step,duplicate_seconds,roi_x,roi_y,roi_w,roi_h,line_y) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',(name.strip(),url,location.strip(),city.strip(),1 if enabled else 0,is_demo,sort_order,1 if lpr_enabled else 0,max(1,min(99,lpr_confidence)),max(1,min(60,frame_step)),max(0,min(3600,duplicate_seconds)),max(0,min(99,roi_x)),max(0,min(99,roi_y)),max(1,min(100-roi_x,roi_w)),max(1,min(100-roi_y,roi_h)),max(0,min(100,line_y))))
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
def edit_cam(camera_id:int,request:Request,name:str=Form(...),rtsp_url:str=Form(''),location:str=Form(''),city:str=Form(''),enabled:str|None=Form(None),is_demo:int=Form(0),sort_order:int=Form(0),lpr_enabled:str|None=Form(None),lpr_confidence:int=Form(60),frame_step:int=Form(5),duplicate_seconds:float=Form(30),roi_x:int=Form(0),roi_y:int=Form(0),roi_w:int=Form(100),roi_h:int=Form(100),line_y:int=Form(50)):
    if not auth(request):return RedirectResponse('/login',302)
    if not has_permission(request,'camera.manage'):return access_denied()
    url='demo://camera' if is_demo else rtsp_url.strip()
    if url.lower().startswith('video://'):
        return RedirectResponse(
            '/cameras?error='+quote(
                'آدرس video:// فقط از مسیر آپلود ویدئو قابل ایجاد است.'
            ),
            303,
        )
    # Uploaded-video replacement, edit, and delete share one lifecycle lock.
    # Otherwise an upload can snapshot this row, then delete it after a
    # concurrent edit has already installed a different stream identity.
    with _VIDEO_CAMERA_HANDOFF_LOCK:
        if manager.remove(camera_id) is not True:
            return RedirectResponse(
                '/cameras?error='+quote('جریان دوربین متوقف نشد؛ ویرایش لغو شد.'),
                303,
            )
        try:
            with connect() as con:con.execute('UPDATE cameras SET name=?,rtsp_url=?,location=?,city=?,enabled=?,is_demo=?,sort_order=?,lpr_enabled=?,lpr_confidence=?,frame_step=?,duplicate_seconds=?,roi_x=?,roi_y=?,roi_w=?,roi_h=?,line_y=? WHERE id=?',(name.strip(),url,location.strip(),city.strip(),1 if enabled else 0,is_demo,sort_order,1 if lpr_enabled else 0,max(1,min(99,lpr_confidence)),max(1,min(60,frame_step)),max(0,min(3600,duplicate_seconds)),max(0,min(99,roi_x)),max(0,min(99,roi_y)),max(1,min(100-roi_x,roi_w)),max(1,min(100-roi_y,roi_h)),max(0,min(100,line_y)),camera_id))
        except Exception:
            manager.start_enabled_cameras()
            raise
    return RedirectResponse('/cameras?msg=1',303)
@app.post('/cameras/{camera_id}/delete')
def delete_cam(camera_id:int,request:Request):
    if not auth(request):return RedirectResponse('/login',302)
    if not has_permission(request,'camera.manage'):return access_denied()
    with _VIDEO_CAMERA_HANDOFF_LOCK:
        if manager.remove(camera_id) is not True:
            return RedirectResponse(
                '/cameras?error='+quote('جریان دوربین متوقف نشد؛ حذف لغو شد.'),
                303,
            )
        try:
            with connect() as con:con.execute('DELETE FROM cameras WHERE id=?',(camera_id,))
        except Exception:
            manager.start_enabled_cameras()
            raise
    return RedirectResponse('/cameras?msg=1',303)


@app.post('/cameras/{camera_id}/toggle')
def toggle_camera(camera_id:int,request:Request):
    if not auth(request):return RedirectResponse('/login',302)
    if not has_permission(request,'camera.manage'):return access_denied()
    with _VIDEO_CAMERA_HANDOFF_LOCK:
        with connect() as con:
            camera=con.execute(
                'SELECT id,name,enabled FROM cameras WHERE id=?',
                (camera_id,),
            ).fetchone()
        if not camera:
            return RedirectResponse(
                '/cameras?error='+quote('دوربین پیدا نشد.'),
                303,
            )
        enable=not bool(camera['enabled'])
        if not enable and manager.remove(camera_id) is not True:
            return RedirectResponse(
                '/cameras?error='+quote(
                    'جریان دوربین متوقف نشد؛ غیرفعال‌سازی لغو شد.'
                ),
                303,
            )
        with connect() as con:
            con.execute(
                'UPDATE cameras SET enabled=? WHERE id=?',
                (1 if enable else 0,camera_id),
            )
        if enable:
            try:
                manager.start_enabled_cameras()
            except Exception:
                with connect() as con:
                    con.execute(
                        'UPDATE cameras SET enabled=0 WHERE id=?',
                        (camera_id,),
                    )
                raise
    audit(
        request,
        'camera_toggle',
        f"camera={camera_id}; enabled={1 if enable else 0}",
    )
    return RedirectResponse('/cameras?msg=1',303)

@app.get('/media')
def media(request:Request,path:str=''):
    if not auth(request): return RedirectResponse('/login',302)
    try:
        raw_path=str(path or '')
        target=Path(raw_path).resolve()
        if target.suffix.lower() not in MEDIA_FILE_EXTENSIONS:
            return JSONResponse({'error':'not found'},404)
        current_roots=[
            Path(get_setting('snapshot_path',str(SNAPSHOT_DIR))).resolve(),
            Path(get_setting('plate_path',str(PLATE_DIR))).resolve(),
            Path(get_setting('video_path',str(VIDEO_DIR))).resolve(),
        ]
        in_current_root=any(
            target.is_relative_to(root) for root in current_roots
        )
        in_historical_root=any(
            target.is_relative_to(root)
            for root in _media_roots_history()
        )
        historically_referenced=False
        if in_historical_root:
            with connect() as con:
                historically_referenced=con.execute(
                    "SELECT 1 FROM plate_events WHERE "
                    "image_path IN (?,?) OR plate_image_path IN (?,?) "
                    "OR video_path IN (?,?) LIMIT 1",
                    (
                        raw_path,str(target),raw_path,str(target),
                        raw_path,str(target),
                    ),
                ).fetchone() is not None
        if not (
            in_current_root or historically_referenced
        ):
            return JSONResponse({'error':'not found'},404)
        read_pin=pin_media_paths((target,))
        if not target.is_file():
            read_pin.close()
            return JSONResponse({'error':'not found'},404)
        try:
            return _PinnedFileResponse(
                target,
                read_pin=read_pin,
            )
        except Exception:
            read_pin.close()
            raise
    except Exception:return JSONResponse({'error':'not found'},404)

@app.get('/events')
def events(
    request:Request,
    q:str='',
    camera:str='',
    city:str='',
    region:str='',
    status:str='',
    vehicle_type:str='',
    vehicle_color:str='',
    date_from:str='',
    time_from:str='',
    date_to:str='',
    time_to:str='',
    events_page:int=1,
    per_page:int=25,
):
    u=auth(request)
    if not u:return RedirectResponse('/login',302)
    q=q.strip();camera=camera.strip();city=city.strip();region=region.strip()
    date_from=date_from.strip();date_to=date_to.strip()
    time_from=time_from.strip();time_to=time_to.strip()
    per_page=per_page if per_page in {25,50,100} else 25
    where=[];params=[];filter_error=''
    normalized_query=normalize_plate(q)
    if q:
        if normalized_query:
            where.append(
                "INSTR(COALESCE(NULLIF(e.plate_norm,''),"
                "e.raw_guess_norm,''),?)>0"
            )
            params.append(normalized_query)
        else:
            filter_error='عبارت واردشده برای جست‌وجوی پلاک معتبر نیست.'
            where.append('1=0')
    if camera:
        where.append('e.camera_name=?');params.append(camera)
    city_expression="COALESCE(NULLIF(e.city,''),NULLIF(c.city,''),'')"
    if city:
        where.append(f'INSTR({city_expression},?)>0');params.append(city)
    normalized_region=''.join(
        char for char in normalize_plate(region) if char.isdigit()
    )
    if region:
        if 1 <= len(normalized_region) <= 2:
            where.append('e.plate_region=?')
            params.append(normalized_region.zfill(2))
        else:
            filter_error=filter_error or 'کد ناحیه پلاک باید یک یا دو رقم باشد.'
            where.append('1=0')
    if status:
        where.append("COALESCE(w.status,'unknown')=?");params.append(status)
    if vehicle_type:
        where.append('e.vehicle_type=?');params.append(vehicle_type)
    if vehicle_color:
        where.append('e.vehicle_color=?');params.append(vehicle_color)
    try:
        start_date=_parse_jalali_date(date_from)
        end_date=_parse_jalali_date(date_to)
        start_time=_parse_time(time_from)
        end_time=_parse_time(time_to)
        start_bound=(
            start_date.replace(
                hour=start_time.hour if start_time else 0,
                minute=start_time.minute if start_time else 0,
                second=0,
                microsecond=0,
            )
            if start_date else None
        )
        end_bound=(
            end_date.replace(
                hour=end_time.hour if end_time else 0,
                minute=end_time.minute if end_time else 0,
                second=0,
                microsecond=0,
            )
            + (timedelta(minutes=1) if end_time else timedelta(days=1))
            if end_date else None
        )
        if start_bound and end_bound and start_bound >= end_bound:
            raise ValueError('ابتدای بازه زمانی باید قبل از انتهای آن باشد.')
        if start_bound:
            where.append('e.created_at>=?')
            params.append(
                _local_to_utc_naive(start_bound).strftime(
                    '%Y-%m-%d %H:%M:%S'
                )
            )
        if end_bound:
            where.append('e.created_at<?')
            params.append(
                _local_to_utc_naive(end_bound).strftime(
                    '%Y-%m-%d %H:%M:%S'
                )
            )
        current_offset=APP_LOCAL_TIMEZONE.utcoffset(
            datetime.now(APP_LOCAL_TIMEZONE)
        ) or timedelta()
        offset_minutes=int(current_offset.total_seconds()//60)
        local_time_sql="TIME(datetime(e.created_at,?))"
        offset_modifier=f"{offset_minutes:+d} minutes"
        if start_time and not start_date:
            where.append(f"{local_time_sql}>=?")
            params.extend([
                offset_modifier,
                start_time.strftime('%H:%M:%S'),
            ])
        if end_time and not end_date:
            end_minutes=end_time.hour*60+end_time.minute+1
            if start_time and (
                start_time.hour*60+start_time.minute >= end_minutes
            ):
                raise ValueError('ابتدای بازه ساعت باید قبل از انتهای آن باشد.')
            where.append(f"{local_time_sql}<?")
            params.extend([
                offset_modifier,
                '24:00:00'
                if end_minutes >= 24*60
                else f'{end_minutes//60:02d}:{end_minutes%60:02d}:00'
            ])
    except ValueError as exc:
        filter_error=filter_error or str(exc)
        where.append('1=0')
    source_sql=""" FROM plate_events e
        LEFT JOIN plate_watchlist w ON w.plate_norm=e.plate_norm
        LEFT JOIN cameras c ON c.id=e.camera_id"""
    where_sql=(' WHERE '+' AND '.join(where)) if where else ''
    with connect() as con:
        total_rows=int(con.execute(
            'SELECT COUNT(*)'+source_sql+where_sql,
            params,
        ).fetchone()[0])
        total_pages=max(1,(total_rows+per_page-1)//per_page)
        events_page=max(1,min(total_pages,int(events_page or 1)))
        sql=(
            "SELECT e.*,COALESCE(w.status,'unknown') watch_status,"
            "w.owner_name,w.vehicle_model,"
            "w.vehicle_color watch_vehicle_color,"
            f"{city_expression} event_city"
            +source_sql+where_sql+
            " ORDER BY e.created_at DESC,e.id DESC LIMIT ? OFFSET ?"
        )
        rows=con.execute(
            sql,
            params+[per_page,(events_page-1)*per_page],
        ).fetchall()
        cameras=[r['camera_name'] for r in con.execute("SELECT DISTINCT camera_name FROM plate_events WHERE camera_name IS NOT NULL AND camera_name<>'' ORDER BY camera_name").fetchall()]
        vehicle_types=[r['vehicle_type'] for r in con.execute("SELECT DISTINCT vehicle_type FROM plate_events WHERE vehicle_type IS NOT NULL AND vehicle_type<>'' ORDER BY vehicle_type").fetchall()]
        vehicle_colors=[r['vehicle_color'] for r in con.execute("SELECT DISTINCT vehicle_color FROM plate_events WHERE vehicle_color IS NOT NULL AND vehicle_color<>'' ORDER BY vehicle_color").fetchall()]
        cities=sorted({
            str(r[0]).strip()
            for r in con.execute(
                "SELECT city FROM plate_events WHERE city<>'' "
                "UNION SELECT city FROM cameras WHERE city<>''"
            ).fetchall()
            if str(r[0]).strip()
        })
    trs=[]
    for r in rows:
        st=r['watch_status'] or 'unknown'; cls='event-blocked' if st=='blocked' else ('event-vip' if st=='vip' else '')
        vehicle=(f"<img class='thumb' onclick=\"showImage(this.src)\" src='/media?path={quote(r['image_path'])}'>" if r['image_path'] and Path(r['image_path']).exists() else '—')
        plateimg=(f"<img class='thumb plate-thumb' onclick=\"showImage(this.src)\" src='/media?path={quote(r['plate_image_path'])}'>" if r['plate_image_path'] and Path(r['plate_image_path']).exists() else '—')
        owner=escape(' / '.join(x for x in [r['owner_name'],r['vehicle_model'],r['watch_vehicle_color']] if x) or '—')
        confirmation=anpr_confirmation_badge(r['review_status'] if 'review_status' in r.keys() else 'confirmed-ai')
        city_label=escape(r['event_city'] or '—')
        region_label=persian_digits(r['plate_region'] or '—')
        trs.append(f"<tr class='{cls}'><td>{persian_digits(r['id'])}</td><td>{vehicle}</td><td>{plateimg}</td><td>{iran_plate_html(r['plate_text'],True)}{confirmation}<br>{event_status_badge(st)}</td><td>{owner}</td><td>{escape(r['vehicle_type'] or 'نامشخص')}<br><span class='muted'>{escape(r['vehicle_color'] or 'نامشخص')}</span></td><td>{persian_digits(int((r['confidence'] or 0)*100))}٪</td><td>{escape(r['camera_name'] or '—')}</td><td>{city_label}<br><span class='muted'>کد پلاک: {region_label}</span></td><td>{persian_digits(jalali_datetime(r['created_at']))}</td><td><a class='btn' href='/events/{r['id']}'>جزئیات و پخش</a></td></tr>")
    trs=''.join(trs) or "<tr><td colspan='11'>رکوردی با این فیلتر پیدا نشد.</td></tr>"
    cam_opts=''.join(f"<option {'selected' if camera==c else ''}>{escape(c)}</option>" for c in cameras)
    type_opts=''.join(f"<option {'selected' if vehicle_type==v else ''}>{escape(v)}</option>" for v in vehicle_types)
    color_opts=''.join(f"<option {'selected' if vehicle_color==v else ''}>{escape(v)}</option>" for v in vehicle_colors)
    city_opts=''.join(f"<option value='{escape(value)}'></option>" for value in cities)
    status_opts=''.join(f"<option value='{v}' {'selected' if status==v else ''}>{l}</option>" for v,l in [('allowed','مجاز'),('blocked','غیرمجاز'),('vip','VIP'),('unknown','ثبت‌نشده')])
    filter_params={
        'q':q,'camera':camera,'city':city,'region':region,'status':status,
        'vehicle_type':vehicle_type,'vehicle_color':vehicle_color,
        'date_from':date_from,'time_from':time_from,
        'date_to':date_to,'time_to':time_to,'per_page':per_page,
    }
    pager=pagination_html(
        '/events',events_page,total_pages,total_rows,filter_params,
        'events_page',per_page,
    )
    error_html=(
        f"<div class='alert'>{escape(filter_error)}</div>"
        if filter_error else ''
    )
    body=f"""<div class='wrap'><div class='toolbar'><h1 style='margin-left:auto'>گزارش ترددها</h1><a class='btn' href='/watchlist'>مدیریت پلاک‌ها</a><a class='btn secondary' href='/events/export.csv'>خروجی CSV</a></div>
    {error_html}<div class='card'><form class='filter-grid'><div><label>پلاک؛ حتی یک یا دو رقم</label><input name='q' value='{escape(q)}' placeholder='مثال: ۱۲ یا ۳۴۵'></div><div><label>دوربین</label><select name='camera'><option value=''>همه</option>{cam_opts}</select></div><div><label>شهر محل ثبت</label><input name='city' list='eventCities' value='{escape(city)}' placeholder='مثال: تهران'><datalist id='eventCities'>{city_opts}</datalist></div><div><label>کد ناحیه پلاک</label><input name='region' inputmode='numeric' maxlength='2' value='{escape(region)}' placeholder='مثال: ۷۴'></div><div><label>وضعیت</label><select name='status'><option value=''>همه</option>{status_opts}</select></div><div><label>نوع خودرو</label><select name='vehicle_type'><option value=''>همه</option>{type_opts}</select></div><div><label>رنگ خودرو</label><select name='vehicle_color'><option value=''>همه</option>{color_opts}</select></div><div><label>از تاریخ شمسی</label><input name='date_from' value='{escape(date_from)}' placeholder='۱۴۰۵/۰۵/۰۸'></div><div><label>از ساعت</label><input type='time' name='time_from' value='{escape(time_from.translate(_ALL_DIGITS))}'></div><div><label>تا تاریخ شمسی</label><input name='date_to' value='{escape(date_to)}' placeholder='۱۴۰۵/۰۵/۰۸'></div><div><label>تا ساعت</label><input type='time' name='time_to' value='{escape(time_to.translate(_ALL_DIGITS))}'></div><div><label>تعداد در هر صفحه</label><select name='per_page'>{''.join(f"<option value='{size}' {'selected' if per_page==size else ''}>{persian_digits(size)}</option>" for size in (25,50,100))}</select></div><div><button>اعمال فیلتر</button> <a class='btn secondary' href='/events'>پاک‌کردن</a></div></form></div>
    <div class='card'><div class='table-wrap'><table><tr><th>ردیف</th><th>تصویر خودرو</th><th>تصویر پلاک</th><th>پلاک/وضعیت</th><th>مالک/خودرو</th><th>تشخیص خودرو</th><th>اطمینان</th><th>دوربین</th><th>شهر / کد ناحیه</th><th>تاریخ و ساعت شمسی</th><th>عملیات</th></tr>{trs}</table></div>{pager}</div></div>
    <div id='imgModal' class='modal-img' onclick='this.classList.remove("open")'><button>بستن</button><img id='modalImage'></div><script>function showImage(src){{document.getElementById('modalImage').src=src;document.getElementById('imgModal').classList.add('open')}}</script>"""
    return page('ترددها',body,u,request)


def _retain_correction_source(endpoint):
    """Keep the event crop alive from lookup through feedback capture."""

    @wraps(endpoint)
    def retained(event_id, *args, **kwargs):
        read_pin = None
        with connect() as con:
            source_row = con.execute(
                "SELECT plate_image_path FROM plate_events WHERE id=?",
                (int(event_id),),
            ).fetchone()
        source = Path(
            source_row["plate_image_path"]
            if source_row and source_row["plate_image_path"]
            else ""
        )
        if str(source):
            try:
                candidate_pin = pin_media_paths((source,))
                if source.is_file():
                    read_pin = candidate_pin
                else:
                    candidate_pin.close()
            except (OSError, RuntimeError, StoragePolicyError, ValueError):
                read_pin = None
        try:
            return endpoint(event_id, *args, **kwargs)
        finally:
            if read_pin is not None:
                read_pin.close()

    return retained


@app.post('/events/{event_id:int}/correct')
@_retain_correction_source
def correct_event_plate(
    event_id: int,
    request: Request,
    corrected_plate: str = Form(...),
    return_to: str = Form('/dashboard'),
):
    username = auth(request)
    if not username:
        return RedirectResponse('/login', 302)
    if not has_permission(request, 'video.process'):
        return access_denied()
    safe_return_to = _safe_dashboard_return_to(return_to)
    try:
        corrected_text, corrected_norm = validate_correction(
            corrected_plate
        )
    except ValueError:
        return page(
            'پلاک نامعتبر',
            "<div class='wrap'><div class='card alert'>"
            "پلاک را با قالب کامل ایران وارد کنید؛ نمونه: "
            "۱۲ ب ۳۴۵ ایران ۶۷</div>"
            f"<a class='btn' href='{escape(safe_return_to, quote=True)}'>"
            "بازگشت</a></div>",
            username,
            request,
        )
    with connect() as con:
        row = con.execute(
            "SELECT * FROM plate_events WHERE id=?",
            (event_id,),
        ).fetchone()
        if not row:
            return JSONResponse({'error': 'event not found'}, 404)
        observed_text = row['plate_text'] or ''
        observed_norm = (
            row['raw_guess_norm']
            if 'raw_guess_norm' in row.keys()
            and row['raw_guess_norm']
            else normalize_plate(observed_text)
        )
        distance = character_distance(observed_norm, corrected_norm)
        current_feedback = con.execute(
            "SELECT id,corrected_norm FROM anpr_feedback "
            "WHERE event_id=? AND status='confirmed' "
            "ORDER BY id DESC LIMIT 1",
            (event_id,),
        ).fetchone()
        if (
            current_feedback
            and normalize_plate(current_feedback['corrected_norm'])
            == corrected_norm
        ):
            # A browser retry of the same confirmation must not inflate the
            # quality denominator or create another training truth row.
            feedback_id = int(current_feedback['id'])
        else:
            if current_feedback:
                con.execute(
                    "UPDATE anpr_feedback SET status='superseded' "
                    "WHERE id=?",
                    (int(current_feedback['id']),),
                )
            feedback_cursor = con.execute(
                "INSERT INTO anpr_feedback("
                "event_id,observed_text,observed_norm,observed_engine,"
                "observed_confidence,observed_model_revision,corrected_text,"
                "corrected_norm,character_distance,exact_match,"
                "plate_image_path,image_path,submitted_by"
                ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    event_id,
                    observed_text,
                    observed_norm,
                    (
                        row['raw_guess_engine']
                        if 'raw_guess_engine' in row.keys()
                        and row['raw_guess_engine']
                        else row['ocr_engine'] or ''
                    ),
                    float(
                        row['raw_guess_confidence']
                        if 'raw_guess_confidence' in row.keys()
                        else row['ocr_confidence'] or 0
                    ),
                    (
                        row['model_revision']
                        if 'model_revision' in row.keys()
                        else row['ocr_engine'] or ''
                    ),
                    corrected_text,
                    corrected_norm,
                    distance,
                    int(bool(observed_norm == corrected_norm)),
                    row['plate_image_path'] or '',
                    row['image_path'] or '',
                    username,
                ),
            )
            feedback_id = int(feedback_cursor.lastrowid)
        columns = {
            column[1]
            for column in con.execute(
                "PRAGMA table_info(plate_events)"
            ).fetchall()
        }
        if "review_status" in columns:
            con.execute(
                "UPDATE plate_events SET plate_text=?,plate_norm=?,"
                "plate_region=?,"
                "review_status='confirmed',"
                "confirmation_source='operator',operator_reviewed=1,"
                "experimental=0,updated_at=? WHERE id=?",
                (
                    corrected_text,
                    corrected_norm,
                    _plate_region(corrected_norm),
                    _utc_now_text(),
                    event_id,
                ),
            )
        else:
            con.execute(
                "UPDATE plate_events SET plate_text=?,plate_norm=? WHERE id=?",
                (corrected_text, corrected_norm, event_id),
            )
    audit(
        request,
        'anpr_feedback',
        f"event={event_id}; corrected={corrected_norm}",
    )
    invalidate_feedback_cache()
    capture_feedback_sample(feedback_id)
    return RedirectResponse(
        _safe_dashboard_return_to(safe_return_to, corrected=True),
        303,
    )


@app.get('/events/{event_id:int}')
def event_detail(event_id:int, request:Request):
    u=auth(request)
    if not u:return RedirectResponse('/login',302)
    with connect() as con:
        r=con.execute("""SELECT e.*,COALESCE(w.status,'unknown') watch_status,w.owner_name,w.phone,w.vehicle_model,w.vehicle_color,w.notes
            FROM plate_events e LEFT JOIN plate_watchlist w ON w.plate_norm=e.plate_norm WHERE e.id=?""",(event_id,)).fetchone()
    if not r:return page('تردد پیدا نشد',"<div class='wrap'><div class='card'><h1>تردد پیدا نشد</h1><a class='btn' href='/events'>بازگشت</a></div></div>",u,request)
    st=r['watch_status'] or 'unknown'
    second=float(r['video_second'] or 0)
    video_ok=bool(r['video_path'] and Path(r['video_path']).is_file())
    image_ok=bool(r['image_path'] and Path(r['image_path']).is_file())
    plate_ok=bool(r['plate_image_path'] and Path(r['plate_image_path']).is_file())
    video=(f"<video id='eventVideo' controls preload='metadata' src='/media?path={quote(r['video_path'])}'></video><div class='replay-controls'><button type='button' onclick='jumpToEvent()'>رفتن به لحظه عبور</button><button type='button' class='secondary' onclick='stepFrame(-1)'>فریم قبل</button><button type='button' class='secondary' onclick='stepFrame(1)'>فریم بعد</button><button type='button' class='secondary' onclick='setSpeed(.25)'>۰٫۲۵×</button><button type='button' class='secondary' onclick='setSpeed(.5)'>۰٫۵×</button><button type='button' class='secondary' onclick='setSpeed(1)'>۱×</button><button type='button' class='secondary' onclick='setSpeed(2)'>۲×</button><span id='speedLabel' class='muted'>سرعت: ۱×</span></div>" if video_ok else "<div class='alert'>فایل ویدئوی این تردد موجود نیست یا طبق تنظیمات نگهداری حذف شده است.</div>")
    vehicle=(f"<img onclick='showImage(this.src)' src='/media?path={quote(r['image_path'])}'>" if image_ok else "<div class='muted'>تصویر خودرو موجود نیست</div>")
    plate=(f"<img onclick='showImage(this.src)' src='/media?path={quote(r['plate_image_path'])}'>" if plate_ok else "<div class='muted'>تصویر پلاک موجود نیست</div>")
    owner=' / '.join(x for x in [r['owner_name'],r['vehicle_model'],r['vehicle_color']] if x) or 'ثبت نشده'
    confirmation=anpr_confirmation_badge(r['review_status'] if 'review_status' in r.keys() else 'confirmed-ai')
    media_notice=(
        f"<div class='alert'>ذخیره بعضی تصاویر این رخداد کامل نشده است: "
        f"{escape(r['media_error'] or 'خطای نامشخص')}</div>"
        if 'media_status' in r.keys()
        and r['media_status'] in {'partial','error'}
        else ''
    )
    correction_value=escape(str(r['plate_text'] or ''))
    correction_form=f"""<form class='correction-form' method='post' action='/events/{r['id']}/correct'><input name='corrected_plate' required maxlength='20' value='{correction_value}' placeholder='مثال: ۱۲ ب ۳۴۵ ایران ۶۷'><button>تأیید/اصلاح و آموزش</button></form>"""
    body=f"""<div class='wrap'><div class='toolbar'><h1 style='margin-left:auto'>جزئیات تردد شماره {r['id']}</h1><a class='btn secondary' href='/events'>بازگشت به ترددها</a></div>
    <div class='replay-layout'><div class='card video-panel'><h3>پخش ویدئو از لحظه عبور</h3><div class='time-badge'>زمان ثبت در ویدئو: {persian_digits(f"{second:.2f}").replace(".", "٫")} ثانیه</div>{video}</div>
    <div>{media_notice}<div class='card'><h3>اطلاعات تردد</h3><div class='event-meta'><div class='meta-item' style='grid-column:1/-1'><small>پلاک</small>{iran_plate_html(r['plate_text'])}{confirmation}</div><div class='meta-item' style='grid-column:1/-1'><small>تأیید یا اصلاح اپراتور</small>{correction_form}</div><div class='meta-item'><small>وضعیت</small>{event_status_badge(st)}</div><div class='meta-item'><small>دوربین</small>{escape(r['camera_name'] or '—')}</div><div class='meta-item'><small>شهر محل ثبت</small>{escape(r['city'] or '—')}</div><div class='meta-item'><small>کد ناحیه پلاک</small>{persian_digits(r['plate_region'] or '—')}</div><div class='meta-item'><small>اطمینان</small>{persian_digits(f"{(r['confidence'] or 0)*100:.1f}")}٪</div><div class='meta-item'><small>تاریخ و ساعت شمسی</small>{persian_digits(jalali_datetime(r['created_at']))}</div><div class='meta-item'><small>روش تشخیص</small>{escape(r['detector_method'] or '—')}</div><div class='meta-item'><small>نوع خودرو</small>{escape(r['vehicle_type'] or 'نامشخص')}</div><div class='meta-item'><small>رنگ خودرو</small>{escape(r['vehicle_color'] or 'نامشخص')}</div><div class='meta-item'><small>اطمینان تشخیص خودرو</small>{persian_digits(f"{(r['vehicle_confidence'] or 0)*100:.1f}")}٪</div><div class='meta-item'><small>مالک / خودرو</small>{escape(owner)}</div><div class='meta-item'><small>شماره تماس</small>{persian_digits(r['phone'] or '—')}</div></div></div>
    <div class='card'><h3>تصاویر ثبت‌شده</h3><div class='detail-images'><div>{vehicle}<small>تصویر خودرو</small></div><div>{plate}<small>تصویر پلاک</small></div></div></div></div></div></div>
    <div id='imgModal' class='modal-img' onclick='this.classList.remove("open")'><button>بستن</button><img id='modalImage'></div>
    <script>const eventSecond={second:.3f};const v=document.getElementById('eventVideo');function jumpToEvent(){{if(!v)return;v.currentTime=Math.max(0,eventSecond-.5);v.play().catch(()=>{{}})}}function stepFrame(dir){{if(!v)return;v.pause();v.currentTime=Math.max(0,v.currentTime+dir/25)}}function setSpeed(rate){{if(!v)return;v.playbackRate=rate;document.getElementById('speedLabel').textContent='سرعت: '+window.faDigits(rate)+'×'}}function showImage(src){{document.getElementById('modalImage').src=src;document.getElementById('imgModal').classList.add('open')}}if(v){{v.addEventListener('loadedmetadata',jumpToEvent,{{once:true}})}}</script>"""
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

def _fsync_file(path):
    with Path(path).open('r+b') as handle:
        os.fsync(handle.fileno())

def _lstat(path):
    try:
        return Path(path).lstat()
    except FileNotFoundError:
        return None

def _regular_file_identity(path, *, expected=None, links=1):
    details=_lstat(path)
    if details is None:
        raise RuntimeError(f'فایل مورد انتظار ایجاد نشد: {path}')
    identity=path_file_identity(path,details=details)
    if (
        not stat.S_ISREG(details.st_mode)
        or int(details.st_nlink)!=int(links)
        or (expected is not None and identity!=expected)
    ):
        raise RuntimeError(f'مالکیت فایل مهاجرت قابل تأیید نیست: {path}')
    return identity

def _unlink_owned_regular(path, identity):
    target=Path(path)
    details=_lstat(target)
    if details is None:
        return False
    if (
        not stat.S_ISREG(details.st_mode)
        or int(details.st_nlink)!=1
        or path_file_identity(target,details=details)!=identity
    ):
        raise RuntimeError(
            f'فایل مهاجرت با فایل دیگری جایگزین شده و حذف نشد: {target}'
        )
    target.unlink()
    fsync_parent_directory(target)
    return True

def _rmdir_owned(path, identity):
    target=Path(path)
    details=_lstat(target)
    if details is None:
        return False
    if (
        not stat.S_ISDIR(details.st_mode)
        or path_file_identity(target,details=details)!=identity
    ):
        raise RuntimeError(
            f'پوشه موقت مهاجرت با مسیر دیگری جایگزین شده است: {target}'
        )
    target.rmdir()
    fsync_parent_directory(target)
    return True

def _copy_private_regular_file(source, destination):
    source=Path(source)
    destination=Path(destination)
    source_details=source.lstat()
    source_identity=path_file_identity(source,details=source_details)
    if (
        not stat.S_ISREG(source_details.st_mode)
        or int(source_details.st_nlink)!=1
    ):
        raise RuntimeError(f'فایل محرمانه منبع ناامن است: {source}')
    created_identity=None
    try:
        with source.open('rb') as reader:
            opened=os.fstat(reader.fileno())
            if (
                not stat.S_ISREG(opened.st_mode)
                or int(opened.st_nlink)!=1
                or descriptor_file_identity(
                    reader.fileno(),details=opened,
                )!=source_identity
            ):
                raise RuntimeError(
                    f'فایل محرمانه منبع هنگام مهاجرت تغییر کرد: {source}'
                )
            with destination.open('xb') as writer:
                created=os.fstat(writer.fileno())
                created_identity=descriptor_file_identity(
                    writer.fileno(),details=created,
                )
                os.fchmod(writer.fileno(),0o600)
                shutil.copyfileobj(reader,writer,1024*1024)
                writer.flush()
                os.fsync(writer.fileno())
        _regular_file_identity(
            destination,expected=created_identity,links=1,
        )
        fsync_parent_directory(destination)
        return created_identity
    except BaseException:
        if created_identity is not None:
            try:
                _unlink_owned_regular(destination,created_identity)
            except Exception:
                pass
        raise

def _publish_staged_file(staged, target, staged_identity):
    staged=Path(staged)
    target=Path(target)
    staged_details=staged.lstat()
    _regular_file_identity(staged,expected=staged_identity,links=1)
    if _lstat(target) is not None:
        raise RuntimeError(
            f'فایل مقصد مهاجرت هم‌زمان ایجاد شد و بازنویسی نشد: {target}'
        )
    hardlinked=False
    try:
        os.link(staged,target,follow_symlinks=False)
        hardlinked=True
    except FileExistsError as exc:
        raise RuntimeError(
            f'فایل مقصد مهاجرت هم‌زمان ایجاد شد و بازنویسی نشد: {target}'
        ) from exc
    except OSError as link_error:
        # FAT/exFAT and some network filesystems do not support hard links.
        # Fall back to an exclusively-created target opened by descriptor;
        # a crash can leave an unpublished partial artifact, but it can never
        # overwrite a foreign path and the storage pointer remains unchanged.
        current=_lstat(target)
        if current is not None and (
            stat.S_ISREG(current.st_mode)
            and int(current.st_nlink)==2
            and path_file_identity(target,details=current)==staged_identity
        ):
            hardlinked=True
        elif current is not None:
            raise RuntimeError(
                f'فایل مقصد مهاجرت هم‌زمان تغییر کرد: {target}'
            ) from link_error
        else:
            descriptor=None
            target_identity=None
            try:
                descriptor=os.open(
                    target,
                    os.O_WRONLY|os.O_CREAT|os.O_EXCL
                    |getattr(os,'O_BINARY',0)
                    |getattr(os,'O_NOFOLLOW',0),
                    stat.S_IMODE(staged_details.st_mode) or 0o600,
                )
                created=os.fstat(descriptor)
                target_identity=descriptor_file_identity(
                    descriptor,details=created,
                )
                if (
                    not stat.S_ISREG(created.st_mode)
                    or int(created.st_nlink)!=1
                ):
                    raise RuntimeError(
                        'فایل مقصد جایگزین مهاجرت خصوصی نیست.'
                    )
                with staged.open('rb') as reader:
                    opened=os.fstat(reader.fileno())
                    if (
                        not stat.S_ISREG(opened.st_mode)
                        or int(opened.st_nlink)!=1
                        or descriptor_file_identity(
                            reader.fileno(),details=opened,
                        )!=staged_identity
                    ):
                        raise RuntimeError(
                            'فایل staging هنگام انتشار تغییر کرد.'
                        )
                    while chunk:=reader.read(1024*1024):
                        remaining=memoryview(chunk)
                        while remaining:
                            written=os.write(descriptor,remaining)
                            if written<=0:
                                raise OSError(
                                    'انتشار مهاجرت پیشرفتی نداشت.'
                                )
                            remaining=remaining[written:]
                os.fsync(descriptor)
                completed=os.fstat(descriptor)
                if int(completed.st_size)!=int(staged_details.st_size):
                    raise RuntimeError(
                        'اندازه فایل مقصد مهاجرت کامل نیست.'
                    )
            except BaseException:
                if descriptor is not None:
                    os.close(descriptor)
                    descriptor=None
                if target_identity is not None:
                    try:
                        _unlink_owned_regular(target,target_identity)
                    except Exception:
                        pass
                raise
            finally:
                if descriptor is not None:
                    os.close(descriptor)
            _regular_file_identity(
                target,expected=target_identity,links=1,
            )
            staged.unlink()
            fsync_parent_directory(target)
            return target_identity
    if not hardlinked:
        raise RuntimeError('انتشار فایل مهاجرت کامل نشد.')
    try:
        _regular_file_identity(target,expected=staged_identity,links=2)
        fsync_parent_directory(target)
        staged.unlink()
        fsync_parent_directory(staged)
        _regular_file_identity(target,expected=staged_identity,links=1)
    except BaseException:
        # Never remove a pathname whose inode no longer belongs to this
        # migration.  Keeping an owned hard link is safer than deleting a
        # concurrently substituted file.
        current=_lstat(target)
        if current is not None and (
            stat.S_ISREG(current.st_mode)
            and path_file_identity(target,details=current)==staged_identity
        ):
            try:
                target.unlink()
                fsync_parent_directory(target)
            except OSError:
                pass
        raise
    return staged_identity

def _storage_pointer_targets_root(config_path, expected_root):
    path=Path(config_path)
    try:
        details=path.lstat()
        if (
            not stat.S_ISREG(details.st_mode)
            or int(details.st_nlink)!=1
            or details.st_size<2
            or details.st_size>4096
        ):
            return False
        identity=path_file_identity(path,details=details)
        with path.open('r',encoding='utf-8') as handle:
            opened=os.fstat(handle.fileno())
            if descriptor_file_identity(
                handle.fileno(),details=opened,
            )!=identity:
                return False
            payload=json.load(handle)
        if not isinstance(payload,dict):
            return False
        value=payload.get('storage_root')
        if not isinstance(value,str) or not value.strip():
            return False
        selected=Path(value.strip()).expanduser()
        if not selected.is_absolute():
            return False
        return selected.resolve()==Path(expected_root).resolve()
    except (OSError,RuntimeError,ValueError,TypeError,json.JSONDecodeError):
        return False

def _mkdir_durable(directory):
    directory=Path(directory)
    missing=[]
    current=directory
    while not current.exists() and current != current.parent:
        missing.append(current)
        current=current.parent
    directory.mkdir(parents=True,exist_ok=True)
    for created in reversed(missing):
        fsync_parent_directory(created)

def _csv_cell(value):
    return csv_safe_cell(value)

MEDIA_FILE_EXTENSIONS={
    '.jpg','.jpeg','.png','.webp','.bmp',
    '.mp4','.avi','.mkv','.mov','.m4v',
}

class _PinnedFileResponse(FileResponse):
    def __init__(self,*args,read_pin,**kwargs):
        self._read_pin=read_pin
        super().__init__(*args,**kwargs)

    async def __call__(self,scope,receive,send):
        try:
            await super().__call__(scope,receive,send)
        finally:
            self._read_pin.close()

def _media_roots_history(*,strict=False):
    try:
        raw=json.loads(get_setting('media_roots_history','[]'))
    except (TypeError,ValueError,json.JSONDecodeError) as exc:
        if strict:
            raise ValueError(
                'سابقه مسیرهای رسانه خراب است و باید پیش از تغییر اصلاح شود.'
            ) from exc
        return []
    if not isinstance(raw,list):
        if strict:
            raise ValueError('ساختار سابقه مسیرهای رسانه معتبر نیست.')
        return []
    roots=[]
    for value in raw:
        try:
            root=Path(str(value)).expanduser().resolve()
            if root == Path(root.anchor):
                raise ValueError('ریشه سیستم در سابقه رسانه مجاز نیست.')
            if root not in roots:
                roots.append(root)
        except (OSError,RuntimeError,ValueError) as exc:
            if strict:
                raise ValueError(
                    f'مسیر سابقه رسانه قابل اعتبارسنجی نیست: {value}'
                ) from exc
            continue
    return roots

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
    try:
        validate_storage_layout(root, paths[1:4], paths[4])
    except StoragePolicyError as exc:
        raise ValueError(f'چیدمان مسیرهای ذخیره‌سازی ناامن است: {exc}') from exc
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

def _video_suffix(filename):
    safe_name=Path(str(filename or '').replace('\\','/')).name
    suffix=Path(safe_name).suffix.lower()
    return suffix if suffix in VIDEO_EXTENSIONS else ''

class _PendingVideoUpload:
    def __init__(
        self,
        target,
        size,
        reservation,
        read_pin,
        created_identity=None,
        acceptance_id=None,
        claim_succeeded=False,
    ):
        self.target=target
        self.size=size
        self._reservation=reservation
        self._read_pin=read_pin
        self._created_identity=created_identity
        self.acceptance_id=acceptance_id
        self._claim_succeeded=bool(claim_succeeded)
        self.committed=False

    def accept(self, connection, *, owner_kind, owner_id):
        if not self.acceptance_id:
            raise RuntimeError('video upload has no database acceptance intent')
        from app.media_acceptance import accept_intent,current_identity

        identity,size_bytes=current_identity(self.target)
        if identity!=self._created_identity or size_bytes!=self.size:
            raise RuntimeError('video upload identity changed before acceptance')
        accept_intent(
            connection,
            self.acceptance_id,
            self.target,
            identity,
            size_bytes,
            owner_kind=owner_kind,
            owner_id=owner_id,
        )

    def commit(self):
        if self.committed:
            return
        reservation=self._reservation
        if reservation is None:
            raise RuntimeError('video upload reservation is unavailable')
        reservation.close(success=True,actual_bytes=self.size)
        self._reservation=None
        self.committed=True

    def rollback(self):
        if self.committed:
            return
        errors=[]
        reservation=self._reservation
        rollback_completed=False
        if reservation is not None:
            try:
                reservation.close(success=False)
            except Exception as exc:
                errors.append(exc)
            else:
                self._reservation=None
                rollback_completed=True
        # The durable reservation normally removes a claimed inode. This
        # identity-checked fallback covers a claim-journal failure without
        # ever unlinking a foreign file that reused the pathname.
        if not self._claim_succeeded or not rollback_completed:
            try:
                try:
                    current=self.target.lstat()
                except FileNotFoundError:
                    current=None
                if current is not None and (
                    stat.S_ISREG(current.st_mode)
                    and int(current.st_nlink)==1
                    and path_file_identity(
                        self.target,details=current,
                    )==self._created_identity
                ):
                    self.target.unlink()
                    fsync_parent_directory(self.target)
            except OSError as exc:
                errors.append(exc)
        if self.acceptance_id and (
            rollback_completed or reservation is None
        ):
            try:
                from app.media_acceptance import discard_intent

                discard_intent(self.acceptance_id)
            except Exception as exc:
                errors.append(exc)
        try:
            self.close_pin()
        except Exception as exc:
            errors.append(exc)
        if errors:
            raise RuntimeError(
                'video upload rollback was incomplete: '
                + '; '.join(str(error) for error in errors)
            ) from errors[0]

    def settle_after_owner_attempt(self):
        if not self.acceptance_id:
            self.rollback()
            return
        from app.media_acceptance import load_intent

        intent=load_intent(self.acceptance_id)
        if intent is not None and intent.get('state')=='accepted':
            self.commit()
        else:
            self.rollback()

    def release_owner(self, connection):
        if not self.acceptance_id:
            return
        connection.execute(
            "DELETE FROM media_acceptance_intents "
            "WHERE acceptance_id=? AND state='accepted'",
            (self.acceptance_id,),
        )

    def detach_pin(self):
        read_pin=self._read_pin
        self._read_pin=None
        return read_pin

    def close_pin(self):
        read_pin=self._read_pin
        self._read_pin=None
        if read_pin is not None:
            read_pin.close()


async def _stage_video_upload(
    video,
    save_dir,
    suffix,
    *,
    create_pin=True,
    acceptance_required=False,
):
    save_dir=Path(save_dir).resolve()
    target=save_dir / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(8)}{suffix}"
    size=0
    reservation=None
    post_write_pin=None
    created_identity=None
    acceptance_id=None
    claim_succeeded=False
    try:
        # The stop action is checked before the destination file (or even its
        # parent directory) is created. As chunks arrive, the reservation is
        # grown so concurrent uploads and image writes cannot oversubscribe
        # the configured byte limit.
        if acceptance_required:
            from app.media_acceptance import create_intent

            acceptance_id=create_intent(target)
            reservation=begin_media_write(
                target,0,acceptance_id=acceptance_id,
            )
        else:
            reservation=begin_media_write(target,0)
        _mkdir_durable(save_dir)
        with target.open('xb') as f:
            created=os.fstat(f.fileno())
            created_identity=descriptor_file_identity(
                f.fileno(),details=created,
            )
            claim_created_path=getattr(
                reservation,'claim_created_path',None,
            )
            if callable(claim_created_path):
                claim_created_path(target)
                claim_succeeded=True
            while chunk:=await video.read(1024*1024):
                next_size=size+len(chunk)
                if next_size>MAX_VIDEO_UPLOAD_BYTES:
                    raise ValueError('حجم ویدئو بیشتر از ۲ گیگابایت است.')
                reservation.grow(next_size)
                f.write(chunk)
                size=next_size
            f.flush()
            os.fsync(f.fileno())
        # The file's directory entry must be durable before a later commit is
        # allowed to permanently delete files quarantined to make room.
        fsync_parent_directory(target)
        # Establish the read lease while the write reservation still protects
        # the same target. No cleanup window exists during the handoff.
        if create_pin:
            post_write_pin=pin_media_paths((target,))
    except BaseException as exc:
        pending=_PendingVideoUpload(
            target,size,reservation,post_write_pin,created_identity,
            acceptance_id,claim_succeeded,
        )
        try:
            pending.rollback()
        except Exception as cleanup_error:
            exc.add_note(f'upload rollback error: {cleanup_error}')
        if isinstance(exc,StoragePolicyError):
            raise ValueError(
                f'سهمیه ذخیره‌سازی اجازه ثبت ویدئو را نمی‌دهد: {exc}'
            ) from exc
        raise
    return _PendingVideoUpload(
        target,size,reservation,post_write_pin,created_identity,
        acceptance_id,claim_succeeded,
    )


async def _save_video_upload(video, save_dir, suffix, *, pin_after_save=False):
    """Compatibility helper for callers that accept the file immediately."""

    pending=await _stage_video_upload(
        video,
        save_dir,
        suffix,
        create_pin=pin_after_save,
    )
    try:
        pending.commit()
    except BaseException:
        pending.rollback()
        pending.close_pin()
        raise
    if pin_after_save:
        return pending.target,pending.detach_pin()
    return pending.target

def _cleanup_old_files(folder, days, storage_root):
    if days <= 0: return 0
    cutoff=time.time()-days*86400
    try:
        return delete_older_than(
            folder,
            cutoff,
            storage_root=storage_root,
        )
    except StoragePolicyError:
        return 0

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
            # Operator-confirmed feedback is durable training truth. Preserve
            # both its row and source event regardless of report retention.
            con.execute(
                "DELETE FROM plate_events "
                "WHERE created_at < datetime('now', ?) "
                "AND NOT EXISTS("
                "SELECT 1 FROM anpr_feedback "
                "WHERE anpr_feedback.event_id=plate_events.id)",
                (f'-{event_days} days',),
            )
    try:
        quota_status=enforce_storage_limit()
        removed += quota_status.deleted_files
    except StoragePolicyError:
        # Invalid/corrupt settings are fail-closed by write reservations. A
        # startup maintenance pass must not make the service unavailable.
        pass
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
    form="""<div class='card'><h3>افزودن کاربر</h3><form method='post' action='/users'><div class='two-col'><div><label>نام کاربری</label><input name='username' required></div><div><label>نام نمایشی</label><input name='display_name' required></div><div><label>رمز عبور</label><input type='password' name='password' minlength='8' required></div><div><label>نقش</label><select name='role'><option value='operator'>اپراتور</option><option value='guard'>نگهبان</option><option value='system'>مدیر سیستم</option><option value='admin'>مدیر کل</option></select></div></div><button>ایجاد کاربر</button></form></div>"""
    return page('مدیریت کاربران',f"<div class='wrap'><div class='toolbar'><h1 style='margin-left:auto'>مدیریت کاربران</h1><a class='btn secondary' href='/audit'>لاگ فعالیت‌ها</a></div>{note}{form}<div class='card'><div class='table-wrap'><table><tr><th>ID</th><th>کاربر</th><th>نقش</th><th>وضعیت</th><th>آخرین ورود</th><th>عملیات</th></tr>{trs}</table></div></div></div>",u,request)

@app.post('/users')
def create_user_route(request:Request,username:str=Form(...),display_name:str=Form(...),password:str=Form(...),role:str=Form('operator')):
    if not require_admin(request):return RedirectResponse('/dashboard',303)
    if role not in ROLE_LABELS or len(password)<8:return RedirectResponse('/users?error='+quote('اطلاعات کاربر معتبر نیست'),303)
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
    body=f"""<div class='wrap'><div class='card'><h1>ویرایش کاربر</h1><form method='post'><label>نام نمایشی</label><input name='display_name' value='{escape(r['display_name'])}' required><label>نقش</label><select name='role'>{opts}</select><label>رمز جدید (اختیاری)</label><input type='password' name='password' minlength='8'><button>ذخیره</button> <a class='btn secondary' href='/users'>بازگشت</a></form></div></div>"""
    return page('ویرایش کاربر',body,u,request)

@app.post('/users/{user_id}/edit')
def edit_user_route(user_id:int,request:Request,display_name:str=Form(...),role:str=Form(...),password:str=Form('')):
    if not require_admin(request):return RedirectResponse('/dashboard',303)
    if role not in ROLE_LABELS or (password and len(password)<8):
        return RedirectResponse('/users?error='+quote('اطلاعات کاربر معتبر نیست'),303)
    with connect() as con:
        if password:con.execute('UPDATE users SET display_name=?,role=?,is_admin=?,password_hash=?,must_change_password=0,session_version=session_version+1 WHERE id=?',(display_name.strip(),role,1 if role=='admin' else 0,hash_password(password),user_id))
        else:con.execute('UPDATE users SET display_name=?,role=?,is_admin=? WHERE id=?',(display_name.strip(),role,1 if role=='admin' else 0,user_id))
    audit(request,'user_update',f'ویرایش کاربر شماره {user_id}')
    return RedirectResponse('/users?msg=1',303)

@app.post('/users/{user_id}/toggle')
def toggle_user(user_id:int,request:Request):
    cu=current_user(request)
    if not cu or not require_admin(request):return RedirectResponse('/dashboard',303)
    if cu['id']==user_id:return RedirectResponse('/users?error='+quote('نمی‌توانید حساب خودتان را غیرفعال کنید'),303)
    with connect() as con:con.execute('UPDATE users SET is_active=CASE is_active WHEN 1 THEN 0 ELSE 1 END,session_version=session_version+1 WHERE id=?',(user_id,))
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
def settings(request:Request,saved:int=0,restart:int=0,error:str='',password_required:int=0):
    u=auth(request)
    if not u:return RedirectResponse('/login',302)
    if not has_permission(request,'system.manage'):return access_denied()
    msg="<div class='card ok'>تنظیمات ذخیره شد.</div>" if saved else ''
    if password_required:
        msg += (
            "<div class='alert'>برای ادامه، رمز پیش‌فرض را با یک رمز "
            "اختصاصی حداقل ۸ کاراکتری جایگزین کنید.</div>"
        )
    if restart: msg += "<div class='alert' style='background:#fff8e5;color:#815b00;border-color:#ffe3a3'>مسیر اصلی ذخیره‌سازی تغییر کرد. برای استفاده کامل از دیتابیس در مسیر جدید، برنامه را یک‌بار ببندید و دوباره اجرا کنید.</div>"
    if error: msg += f"<div class='alert'>{escape(error)}</div>"
    root=get_setting('storage_root',str(DATA_DIR)); snap=get_setting('snapshot_path',str(Path(root)/'snapshots')); plates=get_setting('plate_path',str(Path(root)/'plates')); videos=get_setting('video_path',str(Path(root)/'videos')); backups=get_setting('backup_path',str(Path(root)/'backups'))
    usage=_path_usage(root)
    usage_html=(f"<div class='drive-card'><div class='meter-label'><b>{escape(usage['path'])}</b><span>{usage['percent']}٪</span></div><div class='storage-progress'><span style='width:{usage['percent']}%'></span></div><span class='muted'>آزاد: {_fmt_bytes(usage['free'])} از {_fmt_bytes(usage['total'])}</span></div>" if usage['ok'] else f"<div class='alert'>مسیر ذخیره‌سازی در دسترس نیست: {escape(usage.get('error',''))}</div>")
    checked=lambda k: 'checked' if get_setting(k,'0')=='1' else ''
    selected=lambda k,v: 'selected' if get_setting(k,'')==v else ''
    training=latest_training_status(); training_run=training.get('run')
    with connect() as con:
        quality=feedback_quality_summary(con)
    training_labels={
        'queued':'در صف','running':'در حال آموزش',
        'awaiting-golden':'در انتظار ارزیابی Golden',
        'candidate-ready':'مدل نامزد آماده فعال‌سازی آزمایشی',
        'rejected':'ردشده به‌دلیل افت دقت',
        'applied':'فعال‌شده در مسیر آزمایشی','error':'خطای آموزش',
        'interrupted':'متوقف‌شده با بستن برنامه',
    }
    training_state=(
        training_labels.get(
            training_run['status'],
            training_run['status'],
        )
        if training_run else 'بدون آموزش'
    )
    training_metrics=(
        "<p class='muted'>دقت مدل کنترل آزمایشی: "
        + persian_digits(f"{float(training_run['baseline_accuracy'] or 0)*100:.1f}")
        + "٪ | دقت مدل نامزد: "
        + persian_digits(f"{float(training_run['candidate_accuracy'] or 0)*100:.1f}")
        + "٪</p>"
        if training_run and training_run['status'] in {
            'candidate-ready','rejected','applied'
        }
        else ""
    )
    training_action=(
        f"<form method='post' action='/settings/ai/training/apply'>"
        f"<input type='hidden' name='run_id' value='{training_run['id']}'>"
        "<button>فعال‌سازی آزمایشی مدل نامزد</button></form>"
        if training_run and training_run['status']=='candidate-ready'
        else (
            f"<form method='post' action='/settings/ai/training/evaluate'>"
            f"<input type='hidden' name='run_id' value='{training_run['id']}'>"
            "<button>اجرای ارزیابی مستقل Golden</button></form>"
            if training_run and training_run['status']=='awaiting-golden'
            else (
            "<form method='post' action='/settings/ai/training/start'>"
            "<label>دوره آموزش</label><input type='number' name='epochs' "
            "min='4' max='40' value='12'>"
            "<label><input style='width:auto' type='checkbox' "
            "name='rights_attested' value='1' required> تأیید می‌کنم "
            "سازمان مالک تصاویر است یا مجوز صریح آموزش و توزیع مشتقات "
            "مدل را دارد.</label><button>شروع آموزش کنترل‌شده</button>"
            "</form>"
            if training['ready']
            else "<span class='muted'>با افزایش اصلاحات تأییدشده، آموزش فعال می‌شود.</span>"
            )
        )
    )
    quality_models=''.join(
        "<tr><td>"
        + escape(str(row['model_revision']))
        + "</td><td>"
        + persian_digits(row['reviewed'])
        + "</td><td>"
        + persian_digits(f"{row['coverage']*100:.1f}")
        + "٪</td><td>"
        + persian_digits(f"{row['exact_accuracy']*100:.1f}")
        + "٪</td><td>"
        + persian_digits(row['mean_character_error_end_to_end'])
        + "</td></tr>"
        for row in quality['by_model'][-6:]
    ) or "<tr><td colspan='5'>هنوز ترددی توسط اپراتور بررسی نشده است.</td></tr>"
    body=f"""<div class='wrap'><h1>تنظیمات</h1>{msg}
    <div class='card'><h3>نمایش زنده</h3><form method='post' action='/settings/display'><div class='two-col'><div><label>تعداد ستون نمایش زنده</label><select name='dashboard_grid'>{''.join(f'<option value={x} '+('selected' if get_setting('dashboard_grid','2')==str(x) else '')+f'>{x} ستون</option>' for x in [1,2,3,4])}</select></div><div><label>تعداد سطرهای پلاک در داشبورد</label><input type='number' min='6' max='50' name='dashboard_event_rows' value='{get_setting('dashboard_event_rows','12')}'></div><div><label>تعداد فریم نمایش در ثانیه</label><input type='number' min='1' max='15' name='live_fps' value='{get_setting('live_fps','5')}'></div><div><label>عرض تصویر لایو</label><select name='stream_width'>{''.join(f'<option value={x} '+('selected' if get_setting('stream_width','640')==str(x) else '')+f'>{x}px</option>' for x in [480,640,960,1280])}</select></div><div><label>کیفیت JPEG</label><input type='number' min='30' max='95' name='jpeg_quality' value='{get_setting('jpeg_quality','70')}'></div></div><label>رمز جدید مدیر (حداقل ۸ کاراکتر)</label><input type='password' name='new_password' minlength='8' autocomplete='new-password'><button>ذخیره تنظیمات نمایش</button></form></div>
    <div class='card' id='storage'><h3>ذخیره‌سازی</h3><p class='muted'>درایو یا پوشه اصلی و مسیر جداگانه هر نوع اطلاعات را انتخاب کنید.</p>{usage_html}<form method='post' action='/settings/storage' style='margin-top:18px'><label>مسیر اصلی ذخیره‌سازی</label><input class='code' name='storage_root' value='{escape(root)}' placeholder='D:\\BCVisionData'><div class='storage-grid'><div><label>تصاویر خودرو</label><input class='code' name='snapshot_path' value='{escape(snap)}'></div><div><label>تصاویر پلاک</label><input class='code' name='plate_path' value='{escape(plates)}'></div><div><label>ویدئوها</label><input class='code' name='video_path' value='{escape(videos)}'></div><div><label>نسخه‌های پشتیبان</label><input class='code' name='backup_path' value='{escape(backups)}'></div></div>
    <div class='two-col'><label><input style='width:auto' type='checkbox' name='save_snapshots' value='1' {checked('save_snapshots')}> ذخیره تصویر خودرو</label><label><input style='width:auto' type='checkbox' name='save_plate_images' value='1' {checked('save_plate_images')}> ذخیره تصویر پلاک</label><label><input style='width:auto' type='checkbox' name='save_videos' value='1' {checked('save_videos')}> ذخیره ویدئو</label><div><label>حداکثر فضای مجاز (GB؛ صفر یعنی نامحدود)</label><input type='number' min='0' name='max_storage_gb' value='{get_setting('max_storage_gb','0')}'></div></div>
    <label>وقتی فضا پر شد</label><select name='storage_full_action'><option value='delete_oldest' {selected('storage_full_action','delete_oldest')}>حذف قدیمی‌ترین اطلاعات</option><option value='stop' {selected('storage_full_action','stop')}>توقف ذخیره‌سازی</option><option value='alert' {selected('storage_full_action','alert')}>فقط نمایش هشدار</option></select>
    <h3>مدت نگهداری</h3><div class='storage-grid'><div><label>تصاویر خودرو (روز)</label><input type='number' min='0' name='retention_snapshots_days' value='{get_setting('retention_snapshots_days','90')}'></div><div><label>تصاویر پلاک (روز)</label><input type='number' min='0' name='retention_plates_days' value='{get_setting('retention_plates_days','90')}'></div><div><label>ویدئوها (روز)</label><input type='number' min='0' name='retention_videos_days' value='{get_setting('retention_videos_days','7')}'></div><div><label>رویدادها (روز؛ صفر یعنی نامحدود)</label><input type='number' min='0' name='retention_events_days' value='{get_setting('retention_events_days','0')}'></div></div><button>ذخیره تنظیمات ذخیره‌سازی</button></form></div>
<div class='card'><h3>🧠 تنظیمات هوش مصنوعی</h3>
<form method='post' action='/settings/ai'>
<label>مدل تشخیص پلاک</label>
<select name='anpr_detector_model'>
<option value='yolo11n' {selected('anpr_detector_model','yolo11n')}>YOLO11n (پیش‌فرض پیشنهادی)</option>
<option value='yolov8n' {selected('anpr_detector_model','yolov8n')}>YOLOv8n</option>
<option value='yolox' {selected('anpr_detector_model','yolox')}>YOLOX اختصاصی</option>
</select>
<p class='muted'>مسیر پیشنهادی فعلی: YOLO11n برای تشخیص کادر، سپس Hezar v2
برای خواندن پلاک و فقط در صورت رد یا خطای آن، Platrix ثابت. فقط تشخیصگر
انتخاب‌شده اجرا می‌شود و مدل دیگری پنهانی وارد پردازش نمی‌شود.</p>
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
<label><input style='width:auto' type='checkbox'
name='anpr_auto_confirm_guesses' value='1'
{checked('anpr_auto_confirm_guesses')}>
ثبت حدس‌های کامل چندفریمی به‌عنوان «تأیید خودکار مدل»</label>
<p class='muted'>این نتایج وارد گزارش تردد می‌شوند، اما تا تأیید یا اصلاح
اپراتور به دیتاست آموزشی اضافه نخواهند شد.</p>
<label><input style='width:auto' type='checkbox'
name='anpr_engine_v2_shadow' value='1'
{checked('anpr_engine_v2_shadow')}>
اجرای آزمایشی Engine V2 در حالت Shadow</label>
<p class='muted'>موتور قدیمی Production همیشه پلاک‌ها را می‌خواند و ثبت
می‌کند. Engine V2 فقط هم‌زمان مقایسه و Overlay تولید می‌کند و هیچ رویدادی
در دیتابیس ثبت نمی‌کند.</p>
<button>ذخیره تنظیمات AI</button>
</form>
</div>
<div class='card' id='ai-training'><h3>یادگیری از اصلاحات اپراتور</h3>
<p class='muted'>تصاویر اصلاح‌شده در دیتاست محلی نگهداری می‌شوند. مدل نامزد
فقط پس از آزمون روی مجموعه جدا و بدون افت قابل فعال‌سازی آزمایشی است. این مدل
روی تشخیص زنده اثر ندارد؛ مسیر تولید همیشه HEZAR v2 و سپس Platrix ثابت است.</p>
<div class='stats-grid'>
<div class='stat-card'><span class='muted'>نمونه آموزشی</span><div class='stat'>{persian_digits(training['train_samples'])}</div></div>
<div class='stat-card'><span class='muted'>نمونه اعتبارسنجی</span><div class='stat'>{persian_digits(training['validation_samples'])}</div></div>
<div class='stat-card'><span class='muted'>پلاک یکتا</span><div class='stat'>{persian_digits(training['unique_plates'])}</div></div>
<div class='stat-card'><span class='muted'>وضعیت</span><div style='font-weight:900;margin-top:12px'>{escape(str(training_state))}</div></div>
</div>{training_metrics}{training_action}</div>
<div class='card' id='ai-quality'><h3>اندازه‌گیری خطای حدس‌های پلاک</h3>
<p class='muted'>حدس کامل مدل به‌صورت تأیید خودکار قابل استفاده است، اما
فقط تأیید یا اصلاح اپراتور حقیقت ارزیابی و نمونهٔ آموزشی محسوب می‌شود.</p>
<div class='stats-grid'>
<div class='stat-card'><span class='muted'>تردد بررسی‌شده</span><div class='stat'>{persian_digits(quality['reviewed'])}</div></div>
<div class='stat-card'><span class='muted'>پوشش خوانش معتبر</span><div class='stat'>{persian_digits(f"{quality['coverage']*100:.1f}")}٪</div></div>
<div class='stat-card'><span class='muted'>کاملاً صحیح</span><div class='stat'>{persian_digits(quality['exact'])}</div></div>
<div class='stat-card'><span class='muted'>دقت کامل سرتاسری</span><div class='stat'>{persian_digits(f"{quality['exact_accuracy']*100:.1f}")}٪</div></div>
<div class='stat-card'><span class='muted'>دقت خوانش‌های پذیرفته‌شده</span><div class='stat'>{persian_digits(f"{quality['accepted_precision']*100:.1f}")}٪</div></div>
<div class='stat-card'><span class='muted'>عدم خوانش</span><div class='stat'>{persian_digits(quality['miss_count'])}</div></div>
<div class='stat-card'><span class='muted'>میانگین خطای کاراکتر سرتاسری</span><div class='stat'>{persian_digits(quality['mean_character_error_end_to_end'])}</div></div>
</div>
<div class='table-wrap'><table><tr><th>نسخه مدل</th><th>تردد بررسی‌شده</th><th>پوشش</th><th>دقت سرتاسری</th><th>خطای کاراکتر سرتاسری</th></tr>{quality_models}</table></div>
</div>
<div class='card' id='software-update'>
<h3>⬆️ به‌روزرسانی نرم‌افزار</h3>
<p><b>نسخه نصب‌شده:</b> <span class='version-chip'>{escape(APP_RELEASE_LABEL)}</span>
<span class='muted code'>{escape(APP_VERSION)}</span></p>
<p class='muted'>فایل ZIP رسمی BC Vision را انتخاب کنید. برنامه هش فایل را
بررسی می‌کند، Updater را با دسترسی مدیر اجرا می‌کند و برای تکمیل عملیات بسته
می‌شود. دیتابیس، دوربین‌ها، تصاویر و مدل‌ها حذف نمی‌شوند.</p>
<form method='post' action='/settings/update' enctype='multipart/form-data'>
<input type='file' name='update_zip' accept='.zip,application/zip' required>
<button>نصب فایل ZIP و راه‌اندازی مجدد</button>
</form></div>
<div class='card'><form method='post' action='/backup'><button class='secondary'>دریافت نسخه پشتیبان دیتابیس</button></form></div><div class='card'><b>وضعیت موتور تصویر:</b> {'آماده' if CV_OK else 'OpenCV بارگذاری نشده است'}</div></div>"""
    return page('تنظیمات',body,u,request)


@app.post('/settings/update', response_class=HTMLResponse)
async def install_update_zip(
    request: Request,
    update_zip: UploadFile = File(...),
):
    u = auth(request)
    if not u:
        return RedirectResponse('/login', 302)
    if not has_permission(request, 'system.manage'):
        return access_denied()
    filename = Path(update_zip.filename or '').name
    if not filename.lower().endswith('.zip'):
        return RedirectResponse(
            '/settings?error=' + quote('فقط فایل ZIP رسمی قابل قبول است.'),
            303,
        )
    if not _UPDATE_UPLOAD_LOCK.acquire(blocking=False):
        return RedirectResponse(
            '/settings?error=' + quote('یک به‌روزرسانی دیگر در حال آماده‌سازی است.'),
            303,
        )
    archive = None
    try:
        _UPDATE_STAGE_ROOT.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix='.upload-',
            suffix='.zip',
            dir=_UPDATE_STAGE_ROOT,
        )
        archive = Path(temporary)
        total = 0
        try:
            with os.fdopen(descriptor, 'wb') as target:
                while True:
                    chunk = await update_zip.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_UPDATE_ZIP_BYTES:
                        raise UpdatePackageError('Update ZIP is too large')
                    target.write(chunk)
                target.flush()
                os.fsync(target.fileno())
            staged = await asyncio.to_thread(
                stage_update_zip,
                archive,
                _UPDATE_STAGE_ROOT,
            )
            validate_update_target(staged, APP_VERSION)
        finally:
            await update_zip.close()
            if archive is not None:
                archive.unlink(missing_ok=True)
        backup_root = _configured_storage_child('backup_path', BACKUP_DIR)
        _mkdir_durable(backup_root)
        backup_file = backup_root / (
            f"bcvision-before-update-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            f"-{secrets.token_hex(4)}.db"
        )
        create_database_backup(backup_file)
        audit(
            request,
            'software_update',
            f'{staged.version_label} SHA256={staged.sha256}',
        )
        await _quiesce_services_for_update()
        try:
            launch_staged_update(staged)
        except BaseException:
            await _resume_services_after_update_abort()
            raise
        threading.Thread(
            target=exit_after_update_launch,
            name='bcvision-update-exit',
            daemon=True,
        ).start()
        return page(
            'به‌روزرسانی نرم‌افزار',
            "<div class='wrap'><div class='card ok'><h2>به‌روزرسانی تأیید شد</h2>"
            "<p>پیام UAC ویندوز را تأیید کنید. برنامه بسته می‌شود و پس از "
            "نصب نسخه جدید دوباره اجرا خواهد شد.</p></div></div>",
            u,
            request,
        )
    except (OSError, StoragePolicyError, UpdatePackageError) as exc:
        return RedirectResponse(
            '/settings?error=' + quote(f'فایل آپدیت رد شد: {exc}'),
            303,
        )
    finally:
        _UPDATE_UPLOAD_LOCK.release()


async def _resume_services_after_update_abort():
    """Restore live processing when an updater could not be launched."""
    from app.ai.live_worker import start_live_anpr_worker

    await asyncio.to_thread(start_live_anpr_worker)
    await asyncio.to_thread(manager.start_enabled_cameras)


async def _quiesce_services_for_update():
    """Stop every camera and ANPR producer before setup touches runtime files."""
    from app.ai.live_worker import shutdown_live_anpr_worker

    stop_attempted = False
    try:
        stop_attempted = True
        if await asyncio.to_thread(manager.stop_all) is not True:
            raise UpdatePackageError(
                'یک یا چند جریان دوربین کامل متوقف نشد؛ آپدیت لغو شد.'
            )
        if not await asyncio.to_thread(
            shutdown_live_anpr_worker,
            retry_timeout=5.0,
        ):
            raise UpdatePackageError(
                'سرویس پلاک‌خوان کامل متوقف نشد؛ آپدیت لغو شد.'
            )
        await asyncio.to_thread(require_media_writes_quiescent)
    except BaseException:
        if stop_attempted:
            await _resume_services_after_update_abort()
        raise

@app.post('/settings/display')
def save_display_settings(request:Request,dashboard_grid:int=Form(2),dashboard_event_rows:int=Form(12),live_fps:int=Form(5),stream_width:int=Form(640),jpeg_quality:int=Form(70),new_password:str=Form('')):
    u=auth(request)
    if not u:return RedirectResponse('/login',302)
    if not has_permission(request,'system.manage'):return access_denied()
    with connect() as con:
        account=con.execute(
            'SELECT must_change_password FROM users WHERE username=?',
            (u,),
        ).fetchone()
    password_required=bool(
        account and int(account['must_change_password'] or 0)
    )
    if password_required and not new_password:
        return RedirectResponse(
            '/settings?password_required=1&error='
            + quote('تعویض رمز پیش‌فرض الزامی است.'),
            303,
        )
    if new_password and len(new_password) < 8:
        return RedirectResponse(
            '/settings?password_required='
            + ('1' if password_required else '0')
            + '&error='
            + quote('رمز جدید باید حداقل ۸ کاراکتر باشد.'),
            303,
        )
    set_setting('dashboard_grid',max(1,min(4,dashboard_grid)));set_setting('dashboard_event_rows',max(6,min(50,dashboard_event_rows)));set_setting('live_fps',max(1,min(15,live_fps)));set_setting('stream_width',stream_width);set_setting('jpeg_quality',max(30,min(95,jpeg_quality)))
    new_session_version=None
    if new_password:
        session=read_session(request)
        if not session or session[0]!=u:
            response=RedirectResponse('/login',303)
            response.delete_cookie(COOKIE_NAME,path='/')
            return response
        with connect() as con:
            con.execute('BEGIN IMMEDIATE')
            result=con.execute(
                'UPDATE users SET password_hash=?,must_change_password=0,'
                'session_version=session_version+1 '
                'WHERE username=? AND is_active=1 AND session_version=?',
                (hash_password(new_password),u,session[1]),
            )
            if result.rowcount!=1:
                response=RedirectResponse('/login',303)
                response.delete_cookie(COOKIE_NAME,path='/')
                return response
            new_session_version=session[1]+1
    for cid in list(manager.streams): manager.remove(cid)
    response=RedirectResponse('/settings?saved=1',303)
    if new_session_version is not None:
        response.set_cookie(
            COOKIE_NAME,
            create_token(u,session_version=new_session_version),
            httponly=True,
            samesite='lax',
            secure=False,
            max_age=43200,
            path='/',
        )
    return response

@app.post('/settings/storage')
def save_storage_settings(request:Request,storage_root:str=Form(...),snapshot_path:str=Form(...),plate_path:str=Form(...),video_path:str=Form(...),backup_path:str=Form(...),save_snapshots:str|None=Form(None),save_plate_images:str|None=Form(None),save_videos:str|None=Form(None),max_storage_gb:int=Form(0),storage_full_action:str=Form('delete_oldest'),retention_snapshots_days:int=Form(90),retention_plates_days:int=Form(90),retention_videos_days:int=Form(7),retention_events_days:int=Form(0)):
    if not auth(request):return RedirectResponse('/login',302)
    if not has_permission(request,'system.manage'):return access_denied()
    try:
        paths=_storage_paths(storage_root,snapshot_path,plate_path,video_path,backup_path)
        old_root=Path(get_setting('storage_root',str(DATA_DIR))).resolve(); new_root=paths[0]; restart=old_root!=new_root
        old_media=[
            Path(get_setting('snapshot_path',str(SNAPSHOT_DIR))).resolve(),
            Path(get_setting('plate_path',str(PLATE_DIR))).resolve(),
            Path(get_setting('video_path',str(VIDEO_DIR))).resolve(),
        ]
        history=_media_roots_history(strict=True)
        for old_path,new_path in zip(old_media,paths[1:4]):
            if (
                old_path != new_path
                and old_path != Path(old_path.anchor)
                and old_path not in history
            ):
                history.append(old_path)
        try:
            validate_storage_layout(
                new_root,
                paths[1:4],
                paths[4],
                history_roots=history,
            )
        except StoragePolicyError as exc:
            raise ValueError(
                f'چیدمان مسیرها با سابقه رسانه ناسازگار است: {exc}'
            ) from exc
        # Persist every newly-created ancestor, not only the final child. A
        # durable config pointer must never reference a root whose parent
        # directory entry could still disappear after power loss.
        for x in paths: _mkdir_durable(x)
        values={'storage_root':new_root,'snapshot_path':paths[1],'plate_path':paths[2],'video_path':paths[3],'backup_path':paths[4],'media_roots_history':json.dumps([str(root) for root in history],ensure_ascii=False),'save_snapshots':'1' if save_snapshots else '0','save_plate_images':'1' if save_plate_images else '0','save_videos':'1' if save_videos else '0','max_storage_gb':max(0,max_storage_gb),'storage_full_action':storage_full_action if storage_full_action in {'delete_oldest','stop','alert'} else 'delete_oldest','retention_snapshots_days':max(0,retention_snapshots_days),'retention_plates_days':max(0,retention_plates_days),'retention_videos_days':max(0,retention_videos_days),'retention_events_days':max(0,retention_events_days)}
        if restart:
            with connect() as con:
                active_training=con.execute(
                    "SELECT id,status FROM anpr_training_runs "
                    "WHERE status IN ('queued','running') "
                    "ORDER BY id DESC LIMIT 1"
                ).fetchone()
            if active_training:
                raise ValueError(
                    'تا پایان آموزش فعال، تغییر مسیر اصلی ذخیره‌سازی مجاز نیست.'
                )
            database_target=new_root/'bcvision.db'
            outbox_target=new_root/'bcvision-retry.db'
            primary_database_names=[
                'bcvision.db','bcvision.db-wal','bcvision.db-shm',
                'bcvision.db-journal',
            ]
            retry_database_names=[
                'bcvision-retry.db','bcvision-retry.db-wal',
                'bcvision-retry.db-shm','bcvision-retry.db-journal',
            ]
            persistent_names=[
                *primary_database_names,'.secret',*retry_database_names,
            ]
            migration_marker_path=STORAGE_CONFIG_PATH.with_name(
                STORAGE_MIGRATION_MARKER_NAME,
            )
            if any(_lstat(new_root/name) is not None for name in persistent_names):
                raise ValueError('مسیر جدید از قبل دارای اطلاعات BC Vision است؛ یک پوشه خالی انتخاب کنید.')
            # A marker collision must be diagnosed before producers stop or
            # any destination snapshot is created. Never replace a foreign
            # bootstrap entry after the storage pointer has been published.
            if _lstat(migration_marker_path) is not None:
                ensure_storage_migration_marker(
                    migration_marker_path,
                )
            from app.ai.live_worker import (
                backup_live_anpr_outbox,
                shutdown_live_anpr_worker,
                start_live_anpr_worker,
            )
            migration_committed=False
            created_targets={}
            staged_targets={}
            staging_root=None
            staging_identity=None
            producers_quiesced=False
            try:
                # Stop producers before either SQLite snapshot.  Otherwise an
                # event could commit between the primary DB and outbox copies,
                # leaving neither destination snapshot responsible for it.
                if manager.stop_all() is not True:
                    raise RuntimeError(
                        'یک یا چند جریان دوربین به‌طور کامل متوقف نشد؛ '
                        'مهاجرت ذخیره‌سازی لغو شد.'
                    )
                if not shutdown_live_anpr_worker(retry_timeout=5.0):
                    raise RuntimeError(
                        'صف ثبت پایدار پلاک‌خوان متوقف نشد؛ '
                        'مهاجرت ذخیره‌سازی لغو شد.'
                    )
                require_media_writes_quiescent()
                producers_quiesced=True
                # A filesystem actor can create an artifact while producers
                # are stopping. Recheck with lstat so even a dangling symlink
                # is treated as occupied and is never followed or overwritten.
                if any(
                    _lstat(new_root/name) is not None
                    for name in persistent_names
                ):
                    raise RuntimeError(
                        'مسیر جدید هنگام توقف سرویس‌ها تغییر کرد؛ '
                        'هیچ فایل موجودی بازنویسی نشد.'
                    )
                # Resolve/verify any crash-left quota transaction under the
                # old root before publishing a pointer to the new root. Use
                # every actual same-root current/history media directory so
                # journal originals can be validated and restored.
                recovery_roots=[]
                for candidate in [*old_media,*history]:
                    if (
                        candidate != old_root
                        and candidate.is_relative_to(old_root)
                        and candidate not in recovery_roots
                    ):
                        recovery_roots.append(candidate)
                if not recovery_roots:
                    raise RuntimeError(
                        'هیچ مسیر رسانه معتبری برای بازیابی ریشه قبلی وجود ندارد.'
                    )
                recovery_status=storage_status(
                    force=True,
                    storage_root=old_root,
                    media_roots=recovery_roots,
                    limit_bytes=0,
                    action='stop',
                )
                if not recovery_status.usage_complete:
                    raise RuntimeError(
                        'بازیابی تراکنش ذخیره‌سازی قبلی کامل نشد؛ '
                        'مهاجرت لغو شد.'
                    )
                staging_root=Path(tempfile.mkdtemp(
                    prefix='.bcvision-migration-',dir=new_root,
                ))
                os.chmod(staging_root,0o700)
                staging_details=staging_root.lstat()
                if not stat.S_ISDIR(staging_details.st_mode):
                    raise RuntimeError(
                        'پوشه موقت مهاجرت قابل اعتبارسنجی نیست.'
                    )
                staging_identity=path_file_identity(
                    staging_root,details=staging_details,
                )
                fsync_parent_directory(staging_root)

                staged_database=staging_root/database_target.name
                create_database_backup(staged_database)
                staged_targets[staged_database]=_regular_file_identity(
                    staged_database,links=1,
                )
                set_settings_for_database(staged_database,values)
                _regular_file_identity(
                    staged_database,
                    expected=staged_targets[staged_database],
                    links=1,
                )
                _fsync_file(staged_database)
                fsync_parent_directory(staged_database)

                staged_outbox=staging_root/outbox_target.name
                backup_live_anpr_outbox(staged_outbox)
                staged_targets[staged_outbox]=_regular_file_identity(
                    staged_outbox,links=1,
                )
                _fsync_file(staged_outbox)
                fsync_parent_directory(staged_outbox)
                secret_source=DATA_DIR/'.secret'
                staged_secret=staging_root/'.secret'
                if _lstat(secret_source) is not None:
                    staged_targets[staged_secret]=_copy_private_regular_file(
                        secret_source,staged_secret,
                    )

                expected_stage_names={path.name for path in staged_targets}
                unexpected_stage_entries=[
                    child for child in staging_root.iterdir()
                    if child.name not in expected_stage_names
                ]
                if unexpected_stage_entries:
                    raise RuntimeError(
                        'خروجی موقت ناشناخته در snapshot مهاجرت ایجاد شد؛ '
                        'انتشار لغو شد.'
                    )
                # Publish via hard links, which are atomic and no-clobber.
                # Staging lives below new_root, so every link is same-volume.
                publish_pairs=[
                    (staged_database,database_target),
                    (staged_outbox,outbox_target),
                ]
                if staged_secret in staged_targets:
                    publish_pairs.append((staged_secret,new_root/'.secret'))
                if any(
                    _lstat(new_root/name) is not None
                    for name in persistent_names
                ):
                    raise RuntimeError(
                        'مسیر مقصد پیش از انتشار snapshot تغییر کرد.'
                    )
                for staged,target in publish_pairs:
                    identity=staged_targets[staged]
                    published_identity=_publish_staged_file(
                        staged,target,identity,
                    )
                    staged_targets.pop(staged,None)
                    created_targets[target]=published_identity
                _rmdir_owned(staging_root,staging_identity)
                staging_root=None
                staging_identity=None

                config_temp=STORAGE_CONFIG_PATH.with_name(
                    f'.{STORAGE_CONFIG_PATH.name}.{secrets.token_hex(8)}.tmp'
                )
                config_temp_identity=None
                try:
                    with config_temp.open('x',encoding='utf-8') as handle:
                        opened=os.fstat(handle.fileno())
                        config_temp_identity=descriptor_file_identity(
                            handle.fileno(),details=opened,
                        )
                        os.fchmod(handle.fileno(),0o600)
                        handle.write(json.dumps({'storage_root':str(new_root)},ensure_ascii=False,indent=2))
                        handle.flush()
                        os.fsync(handle.fileno())
                    _regular_file_identity(
                        config_temp,
                        expected=config_temp_identity,
                        links=1,
                    )
                    config_temp.replace(STORAGE_CONFIG_PATH)
                    # From the instant the pointer is replaced, the new state
                    # must be preserved even if its directory fsync reports an
                    # error. Deleting it here could leave a persisted pointer
                    # aimed at an empty root after power loss.
                    migration_committed=True
                    _STORAGE_RESTART_REQUIRED.set()
                    ensure_storage_migration_marker(
                        migration_marker_path,
                    )
                    fsync_parent_directory(STORAGE_CONFIG_PATH)
                finally:
                    if config_temp_identity is not None:
                        _unlink_owned_regular(
                            config_temp,config_temp_identity,
                        )
            except Exception as migration_error:
                # replace()/MoveFileEx can report an error after the atomic
                # rename actually happened. Re-read the durable pointer before
                # deciding that published files are unowned rollback debris.
                if (
                    not migration_committed
                    and _storage_pointer_targets_root(
                        STORAGE_CONFIG_PATH,new_root,
                    )
                ):
                    migration_committed=True
                    _STORAGE_RESTART_REQUIRED.set()
                    try:
                        ensure_storage_migration_marker(
                            migration_marker_path,
                        )
                    except Exception as marker_error:
                        migration_error.add_note(
                            'storage migration marker error: '
                            + str(marker_error)
                        )
                if not migration_committed:
                    rollback_errors=[]
                    for target,identity in reversed(
                        tuple(created_targets.items())
                    ):
                        try:
                            _unlink_owned_regular(target,identity)
                        except Exception as cleanup_error:
                            rollback_errors.append(
                                f'cleanup {target}: {cleanup_error}'
                            )
                    for target,identity in reversed(
                        tuple(staged_targets.items())
                    ):
                        try:
                            _unlink_owned_regular(target,identity)
                        except Exception as cleanup_error:
                            rollback_errors.append(
                                f'cleanup {target}: {cleanup_error}'
                            )
                    if staging_root is not None and staging_identity is not None:
                        try:
                            _rmdir_owned(staging_root,staging_identity)
                        except Exception as cleanup_error:
                            rollback_errors.append(
                                f'cleanup {staging_root}: {cleanup_error}'
                            )
                    # Restart only after every old producer and reservation
                    # proved quiescent. A timed-out retry thread may still own
                    # the outbox/DB; starting a replacement would create two
                    # concurrent owners and lose retry lifecycle visibility.
                    if producers_quiesced:
                        try:
                            start_live_anpr_worker()
                        except Exception as worker_error:
                            rollback_errors.append(
                                f'worker restart: {worker_error}'
                            )
                        try:
                            manager.start_enabled_cameras()
                        except Exception as camera_error:
                            rollback_errors.append(
                                f'camera restart: {camera_error}'
                            )
                    else:
                        _STORAGE_RESTART_REQUIRED.set()
                    if rollback_errors:
                        raise RuntimeError(
                            f'{migration_error}; rollback incomplete: '
                            + '; '.join(rollback_errors)
                        ) from migration_error
                raise
        else:
            # One transaction keeps current paths, historical evidence roots,
            # quota and retention settings mutually consistent after a crash
            # or an injected write failure. All storage-setting saves quiesce
            # background writers: even a quota-only cache invalidation could
            # otherwise rescan a partial upload while it is still growing.
            from app.ai.live_worker import (
                shutdown_live_anpr_worker,
                start_live_anpr_worker,
            )
            producers_quiesced=False
            try:
                if manager.stop_all() is not True:
                    raise RuntimeError(
                        'یک یا چند جریان دوربین به‌طور کامل متوقف نشد؛ '
                        'ذخیره تنظیمات رسانه لغو شد.'
                    )
                if not shutdown_live_anpr_worker(retry_timeout=5.0):
                    raise RuntimeError(
                        'صف ثبت پایدار پلاک‌خوان متوقف نشد؛ '
                        'ذخیره تنظیمات رسانه لغو شد.'
                    )
                require_media_writes_quiescent()
                producers_quiesced=True
                set_settings_for_database(
                    DB_PATH,values,checkpoint_wal=False,
                )
                invalidate_storage_cache()
                run_retention_cleanup()
            finally:
                if producers_quiesced:
                    try:
                        start_live_anpr_worker()
                    finally:
                        manager.start_enabled_cameras()
                else:
                    _STORAGE_RESTART_REQUIRED.set()
        return RedirectResponse('/settings?saved=1'+('&restart=1' if restart else '')+'#storage',303)
    except Exception as e:
        return RedirectResponse('/settings?error='+quote(str(e))+'#storage',303)

@app.get('/api/storage/status')
def api_storage_status(request:Request):
    if not auth(request): return JSONResponse({'error':'unauthorized'},status_code=401)
    root=get_setting('storage_root',str(DATA_DIR)); result=_path_usage(root)
    result.update({'max_storage_gb':_safe_int(get_setting('max_storage_gb','0')),'action':get_setting('storage_full_action','delete_oldest')})
    try:
        managed=storage_status(force=True)
        managed_payload=managed.as_dict()
        policy_error=managed_payload.pop('error','')
        result.update(managed_payload)
        result['policy_error']=policy_error
        result['managed_percent']=(
            round(managed.managed_bytes/managed.limit_bytes*100,1)
            if managed.limit_bytes else 0
        )
    except StoragePolicyError as exc:
        result.update({
            'over_limit':None,
            'write_blocked':True,
            'usage_complete':False,
            'managed_bytes':0,
            'reserved_bytes':0,
            'managed_percent':0,
            'policy_error':str(exc),
        })
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


@app.get('/events/export.csv')
def export_events(request:Request):
    if not auth(request):return RedirectResponse('/login',302)
    filename=f"events-{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    return StreamingResponse(
        iter_event_csv(connect, jalali_datetime),
        media_type='text/csv',
        headers={
            'Content-Disposition':f'attachment; filename="{filename}"',
            'Cache-Control':'no-store',
        },
    )

@app.post('/settings/ai')
def save_ai_settings(request:Request, ai_accelerator:str=Form('auto'), ai_quality:str=Form('balanced'), ai_confidence:int=Form(85), ai_frames:int=Form(5), anpr_auto_confirm_guesses:str|None=Form(None), anpr_detector_model:str=Form('yolo11n'), anpr_engine_v2_shadow:str|None=Form(None)):
    u=auth(request)
    if not u:return RedirectResponse('/login',302)
    if not has_permission(request,'system.manage'):return access_denied()
    detector_model=str(anpr_detector_model or '').strip().lower()
    if detector_model not in {'yolo11n','yolov8n','yolox'}:
        return RedirectResponse(
            '/settings?error='+quote('مدل تشخیص پلاک معتبر نیست.'),
            303,
        )
    if detector_model == 'yolox':
        from app.ai.model_manager import model_status
        yolox_status = model_status(selected_detector='yolox')
        if not yolox_status.get('detector_yolox_ready'):
            return RedirectResponse(
                '/settings?error='+quote(
                    'مدل YOLOX یا manifest هش‌شده آن هنوز نصب و تأیید نشده است.'
                ),
                303,
            )
    detector_changed=(
        get_setting('anpr_detector_model','yolo11n') != detector_model
    )
    if detector_changed:
        from app.ai.live_worker import switch_live_anpr_detector
        try:
            switch_live_anpr_detector(
                detector_model,
                persist_setting=set_setting,
            )
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            return RedirectResponse(
                '/settings?error='+quote(
                    'تغییر مدل انجام نشد: '+str(exc)
                ),
                303,
            )
    set_setting('ai_accelerator', ai_accelerator)
    set_setting('ai_quality', ai_quality)
    set_setting('ai_confidence', max(1,min(99,ai_confidence)))
    set_setting('ai_frames', max(1,min(20,ai_frames)))
    set_setting(
        'anpr_auto_confirm_guesses',
        '1' if anpr_auto_confirm_guesses else '0',
    )
    shadow_enabled=bool(anpr_engine_v2_shadow)
    set_setting(
        'anpr_engine_v2_shadow',
        '1' if shadow_enabled else '0',
    )
    if detector_changed:
        audit(
            request,
            'anpr_detector_switch',
            detector_model+'; execution=exclusive-baseline',
        )
    from app.ai.live_worker import configure_live_engine_v2_shadow
    configure_live_engine_v2_shadow(shadow_enabled)
    audit(
        request,
        'anpr_engine_v2_shadow',
        (
            ('enabled' if shadow_enabled else 'disabled')
            + '; persistence=false'
            + '; mode=shadow-v2; primary=baseline'
            + '; detector=' + detector_model
        ),
    )
    return RedirectResponse('/settings?saved=1',302)


@app.post('/settings/ai/training/start')
def start_ai_training(
    request: Request,
    epochs: int = Form(12),
    rights_attested: str | None = Form(None),
):
    username = auth(request)
    if not username:
        return RedirectResponse('/login', 302)
    if not has_permission(request, 'system.manage'):
        return access_denied()
    if rights_attested != '1':
        return RedirectResponse(
            '/settings?error='
            + quote(
                'برای آموزش، مالکیت یا مجوز صریح استفاده و توزیع '
                'داده‌ها باید تأیید شود.'
            )
            + '#ai-training',
            303,
        )
    try:
        result = start_training(
            device=get_setting('ai_accelerator', 'auto'),
            epochs=epochs,
            rights_attested=True,
            attested_by=username,
        )
        audit(
            request,
            'anpr_training_start',
            f"run={result['run_id']}; epochs={max(4,min(40,epochs))}; "
            f"rights_attested_by={username}",
        )
        return RedirectResponse('/settings?saved=1#ai-training', 303)
    except ValueError as exc:
        return RedirectResponse(
            '/settings?error=' + quote(str(exc)) + '#ai-training',
            303,
        )


@app.post('/settings/ai/training/apply')
def apply_ai_training(
    request: Request,
    run_id: int = Form(...),
):
    username = auth(request)
    if not username:
        return RedirectResponse('/login', 302)
    if not has_permission(request, 'system.manage'):
        return access_denied()
    try:
        result = apply_candidate(run_id, username)
        audit(
            request,
            'anpr_training_apply',
            f"run={run_id}; sha256={result['sha256']}",
        )
        return RedirectResponse('/settings?saved=1#ai-training', 303)
    except ValueError as exc:
        return RedirectResponse(
            '/settings?error=' + quote(str(exc)) + '#ai-training',
            303,
        )


@app.post('/settings/ai/training/evaluate')
def evaluate_ai_training(
    request: Request,
    run_id: int = Form(...),
):
    username = auth(request)
    if not username:
        return RedirectResponse('/login', 302)
    if not has_permission(request, 'system.manage'):
        return access_denied()
    try:
        result = evaluate_candidate_on_golden(run_id)
        audit(
            request,
            'anpr_training_golden_evaluate',
            f"run={run_id}; status={result['status']}",
        )
        return RedirectResponse('/settings?saved=1#ai-training', 303)
    except ValueError as exc:
        return RedirectResponse(
            '/settings?error=' + quote(str(exc)) + '#ai-training',
            303,
        )


@app.post('/backup')
def backup_database(request:Request):
    if not auth(request):return RedirectResponse('/login',302)
    if not has_permission(request,'system.manage'):return access_denied()
    out=_configured_storage_child('backup_path',BACKUP_DIR);_mkdir_durable(out);out=out / f"bcvision-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}-{secrets.token_hex(4)}.db"
    create_database_backup(out)
    return FileResponse(out,media_type='application/octet-stream',filename=out.name)


@app.post('/cameras/video-upload', response_class=HTMLResponse)
async def cameras_video_upload(request: Request, camera_id: int = Form(0), video: UploadFile = File(...)):
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
    if camera_id:
        with connect() as con:
            source_camera=con.execute(
                "SELECT * FROM cameras WHERE id=? AND rtsp_url NOT LIKE 'video://%'",
                (camera_id,),
            ).fetchone()
        if not source_camera:
            return upload_error('دوربین انتخاب‌شده پیدا نشد.')
    else:
        source_camera={
            'name':'پیش‌فرض','city':'','lpr_confidence':60,
            'frame_step':5,'duplicate_seconds':30,
            'roi_x':0,'roi_y':0,'roi_w':100,'roi_h':100,'line_y':50,
        }
    try:
        pending_upload=await _stage_video_upload(
            video,
            _configured_storage_child('video_path',VIDEO_DIR),
            suffix,
            acceptance_required=True,
        )
        target=pending_upload.target
    except ValueError as e:
        return upload_error(e)

    virtual_camera_id=None
    camera_owner_committed=False
    _VIDEO_CAMERA_HANDOFF_LOCK.acquire()
    try:
        from app.ai.video_test import VideoTester
        tester=VideoTester(target)
        try:
            info=tester.info()
        finally:
            tester.close()
        display_name=(Path(video.filename or target.name).stem or 'ویدئو')[:80]
        with connect() as con:
            from app.media_acceptance import require_full_synchronous

            require_full_synchronous(con)
            cursor=con.execute(
                "INSERT INTO cameras("
                "name,rtsp_url,location,city,enabled,is_demo,sort_order,"
                "lpr_enabled,lpr_confidence,frame_step,duplicate_seconds,"
                "roi_x,roi_y,roi_w,roi_h,line_y"
                ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    f"ویدئو: {display_name}",
                    f"video://{target}",
                    f"فایل آپلودی با تنظیمات {source_camera['name']}",
                    source_camera['city'] or '',
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
            pending_upload.accept(
                con,
                owner_kind='virtual-camera',
                owner_id=virtual_camera_id,
            )
        camera_owner_committed=True
        manager.get(
            virtual_camera_id,
            f"video://{target}",
            f"ویدئو: {display_name}",
            int(get_setting('stream_width','640')),
            int(get_setting('live_fps','5')),
            int(get_setting('jpeg_quality','70')),
        )
        # The new camera row and its active decoder now durably own the file.
        # Only at this acceptance boundary may quota eviction be committed.
        pending_upload.commit()
        dashboard_redirect=(
            f'/dashboard?video=1&events_camera={virtual_camera_id}'
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
                'redirect':dashboard_redirect,
            })
        return RedirectResponse(dashboard_redirect,303)
    except Exception as e:
        if virtual_camera_id is not None and camera_owner_committed:
            stream_stopped=False
            try:
                stream_stopped=manager.remove(virtual_camera_id) is True
            except Exception:
                stream_stopped=False
            if not stream_stopped:
                # The decoder may still have the source open. Preserve both
                # its DB owner and file rather than unlinking under a live
                # worker after a stop timeout.
                pending_upload.commit()
                return upload_error(
                    'خطا در آماده‌سازی ویدئو؛ جریان فعال متوقف نشد و '
                    'فایل برای بازیابی حفظ شد.',500
                )
            try:
                with connect() as con:
                    from app.media_acceptance import require_full_synchronous

                    require_full_synchronous(con)
                    con.execute(
                        "DELETE FROM cameras WHERE id=?",
                        (virtual_camera_id,),
                    )
                    pending_upload.release_owner(con)
            except Exception:
                # Resolve an uncertain delete using the same SQLite oracle:
                # keep a surviving owner, otherwise roll the staged file back.
                pending_upload.settle_after_owner_attempt()
                return upload_error(
                    f'خطا در آماده‌سازی ویدئو: {e}',500
                )
        pending_upload.settle_after_owner_attempt()
        return upload_error(f'خطا در آماده‌سازی ویدئو: {e}',500)
    finally:
        _VIDEO_CAMERA_HANDOFF_LOCK.release()
        pending_upload.close_pin()

# ---------- Video AI Test Upload ----------
def _archive_video_test_events(
    events,
    video_path,
    display_name,
    *,
    source_upload=None,
):
    archived=[]
    pending_media = tuple(
        pending
        for raw in events
        for pending in tuple(raw.get('_pending_media') or ())
    )
    uncertain_media = pending_media + (
        (source_upload,) if source_upload is not None else ()
    )
    try:
        with connect() as con:
            if pending_media or source_upload is not None:
                from app.media_acceptance import require_full_synchronous

                require_full_synchronous(con)
            columns={
                row[1]
                for row in con.execute(
                    "PRAGMA table_info(plate_events)"
                ).fetchall()
            }
            for raw in events:
                event=dict(raw)
                candidate_norm=(
                    normalize_plate(event.get('plate_norm'))
                    or normalize_plate(event.get('plate'))
                )
                candidate_parts=split_iran_plate(candidate_norm)
                recognized=bool(
                    candidate_parts and (
                        event.get('valid')
                        or event.get('auto_confirmed')
                    )
                    and not event.get('unreadable_final')
                )
                suggested=bool(event.get('needs_review'))
                if recognized:
                    plate_text=str(event.get('plate') or candidate_norm)
                    normalized=candidate_norm
                elif suggested:
                    plate_text=str(
                        event.get('raw_guess_text')
                        or event.get('plate')
                        or 'ناخوانا'
                    )
                    normalized=normalize_plate(
                        event.get('raw_guess_norm') or plate_text
                    )
                else:
                    plate_text='ناخوانا'
                    normalized=''
                parts=split_iran_plate(normalized)
                plate_norm=normalized if recognized else ''
                review_status=(
                    'auto-confirmed'
                    if recognized and event.get('auto_confirmed')
                    else (
                        'confirmed-ai'
                        if recognized
                        else ('suggested' if suggested else 'unreadable')
                    )
                )
                values={
                    'plate_text':plate_text,
                    'plate_norm':plate_norm,
                    'plate_region':parts['region'] if parts else '',
                    'confidence':max(
                        0.0,min(1.0,float(event.get('confidence') or 0.0))
                    ),
                    'camera_id':None,
                    'camera_name':f"تست ویدئو: {display_name}"[:120],
                    'city':'',
                    'image_path':str(event.get('image_path') or ''),
                    'plate_image_path':str(event.get('plate_path') or ''),
                    'media_status':str(event.get('media_status') or 'missing'),
                    'media_error':str(event.get('media_error') or '')[:1000],
                    'video_path':str(video_path),
                    'video_second':max(
                        0.0,float(event.get('video_second') or 0.0)
                    ),
                    'detector_method':str(event.get('method') or 'video-test'),
                    'ocr_confidence':max(
                        0.0,min(1.0,float(event.get('ocr_confidence') or 0.0))
                    ),
                    'ocr_engine':str(event.get('ocr_engine') or ''),
                    'ocr_alternative':str(event.get('ocr_alternative') or ''),
                    'ocr_disagreement':int(bool(event.get('ocr_disagreement'))),
                    'vehicle_type':str(event.get('vehicle_type') or 'نامشخص'),
                    'vehicle_color':str(event.get('vehicle_color') or 'نامشخص'),
                    'vehicle_brand':str(event.get('vehicle_brand') or 'نامشخص'),
                    'vehicle_confidence':max(
                        0.0,min(
                            1.0,float(event.get('vehicle_confidence') or 0.0)
                        )
                    ),
                    'direction':str(event.get('direction') or 'stationary'),
                    'quality_score':max(
                        0.0,min(1.0,float(event.get('quality_score') or 0.0))
                    ),
                    'consensus_votes':max(
                        0,int(event.get('consensus_votes') or 0)
                    ),
                    'source':'video-test',
                    'processing_ms':max(
                        0.0,float(event.get('processing_ms') or 0.0)
                    ),
                    'review_status':review_status,
                    'confirmation_source':str(
                        event.get('confirmation_source') or 'video-test'
                    ),
                    'operator_reviewed':0,
                    'raw_guess_text':str(
                        event.get('raw_guess_text') or plate_text
                    ),
                    'raw_guess_norm':normalize_plate(
                        event.get('raw_guess_norm')
                        or event.get('raw_guess_text')
                        or plate_norm
                    ),
                    'raw_guess_confidence':max(
                        0.0,min(
                            1.0,
                            float(
                                event.get('raw_guess_confidence')
                                or event.get('ocr_confidence')
                                or 0.0
                            ),
                        )
                    ),
                    'raw_guess_engine':str(
                        event.get('raw_guess_engine')
                        or event.get('ocr_engine')
                        or ''
                    ),
                    'raw_guess_reason':str(
                        event.get('raw_guess_reason') or ''
                    ),
                    'model_revision':str(
                        event.get('model_revision')
                        or event.get('ocr_engine')
                        or ''
                    ),
                    'experimental':int(bool(
                        event.get('experimental')
                        or event.get('needs_review')
                    )),
                }
                selected=[key for key in values if key in columns]
                placeholders=','.join('?' for _ in selected)
                cursor=con.execute(
                    f"INSERT INTO plate_events({','.join(selected)}) "
                    f"VALUES({placeholders})",
                    tuple(values[key] for key in selected),
                )
                event['event_id']=int(cursor.lastrowid)
                for pending in tuple(event.get('_pending_media') or ()):
                    pending.accept(
                        con,
                        owner_kind='plate-event',
                        owner_id=event['event_id'],
                    )
                event.pop('_pending_media',None)
                archived.append(event)
            if source_upload is not None:
                source_upload.accept(
                    con,
                    owner_kind='video-test-run',
                    owner_id=Path(video_path).stem,
                )
    except BaseException:
        from app.media_storage import settle_pending_media

        settle_pending_media(uncertain_media)
        raise
    from app.media_storage import finalize_pending_media

    finalize_pending_media(pending_media)
    return archived


def _persist_and_archive_isolated_video_result(
    transport_result,
    plate_dir,
    snapshot_dir,
    video_path,
    display_name,
    source_upload,
):
    """Validate, publish and archive child output in one quiescent job."""

    from app.ai.video_test import (
        persist_transport_event_media,
        restore_process_video_result,
    )

    info, events = restore_process_video_result(
        transport_result,
        max_events=10_000,
        max_media_bytes=VIDEO_TEST_MEDIA_RESULT_BYTES,
    )
    events = persist_transport_event_media(
        events,
        plate_dir,
        snapshot_dir,
    )
    archived = _archive_video_test_events(
        events,
        video_path,
        display_name,
        source_upload=source_upload if events else None,
    )
    return info, archived


def _video_test_result_row(event, index):
    image_path = str(event.get('image_path') or '')
    vehicle_image = (
        f"<a href='/events/{int(event['event_id'])}'>"
        f"<img class='recent-vehicle-thumb' "
        f"src='/media?path={quote(image_path)}' "
        f"alt='تصویر خودرو ردیف {index}'></a>"
        if image_path and Path(image_path).is_file()
        and event.get('event_id')
        else "<span class='recent-media-missing'>بدون تصویر خودرو</span>"
    )
    plate_path = str(event.get('plate_path') or '')
    plate_href=(
        f"/events/{int(event['event_id'])}"
        if event.get('event_id')
        else f"/media?path={quote(plate_path)}"
    )
    plate_image = (
        f"<a href='{plate_href}'>"
        f"<img class='thumb plate-thumb' "
        f"src='/media?path={quote(plate_path)}' "
        f"alt='تصویر پلاک ردیف {index}'></a>"
        if plate_path and Path(plate_path).is_file()
        else "<span class='recent-media-missing' "
        "style='width:130px;height:48px'>بدون تصویر پلاک</span>"
    )
    recognized = bool(
        split_iran_plate(
            normalize_plate(
                event.get('plate_norm') or event.get('plate')
            )
        )
        and (event.get('valid') or event.get('auto_confirmed'))
        and not event.get('unreadable_final')
    )
    plate_text = str(
        (event.get('plate') if recognized else event.get('raw_guess_text'))
        or event.get('plate')
        or 'ناخوانا'
    )
    confidence = max(0.0, min(1.0, float(event.get('confidence') or 0.0)))
    video_second = max(0.0, float(event.get('video_second') or 0.0))
    engine = str(event.get('ocr_engine') or event.get('method') or '—')
    lane = str(event.get('engine_lane') or 'baseline')
    experimental = bool(
        event.get('experimental')
        or lane == 'candidate-shadow'
    )
    if event.get('auto_confirmed'):
        status = (
            "<span class='read-badge auto-confirmed'>"
            "تأیید خودکار مدل؛ قابل اصلاح</span>"
        )
    elif experimental:
        status = (
            "<span class='read-badge suggested'>"
            "حدس خام مدل آزمایشی؛ قطعی نیست</span>"
        )
    elif event.get('needs_review') and plate_text not in {
        '', 'ناخوانا', 'در حال بررسی'
    }:
        status = (
            "<span class='read-badge suggested'>"
            "حدس آزمایشی؛ نیازمند اصلاح</span>"
        )
    elif event.get('valid'):
        status = "<span class='read-badge confirmed'>خوانش قطعی</span>"
    else:
        status = (
            "<span class='read-badge unreadable'>"
            "ناخوانا / بدون حدس معتبر</span>"
        )
    reason = str(event.get('raw_guess_reason') or '')
    diagnostics = (
        f"<small class='muted code'>{escape(reason)}</small>"
        if reason
        else ""
    )
    lane_label = (
        "مدل جدید / Shadow"
        if lane == 'candidate-shadow'
        else "Baseline فعال"
    )
    return (
        f"<tr><td>{persian_digits(index)}</td>"
        f"<td>{vehicle_image}</td>"
        f"<td><div class='recent-plate-result'>{plate_image}"
        f"<div data-recognized-text='{escape(plate_text)}'>"
        f"{iran_plate_html(plate_text, True)}{status}{diagnostics}"
        f"</div></div></td>"
        f"<td>{persian_digits(round(confidence * 100, 1))}٪</td>"
        f"<td>{persian_digits(round(video_second, 2))} ثانیه</td>"
        f"<td><b>{escape(lane_label)}</b><br>"
        f"<span class='code'>{escape(engine)}</span></td></tr>"
    )


@app.get('/ai/video-test', response_class=HTMLResponse)
def ai_video_test_page(request: Request):
    u=auth(request)
    if not u:
        return RedirectResponse('/login',302)
    if not has_permission(request,'video.process'):return access_denied()
    return page('تست ویدئو',"""
    <div class="wrap">
      <div class="toolbar"><h1>تست واقعی پلاک‌خوان با ویدئو</h1></div>
      <div class="card">
        <p class="muted">تمام فریم‌ها پردازش می‌شوند. از هر تردد، بهترین
        تصویر پلاک و متن تشخیص‌داده‌شده در یک ردیف نمایش داده می‌شود.
        فقط مدل انتخاب‌شده در تنظیمات AI اجرا می‌شود و موتورهای Shadow/Next
        در این تست دخالت ندارند. حدس ناقص همچنان آزمایشی می‌ماند و تا تأیید
        یا اصلاح اپراتور، حقیقت آموزشی محسوب نمی‌شود.</p>
        <form action="/ai/video-test/upload" method="post"
              enctype="multipart/form-data">
          <label>فایل ویدئو</label>
          <input type="file" name="video"
                 accept=".mp4,.avi,.mkv,.mov,.m4v" required>
          <button type="submit">شروع پردازش کامل</button>
        </form>
      </div>
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
        pending_upload=await _stage_video_upload(
            video,
            _configured_storage_child('video_path',VIDEO_DIR),
            suffix,
            acceptance_required=True,
        )
        target=pending_upload.target
    except ValueError as e:
        return page('خطای ویدئو',f"<div class='wrap'><div class='alert'>{escape(str(e))}</div></div>",u,request)
    plate_dir=None
    snapshot_dir=None
    process_slot_acquired=False
    try:
        run_name = target.stem
        plate_dir = (
            _configured_storage_child('plate_path', PLATE_DIR)
            / 'video-tests'
            / run_name
        )
        snapshot_dir = (
            _configured_storage_child('snapshot_path', SNAPSHOT_DIR)
            / 'video-tests'
            / run_name
        )
        started = time.perf_counter()
        await _acquire_video_test_process_slot()
        process_slot_acquired=True
        detector_variant=str(
            get_setting('anpr_detector_model','yolo11n') or 'yolo11n'
        )
        transport_result = await run_module_job_subprocess(
            'app.ai.video_test',
            'process_video_transport',
            str(target),
            str(plate_dir),
            str(snapshot_dir),
            frame_step=1,
            max_events=10000,
            min_confidence=0.20,
            duplicate_seconds=2.5,
            include_candidate_shadow=False,
            detector_variant=detector_variant,
            transport_media_bytes=VIDEO_TEST_MEDIA_RESULT_BYTES,
            timeout_seconds=VIDEO_TEST_JOB_TIMEOUT_SECONDS,
            max_result_bytes=VIDEO_TEST_PROCESS_RESULT_BYTES,
        )
        info, events = await run_to_thread_quiescent(
            _persist_and_archive_isolated_video_result,
            transport_result,
            plate_dir,
            snapshot_dir,
            target,
            Path(video.filename or target.name).stem or 'ویدئو',
            pending_upload,
        )
        if events:
            pending_upload.commit()
        else:
            # A zero-detection test has no durable DB owner for its source.
            pending_upload.rollback()
        elapsed = time.perf_counter() - started
        readable = sum(
            1
            for event in events
            if event.get('valid') and not event.get('needs_review')
        )
        auto_confirmed = sum(
            1
            for event in events
            if event.get('auto_confirmed')
        )
        guesses = sum(
            1
            for event in events
            if (
                event.get('experimental')
                or event.get('needs_review')
            )
            and not event.get('auto_confirmed')
            and str(
                event.get('raw_guess_text')
                or event.get('plate')
                or ''
            ) not in {'', 'ناخوانا', 'در حال بررسی'}
        )
        detector_variant=str(
            info.get('detector_variant') or 'yolo11n'
        ).lower()
        detector_label={
            'yolox': 'YOLOX اختصاصی',
            'yolov8n': 'YOLOv8n',
        }.get(detector_variant, 'YOLO11n')
        rows = ''.join(
            _video_test_result_row(event, index)
            for index, event in enumerate(events, start=1)
        ) or (
            "<tr><td colspan='6'>هیچ پلاکی در این ویدئو تشخیص داده نشد."
            "</td></tr>"
        )
        return page('نتیجه تست پلاک‌خوان',f"""
        <div class="wrap">
          <div class="toolbar">
            <h1>نتیجه تست پلاک‌خوان</h1>
            <a class="btn secondary" href="/ai/video-test">تست ویدئوی دیگر</a>
          </div>
          <div class="card">
            <div class="stat-grid">
              <div class="stat"><small>فریم</small>
                <b>{persian_digits(info['frames'])}</b></div>
              <div class="stat"><small>تردد تشخیص‌داده‌شده</small>
                <b>{persian_digits(len(events))}</b></div>
              <div class="stat"><small>خوانش قطعی</small>
                <b>{persian_digits(readable)}</b></div>
              <div class="stat"><small>تأیید خودکار مدل</small>
                <b>{persian_digits(auto_confirmed)}</b></div>
              <div class="stat"><small>حدس آزمایشی</small>
                <b>{persian_digits(guesses)}</b></div>
              <div class="stat"><small>مدل تشخیص انحصاری</small>
                <b>{detector_label}</b></div>
              <div class="stat"><small>زمان پردازش</small>
                <b>{persian_digits(round(elapsed, 2))} ثانیه</b></div>
            </div>
            <p class="muted">فایل: {escape(video.filename or '')} —
            {persian_digits(info['width'])}×{persian_digits(info['height'])}
            در {persian_digits(info['fps'])} FPS</p>
            <p class="muted">این اجرا فقط از {detector_label} در مسیر
            Baseline استفاده کرد؛ موتورهای Shadow/Next اجرا نشدند.</p>
          </div>
          <div class="card">
            <div class="table-wrap"><table>
              <thead><tr><th>ردیف</th>
                <th>تصویر خودرو</th>
                <th>تصویر پلاک / متن تشخیص‌داده‌شده</th>
                <th>اطمینان</th><th>زمان ویدئو</th><th>موتور / مسیر</th>
              </tr></thead>
              <tbody>{rows}</tbody>
            </table></div>
          </div>
        </div>
        """,u,request)
    except asyncio.CancelledError:
        if not pending_upload.committed:
            pending_upload.settle_after_owner_attempt()
        raise
    except SubprocessJobTimeout:
        if not pending_upload.committed:
            pending_upload.settle_after_owner_attempt()
        return page(
            'خطای ویدئو',
            "<div class='wrap'><div class='alert'>"
            'زمان پردازش از حد امن عبور کرد؛ ویدئو را به بخش‌های '
            'کوتاه‌تر تقسیم کنید.</div></div>',
            u,
            request,
        )
    except Exception as e:
        if not pending_upload.committed:
            pending_upload.settle_after_owner_attempt()
        return page('خطای ویدئو',f"<div class='wrap'><div class='alert'>خطا: {escape(str(e))}</div></div>",u,request)
    finally:
        if process_slot_acquired:
            _VIDEO_TEST_PROCESS_SLOT.release()
        pending_upload.close_pin()
