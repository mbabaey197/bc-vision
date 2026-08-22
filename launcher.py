from __future__ import annotations

import os
import json
import multiprocessing
import socket
import sys
import threading
import time
import traceback
import webbrowser
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from runtime_payload import (
    FAILED_MARKER,
    RuntimePayload,
    RuntimePayloadError,
    install_runtime_importer,
    read_runtime_marker,
    recover_pending_activation,
    select_runtime_payload,
    verify_runtime_payload,
)


def app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


BASE = app_dir()
os.chdir(BASE)
RUNTIME_PIN_SOURCE_ENV = "BCVISION_RUNTIME_PIN_SOURCE"
RUNTIME_PIN_VERSION_ENV = "BCVISION_RUNTIME_PIN_VERSION"
RUNTIME_PIN_ABI_ENV = "BCVISION_RUNTIME_PIN_ABI"
RUNTIME_PIN_ROOT_ENV = "BCVISION_RUNTIME_PIN_ROOT"


def _argument_value(name: str) -> str:
    try:
        index = sys.argv.index(name)
    except ValueError:
        return ""
    if index + 1 >= len(sys.argv):
        return ""
    return sys.argv[index + 1]


def validate_runtime_candidate_request() -> None:
    """Reserve explicit payload selection for isolated installer self-tests."""
    if "--runtime-candidate" not in sys.argv:
        return
    requested = _argument_value("--runtime-candidate").strip()
    isolated_data = _argument_value("--self-test-data-dir").strip()
    if not requested:
        raise RuntimePayloadError("--runtime-candidate requires a version")
    if "--self-test" not in sys.argv or not isolated_data:
        raise RuntimePayloadError(
            "--runtime-candidate is allowed only with --self-test and "
            "an explicit --self-test-data-dir",
        )


def configure_self_test_data_dir() -> None:
    """Keep installer preflight checks away from customer data."""
    value = _argument_value("--self-test-data-dir").strip()
    if not value:
        return
    if "--self-test" not in sys.argv:
        raise RuntimeError("--self-test-data-dir requires --self-test")
    os.environ["BCVISION_DATA_DIR"] = str(Path(value).expanduser().resolve())


validate_runtime_candidate_request()
configure_self_test_data_dir()


def _is_multiprocessing_spawn_helper() -> bool:
    """Recognize the frozen worker command that ``freeze_support`` handles."""
    return any(value == "--multiprocessing-fork" for value in sys.argv[1:])


def _publish_runtime_selection(
    payload: RuntimePayload | None,
    *,
    source: str,
    pin_for_children: bool,
) -> None:
    if source not in {"bundled", "external", "candidate"}:
        raise RuntimePayloadError(f"Invalid runtime source: {source!r}")
    os.environ["BCVISION_ACTIVE_RUNTIME_SOURCE"] = source
    if payload is None:
        if source != "bundled":
            raise RuntimePayloadError("External runtime selection is missing")
        os.environ.pop("BCVISION_ACTIVE_RUNTIME_VERSION", None)
        os.environ.pop("BCVISION_ACTIVE_RUNTIME_ABI", None)
    else:
        if source == "bundled":
            raise RuntimePayloadError("Bundled runtime cannot have a payload")
        os.environ["BCVISION_ACTIVE_RUNTIME_VERSION"] = payload.version
        os.environ["BCVISION_ACTIVE_RUNTIME_ABI"] = payload.runtime_abi

    if not pin_for_children:
        return
    os.environ[RUNTIME_PIN_SOURCE_ENV] = source
    if payload is None:
        os.environ.pop(RUNTIME_PIN_VERSION_ENV, None)
        os.environ.pop(RUNTIME_PIN_ABI_ENV, None)
        os.environ.pop(RUNTIME_PIN_ROOT_ENV, None)
    else:
        os.environ[RUNTIME_PIN_VERSION_ENV] = payload.version
        os.environ[RUNTIME_PIN_ABI_ENV] = payload.runtime_abi
        os.environ[RUNTIME_PIN_ROOT_ENV] = str(payload.root)


def _activate_pinned_spawn_runtime() -> RuntimePayload | None:
    """Recreate the parent's exact runtime importer in a frozen child."""
    source = os.environ.get(RUNTIME_PIN_SOURCE_ENV, "").strip()
    if source == "bundled":
        if any(
            os.environ.get(name, "").strip()
            for name in (
                RUNTIME_PIN_VERSION_ENV,
                RUNTIME_PIN_ABI_ENV,
                RUNTIME_PIN_ROOT_ENV,
            )
        ):
            raise RuntimePayloadError("Bundled runtime pin has payload fields")
        _publish_runtime_selection(
            None,
            source="bundled",
            pin_for_children=False,
        )
        return None
    if source not in {"external", "candidate"}:
        raise RuntimePayloadError("Multiprocessing runtime pin is missing")

    version = os.environ.get(RUNTIME_PIN_VERSION_ENV, "").strip()
    runtime_abi = os.environ.get(RUNTIME_PIN_ABI_ENV, "").strip()
    root_value = os.environ.get(RUNTIME_PIN_ROOT_ENV, "").strip()
    if not version or not runtime_abi or not root_value:
        raise RuntimePayloadError("Multiprocessing runtime pin is incomplete")

    expected_root = (BASE / "runtime" / version).resolve()
    pinned_root = Path(root_value).resolve()
    if pinned_root != expected_root:
        raise RuntimePayloadError("Multiprocessing runtime source changed")
    payload = verify_runtime_payload(
        pinned_root,
        expected_version=version,
        expected_abi=runtime_abi,
    )
    install_runtime_importer(payload)
    _publish_runtime_selection(
        payload,
        source=source,
        pin_for_children=False,
    )
    return payload


def activate_runtime_payload() -> RuntimePayload | None:
    """Prefer a verified versioned app payload in packaged installations."""
    if _is_multiprocessing_spawn_helper():
        return _activate_pinned_spawn_runtime()
    enabled = getattr(sys, "frozen", False) or os.environ.get(
        "BCVISION_ENABLE_RUNTIME_PAYLOAD",
        "0",
    ) == "1"
    if not enabled:
        return None
    requested = _argument_value("--runtime-candidate").strip() or None
    if requested is None:
        recovered = recover_pending_activation(BASE)
        if recovered:
            os.environ["BCVISION_RUNTIME_RECOVERED_FROM"] = recovered
    payload = select_runtime_payload(
        BASE,
        requested_version=requested,
    )
    if payload is None:
        if requested is not None:
            raise RuntimePayloadError(
                f"Requested runtime candidate is invalid: {requested}",
            )
        _publish_runtime_selection(
            None,
            source="bundled",
            pin_for_children=True,
        )
        return None
    install_runtime_importer(payload)
    _publish_runtime_selection(
        payload,
        source="candidate" if requested is not None else "external",
        pin_for_children=True,
    )
    failed = read_runtime_marker(BASE, FAILED_MARKER)
    if failed:
        os.environ["BCVISION_RUNTIME_LAST_FAILURE"] = failed
    return payload


ACTIVE_RUNTIME = activate_runtime_payload()

# PyInstaller re-enters this executable for spawn workers. Runtime activation
# must happen first so the worker unpickles its target through the exact same
# verified app importer as its parent. This still precedes every ``app.*``
# import and all application/database/model initialization.
multiprocessing.freeze_support()

from app.cpu_budget import configure_process_cpu_budget

configure_process_cpu_budget()

from app.config import LOG_PATH

LOG = LOG_PATH
LOG.parent.mkdir(parents=True, exist_ok=True)
PANEL_URL = "http://127.0.0.1:8000/login"
HEALTH_URL = "http://127.0.0.1:8000/api/health"
MODEL_PREPARATION_STATE_ENV = "BCVISION_MODEL_PREPARATION_STATE"
MODEL_PREPARATION_ERROR_ENV = "BCVISION_MODEL_PREPARATION_ERROR"
MODEL_PREPARATION_ATTEMPT_ENV = "BCVISION_MODEL_PREPARATION_ATTEMPT"
MODEL_PREPARATION_MAX_ATTEMPTS = 2
MODEL_PREPARATION_RETRY_SECONDS = 4.0
INSTANCE_LOCK_PATH = LOG.parent / "bcvision.instance.lock"


class SingleInstanceLock:
    """Cross-process lock held before any background producer is started."""

    def __init__(self, path=INSTANCE_LOCK_PATH):
        self.path = Path(path)
        self._handle = None

    def acquire(self) -> bool:
        if self._handle is not None:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"\0")
                    handle.flush()
                    os.fsync(handle.fileno())
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(
                    handle.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
        except OSError:
            handle.close()
            return False
        self._handle = handle
        return True

    def close(self) -> None:
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    def __enter__(self):
        if not self.acquire():
            raise RuntimeError("BC Vision instance lock is already held")
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        self.close()
        return False


def log(message):
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(
            time.strftime("[%Y-%m-%d %H:%M:%S] ")
            + str(message)
            + "\n"
        )


def _publish_model_preparation_status(
    state: str,
    error: str = "",
    attempt: int = 0,
) -> None:
    """Publish bounded bootstrap state to the in-process web application."""

    os.environ[MODEL_PREPARATION_ATTEMPT_ENV] = str(max(0, int(attempt)))
    if error:
        os.environ[MODEL_PREPARATION_ERROR_ENV] = str(error)
    else:
        os.environ.pop(MODEL_PREPARATION_ERROR_ENV, None)
    # Publish the state last so readers never see a new terminal state paired
    # with details left over from the previous attempt.
    os.environ[MODEL_PREPARATION_STATE_ENV] = str(state).strip().lower()


def log_runtime_recovery() -> None:
    """Record read-only recovery in the normal writable application log."""
    interrupted = os.environ.get("BCVISION_RUNTIME_RECOVERED_FROM", "")
    if not interrupted:
        return
    selected = os.environ.get("BCVISION_ACTIVE_RUNTIME_VERSION", "bundled")
    try:
        log(
            "Interrupted runtime activation "
            f"{interrupted!r}; selected verified fallback {selected!r} "
            "without changing installed markers"
        )
    except OSError:
        # Recovery must remain usable even when diagnostics cannot be written.
        pass


log_runtime_recovery()


def hide_service_console() -> bool:
    """Hide an accidentally attached Windows console for packaged runs."""
    if sys.platform != "win32":
        return False
    if not (
        getattr(sys, "frozen", False)
        or os.environ.get("BCVISION_HIDE_CONSOLE", "0") == "1"
    ):
        return False
    try:
        import ctypes

        console = ctypes.windll.kernel32.GetConsoleWindow()
        if not console:
            return False
        ctypes.windll.user32.ShowWindow(console, 0)
        return True
    except Exception:
        return False


def port_open():
    try:
        with socket.create_connection(
            ("127.0.0.1", 8000),
            timeout=0.5,
        ):
            return True
    except OSError:
        return False


def service_ready():
    try:
        with urlopen(HEALTH_URL, timeout=0.75) as response:
            if response.status != 200:
                return False
            payload = json.loads(response.read().decode("utf-8"))
            return (
                payload.get("service") == "bc-vision"
                and payload.get("status") == "ok"
            )
    except (OSError, URLError, ValueError, json.JSONDecodeError):
        return False


def show_startup_error(message):
    text = (
        f"برنامه اجرا نشد:\n{message}\n\n"
        f"فایل گزارش:\n{LOG}"
    )
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(
                0,
                text,
                "خطای اجرای BC Vision",
                0x10,
            )
            return
        except Exception:
            pass
    print(text, file=sys.stderr)


def prepare_anpr_models(
    max_attempts=MODEL_PREPARATION_MAX_ATTEMPTS,
    retry_delay=MODEL_PREPARATION_RETRY_SECONDS,
):
    if os.environ.get("BCVISION_SKIP_MODEL_PREP", "0") == "1":
        _publish_model_preparation_status("skipped")
        log("ANPR model preparation skipped by environment")
        return True

    attempts = max(1, min(3, int(max_attempts)))
    delay = max(0.0, min(30.0, float(retry_delay)))
    previous_error = ""
    for attempt in range(1, attempts + 1):
        _publish_model_preparation_status(
            "preparing" if attempt == 1 else "retrying",
            previous_error,
            attempt,
        )
        try:
            from app.ai.model_manager import model_status, prepare_models

            before = model_status()
            log(f"ANPR model status before preparation: {before}")
            if (
                before["detector_yolo11n_ready"]
                and before["detector_yolov8n_ready"]
                and before["detector_fallback_ready"]
                and before["hezar_ready"]
                and before["crnn_ready"]
                and before["cnn_ready"]
            ):
                _publish_model_preparation_status("ready", attempt=attempt)
                log("ANPR models are already verified and ready")
                return True
            prepared = prepare_models(download=True)
            _publish_model_preparation_status("ready", attempt=attempt)
            log(f"ANPR models prepared successfully: {prepared}")
            return True
        except Exception as exc:
            previous_error = f"{type(exc).__name__}: {exc}"
            terminal = attempt >= attempts
            _publish_model_preparation_status(
                "error" if terminal else "retrying",
                previous_error,
                attempt,
            )
            log(
                "ANPR model preparation failed "
                f"(attempt {attempt}/{attempts}):\n"
                + traceback.format_exc()
            )
            if terminal:
                return False
            if delay:
                time.sleep(delay)
    return False


def run_server():
    try:
        if sys.stdout is None:
            sys.stdout = open(
                os.devnull,
                "w",
                encoding="utf-8",
            )
        if sys.stderr is None:
            sys.stderr = open(
                os.devnull,
                "w",
                encoding="utf-8",
            )

        import uvicorn
        from app.main import app

        log("Starting server")
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8000,
            log_level="info",
            use_colors=False,
        )
    except Exception:
        log(traceback.format_exc())
    finally:
        try:
            from app.ai.live_worker import shutdown_live_anpr_worker

            shutdown_live_anpr_worker(retry_timeout=5.0)
        except Exception:
            log("ANPR retry shutdown failed:\n" + traceback.format_exc())


def run_self_test() -> int:
    """Exercise the packaged runtime without opening the GUI or a browser."""
    requested_candidate = _argument_value("--runtime-candidate").strip()
    result = {
        "ok": False,
        "version": "",
        "runtime_source": os.environ.get(
            "BCVISION_ACTIVE_RUNTIME_SOURCE",
            "external" if ACTIVE_RUNTIME else "bundled",
        ),
        "runtime_version": (
            ACTIVE_RUNTIME.version if ACTIVE_RUNTIME else ""
        ),
        "runtime_abi": (
            ACTIVE_RUNTIME.runtime_abi if ACTIVE_RUNTIME else ""
        ),
        "runtime_recovered_from": os.environ.get(
            "BCVISION_RUNTIME_RECOVERED_FROM",
            "",
        ),
        "runtime_last_failure": os.environ.get(
            "BCVISION_RUNTIME_LAST_FAILURE",
            "",
        ),
        "data_dir": "",
        "database_path": "",
        "database_ready": False,
        "web_app_ready": False,
        "anpr_ready": False,
        "candidate_ready": not bool(requested_candidate),
    }
    try:
        from app.config import (
            APP_VERSION,
            DATA_DIR,
            DB_PATH,
        )
        from app.database import connect, init_db

        init_db()
        with connect() as con:
            user_count = int(
                con.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            )
            table_count = int(
                con.execute(
                    "SELECT COUNT(*) FROM sqlite_master "
                    "WHERE type='table'"
                ).fetchone()[0]
            )
        from app.main import app

        candidate_ready = (
            not requested_candidate
            or (
                ACTIVE_RUNTIME is not None
                and ACTIVE_RUNTIME.version == requested_candidate
                and APP_VERSION == requested_candidate
                and os.environ.get("BCVISION_ACTIVE_RUNTIME_SOURCE")
                == "candidate"
            )
        )

        verify_anpr = "--verify-anpr" in sys.argv
        anpr_ready = not verify_anpr
        if verify_anpr:
            import numpy as np
            import onnxruntime
            from app.ai.model_manager import prepare_models
            from app.ai.onnx_cnn import warmup_cnn
            from app.ai.onnx_crnn import (
                get_crnn_status,
                read_plate_crnn,
            )
            from app.ai.onnx_detector import (
                clear_detector_sessions,
                detect_plates_onnx,
                detector_status,
            )
            from app.ai.onnx_hezar import (
                hezar_status,
                read_plate_hezar_primary,
            )

            models = prepare_models(download=False)
            detect_plates_onnx(
                np.zeros((96, 160, 3), dtype=np.uint8),
                engine_key="packaged-self-test-yolo11n",
                detector_variant="yolo11n",
                raise_on_error=True,
            )
            yolo11n_ready = bool(
                detector_status()["model_loaded"]
                and detector_status()["selected_variant"] == "yolo11n"
            )
            clear_detector_sessions()
            detect_plates_onnx(
                np.zeros((96, 160, 3), dtype=np.uint8),
                engine_key="packaged-self-test-yolov8n",
                detector_variant="yolov8n",
                raise_on_error=True,
            )
            yolov8n_ready = bool(
                detector_status()["model_loaded"]
                and detector_status()["selected_variant"] == "yolov8n"
            )
            read_plate_hezar_primary(
                np.zeros((32, 384, 3), dtype=np.uint8),
                engine_key="packaged-self-test",
            )
            read_plate_crnn(
                np.zeros((32, 128, 3), dtype=np.uint8),
                engine_key="packaged-self-test",
            )
            cnn = warmup_cnn(engine_key="packaged-self-test")
            anpr_ready = bool(
                onnxruntime.__version__
                and models["detector"]
                and models["detector_yolo11n"]
                and models["detector_yolov8n"]
                and models["detector_fallback"]
                and models["hezar"]
                and models["crnn"]
                and models["cnn"]
                and yolo11n_ready
                and yolov8n_ready
                and hezar_status()["model_loaded"]
                and get_crnn_status()["model_loaded"]
                and cnn["model_loaded"]
            )

        result.update({
            "ok": (
                DB_PATH.is_file()
                and table_count >= 6
                and app is not None
                and anpr_ready
                and candidate_ready
            ),
            "version": APP_VERSION,
            "data_dir": str(DATA_DIR),
            "database_path": str(DB_PATH),
            "database_ready": DB_PATH.is_file() and table_count >= 6,
            "web_app_ready": app is not None,
            "anpr_ready": anpr_ready,
            "candidate_ready": candidate_ready,
            "administrator_setup_required": user_count == 0,
            "user_count": user_count,
            "table_count": table_count,
        })
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        log("Self-test failed:\n" + traceback.format_exc())

    payload = json.dumps(result, ensure_ascii=False, indent=2)
    output_path = _argument_value("--self-test-output")
    if output_path:
        destination = Path(output_path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload, encoding="utf-8")
    elif sys.stdout is not None:
        print(payload)
    return 0 if result["ok"] else 1


def open_panel_when_ready():
    for _ in range(80):
        if service_ready():
            webbrowser.open(PANEL_URL)
            return
        time.sleep(0.25)
    log("Web panel did not become ready within 20 seconds")


def main():
    instance_lock = None
    try:
        hide_service_console()

        if service_ready():
            webbrowser.open(PANEL_URL)
            return

        instance_lock = SingleInstanceLock()
        if not instance_lock.acquire():
            # Another BC Vision process won the startup race but may not have
            # bound its health endpoint yet. Wait for that exact service
            # identity instead of importing/starting a second worker set.
            for _ in range(40):
                if service_ready():
                    webbrowser.open(PANEL_URL)
                    return
                time.sleep(0.25)
            raise RuntimeError(
                "نمونه دیگری از BC Vision در حال راه‌اندازی است."
            )

        # Recheck under the interprocess lock: the initial health/port checks
        # were deliberately advisory only.
        if service_ready():
            webbrowser.open(PANEL_URL)
            return

        if port_open():
            raise RuntimeError(
                "پورت 8000 در اختیار برنامه دیگری است."
            )

        threading.Thread(
            target=prepare_anpr_models,
            daemon=True,
            name="anpr-model-preparation",
        ).start()

        threading.Thread(
            target=open_panel_when_ready,
            daemon=True,
            name="bcvision-browser",
        ).start()

        # Uvicorn runs in the main process, so the service stays alive without
        # a Tkinter keep-alive window and shuts camera workers down cleanly.
        run_server()
    except Exception as exc:
        log(traceback.format_exc())
        show_startup_error(str(exc))
    finally:
        if instance_lock is not None:
            instance_lock.close()


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(run_self_test())
    main()
