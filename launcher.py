from __future__ import annotations

import os
import json
import socket
import sys
import threading
import time
import traceback
import webbrowser
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


def app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


BASE = app_dir()
os.chdir(BASE)

from app.cpu_budget import configure_process_cpu_budget

configure_process_cpu_budget()

# RC27.1 is an experimental build. Keep the production licensing subsystem in
# the source tree, but install the temporary unrestricted runtime adapter before
# any module binds license helpers locally. This can later be disabled with
# BCVISION_EXPERIMENTAL_NO_LICENSE=0 without removing the original code.
from app.experimental_license import install_experimental_license_override

install_experimental_license_override()

# Install the proven Hezar Persian CRNN V2 reader before app.ai.pipeline imports
# read_plate_candidate. Existing BC Vision ONNX CRNN/CNN readers remain the
# fallback whenever Hezar cannot produce a valid Iranian plate.
from app.ai.hezar_ocr import install_hezar_primary

install_hezar_primary()

from app.config import LOG_PATH

LOG = LOG_PATH
LOG.parent.mkdir(parents=True, exist_ok=True)
PANEL_URL = "http://127.0.0.1:8000/login"
HEALTH_URL = "http://127.0.0.1:8000/api/health"


def log(message):
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(
            time.strftime("[%Y-%m-%d %H:%M:%S] ")
            + str(message)
            + "\n"
        )


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


def prepare_anpr_models():
    if os.environ.get("BCVISION_SKIP_MODEL_PREP", "0") == "1":
        log("ANPR model preparation skipped by environment")
        return
    try:
        from app.ai.model_manager import model_status, prepare_models

        before = model_status()
        log(f"ANPR model status before preparation: {before}")
        if (
            before["detector_ready"]
            and before["detector_fallback_ready"]
            and before["crnn_ready"]
            and before["cnn_ready"]
        ):
            log("ANPR models are already verified and ready")
            return
        prepared = prepare_models(download=True)
        log(f"ANPR models prepared successfully: {prepared}")
    except Exception:
        # The OpenCV fallback and the web panel remain usable. The exact model
        # failure is recorded for repair without exposing it in the UI.
        log("ANPR model preparation failed:\n" + traceback.format_exc())


def run_server():
    manager = None
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
        from app.streams import manager as stream_manager

        manager = stream_manager
        try:
            started = manager.start_enabled_cameras()
            log(f"Background camera streams started: {started}")
        except Exception:
            # A bad camera URL must not prevent the server and other cameras
            # from starting. Stream threads handle their own reconnection.
            log("Background camera startup failed:\n" + traceback.format_exc())

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
        if manager is not None:
            try:
                manager.stop_all()
                log("Background camera streams stopped")
            except Exception:
                log("Background camera shutdown failed:\n" + traceback.format_exc())


def _argument_value(name: str) -> str:
    try:
        index = sys.argv.index(name)
    except ValueError:
        return ""
    if index + 1 >= len(sys.argv):
        return ""
    return sys.argv[index + 1]


def run_self_test() -> int:
    """Exercise the packaged runtime without opening the GUI or a browser."""
    result = {
        "ok": False,
        "version": "",
        "data_dir": "",
        "database_path": "",
        "database_ready": False,
        "public_key_ready": False,
        "web_app_ready": False,
        "anpr_ready": False,
    }
    try:
        from app.config import (
            APP_VERSION,
            DATA_DIR,
            DB_PATH,
            PUBLIC_KEY_PATH,
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
                detect_plates_onnx,
                detector_status,
            )

            models = prepare_models(download=False)
            detect_plates_onnx(
                np.zeros((96, 160, 3), dtype=np.uint8),
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
                and models["detector_fallback"]
                and models["crnn"]
                and models["cnn"]
                and detector_status()["model_loaded"]
                and get_crnn_status()["model_loaded"]
                and cnn["model_loaded"]
            )

        result.update({
            "ok": (
                DB_PATH.is_file()
                and PUBLIC_KEY_PATH.is_file()
                and table_count >= 6
                and user_count >= 1
                and app is not None
                and anpr_ready
            ),
            "version": APP_VERSION,
            "data_dir": str(DATA_DIR),
            "database_path": str(DB_PATH),
            "database_ready": DB_PATH.is_file() and table_count >= 6,
            "public_key_ready": PUBLIC_KEY_PATH.is_file(),
            "web_app_ready": app is not None,
            "anpr_ready": anpr_ready,
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
    try:
        hide_service_console()

        threading.Thread(
            target=prepare_anpr_models,
            daemon=True,
            name="anpr-model-preparation",
        ).start()

        if service_ready():
            webbrowser.open(PANEL_URL)
            return

        if port_open():
            raise RuntimeError(
                "پورت 8000 در اختیار برنامه دیگری است."
            )

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


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(run_self_test())
    main()
