from __future__ import annotations

import os
import socket
import sys
import threading
import time
import traceback
import webbrowser
from pathlib import Path
import tkinter as tk
from tkinter import messagebox


def app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


BASE = app_dir()
os.chdir(BASE)

from app.config import LOG_PATH

LOG = LOG_PATH
LOG.parent.mkdir(parents=True, exist_ok=True)


def log(message):
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(
            time.strftime("[%Y-%m-%d %H:%M:%S] ")
            + str(message)
            + "\n"
        )


def port_open():
    try:
        with socket.create_connection(
            ("127.0.0.1", 8000),
            timeout=0.5,
        ):
            return True
    except OSError:
        return False


def prepare_anpr_models():
    if os.environ.get("BCVISION_SKIP_MODEL_PREP", "0") == "1":
        log("ANPR model preparation skipped by environment")
        return
    try:
        from app.ai.model_manager import model_status, prepare_models

        before = model_status()
        log(f"ANPR model status before preparation: {before}")
        if before["detector_ready"] and before["easyocr_ready"]:
            log("ANPR models are already verified and ready")
            return
        prepared = prepare_models(download=True)
        log(f"ANPR models prepared successfully: {prepared}")
    except Exception:
        # The OpenCV fallback and the web panel remain usable. The exact model
        # failure is recorded for repair without exposing it in the UI.
        log("ANPR model preparation failed:\n" + traceback.format_exc())


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


def main():
    try:
        threading.Thread(
            target=prepare_anpr_models,
            daemon=True,
            name="anpr-model-preparation",
        ).start()

        if not port_open():
            threading.Thread(
                target=run_server,
                daemon=True,
                name="bcvision-server",
            ).start()

        for _ in range(40):
            if port_open():
                break
            time.sleep(0.25)

        if not port_open():
            raise RuntimeError(
                "Server did not start. See data/BCVision.log"
            )

        webbrowser.open("http://127.0.0.1:8000/login")

        root = tk.Tk()
        root.title("BC Vision | گیلاس آبی البرز")
        root.geometry("560x300")
        root.resizable(False, False)
        tk.Label(
            root,
            text="سامانه پلاک‌خوان در حال اجراست",
            font=("Tahoma", 16, "bold"),
        ).pack(pady=(28, 10))
        tk.Label(
            root,
            text="آدرس پنل: http://127.0.0.1:8000/login",
            font=("Tahoma", 10),
        ).pack(pady=5)
        tk.Label(
            root,
            text="نام کاربری اولیه: admin     رمز اولیه: 123456",
            font=("Tahoma", 10),
        ).pack(pady=5)
        tk.Label(
            root,
            text=(
                "در اولین اجرا، مدل‌های هوش مصنوعی در پس‌زمینه "
                "آماده می‌شوند."
            ),
            font=("Tahoma", 9),
            fg="#555",
        ).pack(pady=4)
        tk.Button(
            root,
            text="باز کردن پنل",
            width=24,
            height=2,
            command=lambda: webbrowser.open(
                "http://127.0.0.1:8000/login"
            ),
        ).pack(pady=14)
        tk.Label(
            root,
            text="برای توقف سرویس این پنجره را ببندید.",
            fg="#666",
        ).pack()
        root.mainloop()
    except Exception as exc:
        log(traceback.format_exc())
        messagebox.showerror(
            "خطای اجرا",
            f"برنامه اجرا نشد:\n{exc}\n\nفایل گزارش:\n{LOG}",
        )


if __name__ == "__main__":
    main()
