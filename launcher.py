from __future__ import annotations
import os, sys, threading, time, traceback, webbrowser, socket
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

def app_dir() -> Path:
    return Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent

BASE = app_dir()
os.chdir(BASE)
from app.config import LOG_PATH
LOG = LOG_PATH
LOG.parent.mkdir(parents=True, exist_ok=True)

def log(msg):
    with LOG.open("a", encoding="utf-8") as f:
        f.write(time.strftime("[%Y-%m-%d %H:%M:%S] ") + msg + "\n")

def port_open():
    try:
        with socket.create_connection(("127.0.0.1", 8000), timeout=.5):
            return True
    except OSError:
        return False

def run_server():
    try:
        if sys.stdout is None:
            sys.stdout = open(os.devnull, "w", encoding="utf-8")
        if sys.stderr is None:
            sys.stderr = open(os.devnull, "w", encoding="utf-8")

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
        if not port_open():
            threading.Thread(target=run_server, daemon=True).start()
        for _ in range(40):
            if port_open():
                break
            time.sleep(.25)
        if not port_open():
            raise RuntimeError("Server did not start. See data/BCVision.log")
        webbrowser.open("http://127.0.0.1:8000/login")
        root = tk.Tk()
        root.title("BC Vision | Ú¯ÛŒÙ„Ø§Ø³ Ø¢Ø¨ÛŒ Ø§Ù„Ø¨Ø±Ø²")
        root.geometry("530x260")
        root.resizable(False, False)
        tk.Label(root, text="Ø³Ø§Ù…Ø§Ù†Ù‡ Ù¾Ù„Ø§Ú©â€ŒØ®ÙˆØ§Ù† Ø¯Ø± Ø­Ø§Ù„ Ø§Ø¬Ø±Ø§Ø³Øª", font=("Tahoma", 16, "bold")).pack(pady=(30,10))
        tk.Label(root, text="Ø¢Ø¯Ø±Ø³ Ù¾Ù†Ù„: http://127.0.0.1:8000/login", font=("Tahoma", 10)).pack(pady=5)
        tk.Label(root, text="Ù†Ø§Ù… Ú©Ø§Ø±Ø¨Ø±ÛŒ Ø§ÙˆÙ„ÛŒÙ‡: admin     Ø±Ù…Ø² Ø§ÙˆÙ„ÛŒÙ‡: 123456", font=("Tahoma", 10)).pack(pady=5)
        tk.Button(root, text="Ø¨Ø§Ø² Ú©Ø±Ø¯Ù† Ù¾Ù†Ù„", width=24, height=2, command=lambda:webbrowser.open("http://127.0.0.1:8000/login")).pack(pady=16)
        tk.Label(root, text="Ø¨Ø±Ø§ÛŒ ØªÙˆÙ‚Ù Ø³Ø±ÙˆÛŒØ³ Ø§ÛŒÙ† Ù¾Ù†Ø¬Ø±Ù‡ Ø±Ø§ Ø¨Ø¨Ù†Ø¯ÛŒØ¯.", fg="#666").pack()
        root.mainloop()
    except Exception as e:
        log(traceback.format_exc())
        messagebox.showerror("Ø®Ø·Ø§ÛŒ Ø§Ø¬Ø±Ø§", f"Ø¨Ø±Ù†Ø§Ù…Ù‡ Ø§Ø¬Ø±Ø§ Ù†Ø´Ø¯:\n{e}\n\nÙØ§ÛŒÙ„ Ú¯Ø²Ø§Ø±Ø´:\n{LOG}")

if __name__ == "__main__":
    main()
