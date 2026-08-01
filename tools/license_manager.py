from __future__ import annotations

import json
import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.license_service import (
    ALL_FEATURES,
    FEATURES,
    LIMITS,
    LicenseLedger,
    LicenseRequest,
    issue_license,
    machine_ids_from_request_file,
    normalize_machine_ids,
)

APP_DIR = Path.home() / "BCVisionLicenseManager"
KEY_DIR = Path(__file__).resolve().parent / "keys"
DEFAULT_PRIVATE = KEY_DIR / "license_private_key.pem"
DEFAULT_PUBLIC = ROOT / "license_public_key.pem"
DEFAULT_LEDGER = APP_DIR / "license_manager.db"
DEFAULT_OUTPUT = APP_DIR / "issued"


class LicenseManagerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("BC Vision License Manager")
        self.geometry("1120x720")
        self.minsize(960, 620)
        self.ledger = LicenseLedger(DEFAULT_LEDGER)
        self.machine_ids: tuple[str, ...] = ()
        self.feature_vars = {
            feature: tk.BooleanVar(value=False) for feature in ALL_FEATURES
        }
        self._build_ui()
        self._apply_plan_defaults()
        self._refresh_history()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        notebook = ttk.Notebook(self)
        notebook.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)

        issue_tab = ttk.Frame(notebook, padding=16)
        history_tab = ttk.Frame(notebook, padding=12)
        notebook.add(issue_tab, text="صدور لایسنس")
        notebook.add(history_tab, text="تاریخچه")

        issue_tab.columnconfigure(1, weight=1)
        issue_tab.columnconfigure(3, weight=1)

        self.customer_var = tk.StringVar()
        self.contract_var = tk.StringVar()
        self.plan_var = tk.StringVar(value="basic")
        self.camera_var = tk.IntVar(value=LIMITS["basic"])
        self.days_var = tk.IntVar(value=365)
        self.perpetual_var = tk.BooleanVar(value=False)
        self.machine_summary_var = tk.StringVar(value="هیچ شناسه دستگاهی انتخاب نشده")
        self.private_key_var = tk.StringVar(value=str(DEFAULT_PRIVATE))
        self.public_key_var = tk.StringVar(value=str(DEFAULT_PUBLIC))
        self.output_dir_var = tk.StringVar(value=str(DEFAULT_OUTPUT))

        row = 0
        ttk.Label(issue_tab, text="نام مشتری").grid(row=row, column=0, sticky="e", padx=6, pady=6)
        ttk.Entry(issue_tab, textvariable=self.customer_var).grid(row=row, column=1, sticky="ew", padx=6, pady=6)
        ttk.Label(issue_tab, text="شماره قرارداد").grid(row=row, column=2, sticky="e", padx=6, pady=6)
        ttk.Entry(issue_tab, textvariable=self.contract_var).grid(row=row, column=3, sticky="ew", padx=6, pady=6)

        row += 1
        ttk.Label(issue_tab, text="پلن").grid(row=row, column=0, sticky="e", padx=6, pady=6)
        plan_box = ttk.Combobox(
            issue_tab,
            state="readonly",
            textvariable=self.plan_var,
            values=tuple(LIMITS),
        )
        plan_box.grid(row=row, column=1, sticky="ew", padx=6, pady=6)
        plan_box.bind("<<ComboboxSelected>>", lambda _event: self._apply_plan_defaults())
        ttk.Label(issue_tab, text="تعداد دوربین").grid(row=row, column=2, sticky="e", padx=6, pady=6)
        ttk.Spinbox(issue_tab, from_=1, to=4096, textvariable=self.camera_var).grid(row=row, column=3, sticky="ew", padx=6, pady=6)

        row += 1
        ttk.Label(issue_tab, text="مدت اعتبار (روز)").grid(row=row, column=0, sticky="e", padx=6, pady=6)
        ttk.Spinbox(issue_tab, from_=1, to=36500, textvariable=self.days_var).grid(row=row, column=1, sticky="ew", padx=6, pady=6)
        ttk.Checkbutton(
            issue_tab,
            text="دائمی",
            variable=self.perpetual_var,
            command=self._toggle_perpetual,
        ).grid(row=row, column=2, columnspan=2, sticky="w", padx=6, pady=6)

        row += 1
        machine_frame = ttk.LabelFrame(issue_tab, text="شناسه دستگاه", padding=10)
        machine_frame.grid(row=row, column=0, columnspan=4, sticky="ew", padx=6, pady=10)
        machine_frame.columnconfigure(0, weight=1)
        ttk.Label(machine_frame, textvariable=self.machine_summary_var).grid(row=0, column=0, sticky="w")
        ttk.Button(machine_frame, text="ورود فایل درخواست", command=self._load_request_file).grid(row=0, column=1, padx=4)
        ttk.Button(machine_frame, text="ورود دستی", command=self._enter_machine_ids).grid(row=0, column=2, padx=4)

        row += 1
        features_frame = ttk.LabelFrame(issue_tab, text="قابلیت‌ها", padding=10)
        features_frame.grid(row=row, column=0, columnspan=4, sticky="ew", padx=6, pady=8)
        for index, feature in enumerate(ALL_FEATURES):
            ttk.Checkbutton(
                features_frame,
                text=feature,
                variable=self.feature_vars[feature],
            ).grid(row=index // 4, column=index % 4, sticky="w", padx=8, pady=4)

        row += 1
        keys_frame = ttk.LabelFrame(issue_tab, text="کلیدها و خروجی", padding=10)
        keys_frame.grid(row=row, column=0, columnspan=4, sticky="ew", padx=6, pady=8)
        keys_frame.columnconfigure(1, weight=1)
        ttk.Label(keys_frame, text="کلید خصوصی").grid(row=0, column=0, sticky="e", padx=4, pady=4)
        ttk.Entry(keys_frame, textvariable=self.private_key_var).grid(row=0, column=1, sticky="ew", padx=4, pady=4)
        ttk.Button(keys_frame, text="انتخاب", command=lambda: self._pick_file(self.private_key_var)).grid(row=0, column=2, padx=4)
        ttk.Label(keys_frame, text="کلید عمومی").grid(row=1, column=0, sticky="e", padx=4, pady=4)
        ttk.Entry(keys_frame, textvariable=self.public_key_var).grid(row=1, column=1, sticky="ew", padx=4, pady=4)
        ttk.Button(keys_frame, text="انتخاب", command=lambda: self._pick_file(self.public_key_var)).grid(row=1, column=2, padx=4)
        ttk.Label(keys_frame, text="پوشه خروجی").grid(row=2, column=0, sticky="e", padx=4, pady=4)
        ttk.Entry(keys_frame, textvariable=self.output_dir_var).grid(row=2, column=1, sticky="ew", padx=4, pady=4)
        ttk.Button(keys_frame, text="انتخاب", command=self._pick_output_dir).grid(row=2, column=2, padx=4)

        row += 1
        ttk.Button(issue_tab, text="صدور license.dat", command=self._issue).grid(row=row, column=0, columnspan=4, pady=18)

        history_tab.columnconfigure(0, weight=1)
        history_tab.rowconfigure(0, weight=1)
        columns = ("customer", "plan", "cameras", "issued", "expires", "license_id")
        self.history = ttk.Treeview(history_tab, columns=columns, show="headings")
        headings = {
            "customer": "مشتری",
            "plan": "پلن",
            "cameras": "دوربین",
            "issued": "صدور",
            "expires": "انقضا",
            "license_id": "شناسه لایسنس",
        }
        for column in columns:
            self.history.heading(column, text=headings[column])
            self.history.column(column, width=150, anchor="center")
        self.history.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(history_tab, orient="vertical", command=self.history.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.history.configure(yscrollcommand=scrollbar.set)
        ttk.Button(history_tab, text="به‌روزرسانی", command=self._refresh_history).grid(row=1, column=0, pady=8)

    def _apply_plan_defaults(self) -> None:
        plan = self.plan_var.get()
        self.camera_var.set(LIMITS[plan])
        defaults = set(FEATURES[plan])
        for feature, variable in self.feature_vars.items():
            variable.set(feature in defaults)

    def _toggle_perpetual(self) -> None:
        if self.perpetual_var.get():
            self.days_var.set(365)

    def _load_request_file(self) -> None:
        filename = filedialog.askopenfilename(
            title="انتخاب فایل درخواست دستگاه",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not filename:
            return
        try:
            self.machine_ids = machine_ids_from_request_file(Path(filename))
            self._update_machine_summary()
        except Exception as exc:
            messagebox.showerror("خطا", str(exc))

    def _enter_machine_ids(self) -> None:
        dialog = tk.Toplevel(self)
        dialog.title("ورود شناسه دستگاه")
        dialog.transient(self)
        dialog.grab_set()
        ttk.Label(dialog, text="هر شناسه را در یک خط وارد کنید").pack(padx=12, pady=(12, 4))
        text = tk.Text(dialog, width=70, height=10)
        text.pack(padx=12, pady=4)
        text.insert("1.0", "\n".join(self.machine_ids))

        def save() -> None:
            try:
                self.machine_ids = normalize_machine_ids(text.get("1.0", "end").splitlines())
                self._update_machine_summary()
                dialog.destroy()
            except Exception as exc:
                messagebox.showerror("خطا", str(exc), parent=dialog)

        ttk.Button(dialog, text="ثبت", command=save).pack(pady=12)

    def _update_machine_summary(self) -> None:
        first = self.machine_ids[0] if self.machine_ids else ""
        self.machine_summary_var.set(
            f"{len(self.machine_ids)} دستگاه — {first[:48]}"
        )

    def _pick_file(self, variable: tk.StringVar) -> None:
        filename = filedialog.askopenfilename()
        if filename:
            variable.set(filename)

    def _pick_output_dir(self) -> None:
        directory = filedialog.askdirectory()
        if directory:
            self.output_dir_var.set(directory)

    def _issue(self) -> None:
        try:
            selected_features = tuple(
                feature for feature, variable in self.feature_vars.items() if variable.get()
            )
            customer = self.customer_var.get().strip()
            safe_name = "_".join(customer.split()) or "customer"
            output = Path(self.output_dir_var.get()) / f"{safe_name}.license.dat"
            request = LicenseRequest(
                customer=customer,
                contract=self.contract_var.get(),
                machine_ids=self.machine_ids,
                plan=self.plan_var.get(),
                days=int(self.days_var.get()),
                perpetual=bool(self.perpetual_var.get()),
                camera_limit=int(self.camera_var.get()),
                features=selected_features,
            )
            issued = issue_license(
                request,
                private_key_path=Path(self.private_key_var.get()),
                public_key_path=Path(self.public_key_var.get()),
                output_path=output,
            )
            self.ledger.record(issued)
            self._refresh_history()
            messagebox.showinfo(
                "انجام شد",
                f"لایسنس صادر شد:\n{issued.output_path}\n\nLicense ID: {issued.license_id}",
            )
        except Exception as exc:
            messagebox.showerror("خطا در صدور لایسنس", str(exc))

    def _refresh_history(self) -> None:
        for item in self.history.get_children():
            self.history.delete(item)
        for row in self.ledger.list_recent():
            self.history.insert(
                "",
                "end",
                values=(
                    row["customer"],
                    row["plan"],
                    row["camera_limit"],
                    row["issued_at"],
                    row["expires_at"],
                    row["license_id"],
                ),
            )


def main() -> None:
    DEFAULT_OUTPUT.mkdir(parents=True, exist_ok=True)
    LicenseManagerApp().mainloop()


if __name__ == "__main__":
    main()
