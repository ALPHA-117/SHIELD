"""
SHIELD — app.py
Main Tkinter application.

Tabs:
  1. Export (Simple)  — Basic GEE export: rainfall + elevation + soil
  2. Export (Full)    — Full GEE export: adds river / water-body features
  3. Train            — Load CSV → train LSTM+XGBoost → view metrics
  4. Predict          — Load CSV → 15-day forecast → chart + risk table + CSV export

Run from project root:
  python -m shield.app
  OR
  python shield/app.py
"""

import csv as _csv
import logging
import os
import sys
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Allow running as a script directly from the shield/ folder
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
if _root not in sys.path:
    sys.path.insert(0, _root)

from shield.gee_simple import GEESimpleFrame
from shield.gee_full   import GEEFullFrame
from shield.config     import MODEL_DIR, get_risk_level

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────
CLR = {
    "bg":          "#F5F7FA",
    "header":      "#1A237E",
    "accent":      "#1565C0",
    "btn_train":   "#1B5E20",
    "btn_predict": "#4A148C",
    "btn_export":  "#E65100",
    "text":        "#212121",
    "subtext":     "#757575",
    "risk_high":   "#B71C1C",
    "risk_med":    "#E65100",
    "risk_low":    "#1B5E20",
}

FONT_TITLE  = ("Segoe UI", 18, "bold")
FONT_HDR    = ("Segoe UI", 12, "bold")
FONT_BODY   = ("Segoe UI", 10)
FONT_MONO   = ("Consolas", 9)


def _btn(parent, text, command, bg, **kw):
    return tk.Button(
        parent, text=text, command=command,
        bg=bg, fg="white", relief="flat",
        font=("Segoe UI", 10, "bold"),
        padx=16, pady=6, cursor="hand2", **kw
    )


# ─────────────────────────────────────────────────────────────────────────────
# Train Tab
# ─────────────────────────────────────────────────────────────────────────────

class TrainFrame(tk.Frame):

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=CLR["bg"], **kw)
        self._models = None
        self._build()

    def _build(self):
        pad = {"padx": 12, "pady": 6}

        # ── Header ────────────────────────────────────────────────────────────
        tk.Label(self, text="Train SHIELD Model", font=FONT_TITLE,
                 bg=CLR["bg"], fg=CLR["header"]).pack(pady=(16, 2))
        tk.Label(self,
                 text="Select a GEE-exported CSV to train the LSTM + XGBoost hybrid model.",
                 font=FONT_BODY, bg=CLR["bg"], fg=CLR["subtext"]).pack()

        # ── File selector ─────────────────────────────────────────────────────
        frm = tk.Frame(self, bg=CLR["bg"])
        frm.pack(pady=8)
        self._csv_var = tk.StringVar(value="No file selected")
        tk.Label(frm, textvariable=self._csv_var, font=FONT_BODY,
                 bg=CLR["bg"], fg=CLR["text"], width=52, anchor="w").grid(row=0, column=0, padx=4)
        _btn(frm, "Browse File", self._browse_file, CLR["accent"]).grid(row=0, column=1, padx=4)
        _btn(frm, "Browse Folder", self._browse_folder, CLR["btn_predict"]).grid(row=0, column=2, padx=4)

        # ── Region / known dates ───────────────────────────────────────────────
        opt_frm = tk.LabelFrame(self, text="Optional: Label Settings",
                                font=FONT_BODY, bg=CLR["bg"])
        opt_frm.pack(fill="x", padx=24, pady=6)

        tk.Label(opt_frm, text="Region (e.g. Barpeta):", font=FONT_BODY,
                 bg=CLR["bg"]).grid(row=0, column=0, sticky="w", **pad)
        self._region_var = tk.StringVar(value="Barpeta")
        tk.Entry(opt_frm, textvariable=self._region_var, width=20).grid(
            row=0, column=1, sticky="w", **pad)

        tk.Label(opt_frm, text="Extra flood dates (comma-sep YYYY-MM-DD):",
                 font=FONT_BODY, bg=CLR["bg"]).grid(row=1, column=0, sticky="w", **pad)
        self._dates_var = tk.StringVar(value="")
        tk.Entry(opt_frm, textvariable=self._dates_var, width=40).grid(
            row=1, column=1, sticky="w", **pad)

        # ── Train button ──────────────────────────────────────────────────────
        _btn(self, "▶  Train Model", self._train, CLR["btn_train"]).pack(pady=10)

        # ── Progress ──────────────────────────────────────────────────────────
        self._status_var   = tk.StringVar(value="Ready.")
        self._progress_var = tk.DoubleVar()
        tk.Label(self, textvariable=self._status_var, font=FONT_BODY,
                 bg=CLR["bg"], fg=CLR["subtext"], wraplength=580).pack()
        ttk.Progressbar(self, variable=self._progress_var, maximum=100,
                        length=580).pack(pady=4)

        # ── Results area ──────────────────────────────────────────────────────
        tk.Label(self, text="Training Results", font=FONT_HDR,
                 bg=CLR["bg"], fg=CLR["header"]).pack(pady=(12, 2))
        self._results_box = scrolledtext.ScrolledText(
            self, height=16, width=80, font=FONT_MONO,
            bg="#ECEFF1", relief="flat"
        )
        self._results_box.pack(padx=20, pady=4, fill="both", expand=False)

    def _browse_file(self):
        path = filedialog.askopenfilename(
            title="Select GEE CSV", filetypes=[("CSV files", "*.csv")]
        )
        if path:
            self._csv_var.set(path)
            # Try to auto-detect region name from filename (e.g. barpeta_2023 -> Barpeta)
            base = os.path.basename(path).split("_")[0]
            if base and base.isidentifier():
                self._region_var.set(base.capitalize())

    def _browse_folder(self):
        path = filedialog.askdirectory(title="Select Training Data Folder")
        if path:
            self._csv_var.set(path)
            self._region_var.set("Auto/Batch")

    def _train(self):
        path = self._csv_var.get()
        if not (os.path.isfile(path) or os.path.isdir(path)):
            messagebox.showerror("No Data", "Please select a CSV file or a folder containing CSVs.")
            return

        region = self._region_var.get().strip() or None
        dates_raw = self._dates_var.get().strip()
        extra_dates = [d.strip() for d in dates_raw.split(",") if d.strip()] if dates_raw else None

        self._results_box.delete("1.0", tk.END)
        root = self.winfo_toplevel()

        def _run():
            from shield.train import train_pipeline

            def cb(msg, pct):
                self._status_var.set(msg)
                self._progress_var.set(pct)
                root.update_idletasks()

            try:
                result = train_pipeline(
                    csv_path=path,
                    region=region,
                    extra_flood_dates=extra_dates,
                    progress_cb=cb,
                )
                m = result["metrics"]
                text = (
                    f"{'─'*60}\n"
                    f"  SHIELD Training Complete\n"
                    f"{'─'*60}\n"
                    f"  Rows: train={m['train_rows']}, test={m['test_rows']}\n"
                    f"  Flood days: {m['flood_days_total']},  "
                    f"Safe days: {m['safe_days_total']}\n"
                    f"  ROC-AUC : {m['roc_auc']}\n"
                    f"  F1-Score: {m['f1_score']}\n"
                    f"{'─'*60}\n"
                    f"Confusion Matrix:\n"
                    f"  {m['confusion_matrix']}\n"
                    f"{'─'*60}\n"
                    f"Classification Report:\n"
                    f"{m['classification_report']}\n"
                    f"Models saved to: {MODEL_DIR}\n"
                )
                self._results_box.insert(tk.END, text)
                messagebox.showinfo("Training Complete",
                                    f"✅ Model trained successfully!\n"
                                    f"ROC-AUC: {m['roc_auc']}   F1: {m['f1_score']}")
            except Exception as e:
                self._status_var.set(f"❌ {e}")
                self._results_box.insert(tk.END, f"ERROR:\n{traceback.format_exc()}")
                messagebox.showerror("Training Error", str(e))

        threading.Thread(target=_run, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
# Predict Tab
# ─────────────────────────────────────────────────────────────────────────────

class PredictFrame(tk.Frame):

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=CLR["bg"], **kw)
        self._predictions = []
        self._build()

    def _build(self):
        pad = {"padx": 12, "pady": 6}

        # ── Header ────────────────────────────────────────────────────────────
        tk.Label(self, text="15-Day Flood Forecast", font=FONT_TITLE,
                 bg=CLR["bg"], fg=CLR["header"]).pack(pady=(16, 2))
        tk.Label(self,
                 text="Select a recent GEE CSV to generate a 15-day deterministic flood forecast.",
                 font=FONT_BODY, bg=CLR["bg"], fg=CLR["subtext"]).pack()

        # ── File selector ─────────────────────────────────────────────────────
        frm = tk.Frame(self, bg=CLR["bg"])
        frm.pack(pady=8)
        self._csv_var = tk.StringVar(value="No file selected")
        tk.Label(frm, textvariable=self._csv_var, font=FONT_BODY,
                 bg=CLR["bg"], fg=CLR["text"], width=54, anchor="w").grid(row=0, column=0, padx=4)
        _btn(frm, "Browse CSV", self._browse, CLR["accent"]).grid(row=0, column=1, padx=4)

        # ── Forecast button ───────────────────────────────────────────────────
        _btn(self, "🌊  Run 15-Day Forecast", self._predict, CLR["btn_predict"]).pack(pady=10)

        # ── Progress ──────────────────────────────────────────────────────────
        self._status_var   = tk.StringVar(value="Ready.")
        self._progress_var = tk.DoubleVar()
        tk.Label(self, textvariable=self._status_var, font=FONT_BODY,
                 bg=CLR["bg"], fg=CLR["subtext"], wraplength=580).pack()
        ttk.Progressbar(self, variable=self._progress_var, maximum=100,
                        length=580).pack(pady=4)

        # ── Result area: chart left + risk table right ────────────────────────
        results_frame = tk.Frame(self, bg=CLR["bg"])
        results_frame.pack(fill="both", expand=True, padx=12, pady=8)

        # Chart
        self._fig, self._ax = plt.subplots(figsize=(6.5, 3.5))
        self._canvas = FigureCanvasTkAgg(self._fig, master=results_frame)
        self._canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        # Risk table
        tbl_frame = tk.Frame(results_frame, bg=CLR["bg"])
        tbl_frame.grid(row=0, column=1, sticky="nsew")
        tk.Label(tbl_frame, text="Flood Risk Table", font=FONT_HDR,
                 bg=CLR["bg"], fg=CLR["header"]).pack()

        cols = ("Date", "Probability", "Risk Level")
        self._tree = ttk.Treeview(tbl_frame, columns=cols, show="headings", height=14)
        for c in cols:
            self._tree.heading(c, text=c)
            self._tree.column(c, width=110 if c != "Risk Level" else 160)
        self._tree.pack(fill="both", expand=True)

        results_frame.columnconfigure(0, weight=3)
        results_frame.columnconfigure(1, weight=2)

        # Export button
        _btn(self, "💾  Export Results to CSV", self._export_csv, CLR["btn_export"]).pack(pady=6)

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select GEE CSV", filetypes=[("CSV files", "*.csv")]
        )
        if path:
            self._csv_var.set(path)

    def _predict(self):
        path = self._csv_var.get()
        if not os.path.isfile(path):
            messagebox.showerror("No File", "Please browse and select a CSV file first.")
            return

        root = self.winfo_toplevel()

        def _run():
            from shield.predict import predict_flood

            def cb(msg, pct):
                self._status_var.set(msg)
                self._progress_var.set(pct)
                root.update_idletasks()

            try:
                preds = predict_flood(csv_path=path, progress_cb=cb)
                self._predictions = preds
                root.after(0, lambda: self._display(preds))
            except FileNotFoundError as e:
                self._status_var.set("❌ Models not found — please train first.")
                messagebox.showerror("Models Not Found", str(e))
            except Exception as e:
                self._status_var.set(f"❌ {e}")
                messagebox.showerror("Forecast Error", f"{e}\n\n{traceback.format_exc()}")

        threading.Thread(target=_run, daemon=True).start()

    def _display(self, preds):
        """Render chart and populate risk table (called from main thread via after())."""
        if not preds:
            messagebox.showinfo("No Predictions", "No flood predictions were generated.")
            return

        dates  = [p[0] for p in preds]
        probs  = [p[1] for p in preds]
        labels = [p[2] for p in preds]

        # ── Chart ──────────────────────────────────────────────────────────────
        self._ax.clear()
        colours = []
        for p in probs:
            if p >= 0.50:   colours.append("#B71C1C")
            elif p >= 0.30: colours.append("#E65100")
            elif p >= 0.10: colours.append("#F9A825")
            else:            colours.append("#388E3C")

        self._ax.bar(range(len(dates)), probs, color=colours, alpha=0.85)
        self._ax.axhline(0.50, color="#B71C1C", linestyle="--", linewidth=1, label="Extreme (50%)")
        self._ax.axhline(0.30, color="#E65100", linestyle="--", linewidth=1, label="High (30%)")
        self._ax.set_xticks(range(len(dates)))
        self._ax.set_xticklabels(
            [d.strftime("%d %b") for d in dates], rotation=45, fontsize=7
        )
        self._ax.set_ylim(0, 1)
        self._ax.set_ylabel("Flood Probability")
        self._ax.set_title("15-Day Flood Risk Forecast — SHIELD", fontsize=10, fontweight="bold")
        self._ax.legend(fontsize=7)
        self._fig.tight_layout()
        self._canvas.draw()

        # ── Risk table ─────────────────────────────────────────────────────────
        for row in self._tree.get_children():
            self._tree.delete(row)
        for d, p, lbl, clr, _ in preds:
            self._tree.insert(
                "", tk.END,
                values=(d.strftime("%Y-%m-%d"), f"{p:.1%}", lbl)
            )

    def _export_csv(self):
        if not self._predictions:
            messagebox.showwarning("No Predictions", "Run a forecast first.")
            return
        save_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile="shield_forecast.csv",
            title="Save Forecast CSV",
        )
        if save_path:
            with open(save_path, "w", newline="") as f:
                writer = _csv.writer(f)
                writer.writerow(["date", "flood_probability", "risk_level"])
                for d, p, lbl, clr, r in self._predictions:
                    writer.writerow([d.strftime("%Y-%m-%d"), round(p, 4), lbl])
            messagebox.showinfo("Saved", f"Forecast saved to:\n{save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────────────────────────────────────

class SHIELDApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("SHIELD — Smart Hydro-climate Insights for Early Warning & Local Defense")
        self.geometry("900x750")
        self.resizable(True, True)
        self.configure(bg=CLR["bg"])
        self._build_header()
        self._build_notebook()
        self._build_footer()

    def _build_header(self):
        hdr = tk.Frame(self, bg=CLR["header"], height=60)
        hdr.pack(fill="x")
        tk.Label(
            hdr,
            text="🌊  SHIELD  |  Flood Early Warning System",
            font=("Segoe UI", 15, "bold"),
            bg=CLR["header"], fg="white",
        ).pack(side="left", padx=20, pady=12)
        tk.Label(
            hdr,
            text="Smart Hydro-climate Insights for Early Warning & Local Defense",
            font=("Segoe UI", 9),
            bg=CLR["header"], fg="#90CAF9",
        ).pack(side="left", pady=12)

    def _build_notebook(self):
        style = ttk.Style()
        style.configure("TNotebook.Tab", font=("Segoe UI", 10), padding=[12, 6])

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=8)

        self._tab_simple  = GEESimpleFrame(nb, bg=CLR["bg"])
        self._tab_full    = GEEFullFrame(nb, bg=CLR["bg"])
        self._tab_train   = TrainFrame(nb)
        self._tab_predict = PredictFrame(nb)

        nb.add(self._tab_simple,  text="📡 Export (Simple)")
        nb.add(self._tab_full,    text="🛰️  Export (Full)")
        nb.add(self._tab_train,   text="🧠 Train")
        nb.add(self._tab_predict, text="🌊 Predict")

    def _build_footer(self):
        footer = tk.Frame(self, bg=CLR["header"], height=28)
        footer.pack(fill="x", side="bottom")
        tk.Label(
            footer,
            text="SHIELD v1.0  |  Phase 1: XGBoost  |  Phase 2: LSTM+XGBoost  "
                 "|  Data: NASA GPM + SRTM + JRC + SoilGrids",
            font=("Segoe UI", 8),
            bg=CLR["header"], fg="#90CAF9",
        ).pack(pady=4)


def main():
    app = SHIELDApp()
    app.mainloop()


if __name__ == "__main__":
    main()
