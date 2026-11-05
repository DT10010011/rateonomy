"""RateOnomy - Autonomy Decision Support Tool

Three components (ROI, Drivability, Legal) feed a weighted composite score.
ROI and drivability use calculated inputs; legal feasibility remains a dummy.
"""

import os
import math
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, font as tkfont

from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageChops, ImageFont, ImageTk
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

# Load ROBOFLOW_API_KEY from a .env file next to this script
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Weight vector from the midterm report: [ROI, Drivability, Legal]
WEIGHTS = {"roi": 0.3, "drivability": 0.2, "legal": 0.5}
MAX_SCORE = 10.0

# Static panel descriptions shown under each score
SUMMARIES = {
    "roi": "The ROI Score predicts the general return-on-investment of "
           "autonomous vehicles for your farm.",
    "drivability": "The Drivability Score is based on your farm's total "
                   "obstacle coverage.",
    "legal": "The Legal Feasibility Score searches the laws and regulations "
             "in your area to help you set realistic compliance goals. The "
             "Legal Feasibility Score is not a substitute for real legal advice.",
}

# Letter-grade cutoffs (score out of 10). Anything below the last cutoff is an F.
GRADE_THRESHOLDS = [(9.0, "A"), (8.0, "B"), (7.0, "C"), (6.0, "D")]


def grade_from_score(score):
    for cutoff, letter in GRADE_THRESHOLDS:
        if score >= cutoff:
            return letter
    return "F"

# Roboflow (put ROBOFLOW_API_KEY in your .env file; never hardcode it)
ROBOFLOW_API_URL = "https://serverless.roboflow.com"
ROBOFLOW_WORKSPACE = "rgibson6-illinois-edu"
ROBOFLOW_WORKFLOW_ID = "instancedriveablityscore"


# ---------------------------------------------------------------------------
# Component stubs - replace these with the real implementations
# ---------------------------------------------------------------------------
def compute_roi_score(payback_years=None):
    """Return dict with ROI score (0-10) and display details."""
    if payback_years is None:
        score = 0
        payback_label = "Never pays back"
    elif payback_years <= 0.5:
        score = 1
        payback_label = f"{payback_years:,.1f} years"
    elif payback_years < 2:
        score = 2
        payback_label = f"{payback_years:,.1f} years"
    elif payback_years < 3:
        score = 3
        payback_label = f"{payback_years:,.1f} years"
    elif payback_years < 4:
        score = 4
        payback_label = f"{payback_years:,.1f} years"
    elif payback_years < 5:
        score = 5
        payback_label = f"{payback_years:,.1f} years"
    elif payback_years < 6:
        score = 6
        payback_label = f"{payback_years:,.1f} years"
    elif payback_years < 7:
        score = 7
        payback_label = f"{payback_years:,.1f} years"
    elif payback_years < 8:
        score = 8
        payback_label = f"{payback_years:,.1f} years"
    elif payback_years < 9:
        score = 9
        payback_label = f"{payback_years:,.1f} years"
    else:
        score = 10
        payback_label = f"{payback_years:,.1f} years"

    return {
        "score": float(score),
        "summary": "ROI rating is based on the calculated payback period.",
        "details": [("Payback period", payback_label), ("ROI rating", f"{score} / 10")],
    }


def _find_predictions(obj):
    """Recursively find the first list of polygon predictions in a workflow result."""
    if isinstance(obj, dict):
        preds = obj.get("predictions")
        if isinstance(preds, list) and (
            not preds or (isinstance(preds[0], dict) and "points" in preds[0])
        ):
            return preds
        for value in obj.values():
            found = _find_predictions(value)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_predictions(item)
            if found is not None:
                return found
    return None


def _class_mask(predictions, size, class_filter):
    """Rasterize all polygons of matching classes into one 0/1 mask.

    Using a mask means overlapping detections are not double counted.
    """
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for pred in predictions:
        if not class_filter(pred.get("class", "")):
            continue
        pts = [(p["x"], p["y"]) for p in pred.get("points", [])]
        if len(pts) >= 3:
            draw.polygon(pts, fill=1)
    return mask


def compute_drivability_score(image_path=None, field_area=None):
    """Return dict with drivability score (0-10) and display details.

    Drivable pixels = area the model labels Drivable, minus anything it also
    labels NonDrivable. Everything else in the image counts as obstacle.
    Drivable coverage is mapped to a 1-10 image rating. The image rating
    contributes 90%; field size contributes 10%, scaled to 10 points at
    1,000 acres.
    Raises an exception if the Roboflow call fails (caller shows the error).
    """
    if not image_path:
        return {
            "score": 10.0,
            "summary": "No image selected - using dummy score.",
            "details": [("Obstacle coverage", "N/A (dummy)")],
        }
    if field_area is None or not math.isfinite(field_area) or field_area <= 0:
        raise ValueError("Enter a field size greater than 0 acres.")

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY not found. Add it to the .env file next to app.py.")

    client = InferenceHTTPClient(
        api_url=ROBOFLOW_API_URL, api_key=api_key
    ).configure(InferenceConfiguration(api_key_transport="header"))
    result = client.run_workflow(
        workspace_name=ROBOFLOW_WORKSPACE,
        workflow_id=ROBOFLOW_WORKFLOW_ID,
        images={"image": image_path},
        use_cache=True,
    )

    preds = _find_predictions(result)
    if preds is None:
        raise RuntimeError(
            "Could not find predictions in the workflow response: " + str(result)[:300]
        )

    with Image.open(image_path) as img:
        width, height = img.size
    total = float(width * height)

    is_nondrivable = lambda c: "non" in c.lower()
    # Matches "Drivable" and "Driveable" spellings; excludes "NonDrivable"
    is_drivable = lambda c: "driv" in c.lower() and "non" not in c.lower()

    drivable_mask = _class_mask(preds, (width, height), is_drivable)
    nondrivable_mask = _class_mask(preds, (width, height), is_nondrivable)
    clear_mask = ImageChops.subtract(drivable_mask, nondrivable_mask)

    drivable_frac = clear_mask.histogram()[1] / total
    obstacle_frac = 1.0 - drivable_frac
    image_coverage = drivable_frac * 100
    ratings = [
        (98, 10), (96, 9), (94, 8), (92, 7), (90, 6),
        (88, 5), (86, 4), (84, 3), (82, 2), (0, 1),
    ]
    image_rating = 1
    for limit, rating in ratings:
        if image_coverage >= limit:
            image_rating = rating
            break

    field_size_score = min(MAX_SCORE, (field_area / 1000) * MAX_SCORE)
    score = 0.9 * image_rating + 0.1 * field_size_score

    return {
        "score": round(score, 1),
        "summary": "Score combines image rating (90%) and field size (10%).",
        "details": [
            ("Obstacle coverage", f"{obstacle_frac:.1%}"),
            ("Drivable coverage", f"{drivable_frac:.1%}"),
            ("Image drivability rating", f"{image_rating} / 10"),
            ("Field size", f"{field_area:,.1f} acres"),
            ("Field size rating", f"{field_size_score:.1f} / 10"),
            ("Regions detected", str(len(preds))),
            ("Classes found", ", ".join(sorted({p.get("class", "?") for p in preds})) or "none"),
        ],
    }


def compute_legal_score(location=None, equipment=None):
    """Return dict with legal score (0-10), letter grade, and feasibility."""
    # TODO: query curated regulatory database
    return {
        "score": 10.0,
        "grade": "A",
        "label": "Workable",
        "feasible": "Legally feasible",
        "summary": "Dummy legal score (regulatory database not yet built).",
        "details": [("Restrictions", "None (dummy)")],
    }


def compute_composite(roi, drivability, legal):
    return (
        WEIGHTS["roi"] * roi
        + WEIGHTS["drivability"] * drivability
        + WEIGHTS["legal"] * legal
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
# First installed font in this list is used (Bahnschrift is a DIN-style
# industrial typeface on Windows 10+; the rest are fallbacks).
PREFERRED_FONTS = ["Bahnschrift", "Segoe UI", "Helvetica", "Arial"]
FONT = "Segoe UI"  # replaced at startup by pick_font()


def pick_font():
    installed = set(tkfont.families())
    for name in PREFERRED_FONTS:
        if name in installed:
            return name
    return "TkDefaultFont"


BG = "#F2F5F1"
CARD = "#FFFFFF"
HEADER = "#2F5D3A"
HEADER_SUB = "#CFE3D3"
ACCENT = "#3F8A4F"
ACCENT_DARK = "#2F6E3D"
TEXT = "#1F2A24"
MUTED = "#6B7770"
BORDER = "#DCE3DC"
TROUGH = "#E6EBE5"
PENDING = "#9AA59E"
DANGER = "#C0392B"

GRADE_COLORS = {
    "A": "#2E8B57",
    "B": "#6BA43A",
    "C": "#D4A017",
    "D": "#E07B28",
    "F": "#C0392B",
}


def setup_styles(root):
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("Accent.TButton", font=(FONT, 11, "bold"), foreground="white",
                    background=ACCENT, borderwidth=0, focusthickness=0, padding=(20, 9))
    style.map("Accent.TButton",
              background=[("disabled", "#A9C4AF"), ("active", ACCENT_DARK), ("pressed", ACCENT_DARK)],
              foreground=[("disabled", "#F2F5F1")])
    style.configure("Secondary.TButton", font=(FONT, 10), foreground=TEXT,
                    background="#E4E9E3", borderwidth=0, focusthickness=0, padding=(14, 9))
    style.map("Secondary.TButton",
              background=[("disabled", "#EEF1ED"), ("active", "#D5DCD4"), ("pressed", "#D5DCD4")],
              foreground=[("disabled", PENDING)])


def make_card(parent):
    return tk.Frame(parent, bg=CARD, highlightbackground=BORDER, highlightthickness=1)


class Spinner(tk.Canvas):
    """A small rotating-arc loading indicator."""

    def __init__(self, parent, size=22, color=ACCENT, bg=CARD):
        super().__init__(parent, width=size, height=size, bg=bg, highlightthickness=0)
        pad = 3
        box = (pad, pad, size - pad, size - pad)
        self.create_oval(*box, outline=TROUGH, width=3)
        self._arc = self.create_arc(*box, start=90, extent=100, style="arc",
                                    outline=color, width=3)
        self._angle = 90
        self._job = None

    def start(self):
        if self._job is None:
            self._step()

    def _step(self):
        self._angle = (self._angle - 18) % 360
        self.itemconfigure(self._arc, start=self._angle)
        self._job = self.after(40, self._step)

    def stop(self):
        if self._job is not None:
            self.after_cancel(self._job)
            self._job = None


class GradeBar(tk.Canvas):
    """A score bar (0-10) with tick marks at each letter-grade cutoff."""

    BAR_TOP = 2
    BAR_H = 10

    def __init__(self, parent, bg=CARD):
        super().__init__(parent, height=34, bg=bg, highlightthickness=0)
        self._value = 0.0
        self._color = PENDING
        self.bind("<Configure>", lambda _e: self._draw())

    def set(self, value, color):
        self._value = value
        self._color = color
        self._draw()

    def _draw(self):
        self.delete("all")
        w = self.winfo_width()
        if w <= 1:
            return
        top, bottom = self.BAR_TOP, self.BAR_TOP + self.BAR_H
        self.create_rectangle(0, top, w, bottom, fill=TROUGH, outline="")
        if self._value > 0:
            fill_w = max(0, min(w, w * self._value / MAX_SCORE))
            self.create_rectangle(0, top, fill_w, bottom, fill=self._color, outline="")
        # One tick per cutoff; the letter marks the grade reached at that point.
        for cutoff, letter in GRADE_THRESHOLDS:
            x = round(w * cutoff / MAX_SCORE)
            self.create_line(x, top - 2, x, bottom + 4, fill=TEXT, width=1)
            self.create_text(x, bottom + 6, text=letter, anchor="n",
                             font=(FONT, 8), fill=MUTED)


ICON_SIZE = 30        # final icon size in pixels
ICON_SCALE = 8        # drawn 8x larger, then shrunk, which smooths the edges
ICON_FONT_FILES = ("bahnschrift.ttf", "segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf")


def render_icon(kind, color=ACCENT, bg=CARD):
    """Draw a score icon (roi / drivability / legal) as an anti-aliased PIL image.

    Tk's canvas has no anti-aliasing, so the icon is drawn at high resolution
    with Pillow and downsampled instead.
    """
    S = ICON_SCALE
    img = Image.new("RGB", (ICON_SIZE * S, ICON_SIZE * S), bg)
    d = ImageDraw.Draw(img)
    off = 2  # keeps strokes from touching the edge; icons are drawn in a 26-unit box

    def pt(x, y):
        return ((x + off) * S, (y + off) * S)

    def box(x0, y0, x1, y1):
        return [pt(x0, y0), pt(x1, y1)]

    def line(x0, y0, x1, y1, w=2):
        d.line([pt(x0, y0), pt(x1, y1)], fill=color, width=round(w * S))
        r = w * S / 2  # round the line ends
        for x, y in ((x0, y0), (x1, y1)):
            cx, cy = pt(x, y)
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)

    if kind == "roi":  # coin
        d.ellipse(box(0, 0, 26, 26), outline=color, width=2 * S)
        font = None
        for name in ICON_FONT_FILES:
            try:
                font = ImageFont.truetype(name, 15 * S)
                break
            except OSError:
                continue
        if font is not None:
            d.text(pt(13, 13.5), "$", fill=color, font=font, anchor="mm")
        else:  # no usable font found: draw an inner ring instead of the $
            d.ellipse(box(6, 6, 20, 20), outline=color, width=S)
    elif kind == "drivability":  # steering wheel
        d.ellipse(box(0, 0, 26, 26), outline=color, width=2 * S)
        d.ellipse(box(9, 9, 17, 17), fill=color)
        line(0.5, 13, 9, 13)
        line(17, 13, 25.5, 13)
        line(13, 17, 13, 25.5)
    elif kind == "legal":  # scales of justice
        line(13, 3, 13, 22)
        line(7, 23, 19, 23)
        line(3, 6, 23, 6)
        for cx in (3, 23):
            line(cx, 6, cx - 3, 15, w=1)
            line(cx, 6, cx + 3, 15, w=1)
            d.arc(box(cx - 3, 12, cx + 3, 18), start=0, end=180, fill=color, width=2 * S)
    return img.resize((ICON_SIZE, ICON_SIZE), Image.Resampling.LANCZOS)


class ScoreIcon(tk.Label):
    """Smooth icon for a score card (see render_icon)."""

    def __init__(self, parent, kind, color=ACCENT, bg=CARD):
        super().__init__(parent, bg=bg, bd=0, highlightthickness=0)
        self._photo = ImageTk.PhotoImage(render_icon(kind, color, bg))  # keep a reference
        self.config(image=self._photo)


class ScorePanel(tk.Frame):
    """A card showing one component's score, grade, summary, and detail rows."""

    def __init__(self, parent, key, title, summary):
        super().__init__(parent, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        inner = tk.Frame(self, bg=CARD)
        inner.pack(fill="both", expand=True, padx=16, pady=14)

        head = tk.Frame(inner, bg=CARD)
        head.pack(fill="x")
        tk.Label(head, text=title.upper(), font=(FONT, 10, "bold"),
                 fg=MUTED, bg=CARD).pack(side="left")
        ScoreIcon(head, key).pack(side="right")

        self.score_label = tk.Label(inner, text="--", font=(FONT, 38, "bold"),
                                    fg=PENDING, bg=CARD)
        self.score_label.pack(pady=(10, 0))
        tk.Label(inner, text="out of 10", font=(FONT, 9), fg=MUTED, bg=CARD).pack()

        badge_row = tk.Frame(inner, bg=CARD)
        badge_row.pack(pady=(10, 10))
        self.badge = tk.Label(badge_row, text="-", font=(FONT, 12, "bold"),
                              fg="white", bg=PENDING, width=3)
        self.badge.pack(side="left")
        self.note = tk.Label(badge_row, text="", font=(FONT, 11), fg=TEXT, bg=CARD)
        self.note.pack(side="left", padx=(8, 0))

        self.bar = GradeBar(inner)
        self.bar.pack(fill="x", pady=(0, 10))

        # Summary sits in a frame whose height is equalized across panels
        # (see set_summary_height) so everything below it lines up.
        self.summary_frame = tk.Frame(inner, bg=CARD)
        self.summary_frame.pack(fill="x")
        self.summary_label = tk.Label(self.summary_frame, text=summary, font=(FONT, 9),
                                      fg=MUTED, bg=CARD, wraplength=230, justify="left")
        self.summary_label.pack(anchor="nw")

        tk.Frame(inner, bg=BORDER, height=1).pack(fill="x", pady=10)
        self.details = tk.Frame(inner, bg=CARD)
        self.details.pack(fill="x")

    def set_summary_height(self, height):
        self.summary_frame.config(height=height)
        self.summary_frame.pack_propagate(False)

    def _clear_details(self):
        for child in self.details.winfo_children():
            child.destroy()

    def update_result(self, result, grade, note=""):
        color = GRADE_COLORS.get(grade, PENDING)
        self.score_label.config(text=f"{result['score']:.1f}", fg=color)
        self.badge.config(text=grade, bg=color)
        self.note.config(text=note)
        self.bar.set(result["score"], color)
        self._clear_details()
        for label, value in result["details"]:
            row = tk.Frame(self.details, bg=CARD)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=label, font=(FONT, 10), fg=MUTED, bg=CARD).pack(side="left")
            tk.Label(row, text=value, font=(FONT, 10, "bold"), fg=TEXT,
                     bg=CARD).pack(side="right")

    def clear(self):
        self.score_label.config(text="--", fg=PENDING)
        self.badge.config(text="-", bg=PENDING)
        self.note.config(text="")
        self.bar.set(0, PENDING)
        self._clear_details()


CONVENTIONAL_FIELDS = [
    ("conEquipCost", "Machine purchase price or remaining loan ($)", int, None),
    ("conSubCost", "Subscriptions: autosteer, RTK, etc. ($)", int, None),
    ("conNumMachines", "Number of machines that could be autonomy enabled", int, None),
    ("conLaborRate", "Labor rate ($/hr)", float, None),
]

AUTONOMY_FIELDS = [
    ("autEquipCost", "Machine purchase price or remaining loan ($)", int, None),
    ("autRetroCost", "Autonomy kit install cost, blank if none ($)", int, 0),
    ("autSubCost", "Required subscriptions: autosteer, RTK, etc. ($)", int, None),
    ("autNumMachines", "Number of machines that could be autonomy enabled", int, None),
    ("autMachinesPerOperator", "Machines per operator", int, None),
    ("dedicatedTimeRaw", "% of time operator spends supervising (1-100)", int, None),
    ("autLaborRate", "Labor rate ($/hr)", float, None),
]


class ROIApp(tk.Toplevel):
    def __init__(self, master, on_result=None, font_family="Segoe UI"):
        super().__init__(master)
        self.title("Autonomy ROI Calculator")
        self.geometry("900x700")
        self.resizable(False, False)
        self.configure(bg=BG)
        self.entries = {}
        self.result_vars = {}
        self.payback_years = None
        self.on_result = on_result
        self.font_family = font_family
        self._setup_styles()

        header = tk.Frame(self, bg=HEADER)
        header.pack(fill="x")
        tk.Label(header, text="Autonomy ROI Calculator",
                 font=(font_family, 20, "bold"), fg="white", bg=HEADER).pack(
                     anchor="w", padx=24, pady=(14, 2)
                 )
        tk.Label(header, text="Compare conventional and autonomous operating costs",
                 font=(font_family, 10), fg=HEADER_SUB, bg=HEADER).pack(
                     anchor="w", padx=24, pady=(0, 14)
                 )

        frame = tk.Frame(self, bg=BG)
        frame.pack(fill="both", expand=True, padx=18, pady=16)
        self._build_section(frame, "Conventional Operations", CONVENTIONAL_FIELDS, 0)
        self._build_section(frame, "Autonomous Operations", AUTONOMY_FIELDS, 1)
        self._build_results(frame)

        btns = tk.Frame(frame, bg=BG)
        btns.grid(row=2, column=0, columnspan=2, pady=(10, 0))
        ttk.Button(btns, text="Calculate", style="ROI.Accent.TButton",
                   command=self.calculate).pack(side="left", padx=4)
        ttk.Button(btns, text="Clear", style="ROI.Secondary.TButton",
                   command=self.clear).pack(side="left", padx=4)

    def _setup_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("ROI.TLabelframe", background=CARD, bordercolor=BORDER,
                        relief="solid", padding=10)
        style.configure("ROI.TLabelframe.Label", background=CARD, foreground=MUTED,
                        font=(self.font_family, 10, "bold"))
        style.configure("ROI.TLabel", background=CARD, foreground=TEXT,
                        font=(self.font_family, 9))
        style.configure("ROI.TEntry", fieldbackground=CARD, padding=4)
        style.configure("ROI.Accent.TButton", font=(self.font_family, 10, "bold"),
                        foreground="white", background=ACCENT, borderwidth=0,
                        padding=(18, 8))
        style.map("ROI.Accent.TButton", background=[("active", ACCENT_DARK)])
        style.configure("ROI.Secondary.TButton", font=(self.font_family, 10),
                        foreground=TEXT, background="#E4E9E3", borderwidth=0,
                        padding=(14, 8))
        style.map("ROI.Secondary.TButton", background=[("active", "#D5DCD4")])

    def _build_section(self, parent, title, fields, col):
        box = ttk.LabelFrame(parent, text=title, style="ROI.TLabelframe")
        box.grid(row=0, column=col, padx=6, pady=6, sticky="nsew")
        for r, (key, label, _type, _default) in enumerate(fields):
            ttk.Label(box, text=label, wraplength=260, justify="left",
                      style="ROI.TLabel").grid(row=r, column=0, sticky="w", pady=3)
            entry = ttk.Entry(box, width=14, style="ROI.TEntry")
            entry.grid(row=r, column=1, padx=(8, 0), pady=3)
            self.entries[key] = entry

    def _build_results(self, parent):
        box = ttk.LabelFrame(parent, text="Analysis", style="ROI.TLabelframe")
        box.grid(row=1, column=0, columnspan=2, padx=6, pady=6, sticky="ew")
        rows = [
            ("conTotalCost", "Conventional total cost"),
            ("conLaborCost", "Conventional labor cost"),
            ("autTotalCost", "Autonomy total cost"),
            ("autLaborCost", "Autonomy labor cost"),
            ("savings", "Autonomy labor savings ($/hr)"),
            ("payback", "Payback period (hours)"),
            ("paybackWeeks", "Payback period (weeks, 40 hr/week)"),
            ("paybackYears", "Payback period (years)"),
        ]
        for r, (key, label) in enumerate(rows):
            ttk.Label(box, text=label + ":", style="ROI.TLabel").grid(
                row=r, column=0, sticky="w", pady=2
            )
            var = tk.StringVar(value="-")
            ttk.Label(box, textvariable=var, style="ROI.TLabel",
                      font=(self.font_family, 10, "bold")).grid(
                          row=r, column=1, sticky="e", padx=(20, 0)
                      )
            self.result_vars[key] = var

    def _read(self, fields):
        values = {}
        for key, label, typ, default in fields:
            text = self.entries[key].get().strip()
            if not text:
                if default is None:
                    raise ValueError(f"Missing value: {label}")
                values[key] = default
                continue
            try:
                values[key] = typ(text)
            except ValueError:
                raise ValueError(f"Invalid number for: {label}")
        return values

    def calculate(self):
        try:
            con = self._read(CONVENTIONAL_FIELDS)
            aut = self._read(AUTONOMY_FIELDS)
        except ValueError as e:
            messagebox.showerror("Input error", str(e), parent=self)
            return

        if not 1 <= aut["dedicatedTimeRaw"] <= 100:
            messagebox.showerror("Input error", "Supervising time must be between 1 and 100.",
                                 parent=self)
            return
        if aut["autNumMachines"] == 0:
            messagebox.showerror("Input error", "Number of machines cannot be 0.", parent=self)
            return

        conTotalCost = con["conEquipCost"] + con["conSubCost"]
        conLaborCost = con["conNumMachines"] * con["conLaborRate"]
        autTotalCost = aut["autEquipCost"] + aut["autRetroCost"] + aut["autSubCost"]
        dedicatedTime = aut["dedicatedTimeRaw"] / 100
        autLaborCost = (aut["autLaborRate"] * dedicatedTime) / aut["autNumMachines"]

        savings = conLaborCost - autLaborCost
        costdif = autTotalCost - conTotalCost

        self.result_vars["conTotalCost"].set(f"${conTotalCost:,.2f}")
        self.result_vars["conLaborCost"].set(f"${conLaborCost:,.2f}")
        self.result_vars["autTotalCost"].set(f"${autTotalCost:,.2f}")
        self.result_vars["autLaborCost"].set(f"${autLaborCost:,.2f}")
        self.result_vars["savings"].set(f"${savings:,.2f}")

        if savings <= 0:
            self.payback_years = None
            self.result_vars["payback"].set("Never pays back")
            self.result_vars["paybackWeeks"].set("Never pays back")
            self.result_vars["paybackYears"].set("Never pays back")
        else:
            paybackHours = costdif / savings
            paybackWeeks = paybackHours / 40
            self.payback_years = paybackWeeks / 52
            self.result_vars["payback"].set(f"{paybackHours:,.1f} hours")
            self.result_vars["paybackWeeks"].set(f"{paybackWeeks:,.1f} weeks")
            self.result_vars["paybackYears"].set(f"{self.payback_years:,.1f} years")

        if self.on_result:
            self.on_result(self.payback_years, True)

    def clear(self):
        for entry in self.entries.values():
            entry.delete(0, tk.END)
        for var in self.result_vars.values():
            var.set("-")
        self.payback_years = None
        if self.on_result:
            self.on_result(None, False)


class RateOnomyApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RateOnomy - Autonomy Decision Support")
        self.geometry("980x760")
        self.minsize(920, 720)
        self.configure(bg=BG)
        global FONT
        FONT = pick_font()
        setup_styles(self)
        self.roi_window = None
        self.roi_payback_years = None
        self.roi_calculated = False

        # Header banner
        header = tk.Frame(self, bg=HEADER)
        header.pack(fill="x")
        tk.Label(header, text="RateOnomy", font=(FONT, 22, "bold"),
                 fg="white", bg=HEADER).pack(anchor="w", padx=24, pady=(14, 0))
        tk.Label(header, text="Decision support for adopting autonomous field equipment",
                 font=(FONT, 11), fg=HEADER_SUB, bg=HEADER).pack(anchor="w", padx=24, pady=(0, 14))

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=20, pady=16)

        # Controls card: image picker on the left, actions on the right
        self.image_path = None
        self.image_var = tk.StringVar(value="No field image selected")
        controls = make_card(body)
        controls.pack(fill="x")
        cin = tk.Frame(controls, bg=CARD)
        cin.pack(fill="x", padx=16, pady=12)
        self.roi_btn = ttk.Button(cin, text="ROI Calculator",
                      style="Secondary.TButton",
                      command=self.open_roi_calculator)
        self.roi_btn.pack(side="left", padx=(0, 8))
        self.pick_btn = ttk.Button(cin, text="Select Field Image...",
                                   style="Secondary.TButton", command=self.select_image)
        self.pick_btn.pack(side="left")
        tk.Label(cin, textvariable=self.image_var, font=(FONT, 10), fg=MUTED,
                 bg=CARD).pack(side="left", padx=12)
        self.run_btn = ttk.Button(cin, text="Run Assessment", style="Accent.TButton",
                                  command=self.run_assessment)
        self.run_btn.pack(side="right")
        self.clear_btn = ttk.Button(cin, text="Clear", style="Secondary.TButton",
                                    command=self.clear)
        self.clear_btn.pack(side="right", padx=(0, 10))

        size_row = tk.Frame(controls, bg=CARD)
        size_row.pack(fill="x", padx=16, pady=(0, 12))
        tk.Label(size_row, text="Field size (acres)", font=(FONT, 10),
             fg=TEXT, bg=CARD).pack(side="left")
        self.field_area_entry = ttk.Entry(size_row, width=12)
        self.field_area_entry.pack(side="left", padx=(8, 0))

        # Loading indicator (only packed while an assessment is running)
        self._running = False
        self._worker = None
        self._outcome = {}
        self.status_frame = tk.Frame(cin, bg=CARD)
        self.spinner = Spinner(self.status_frame)
        self.spinner.pack(side="left")
        tk.Label(self.status_frame, text="Running assessment...", font=(FONT, 10),
                 fg=MUTED, bg=CARD).pack(side="left", padx=(8, 0))

        # Three component panels
        panels = tk.Frame(body, bg=BG)
        panels.pack(fill="both", expand=True, pady=(12, 0))
        for i in range(3):
            panels.columnconfigure(i, weight=1, uniform="p")
        panels.rowconfigure(0, weight=1)

        self.roi_panel = ScorePanel(panels, "roi", "ROI", SUMMARIES["roi"])
        self.driv_panel = ScorePanel(panels, "drivability", "Drivability",
                                     SUMMARIES["drivability"])
        self.legal_panel = ScorePanel(panels, "legal", "Legal", SUMMARIES["legal"])
        all_panels = (self.roi_panel, self.driv_panel, self.legal_panel)
        for col, panel in enumerate(all_panels):
            panel.grid(row=0, column=col, sticky="nsew",
                       padx=(0 if col == 0 else 6, 0 if col == 2 else 6))

        # Give every summary the height of the tallest one so the fields
        # below them start at the same vertical position.
        self.update_idletasks()
        tallest = max(p.summary_label.winfo_reqheight() for p in all_panels)
        for panel in all_panels:
            panel.set_summary_height(tallest)

        # Composite card
        comp = make_card(body)
        comp.pack(fill="x", pady=(12, 0))
        cin = tk.Frame(comp, bg=CARD)
        cin.pack(fill="x", padx=20, pady=14)

        left = tk.Frame(cin, bg=CARD)
        left.pack(side="left", padx=(0, 28))
        tk.Label(left, text="COMPOSITE AUTONOMY SCORE", font=(FONT, 10, "bold"),
                 fg=MUTED, bg=CARD).pack(anchor="w")
        score_row = tk.Frame(left, bg=CARD)
        score_row.pack(anchor="w")
        self.composite_label = tk.Label(score_row, text="--", font=(FONT, 40, "bold"),
                                        fg=PENDING, bg=CARD)
        self.composite_label.pack(side="left")
        tk.Label(score_row, text="/ 10", font=(FONT, 14), fg=MUTED,
                 bg=CARD).pack(side="left", anchor="s", pady=(0, 10), padx=(4, 12))
        self.composite_badge = tk.Label(score_row, text="-", font=(FONT, 14, "bold"),
                                        fg="white", bg=PENDING, width=3)
        self.composite_badge.pack(side="left", anchor="center")

        right = tk.Frame(cin, bg=CARD)
        right.pack(side="left", fill="x", expand=True)
        self.composite_bar = GradeBar(right)
        self.composite_bar.pack(fill="x", pady=(4, 6))
        self.feasibility_label = tk.Label(right, text="", font=(FONT, 10, "bold"),
                                          fg="white", bg=CARD)
        self.feasibility_label.pack(anchor="w", pady=(0, 6))
        tk.Label(
            right,
            text="Weighted average of the three scores. Legal feasibility is reported "
                 "separately so a high composite cannot hide a legal restriction.",
            font=(FONT, 9), fg=MUTED, bg=CARD, wraplength=520, justify="left",
        ).pack(anchor="w")

    def open_roi_calculator(self):
        if self.roi_window is not None and self.roi_window.winfo_exists():
            self.roi_window.lift()
            self.roi_window.focus_force()
            return
        self.roi_window = ROIApp(
            self, self._on_roi_result, font_family=FONT
        )
        self.roi_window.transient(self)

    def _on_roi_result(self, payback_years, calculated):
        self.roi_payback_years = payback_years
        self.roi_calculated = calculated
        if calculated:
            result = compute_roi_score(payback_years)
            self.roi_panel.update_result(result, grade_from_score(result["score"]))
        else:
            self.roi_panel.clear()

    def select_image(self):
        path = filedialog.askopenfilename(
            title="Select satellite image of field",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.tif *.tiff"), ("All files", "*.*")],
        )
        if path:
            self.image_path = path
            self.image_var.set(os.path.basename(path))
            self.field_area_entry.delete(0, tk.END)

    def _set_running(self, running):
        self._running = running
        state = "disabled" if running else "normal"
        for btn in (self.run_btn, self.clear_btn, self.pick_btn, self.roi_btn,
                self.field_area_entry):
            btn.config(state=state)
        if running:
            self.status_frame.pack(side="right", padx=(0, 16))
            self.spinner.start()
            self.config(cursor="watch")
        else:
            self.spinner.stop()
            self.status_frame.pack_forget()
            self.config(cursor="")

    def run_assessment(self):
        if self._running:
            return
        if not self.roi_calculated:
            messagebox.showerror(
                "ROI required",
                "Open the ROI Calculator and click Calculate before running the assessment.",
            )
            return
        image_path = self.image_path
        roi_payback_years = self.roi_payback_years
        field_area = None
        if image_path:
            try:
                field_area = float(self.field_area_entry.get().strip())
            except ValueError:
                messagebox.showerror("Input error", "Enter the field size in acres.")
                return
            if not math.isfinite(field_area) or field_area <= 0:
                messagebox.showerror("Input error", "Field size must be greater than 0 acres.")
                return

        self._set_running(True)
        self._outcome = {}

        def work():
            # Runs off the UI thread so the spinner keeps animating.
            # Do not touch any tkinter widgets in here.
            try:
                self._outcome["roi"] = compute_roi_score(roi_payback_years)
                self._outcome["driv"] = compute_drivability_score(image_path, field_area)
                self._outcome["legal"] = compute_legal_score()
            except Exception as exc:
                self._outcome["error"] = exc

        self._worker = threading.Thread(target=work, daemon=True)
        self._worker.start()
        self.after(100, self._poll_worker)

    def _poll_worker(self):
        if self._worker.is_alive():
            self.after(100, self._poll_worker)
            return
        self._set_running(False)
        if "error" in self._outcome:
            messagebox.showerror("Assessment error", str(self._outcome["error"]))
            return
        self._show_results(self._outcome["roi"], self._outcome["driv"],
                           self._outcome["legal"])

    def _show_results(self, roi, driv, legal):
        self.roi_panel.update_result(roi, grade_from_score(roi["score"]))
        self.driv_panel.update_result(driv, grade_from_score(driv["score"]))
        self.legal_panel.update_result(legal, legal["grade"], legal["label"])

        composite = compute_composite(roi["score"], driv["score"], legal["score"])
        grade = grade_from_score(composite)
        color = GRADE_COLORS[grade]
        self.composite_label.config(text=f"{composite:.1f}", fg=color)
        self.composite_badge.config(text=grade, bg=color)
        self.composite_bar.set(composite, color)

        feasible = legal["feasible"]
        pill_color = ACCENT if feasible.lower().startswith("legally feasible") else DANGER
        self.feasibility_label.config(text=f"  {feasible}  ", bg=pill_color)

    def clear(self):
        if self.roi_window is not None and self.roi_window.winfo_exists():
            self.roi_window.clear()
        else:
            self._on_roi_result(None, False)
        for panel in (self.roi_panel, self.driv_panel, self.legal_panel):
            panel.clear()
        self.composite_label.config(text="--", fg=PENDING)
        self.composite_badge.config(text="-", bg=PENDING)
        self.composite_bar.set(0, PENDING)
        self.feasibility_label.config(text="", bg=CARD)


if __name__ == "__main__":
    RateOnomyApp().mainloop()
