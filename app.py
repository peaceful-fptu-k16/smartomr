"""
SmartOMR - Optical Mark Recognition Desktop App
================================================
A GUI application for automatic grading of multiple-choice answer sheets.
Supports 120-question sheets (4 columns × 30 rows, choices A/B/C/D).

Run: python app.py
"""

import os
import sys
import math
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from functools import partial

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from smart_omr import run as omr_run, NUM_QUESTIONS
from modules.grader import (
    load_answer_key, save_answer_key, grade,
    draw_graded_annotated, CHOICE_LABELS
)


# ─────────────────────────────────────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────────────────────────────────────
class T:
    # Backgrounds
    BG        = "#0d1017"
    SURFACE   = "#151922"
    CARD      = "#1c2030"
    ELEVATED  = "#272d42"
    INPUT     = "#12151f"
    BORDER    = "#2a3050"

    # Accent
    PRIMARY   = "#7c6fff"
    PRIMARY_D = "#5b52cc"
    SUCCESS   = "#00d68f"
    SUCCESS_D = "#00a86e"
    DANGER    = "#ff5370"
    WARNING   = "#ffb74d"
    INFO      = "#64b5f6"

    # Text
    FG        = "#e8eaf6"
    FG2       = "#8f94b2"
    FG3       = "#4a4f6e"

    # Answer tag colors
    CLR_CORRECT = "#00e396"
    CLR_WRONG   = "#ff6178"
    CLR_BLANK   = "#ffb74d"
    CLR_NOKEY   = "#4a4f6e"

    # Fonts
    H1        = ("Segoe UI", 14, "bold")
    H2        = ("Segoe UI", 12, "bold")
    H3        = ("Segoe UI", 10, "bold")
    BODY      = ("Segoe UI", 10)
    SMALL     = ("Segoe UI", 9)
    MONO      = ("Consolas", 10)
    MONO_SM   = ("Consolas", 9)
    SCORE_XL  = ("Segoe UI", 42, "bold")
    SCORE_MD  = ("Segoe UI", 20, "bold")
    SCORE_SM  = ("Segoe UI", 11)



RESULT_LABELS = {
    "correct": "✓  Correct",
    "wrong":   "✗  Wrong",
    "blank":   "○  Blank",
    "no_key":  "—",
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _lighten(hex_color, factor=0.18):
    """Lighten a hex color by the given factor."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r = min(255, int(r + (255 - r) * factor))
    g = min(255, int(g + (255 - g) * factor))
    b = min(255, int(b + (255 - b) * factor))
    return f"#{r:02x}{g:02x}{b:02x}"


# ─────────────────────────────────────────────────────────────────────────────
# WIDGET HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _btn(parent, text, cmd, bg=T.PRIMARY, fg=T.FG, font=T.BODY,
         px=14, py=7, hover_bg=None, **kw):
    hbg = hover_bg or _lighten(bg)
    b = tk.Button(parent, text=text, command=cmd, bg=bg, fg=fg, font=font,
                  relief="flat", bd=0, cursor="hand2", padx=px, pady=py,
                  activebackground=hbg, activeforeground=fg, **kw)
    _orig_bg = bg
    b.bind("<Enter>", lambda e: b.config(bg=hbg))
    b.bind("<Leave>", lambda e: b.config(bg=_orig_bg))
    return b


def _lbl(parent, text, font=T.BODY, fg=T.FG, bg=None, **kw):
    return tk.Label(parent, text=text, font=font, fg=fg,
                    bg=bg or T.SURFACE, **kw)


def _inp(parent, var, width=22, **kw):
    return tk.Entry(parent, textvariable=var, width=width,
                    bg=T.INPUT, fg=T.FG, insertbackground=T.FG,
                    relief="flat", font=T.BODY,
                    highlightthickness=1,
                    highlightcolor=T.PRIMARY,
                    highlightbackground=T.BORDER, **kw)


def _badge(parent, text, color, font=None):
    font = font or ("Segoe UI", 8)
    return tk.Label(parent, text=f"  {text}  ", font=font,
                    fg=T.BG, bg=color, relief="flat", padx=3, pady=1)


def _card(parent, **kw):
    return tk.Frame(parent, bg=T.CARD, bd=0,
                    highlightthickness=1, highlightbackground=T.BORDER, **kw)


def _section(parent, icon, title, color=T.PRIMARY):
    """Section header with colored left accent bar."""
    wrap = tk.Frame(parent, bg=T.SURFACE)
    wrap.pack(fill="x", padx=14, pady=(16, 6))
    # Accent bar
    tk.Frame(wrap, bg=color, width=3).pack(side="left", fill="y", padx=(0, 10))
    tk.Label(wrap, text=f"{icon}  {title}", font=T.H3, fg=T.FG2,
             bg=T.SURFACE).pack(side="left", anchor="w")


# ─────────────────────────────────────────────────────────────────────────────
# CIRCULAR SCORE RING
# ─────────────────────────────────────────────────────────────────────────────

class ScoreRing(tk.Canvas):
    """A circular progress ring that displays a percentage."""

    SIZE = 130
    LW   = 8

    def __init__(self, parent, **kw):
        super().__init__(parent, width=self.SIZE, height=self.SIZE,
                         bg=T.CARD, highlightthickness=0, **kw)
        self._score_text = None
        self._sub_text   = None
        self._arc_bg     = None
        self._arc_fg     = None
        self._draw_empty()

    def _draw_empty(self):
        cx, cy = self.SIZE // 2, self.SIZE // 2
        r = (self.SIZE - self.LW) // 2 - 4
        self._arc_bg = self.create_oval(
            cx - r, cy - r, cx + r, cy + r,
            outline=T.ELEVATED, width=self.LW)
        self._arc_fg = self.create_arc(
            cx - r, cy - r, cx + r, cy + r,
            start=90, extent=0, outline=T.FG3,
            width=self.LW, style="arc")
        self._score_text = self.create_text(
            cx, cy - 6, text="—", font=T.SCORE_MD,
            fill=T.FG3, anchor="center")
        self._sub_text = self.create_text(
            cx, cy + 18, text="no result", font=("Segoe UI", 8),
            fill=T.FG3, anchor="center")

    def update_score(self, score=None, score_max=None, pct=None):
        cx, cy = self.SIZE // 2, self.SIZE // 2
        r = (self.SIZE - self.LW) // 2 - 4

        if score is None:
            self.delete(self._arc_fg)
            self._arc_fg = self.create_arc(
                cx - r, cy - r, cx + r, cy + r,
                start=90, extent=0, outline=T.FG3,
                width=self.LW, style="arc")
            self.delete(self._score_text)
            self._score_text = self.create_text(
                cx, cy - 6, text="—", font=T.SCORE_MD,
                fill=T.FG3, anchor="center")
            self.delete(self._sub_text)
            self._sub_text = self.create_text(
                cx, cy + 18, text="no result", font=("Segoe UI", 8),
                fill=T.FG3, anchor="center")
            return

        color = T.CLR_CORRECT if pct >= 50 else T.CLR_WRONG
        extent = -(pct / 100) * 360

        self.delete(self._arc_fg)
        self._arc_fg = self.create_arc(
            cx - r, cy - r, cx + r, cy + r,
            start=90, extent=extent, outline=color,
            width=self.LW, style="arc")

        self.delete(self._score_text)
        self._score_text = self.create_text(
            cx, cy - 6, text=f"{pct}%", font=T.SCORE_MD,
            fill=color, anchor="center")

        self.delete(self._sub_text)
        self._sub_text = self.create_text(
            cx, cy + 18, text=f"{score:.2f} / {score_max}",
            font=("Segoe UI", 8), fill=T.FG2, anchor="center")


# ─────────────────────────────────────────────────────────────────────────────
# ANSWER KEY EDITOR DIALOG
# ─────────────────────────────────────────────────────────────────────────────

class KeyEditor(tk.Toplevel):
    """Interactive answer-key editor."""

    def __init__(self, parent, filepath=None, n_questions=120, callback=None):
        super().__init__(parent)
        self.title("Answer Key Editor — SmartOMR")
        self.configure(bg=T.BG)
        self.resizable(True, True)
        self.callback  = callback
        self.filepath  = filepath or ""
        self.n_questions = n_questions

        W, H = 880, 680
        sx = (self.winfo_screenwidth()  - W) // 2
        sy = (self.winfo_screenheight() - H) // 2
        self.geometry(f"{W}x{H}+{sx}+{sy}")
        self.grab_set()

        self._vars = {}
        self._cfg  = dict(
            name          = tk.StringVar(value="Exam"),
            total_score   = tk.StringVar(value="10"),
            correct_score = tk.StringVar(value="0.0833"),
            wrong_penalty = tk.StringVar(value="0"),
        )
        self._build()
        if filepath and os.path.isfile(filepath):
            self._load(filepath)

    # ── layout ──────────────────────────────────────────────────────

    def _build(self):
        # Header
        hdr = tk.Frame(self, bg=T.SURFACE, pady=12)
        hdr.pack(fill="x")
        tk.Label(hdr, text="  ✏  Answer Key Editor",
                 font=T.H1, fg=T.FG, bg=T.SURFACE).pack(side="left", padx=14)
        btns = tk.Frame(hdr, bg=T.SURFACE)
        btns.pack(side="right", padx=14)
        _btn(btns, "📂 Open…",  self._browse,  T.ELEVATED, T.FG2, T.SMALL, px=10, py=5).pack(side="left", padx=3)
        _btn(btns, "💾 Save",   self._save,    T.SUCCESS, fg=T.BG, px=12, py=5).pack(side="left", padx=3)
        _btn(btns, "Save As…",  self._save_as, T.ELEVATED, T.FG2, T.SMALL, px=10, py=5).pack(side="left", padx=3)

        # Config strip
        cfg = tk.Frame(self, bg=T.CARD, pady=10)
        cfg.pack(fill="x")
        for col, (key, label) in enumerate([
            ("name", "Exam name"),
            ("total_score", "Total pts"),
            ("correct_score", "Per correct"),
            ("wrong_penalty", "Penalty/wrong"),
        ]):
            tk.Label(cfg, text=label, font=T.SMALL, fg=T.FG2, bg=T.CARD
                     ).grid(row=0, column=col * 2, padx=(18, 4), sticky="e")
            _inp(cfg, self._cfg[key], width=14 if key == "name" else 8
                 ).grid(row=0, column=col * 2 + 1, padx=(0, 6), ipady=3)

        # Scrollable grid
        wrap = tk.Frame(self, bg=T.BG)
        wrap.pack(fill="both", expand=True)
        cvs = tk.Canvas(wrap, bg=T.BG, highlightthickness=0)
        sb  = ttk.Scrollbar(wrap, orient="vertical", command=cvs.yview)
        cvs.configure(yscrollcommand=sb.set)
        cvs.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        inner = tk.Frame(cvs, bg=T.BG)
        cvs.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: cvs.configure(scrollregion=cvs.bbox("all")))
        cvs.bind_all("<MouseWheel>",
                     lambda e: cvs.yview_scroll(-1 * (e.delta // 120), "units"))
        self._build_grid(inner)

        # Footer
        foot = tk.Frame(self, bg=T.SURFACE, pady=10)
        foot.pack(fill="x")
        _btn(foot, "Clear All", self._clear_all, T.ELEVATED, T.FG2, T.SMALL,
             px=10, py=4).pack(side="left", padx=14)
        _btn(foot, "Close", self.destroy, T.ELEVATED, px=10, py=4
             ).pack(side="right", padx=12)
        if self.callback:
            _btn(foot, "✓  Apply & Close", self._apply, T.SUCCESS, fg=T.BG, px=14, py=5
                 ).pack(side="right", padx=4)

    def _build_grid(self, parent):
        N   = 4
        per = (self.n_questions + N - 1) // N
        wrap = tk.Frame(parent, bg=T.BG)
        wrap.pack(padx=12, pady=12, fill="x")

        for blk in range(N):
            start = blk * per + 1
            end   = min((blk + 1) * per, self.n_questions)
            col_f = tk.Frame(wrap, bg=T.CARD,
                             highlightthickness=1, highlightbackground=T.BORDER)
            col_f.grid(row=0, column=blk, padx=6, sticky="n")
            # column title
            tk.Label(col_f, text=f"  Q {start} – {end}",
                     font=T.H3, fg=T.PRIMARY, bg=T.CARD, pady=8
                     ).grid(row=0, column=0, columnspan=7, sticky="ew")
            tk.Frame(col_f, bg=T.BORDER, height=1
                     ).grid(row=1, column=0, columnspan=7, sticky="ew")

            for ri in range(per):
                q = blk * per + ri + 1
                if q > self.n_questions:
                    break
                var = tk.StringVar(value="")
                self._vars[q] = var

                row_bg = T.CARD if ri % 2 == 0 else _lighten(T.CARD, 0.04)

                tk.Label(col_f, text=f"{q:3d}", font=T.MONO_SM,
                         fg=T.FG3, bg=row_bg, width=3, anchor="e"
                         ).grid(row=ri + 2, column=0, padx=(8, 4), pady=1, sticky="ew")

                for ci, ch in enumerate(CHOICE_LABELS):
                    rb = tk.Radiobutton(
                        col_f, text=ch, variable=var, value=ch,
                        font=("Segoe UI", 9, "bold"),
                        fg=T.FG2, bg=row_bg,
                        selectcolor=T.PRIMARY,
                        activebackground=T.CARD, activeforeground=T.FG,
                        indicatoron=False, relief="flat",
                        padx=5, pady=2, cursor="hand2", bd=0, width=2,
                    )
                    rb.grid(row=ri + 2, column=ci + 1, padx=1, pady=1, sticky="ew")

                tk.Button(
                    col_f, text="×", font=T.SMALL, fg=T.FG3, bg=row_bg,
                    relief="flat", bd=0, cursor="hand2",
                    command=partial(self._clear_q, q),
                ).grid(row=ri + 2, column=5, padx=(2, 8), sticky="ew")

    # ── data helpers ────────────────────────────────────────────────

    def _clear_q(self, q):   self._vars[q].set("")
    def _clear_all(self):
        for v in self._vars.values(): v.set("")

    def _answers(self):
        return {q: (v.get() or None) for q, v in self._vars.items()}

    def _config(self):
        def f(k, d):
            try: return float(self._cfg[k].get())
            except: return d
        return {
            "name":          self._cfg["name"].get(),
            "total_score":   f("total_score", 10.0),
            "correct_score": f("correct_score", 0.0833),
            "wrong_penalty": f("wrong_penalty", 0.0),
        }

    # ── I/O ─────────────────────────────────────────────────────────

    def _load(self, fp):
        try:
            data = load_answer_key(fp)
            for q, v in data["answers"].items():
                if q in self._vars:
                    self._vars[q].set(v or "")
            c = data.get("config", {})
            if "name"          in c: self._cfg["name"].set(c["name"])
            if "total_score"   in c: self._cfg["total_score"].set(str(c["total_score"]))
            if "correct_score" in c and c["correct_score"]:
                self._cfg["correct_score"].set(str(round(c["correct_score"], 4)))
            if "wrong_penalty" in c:
                self._cfg["wrong_penalty"].set(str(c["wrong_penalty"]))
        except Exception as e:
            messagebox.showerror("Load Error", str(e), parent=self)

    def _browse(self):
        fp = filedialog.askopenfilename(
            title="Open answer key",
            filetypes=[("Text", "*.txt"), ("JSON", "*.json"), ("All", "*.*")],
            parent=self)
        if fp:
            self.filepath = fp
            self._load(fp)

    def _save(self):
        if not self.filepath:
            self._save_as()
        else:
            self._do_save(self.filepath)

    def _save_as(self):
        fp = filedialog.asksaveasfilename(
            title="Save answer key",
            defaultextension=".txt",
            filetypes=[("Text", "*.txt"), ("JSON", "*.json")],
            initialdir=os.path.join(HERE, "answer_keys"),
            parent=self)
        if fp:
            self.filepath = fp
            self._do_save(fp)

    def _do_save(self, path):
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            save_answer_key(self._answers(), path, config=self._config())
            messagebox.showinfo("Saved", f"Answer key saved:\n{path}", parent=self)
        except Exception as e:
            messagebox.showerror("Save Error", str(e), parent=self)

    def _apply(self):
        if self.callback:
            self.callback(self.filepath)
        self.destroy()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("SmartOMR — Automatic Exam Grader")
        self.configure(bg=T.BG)

        W, H = 1340, 860
        self.minsize(1000, 660)
        sx = (self.winfo_screenwidth()  - W) // 2
        sy = (self.winfo_screenheight() - H) // 2
        self.geometry(f"{W}x{H}+{sx}+{sy}")

        # State
        self.v_image          = tk.StringVar()
        self.v_key            = tk.StringVar()
        self.v_status         = tk.StringVar(value="Ready · select an image to begin")
        self.v_grading_method = tk.StringVar(value="standard")

        self._result  = None
        self._grading = None
        self._img_pil = None
        self._photo   = None
        self._scale   = 1.0

        # Step viewer state
        self._steps       = []   # list of (name, desc, cv2_bgr_img)
        self._step_idx    = 0
        self._step_pil    = None
        self._step_photo  = None
        self._step_scale  = 1.0

        self._apply_styles()
        self._build()
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    # ─── TTK STYLES ─────────────────────────────────────────────────────────

    def _apply_styles(self):
        s = ttk.Style(self)
        s.theme_use("clam")

        _TAB_PAD = [32, 10]
        s.configure("Tab.TNotebook",
                    background=T.SURFACE, borderwidth=0,
                    tabmargins=[0, 0, 0, 0])
        s.configure("Tab.TNotebook.Tab",
                    background=T.ELEVATED, foreground=T.FG2,
                    padding=_TAB_PAD, font=T.BODY, borderwidth=0,
                    lightcolor=T.ELEVATED, darkcolor=T.ELEVATED,
                    bordercolor=T.BG, focuscolor=T.ELEVATED)
        s.map("Tab.TNotebook.Tab",
              background=[("selected", T.PRIMARY),   ("!selected", T.ELEVATED)],
              foreground=[("selected", T.FG),         ("!selected", T.FG2)],
              focuscolor=[("selected", T.PRIMARY),    ("!selected", T.ELEVATED)],
              padding=   [("selected", _TAB_PAD),     ("!selected", _TAB_PAD)])
        # Remove the dashed focus rectangle from tabs
        s.layout("Tab.TNotebook.Tab", [
            ("Notebook.tab", {"children": [
                ("Notebook.padding", {"side": "top", "children": [
                    ("Notebook.label", {"side": "top", "sticky": ""})
                ], "sticky": "nswe"})
            ], "sticky": "nswe"})
        ])

        s.configure("Grid.Treeview",
                    background=T.CARD, foreground=T.FG,
                    fieldbackground=T.CARD, font=T.MONO_SM,
                    rowheight=28, borderwidth=0)
        s.configure("Grid.Treeview.Heading",
                    background=T.ELEVATED, foreground=T.FG2,
                    font=("Segoe UI", 9, "bold"), relief="flat", borderwidth=0)
        s.map("Grid.Treeview",
              background=[("selected", T.PRIMARY_D)],
              foreground=[("selected", T.FG)])

        s.configure("S.Vertical.TScrollbar",
                    troughcolor=T.BG, background=T.ELEVATED,
                    borderwidth=0, arrowsize=11)
        s.configure("S.Horizontal.TScrollbar",
                    troughcolor=T.BG, background=T.ELEVATED,
                    borderwidth=0, arrowsize=11)

        s.configure("Run.Horizontal.TProgressbar",
                    troughcolor=T.CARD, background=T.PRIMARY,
                    borderwidth=0, lightcolor=T.PRIMARY, darkcolor=T.PRIMARY)

    # ─── BUILD ──────────────────────────────────────────────────────────────

    def _build(self):
        self._build_navbar()

        body = tk.Frame(self, bg=T.BG)
        body.pack(fill="both", expand=True)

        self._build_sidebar(body)
        tk.Frame(body, bg=T.BORDER, width=1).pack(side="left", fill="y")
        self._build_workspace(body)

        self._build_statusbar()

    # ── Nav bar ────────────────────────────────────────────────────

    def _build_navbar(self):
        nav = tk.Frame(self, bg=T.SURFACE, height=56)
        nav.pack(fill="x")
        nav.pack_propagate(False)

        # Logo
        logo = tk.Frame(nav, bg=T.SURFACE)
        logo.pack(side="left", padx=22, pady=10)
        tk.Label(logo, text="Smart", font=("Segoe UI", 20, "bold"),
                 fg=T.PRIMARY, bg=T.SURFACE).pack(side="left")
        tk.Label(logo, text="OMR",   font=("Segoe UI", 20, "bold"),
                 fg=T.SUCCESS, bg=T.SURFACE).pack(side="left")

        # Separator dot + subtitle
        tk.Label(nav, text="·", font=("Segoe UI", 16),
                 fg=T.FG3, bg=T.SURFACE).pack(side="left", padx=(0, 6))
        tk.Label(nav, text="Optical Mark Recognition",
                 font=("Segoe UI", 10), fg=T.FG3, bg=T.SURFACE).pack(side="left")

        # Version badge
        tk.Label(nav, text=" v3.0 ", font=("Segoe UI", 8),
                 fg=T.PRIMARY, bg=T.ELEVATED).pack(side="left", padx=10, pady=18)

        # Right action buttons
        right = tk.Frame(nav, bg=T.SURFACE)
        right.pack(side="right", padx=16, pady=12)
        _btn(right, "⊕  New Key",  self._create_key, T.ELEVATED, T.FG2, T.SMALL, px=12, py=6
             ).pack(side="left", padx=4)
        _btn(right, "✏  Edit Key", self._edit_key,   T.ELEVATED, T.FG2, T.SMALL, px=12, py=6
             ).pack(side="left", padx=4)

    # ── Sidebar ────────────────────────────────────────────────────

    def _build_sidebar(self, parent):
        sb_wrap = tk.Frame(parent, bg=T.SURFACE, width=290)
        sb_wrap.pack(side="left", fill="y")
        sb_wrap.pack_propagate(False)

        cvs = tk.Canvas(sb_wrap, bg=T.SURFACE, highlightthickness=0)
        scrollbar_y = ttk.Scrollbar(sb_wrap, orient="vertical", command=cvs.yview)
        scrollbar_x = ttk.Scrollbar(sb_wrap, orient="horizontal", command=cvs.xview)
        
        sb = tk.Frame(cvs, bg=T.SURFACE)
        
        sb.bind(
            "<Configure>",
            lambda e: cvs.configure(scrollregion=cvs.bbox("all"))
        )
        
        cvs.create_window((0, 0), window=sb, anchor="nw")
        cvs.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        
        scrollbar_x.pack(side="bottom", fill="x")
        scrollbar_y.pack(side="right", fill="y")
        cvs.pack(side="left", fill="both", expand=True)
        
        def _on_mousewheel_y(event):
            cvs.yview_scroll(int(-1*(event.delta/120)), "units")
        def _on_mousewheel_x(event):
            cvs.xview_scroll(int(-1*(event.delta/120)), "units")
            
        cvs.bind("<Enter>", lambda _: (cvs.bind_all("<MouseWheel>", _on_mousewheel_y), cvs.bind_all("<Shift-MouseWheel>", _on_mousewheel_x)))
        cvs.bind("<Leave>", lambda _: (cvs.unbind_all("<MouseWheel>"), cvs.unbind_all("<Shift-MouseWheel>")))

        # ─ Input Files ─
        _section(sb, "📂", "INPUT FILES", T.PRIMARY)
        self._add_file_row(sb, "Answer Sheet",
                           self.v_image, self._browse_image,
                           "image (JPG / PNG)")
        self._add_file_row(sb, "Answer Key",
                           self.v_key, self._browse_key,
                           "key file (TXT / JSON)", optional=True)



        # ─ Score Card ─
        _section(sb, "🏆", "SCORE", T.WARNING)
        self._build_scorecard(sb)

        # ─ Options ─
        _section(sb, "⚙️", "GRADING METHOD", T.PRIMARY)
        opts = tk.Frame(sb, bg=T.SURFACE)
        opts.pack(fill="x", padx=16, pady=(4, 10))
        tk.Radiobutton(opts, text="Standard Threshold", variable=self.v_grading_method, 
                       value="standard", bg=T.SURFACE, fg=T.FG, font=T.BODY, 
                       activebackground=T.SURFACE, activeforeground=T.FG, 
                       selectcolor=T.BG).pack(anchor="w", pady=2)
        tk.Radiobutton(opts, text="Crop & Clean Threshold", variable=self.v_grading_method, 
                       value="crop_clean", bg=T.SURFACE, fg=T.FG, font=T.BODY,
                       activebackground=T.SURFACE, activeforeground=T.FG, 
                       selectcolor=T.BG).pack(anchor="w", pady=2)

        # ─ Run ─
        run_wrap = tk.Frame(sb, bg=T.SURFACE, padx=18)
        run_wrap.pack(fill="x", pady=(16, 8))
        self._run_btn = _btn(run_wrap, "▶   Grade Now",
                             self._run, T.PRIMARY, T.FG, T.H2,
                             px=0, py=14, hover_bg=_lighten(T.PRIMARY, 0.22))
        self._run_btn.pack(fill="x")

        # ─ Export ─
        _section(sb, "💾", "EXPORT", T.FG3)
        ex = tk.Frame(sb, bg=T.SURFACE, padx=18)
        ex.pack(fill="x", pady=(2, 10))
        for text, cmd in [
            ("📄  Save Grading Report",  self._export_report),
            ("🖼  Save Graded Image",     self._export_image),
            ("📂  Open Output Folder",    self._open_output),
            ("📂  Mở thư mục 120 câu đã cắt", self._open_questions_folder),
        ]:
            _btn(ex, text, cmd, T.CARD, T.FG2, T.SMALL, px=12, py=7
                 ).pack(fill="x", pady=2)

    def _add_file_row(self, parent, label, var, cmd, hint, optional=False):
        f = tk.Frame(parent, bg=T.SURFACE)
        f.pack(fill="x", padx=18, pady=(4, 0))

        row_h = tk.Frame(f, bg=T.SURFACE)
        row_h.pack(fill="x")
        tk.Label(row_h, text=label, font=T.SMALL, fg=T.FG2,
                 bg=T.SURFACE).pack(side="left")
        if optional:
            _badge(row_h, "optional", T.FG3, ("Segoe UI", 7)
                   ).pack(side="left", padx=6)

        row = tk.Frame(f, bg=T.SURFACE, pady=3)
        row.pack(fill="x")
        _inp(row, var).pack(side="left", fill="x", expand=True, padx=(0, 5), ipady=4)
        _btn(row, "…", cmd, T.ELEVATED, T.FG2, T.SMALL, px=9, py=4
             ).pack(side="left")

        status = tk.Label(f, text=f"— no {hint} selected",
                          font=("Segoe UI", 8), fg=T.FG3, bg=T.SURFACE, anchor="w")
        status.pack(fill="x", pady=(0, 4))

        def _on(*_):
            v = var.get().strip()
            if v and os.path.isfile(v):
                name = os.path.basename(v)
                sz   = os.path.getsize(v)
                nice = f"{sz/1024:.0f} KB" if sz < 1_000_000 else f"{sz/1e6:.1f} MB"
                status.config(text=f"✓  {name}  ({nice})", fg=T.SUCCESS)
            elif v:
                status.config(text="✗  File not found", fg=T.DANGER)
            else:
                status.config(text=f"— no {hint} selected", fg=T.FG3)

        var.trace("w", _on)
        _on()

    def _build_scorecard(self, parent):
        sc = _card(parent)
        sc.pack(fill="x", padx=18, pady=6)

        # Score ring
        ring_frame = tk.Frame(sc, bg=T.CARD)
        ring_frame.pack(pady=(14, 6))
        self._score_ring = ScoreRing(ring_frame)
        self._score_ring.pack()

        # Stats row
        stats = tk.Frame(sc, bg=T.CARD)
        stats.pack(fill="x", padx=8, pady=(4, 0))
        stats.columnconfigure((0, 1, 2), weight=1)
        self._lbl_c = self._stat(stats, "Correct", T.CLR_CORRECT, 0)
        self._lbl_w = self._stat(stats, "Wrong",   T.CLR_WRONG,   1)
        self._lbl_b = self._stat(stats, "Blank",   T.CLR_BLANK,   2)

        # Progress bar
        bar_bg = tk.Frame(sc, bg=T.ELEVATED, height=4)
        bar_bg.pack(fill="x", padx=18, pady=(10, 16))
        bar_bg.pack_propagate(False)
        self._bar = tk.Frame(bar_bg, bg=T.FG3, height=4)
        self._bar.place(x=0, y=0, relheight=1, relwidth=0)

    def _stat(self, parent, title, color, col):
        f = tk.Frame(parent, bg=T.CARD)
        f.grid(row=0, column=col, padx=4, pady=6, sticky="ew")
        v = tk.Label(f, text="—", font=T.SCORE_MD, fg=color, bg=T.CARD)
        v.pack()
        tk.Label(f, text=title, font=T.SMALL, fg=T.FG3, bg=T.CARD).pack()
        return v

    def _update_scorecard(self, gr):
        if gr is None:
            self._score_ring.update_score()
            for w in (self._lbl_c, self._lbl_w, self._lbl_b):
                w.config(text="—")
            self._bar.place(relwidth=0)
            return
        color = T.CLR_CORRECT if gr.score_pct >= 50 else T.CLR_WRONG
        self._score_ring.update_score(gr.score, gr.score_max, gr.score_pct)
        self._lbl_c.config(text=str(gr.n_correct))
        self._lbl_w.config(text=str(gr.n_wrong))
        self._lbl_b.config(text=str(gr.n_blank))
        self._bar.config(bg=color)
        self._bar.place(relwidth=min(gr.score_pct / 100, 1.0))

    # ── Workspace ─────────────────────────────────────────────────

    def _build_workspace(self, parent):
        ws = tk.Frame(parent, bg=T.BG)
        ws.pack(side="left", fill="both", expand=True)

        self._nb = ttk.Notebook(ws, style="Tab.TNotebook")
        self._nb.pack(fill="both", expand=True)

        # Equal-width tab labels (center-padded to same length)
        _TAB_W = 18
        t1 = tk.Frame(self._nb, bg=T.BG)
        self._nb.add(t1, text=f"{'Result Image':^{_TAB_W}}")
        self._build_image_tab(t1)

        t4 = tk.Frame(self._nb, bg=T.BG)
        self._nb.add(t4, text=f"{'Processing Steps':^{_TAB_W}}")
        self._build_steps_tab(t4)

        t2 = tk.Frame(self._nb, bg=T.BG)
        self._nb.add(t2, text=f"{'Answer Details':^{_TAB_W}}")
        self._build_table_tab(t2)

        t3 = tk.Frame(self._nb, bg=T.BG)
        self._nb.add(t3, text=f"{'Console':^{_TAB_W}}")
        self._build_log_tab(t3)

    # ── Image tab ─────────────────────────────────────────────────

    def _build_image_tab(self, parent):
        # Toolbar
        tb = tk.Frame(parent, bg=T.SURFACE, height=46)
        tb.pack(fill="x")
        tb.pack_propagate(False)

        zt = tk.Frame(tb, bg=T.SURFACE)
        zt.pack(side="left", padx=16, pady=8)
        for text, cmd in [(" − ", self._zoom_out), ("⊡ Fit", self._zoom_fit), (" + ", self._zoom_in)]:
            b = _btn(zt, text, cmd, T.ELEVATED, T.FG, T.SMALL, px=10, py=4)
            b.config(height=1)
            b.pack(side="left", padx=2)
        self._lbl_zoom = tk.Label(tb, text="", font=T.SMALL, fg=T.FG3, bg=T.SURFACE)
        self._lbl_zoom.pack(side="left", padx=8)

        # Legend
        leg = tk.Frame(tb, bg=T.SURFACE)
        leg.pack(side="right", padx=16)
        for txt, col in [
            ("Correct",      T.CLR_CORRECT),
            ("Wrong choice", T.CLR_WRONG),
            ("Right answer", T.INFO),
            ("Blank",        T.CLR_BLANK),
        ]:
            dot = tk.Canvas(leg, width=10, height=10,
                            bg=T.SURFACE, highlightthickness=0)
            dot.create_oval(1, 1, 9, 9, fill=col, outline="")
            dot.pack(side="left", padx=(10, 2), pady=14)
            tk.Label(leg, text=txt, font=T.SMALL,
                     fg=T.FG2, bg=T.SURFACE).pack(side="left")

        # Canvas
        cf = tk.Frame(parent, bg="#080b12")
        cf.pack(fill="both", expand=True)

        self._cvs = tk.Canvas(cf, bg="#080b12", cursor="fleur",
                               highlightthickness=0)
        sbv = ttk.Scrollbar(cf, orient="vertical",   command=self._cvs.yview,
                            style="S.Vertical.TScrollbar")
        sbh = ttk.Scrollbar(cf, orient="horizontal", command=self._cvs.xview,
                            style="S.Horizontal.TScrollbar")
        self._cvs.configure(yscrollcommand=sbv.set, xscrollcommand=sbh.set)
        sbv.pack(side="right",  fill="y")
        sbh.pack(side="bottom", fill="x")
        self._cvs.pack(fill="both", expand=True)

        # Empty state prompt
        self._prompt_id = self._cvs.create_text(
            400, 260, text="📷",
            font=("Segoe UI", 48), fill=T.FG3, tags="prompt")
        self._cvs.create_text(
            400, 330,
            text="Select an answer sheet and press  ▶ Grade Now",
            font=("Segoe UI", 12), fill=T.FG3, tags="prompt")
        self._cvs.create_text(
            400, 358,
            text="Supports JPG, PNG  ·  120 questions  ·  4 choices A B C D",
            font=("Segoe UI", 9), fill=T.FG3, tags="prompt")

        self._cvs.bind("<ButtonPress-1>",  self._pan_start)
        self._cvs.bind("<B1-Motion>",      self._pan_move)
        self._cvs.bind("<MouseWheel>",     self._on_wheel)
        self._cvs.bind("<Configure>",      self._cvs_resize)

    # ── Table tab ─────────────────────────────────────────────────

    def _build_table_tab(self, parent):
        # Filter bar
        fb = tk.Frame(parent, bg=T.SURFACE, pady=10)
        fb.pack(fill="x", padx=8)

        tk.Label(fb, text="Filter:", font=T.SMALL, fg=T.FG2,
                 bg=T.SURFACE).pack(side="left", padx=(10, 6))
        self.v_filter = tk.StringVar(value="all")
        for text, val in [("All","all"),("Wrong ✗","wrong"),
                           ("Blank ○","blank"),("Correct ✓","correct")]:
            tk.Radiobutton(
                fb, text=text, variable=self.v_filter, value=val,
                font=T.SMALL, fg=T.FG, bg=T.SURFACE,
                selectcolor=T.PRIMARY, activebackground=T.SURFACE,
                command=self._refresh_table,
            ).pack(side="left", padx=5)

        tk.Label(fb, text="  Q#:", font=T.SMALL, fg=T.FG2,
                 bg=T.SURFACE).pack(side="left", padx=(14, 4))
        self.v_search = tk.StringVar()
        self.v_search.trace("w", lambda *_: self._refresh_table())
        _inp(fb, self.v_search, width=6).pack(side="left", ipady=3)

        self._badge_row = tk.Frame(fb, bg=T.SURFACE)
        self._badge_row.pack(side="right", padx=10)

        # Table
        cols = ("q", "student", "correct", "result", "pts")
        self._tree = ttk.Treeview(parent, columns=cols, show="headings",
                                   style="Grid.Treeview", selectmode="browse")
        for col, head, w, stretch in [
            ("q",       "#",              60,  False),
            ("student", "Student Answer", 150, False),
            ("correct", "Correct Answer", 150, False),
            ("result",  "Result",         130, False),
            ("pts",     "Points",         90,  True),
        ]:
            self._tree.heading(col, text=head, anchor="center")
            self._tree.column(col, width=w, anchor="center",
                              stretch=stretch)

        sbv2 = ttk.Scrollbar(parent, orient="vertical",
                              command=self._tree.yview,
                              style="S.Vertical.TScrollbar")
        self._tree.configure(yscrollcommand=sbv2.set)
        sbv2.pack(side="right", fill="y")
        self._tree.pack(fill="both", expand=True, padx=6, pady=6)

        self._tree.tag_configure("correct", foreground=T.CLR_CORRECT)
        self._tree.tag_configure("wrong",   foreground=T.CLR_WRONG)
        self._tree.tag_configure("blank",   foreground=T.CLR_BLANK)
        self._tree.tag_configure("no_key",  foreground=T.CLR_NOKEY)
        self._tree.tag_configure("alt",     background="#181d30")

    # ── Log tab ───────────────────────────────────────────────────

    def _build_log_tab(self, parent):
        tb = tk.Frame(parent, bg=T.SURFACE, pady=8)
        tb.pack(fill="x")
        tk.Label(tb, text="  ⚡  Console Output",
                 font=T.SMALL, fg=T.FG2, bg=T.SURFACE).pack(side="left", padx=12)
        _btn(tb, "🗑  Clear", self._clear_log, T.CARD, T.FG3, T.SMALL, px=10, py=3
             ).pack(side="right", padx=12)

        self._log = scrolledtext.ScrolledText(
            parent, bg="#080b12", fg="#b8bcd8", font=T.MONO_SM,
            wrap="word", state="disabled", relief="flat",
            insertbackground=T.FG, padx=14, pady=12)
        self._log.pack(fill="both", expand=True)

        for tag, color in [
            ("green",  T.SUCCESS), ("red",   T.DANGER),
            ("yellow", T.WARNING), ("blue",  T.INFO),
            ("dim",    T.FG3),     ("white", T.FG),
        ]:
            self._log.tag_configure(tag, foreground=color)

    # ── Steps tab ─────────────────────────────────────────────────

    def _build_steps_tab(self, parent):
        """Tab hiển thị từng bước xử lý ảnh."""
        # Toolbar
        tb = tk.Frame(parent, bg=T.SURFACE, height=52)
        tb.pack(fill="x")
        tb.pack_propagate(False)

        # Navigation buttons
        nav = tk.Frame(tb, bg=T.SURFACE)
        nav.pack(side="left", padx=16, pady=8)

        self._btn_prev = _btn(nav, "  ◀  Prev  ", self._step_prev,
                              T.ELEVATED, T.FG, T.BODY, px=14, py=5)
        self._btn_prev.pack(side="left", padx=4)

        self._btn_next = _btn(nav, "  Next  ▶  ", self._step_next,
                              T.PRIMARY, T.FG, T.BODY, px=14, py=5)
        self._btn_next.pack(side="left", padx=4)

        # Step counter
        self._lbl_step_counter = tk.Label(
            tb, text="Step — / —", font=T.H3, fg=T.FG2, bg=T.SURFACE)
        self._lbl_step_counter.pack(side="left", padx=16)

        # Zoom buttons (right side)
        zt = tk.Frame(tb, bg=T.SURFACE)
        zt.pack(side="right", padx=16, pady=8)
        for text, cmd in [(" − ", self._step_zoom_out),
                          ("⊡ Fit", self._step_zoom_fit),
                          (" + ", self._step_zoom_in)]:
            b = _btn(zt, text, cmd, T.ELEVATED, T.FG, T.SMALL, px=10, py=4)
            b.config(height=1)
            b.pack(side="left", padx=2)
        self._lbl_step_zoom = tk.Label(
            tb, text="", font=T.SMALL, fg=T.FG3, bg=T.SURFACE)
        self._lbl_step_zoom.pack(side="right", padx=4)

        # Step name bar
        name_bar = tk.Frame(parent, bg=T.CARD, height=42)
        name_bar.pack(fill="x")
        name_bar.pack_propagate(False)
        tk.Frame(name_bar, bg=T.PRIMARY, width=4).pack(side="left", fill="y")
        self._lbl_step_name = tk.Label(
            name_bar, text="Chưa có kết quả",
            font=T.H2, fg=T.FG, bg=T.CARD, anchor="w")
        self._lbl_step_name.pack(side="left", padx=14, fill="x", expand=True)

        # Main area: image left, description right
        body = tk.Frame(parent, bg="#080b12")
        body.pack(fill="both", expand=True)

        # Image canvas
        cf = tk.Frame(body, bg="#080b12")
        cf.pack(side="left", fill="both", expand=True)

        self._step_cvs = tk.Canvas(cf, bg="#080b12", cursor="fleur",
                                    highlightthickness=0)
        sbv = ttk.Scrollbar(cf, orient="vertical",
                            command=self._step_cvs.yview,
                            style="S.Vertical.TScrollbar")
        sbh = ttk.Scrollbar(cf, orient="horizontal",
                            command=self._step_cvs.xview,
                            style="S.Horizontal.TScrollbar")
        self._step_cvs.configure(yscrollcommand=sbv.set,
                                 xscrollcommand=sbh.set)
        sbv.pack(side="right", fill="y")
        sbh.pack(side="bottom", fill="x")
        self._step_cvs.pack(fill="both", expand=True)

        # Empty state
        self._step_cvs.create_text(
            300, 220, text="🔬",
            font=("Segoe UI", 48), fill=T.FG3, tags="step_prompt")
        self._step_cvs.create_text(
            300, 290,
            text="Xử lý ảnh để xem các bước",
            font=("Segoe UI", 12), fill=T.FG3, tags="step_prompt")

        self._step_cvs.bind("<ButtonPress-1>",
                            lambda e: self._step_cvs.scan_mark(e.x, e.y))
        self._step_cvs.bind("<B1-Motion>",
                            lambda e: self._step_cvs.scan_dragto(e.x, e.y, gain=1))
        self._step_cvs.bind("<MouseWheel>",
                            lambda e: (self._step_zoom_in() if e.delta > 0
                                       else self._step_zoom_out()))
        self._step_cvs.bind("<Configure>",
                            lambda e: self._step_render_fit_if_needed())

        # Description panel (right side)
        desc_frame = tk.Frame(body, bg=T.CARD, width=280)
        desc_frame.pack(side="right", fill="y")
        desc_frame.pack_propagate(False)

        tk.Frame(desc_frame, bg=T.BORDER, width=1).pack(side="left", fill="y")

        desc_inner = tk.Frame(desc_frame, bg=T.CARD)
        desc_inner.pack(fill="both", expand=True, padx=16, pady=16)

        tk.Label(desc_inner, text="📝  Mô tả bước",
                 font=T.H3, fg=T.PRIMARY, bg=T.CARD, anchor="w"
                 ).pack(fill="x", pady=(0, 10))
        tk.Frame(desc_inner, bg=T.BORDER, height=1).pack(fill="x", pady=(0, 12))

        self._lbl_step_desc = tk.Label(
            desc_inner, text="Chọn ảnh và nhấn Grade Now\nđể xem các bước xử lý.",
            font=T.BODY, fg=T.FG2, bg=T.CARD,
            anchor="nw", justify="left", wraplength=240)
        self._lbl_step_desc.pack(fill="both", expand=True)

        # Step list (thumbnails)
        tk.Frame(desc_inner, bg=T.BORDER, height=1).pack(fill="x", pady=(12, 8))
        tk.Label(desc_inner, text="📋  Danh sách bước",
                 font=T.H3, fg=T.PRIMARY, bg=T.CARD, anchor="w"
                 ).pack(fill="x", pady=(0, 6))

        list_wrap = tk.Frame(desc_inner, bg=T.CARD)
        list_wrap.pack(fill="both", expand=True)

        list_cvs = tk.Canvas(list_wrap, bg=T.CARD, highlightthickness=0)
        list_sb = ttk.Scrollbar(list_wrap, orient="vertical",
                                command=list_cvs.yview,
                                style="S.Vertical.TScrollbar")
        list_cvs.configure(yscrollcommand=list_sb.set)
        list_sb.pack(side="right", fill="y")
        list_cvs.pack(side="left", fill="both", expand=True)

        self._step_list_inner = tk.Frame(list_cvs, bg=T.CARD)
        list_cvs.create_window((0, 0), window=self._step_list_inner,
                               anchor="nw")
        self._step_list_inner.bind(
            "<Configure>",
            lambda e: list_cvs.configure(
                scrollregion=list_cvs.bbox("all")))
        list_cvs.bind_all("<MouseWheel>",
                          lambda e: list_cvs.yview_scroll(
                              -1 * (e.delta // 120), "units"),
                          add="+")
        self._step_list_cvs = list_cvs
        self._step_list_btns = []

    # ── Step navigation ───────────────────────────────────────────

    def _load_steps(self, step_images):
        """Nạp danh sách ảnh bước xử lý từ kết quả."""
        self._steps = step_images or []
        self._step_idx = 0
        self._rebuild_step_list()
        if self._steps:
            self._show_step(0)
        else:
            self._lbl_step_name.config(text="Không có dữ liệu")
            self._lbl_step_desc.config(text="")
            self._lbl_step_counter.config(text="Step — / —")

    def _rebuild_step_list(self):
        """Xây lại danh sách bước ở panel phải."""
        for w in self._step_list_btns:
            w.destroy()
        self._step_list_btns = []

        for i, (name, desc, _) in enumerate(self._steps):
            bg = T.PRIMARY_D if i == self._step_idx else T.ELEVATED
            fg = T.FG if i == self._step_idx else T.FG2
            btn = tk.Button(
                self._step_list_inner,
                text=f" {i+1}. {name}",
                font=T.SMALL, fg=fg, bg=bg,
                anchor="w", relief="flat", bd=0,
                cursor="hand2", padx=8, pady=4,
                command=lambda idx=i: self._show_step(idx))
            btn.pack(fill="x", pady=1)
            self._step_list_btns.append(btn)

    def _show_step(self, idx):
        """Hiển thị ảnh bước thứ idx."""
        if not self._steps or idx < 0 or idx >= len(self._steps):
            return
        self._step_idx = idx
        name, desc, bgr_img = self._steps[idx]

        # Update UI labels
        self._lbl_step_counter.config(
            text=f"Step {idx + 1} / {len(self._steps)}")
        self._lbl_step_name.config(text=f"  {name}")
        self._lbl_step_desc.config(text=desc)

        # Highlight active in list
        for i, btn in enumerate(self._step_list_btns):
            if i == idx:
                btn.config(bg=T.PRIMARY_D, fg=T.FG)
            else:
                btn.config(bg=T.ELEVATED, fg=T.FG2)

        # Convert and display image
        import cv2 as _cv2
        rgb = _cv2.cvtColor(bgr_img, _cv2.COLOR_BGR2RGB)
        self._step_pil = Image.fromarray(rgb)
        self._step_cvs.delete("step_prompt")
        self._step_cvs.delete("step_img")
        self.after(60, self._step_zoom_fit)

        # Update button states
        self._btn_prev.config(
            state="normal" if idx > 0 else "disabled")
        self._btn_next.config(
            state="normal" if idx < len(self._steps) - 1 else "disabled")

    def _step_prev(self):
        if self._step_idx > 0:
            self._show_step(self._step_idx - 1)

    def _step_next(self):
        if self._step_idx < len(self._steps) - 1:
            self._show_step(self._step_idx + 1)

    def _step_render(self, scale=None):
        if self._step_pil is None:
            return
        if scale is not None:
            self._step_scale = scale
        iw, ih = self._step_pil.size
        nw = max(1, int(iw * self._step_scale))
        nh = max(1, int(ih * self._step_scale))
        resized = self._step_pil.resize((nw, nh), Image.LANCZOS)
        self._step_photo = ImageTk.PhotoImage(resized)
        self._step_cvs.delete("step_img")
        self._step_cvs.create_image(0, 0, image=self._step_photo,
                                     anchor="nw", tags="step_img")
        self._step_cvs.configure(scrollregion=(0, 0, nw, nh))
        self._lbl_step_zoom.config(
            text=f"zoom: {int(self._step_scale * 100)}%")

    def _step_zoom_in(self):
        self._step_render(min(self._step_scale * 1.25, 6.0))

    def _step_zoom_out(self):
        self._step_render(max(self._step_scale * 0.80, 0.05))

    def _step_zoom_fit(self):
        if self._step_pil is None:
            return
        cw = self._step_cvs.winfo_width()
        ch = self._step_cvs.winfo_height()
        if cw < 10:
            self.after(80, self._step_zoom_fit)
            return
        iw, ih = self._step_pil.size
        self._step_render(min(cw / iw, ch / ih, 1.0))

    def _step_render_fit_if_needed(self):
        if self._step_pil:
            self._step_zoom_fit()

    # ── Status bar ────────────────────────────────────────────────

    def _build_statusbar(self):
        sb = tk.Frame(self, bg=T.SURFACE, height=30)
        sb.pack(fill="x", side="bottom")
        sb.pack_propagate(False)
        tk.Frame(sb, bg=T.PRIMARY, width=3).pack(side="left", fill="y")
        tk.Label(sb, textvariable=self.v_status,
                 font=T.SMALL, fg=T.FG2, bg=T.SURFACE).pack(side="left", padx=12)
        tk.Label(sb, text="SmartOMR v3.0", font=("Segoe UI", 8),
                 fg=T.FG3, bg=T.SURFACE).pack(side="right", padx=14)
        self._pgbar = ttk.Progressbar(sb, length=130, mode="indeterminate",
                                       style="Run.Horizontal.TProgressbar")
        self._pgbar.pack(side="right", padx=10, pady=6)

    # ─── FILE BROWSERS ─────────────────────────────────────────────────────

    def _browse_image(self):
        fp = filedialog.askopenfilename(
            title="Select answer sheet image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("All", "*.*")],
            initialdir=os.path.join(HERE, "input"))
        if fp: self.v_image.set(fp)

    def _browse_key(self):
        fp = filedialog.askopenfilename(
            title="Select answer key",
            filetypes=[("Text","*.txt"),("JSON","*.json"),
                       ("CSV","*.csv"),("All","*.*")],
            initialdir=os.path.join(HERE, "answer_keys"))
        if fp: self.v_key.set(fp)



    # ─── ANSWER KEY ────────────────────────────────────────────────────────

    def _create_key(self):
        def cb(fp):
            if fp: self.v_key.set(fp)
        KeyEditor(self, callback=cb)

    def _edit_key(self):
        fp = self.v_key.get().strip()
        def cb(p):
            if p: self.v_key.set(p)
        KeyEditor(self,
                  filepath=fp if os.path.isfile(fp) else None,
                  callback=cb)

    # ─── GRADING ───────────────────────────────────────────────────────────

    def _run(self):
        img  = self.v_image.get().strip()
        key  = self.v_key.get().strip() or None

        if not img or not os.path.isfile(img):
            messagebox.showwarning(
                "No Image", "Please select a valid answer sheet image.",
                parent=self)
            return

        self._run_btn.config(state="disabled", text="⏳  Processing…")
        self._pgbar.start(10)
        self.v_status.set("Processing…")
        self._clear_log()
        self._nb.select(3)  # show console (tab 3 after adding Steps tab)

        def worker():
            import io
            from contextlib import redirect_stdout

            self._log_write("SmartOMR — starting\n", "blue")
            self._log_write(f"  Image : {img}\n",              "dim")
            self._log_write(f"  Key   : {key or '(none)'}\n",  "dim")
            self._log_write("─" * 56 + "\n", "dim")

            buf, res = io.StringIO(), [None]
            try:
                with redirect_stdout(buf):
                    res[0] = omr_run(
                        image_path=img, debug=False, save=True,
                        answer_key=key if key and os.path.isfile(key) else None,
                        grading_method=self.v_grading_method.get(),
                    )
            except Exception as ex:
                self._log_write(f"\n[ERROR] {ex}\n", "red")
                self.after(0, self._done_processing)
                return

            for line in buf.getvalue().splitlines():
                tag = ("green"  if any(k in line for k in ("Correct", "dung")) else
                       "red"    if any(k in line for k in ("ERROR", "SAI"))    else
                       "yellow" if "WARN" in line                              else "dim")
                self._log_write(line + "\n", tag)

            self.after(0, lambda: self._on_result(res[0]))

        threading.Thread(target=worker, daemon=True).start()

    def _on_result(self, result):
        self._done_processing()
        if result is None:
            messagebox.showerror("Error",
                "Could not process the image.\nSee Console for details.",
                parent=self)
            return

        self._result = result
        gr = None
        kp = self.v_key.get().strip()
        if kp and os.path.isfile(kp):
            try:
                gr = grade(result["answers"], load_answer_key(kp))
            except Exception as e:
                self._log_write(f"[WARN] Grading error: {e}\n", "yellow")
        self._grading = gr

        self._update_scorecard(gr)
        self._refresh_table()
        self._show_result_image(result, gr)
        self._load_steps(result.get('step_images', []))

        s = result["stats"]
        if gr:
            msg = (f"Score: {gr.score:.2f} / {gr.score_max}  ({gr.score_pct}%)  ·  "
                   f"Correct: {gr.n_correct}  Wrong: {gr.n_wrong}  Blank: {gr.n_blank}"
                   f"  ·  {s['processing_time']:.1f}s")
        else:
            msg = (f"Detected {s['answered']} / {s['total_questions']} answers"
                   f"  ·  {s['processing_time']:.1f}s")
        self.v_status.set(msg)

        sep = "─" * 56
        self._log_write(f"\n{sep}\n", "dim")
        if gr:
            self._log_write(f"  Score   : {gr.score:.2f} / {gr.score_max}  ({gr.score_pct}%)\n", "yellow")
            self._log_write(f"  Correct : {gr.n_correct}\n", "green")
            self._log_write(f"  Wrong   : {gr.n_wrong}\n",   "red")
            self._log_write(f"  Blank   : {gr.n_blank}\n",   "dim")
        self._log_write(f"\n✅  Done — {s['processing_time']:.2f}s\n", "green")

        self._nb.select(0)

    def _done_processing(self):
        self._pgbar.stop()
        self._run_btn.config(state="normal", text="▶   Grade Now")

    # ─── IMAGE ─────────────────────────────────────────────────────────────

    def _show_result_image(self, result, gr):
        import cv2
        ann = (draw_graded_annotated(result["image_orig"], result["grid"],
                                      result["answers"], gr)
               if gr is not None else result["annotated"])
        rgb = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
        self._img_pil = Image.fromarray(rgb)
        self._cvs.delete("prompt")
        self._cvs.delete("img")
        self.after(60, self._zoom_fit)

    def _render(self, scale=None):
        if self._img_pil is None: return
        if scale is not None: self._scale = scale
        iw, ih  = self._img_pil.size
        nw, nh  = max(1, int(iw * self._scale)), max(1, int(ih * self._scale))
        resized = self._img_pil.resize((nw, nh), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(resized)
        self._cvs.delete("img")
        self._cvs.create_image(0, 0, image=self._photo, anchor="nw", tags="img")
        self._cvs.configure(scrollregion=(0, 0, nw, nh))
        self._lbl_zoom.config(text=f"zoom: {int(self._scale * 100)}%")

    def _zoom_in(self):  self._render(min(self._scale * 1.25, 6.0))
    def _zoom_out(self): self._render(max(self._scale * 0.80, 0.05))

    def _zoom_fit(self):
        if self._img_pil is None: return
        cw, ch = self._cvs.winfo_width(), self._cvs.winfo_height()
        if cw < 10: self.after(80, self._zoom_fit); return
        iw, ih = self._img_pil.size
        self._render(min(cw / iw, ch / ih, 1.0))

    def _cvs_resize(self, _):
        if self._img_pil:
            self._zoom_fit()

    def _pan_start(self, e): self._cvs.scan_mark(e.x, e.y)
    def _pan_move(self,  e): self._cvs.scan_dragto(e.x, e.y, gain=1)
    def _on_wheel(self,  e):
        (self._zoom_in if e.delta > 0 else self._zoom_out)()

    # ─── TABLE ─────────────────────────────────────────────────────────────

    def _refresh_table(self):
        self._tree.delete(*self._tree.get_children())
        for w in self._badge_row.winfo_children(): w.destroy()
        if self._grading is None: return

        flt    = self.v_filter.get()
        search = self.v_search.get().strip()
        alt    = False

        for q in sorted(self._grading.details):
            d = self._grading.details[q]
            if flt != "all" and d["status"] != flt: continue
            if search and str(q) != search: continue
            s   = d["student"] or "—"
            c   = d["correct"] or "—"
            res = RESULT_LABELS.get(d["status"], "")
            pts = f"{d['earned']:+.3f}" if d["status"] != "no_key" else "—"
            tags = (d["status"],) + (("alt",) if alt else ())
            self._tree.insert("", "end", values=(q, s, c, res, pts), tags=tags)
            alt = not alt

        gr = self._grading
        for txt, col in [(f"✓ {gr.n_correct}", T.CLR_CORRECT),
                          (f"✗ {gr.n_wrong}",   T.CLR_WRONG),
                          (f"○ {gr.n_blank}",   T.CLR_BLANK)]:
            _badge(self._badge_row, txt, col).pack(side="left", padx=3)

    # ─── LOG ───────────────────────────────────────────────────────────────

    def _log_write(self, text, tag="dim"):
        def _do():
            self._log.config(state="normal")
            self._log.insert("end", text, tag)
            self._log.see("end")
            self._log.config(state="disabled")
        self.after(0, _do)

    def _clear_log(self):
        self._log.config(state="normal")
        self._log.delete("1.0", "end")
        self._log.config(state="disabled")

    # ─── EXPORT ────────────────────────────────────────────────────────────

    def _export_report(self):
        if self._grading is None:
            messagebox.showinfo("No Result", "Grade an exam first.", parent=self)
            return
        fp = filedialog.asksaveasfilename(
            title="Save grading report",
            defaultextension=".txt",
            filetypes=[("Text", "*.txt"), ("All", "*.*")],
            initialdir=os.path.join(HERE, "output"))
        if fp:
            self._grading.save_report(fp)
            messagebox.showinfo("Saved", fp, parent=self)

    def _export_image(self):
        if self._img_pil is None:
            messagebox.showinfo("No Image", "Grade an exam first.", parent=self)
            return
        fp = filedialog.asksaveasfilename(
            title="Save graded image",
            defaultextension=".jpg",
            filetypes=[("JPEG","*.jpg"),("PNG","*.png"),("All","*.*")],
            initialdir=os.path.join(HERE, "output"))
        if fp:
            import cv2, numpy as np
            bgr = cv2.cvtColor(np.array(self._img_pil), cv2.COLOR_RGB2BGR)
            cv2.imwrite(fp, bgr)
            messagebox.showinfo("Saved", fp, parent=self)

    def _open_output(self):
        out = os.path.join(HERE, "output")
        os.makedirs(out, exist_ok=True)
        os.startfile(out)

    def _open_questions_folder(self):
        if self._result and 'question_images' in self._result:
            QuestionGalleryDialog(self, self._result['question_images'])
        elif self._result and 'q_dir' in self._result and os.path.isdir(self._result['q_dir']):
            os.startfile(self._result['q_dir'])
        else:
            self._open_output()

class QuestionGalleryDialog(tk.Toplevel):
    def __init__(self, parent, q_dict):
        super().__init__(parent)
        self.title("120 Câu Hỏi Đã Xử Lý")
        self.geometry("900x700")
        self.configure(bg=T.BG)
        self.transient(parent)
        self.grab_set()

        lbl = tk.Label(self, text="Danh sách 120 ảnh đã xử lý", font=T.H2, bg=T.BG, fg=T.PRIMARY)
        lbl.pack(pady=10)

        container = tk.Frame(self, bg=T.BG)
        container.pack(fill="both", expand=True, padx=20, pady=10)

        canvas = tk.Canvas(container, bg=T.BG, highlightthickness=0)
        scrollbar_y = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollbar_x = ttk.Scrollbar(container, orient="horizontal", command=canvas.xview)
        scrollable_frame = tk.Frame(canvas, bg=T.BG)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

        scrollbar_x.pack(side="bottom", fill="x")
        scrollbar_y.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        def _on_mousewheel(event):
            # Roll mouse wheel to scroll horizontally
            canvas.xview_scroll(int(-1*(event.delta/120)), "units")
            
        # Bind only to this window
        self.bind("<Enter>", lambda _: self.bind_all("<MouseWheel>", _on_mousewheel))
        self.bind("<Leave>", lambda _: self.unbind_all("<MouseWheel>"))

        import cv2
        from PIL import Image, ImageTk
        self.images = [] 
        
        row_frame = tk.Frame(scrollable_frame, bg=T.BG)
        row_frame.pack(fill="y", expand=True, padx=10, pady=20)
        
        for q_num in sorted(q_dict.keys()):
            box = tk.Frame(row_frame, bg=T.SURFACE, bd=1, relief="solid")
            box.pack(side="left", padx=10, pady=5)
            
            ans = q_dict[q_num].get('answer') or "_"
            tk.Label(box, text=f"Câu {q_num}: {ans}", bg=T.SURFACE, fg=T.FG, font=T.BODY).pack(pady=(5,0))
            
            bgr = q_dict[q_num]['image']
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            
            # Phóng to một chút để xem rõ hơn trên dải ngang
            if pil_img.width < 150:
                scale = 150 / pil_img.width
                pil_img = pil_img.resize((int(pil_img.width * scale), int(pil_img.height * scale)), Image.LANCZOS)
            
            tk_img = ImageTk.PhotoImage(pil_img)
            self.images.append(tk_img)
            lbl_img = tk.Label(box, image=tk_img, bg=T.SURFACE)
            lbl_img.pack(padx=10, pady=10)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if not PIL_AVAILABLE:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Missing Dependency",
            "Pillow is required to run SmartOMR.\n\nInstall it with:\n  pip install Pillow")
        sys.exit(1)
    App().mainloop()


if __name__ == "__main__":
    main()
