"""
predict.py
==========
A desktop window for predicting international football (FIFA) match results.

Pick two teams and hit *Predict*. The app shows, side by side:

  * the **Elo model**    — calibrated Win / Draw / Loss probabilities, ratings;
  * the **Neural net**   — the entity-embedding network's W/D/L probabilities
    (shown if a trained nn_model.pt exists — run train_nn.py to create it);
  * the **Logic graph**  — the predicted winner plus the chain of past results
    that justifies it (t1 → t2 → … → winner), drawn live below.

All models are trained on the last 10 years of international results (see
data.py). Run with:

    python predict.py
"""
from __future__ import annotations

import os
import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk,
)

import graph_visualize as gv
from data import load_matches, team_list
from elo_model import build_elo
from graph_model import build_graph

NN_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "nn_model.pt")

BG = "#f4f6f7"
HOME_C = "#27ae60"   # green
DRAW_C = "#95a5a6"   # grey
AWAY_C = "#c0392b"   # red


class PredictApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("FIFA Match Predictor — Elo + Neural Net + Logic Graph")
        root.geometry("1180x860")
        root.configure(bg=BG)
        root.minsize(980, 720)

        self._splash()
        root.update()

        # Train both models once on the shared dataset.
        self.df = load_matches()
        self.elo = build_elo(self.df)
        self.graph = build_graph(self.df)
        self.teams = team_list(self.df)
        self._ranks = {r.team: r.rank
                       for r in self.elo.ranking_frame().itertuples(index=False)}

        # Load the trained neural net if it exists (optional third model).
        self.nn = None
        if os.path.exists(NN_MODEL_PATH):
            try:
                from nn_model import NNPredictor
                self.nn = NNPredictor.load(NN_MODEL_PATH)
            except Exception:
                self.nn = None

        for w in root.winfo_children():
            w.destroy()
        self._build_ui()
        self._show_overview()

    # ------------------------------------------------------------- startup
    def _splash(self) -> None:
        self._splash_lbl = tk.Label(
            self.root, text="Loading 10 years of results and training the "
            "Elo + graph models…", bg=BG, fg="#2c3e50",
            font=("Helvetica", 15))
        self._splash_lbl.pack(expand=True)

    # -------------------------------------------------------------- layout
    def _build_ui(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TButton", font=("Helvetica", 11))
        style.configure("Predict.TButton", font=("Helvetica", 12, "bold"))

        header = tk.Label(
            self.root, text="FIFA Match Predictor",
            bg=BG, fg="#1a5276", font=("Helvetica", 20, "bold"))
        header.pack(pady=(12, 2))
        models = "Elo + neural-net + logic-graph" if self.nn else \
            "Elo + logic-graph"
        sub = tk.Label(
            self.root,
            text=f"{models} models · last 10 years · "
                 f"{len(self.df):,} matches · {len(self.teams)} teams",
            bg=BG, fg="#566573", font=("Helvetica", 11))
        sub.pack(pady=(0, 8))

        # ---- controls ----
        ctrl = tk.Frame(self.root, bg=BG)
        ctrl.pack(fill="x", padx=18)

        tk.Label(ctrl, text="Team A", bg=BG, font=("Helvetica", 11, "bold")
                 ).grid(row=0, column=0, padx=4, sticky="w")
        self.cb_a = self._make_combo(ctrl)
        self.cb_a.grid(row=1, column=0, padx=4)

        tk.Label(ctrl, text="vs", bg=BG, font=("Helvetica", 12, "bold")
                 ).grid(row=1, column=1, padx=8)

        tk.Label(ctrl, text="Team B", bg=BG, font=("Helvetica", 11, "bold")
                 ).grid(row=0, column=2, padx=4, sticky="w")
        self.cb_b = self._make_combo(ctrl)
        self.cb_b.grid(row=1, column=2, padx=4)

        self.neutral_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="Neutral venue", variable=self.neutral_var
                        ).grid(row=1, column=3, padx=14)

        ttk.Button(ctrl, text="Swap", command=self._swap
                   ).grid(row=1, column=4, padx=4)
        ttk.Button(ctrl, text="Predict ▶", style="Predict.TButton",
                   command=self.predict).grid(row=1, column=5, padx=10)

        # sensible defaults
        if self.teams:
            self.cb_a.set("Brazil" if "Brazil" in self.teams else self.teams[0])
            self.cb_b.set("France" if "France" in self.teams else self.teams[-1])

        tk.Label(self.root, bg=BG, fg="#7f8c8d", font=("Helvetica", 9, "italic"),
                 text="Tip: type part of a name to filter the dropdown, then "
                      "press Enter or Predict.").pack(pady=(6, 0))

        # ---- results panels (Elo, [Neural net], Logic graph) ----
        panels = tk.Frame(self.root, bg=BG)
        panels.pack(fill="x", padx=18, pady=(12, 6))
        ncols = 3 if self.nn else 2
        for col in range(ncols):
            panels.columnconfigure(col, weight=1, uniform="p")

        self._build_elo_panel(panels, column=0)
        graph_col = 1
        if self.nn:
            self._build_nn_panel(panels, column=1)
            graph_col = 2
        self._build_graph_panel(panels, column=graph_col)

        # ---- live graph ----
        gframe = tk.LabelFrame(self.root, text="Live dominance graph",
                               bg=BG, font=("Helvetica", 11, "bold"),
                               fg="#1a5276")
        gframe.pack(fill="both", expand=True, padx=18, pady=(6, 12))
        self.fig = Figure(figsize=(9, 4.2), facecolor="white")
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=gframe)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, gframe, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")

    def _make_combo(self, parent) -> ttk.Combobox:
        cb = ttk.Combobox(parent, values=self.teams, width=26,
                          font=("Helvetica", 11))
        cb.bind("<KeyRelease>", self._on_combo_type)
        cb.bind("<Return>", lambda e: self.predict())
        return cb

    def _make_prob_bar(self, parent) -> tk.Canvas:
        """A horizontal W/D/L bar that repaints itself on resize."""
        c = tk.Canvas(parent, height=34, bg="white", highlightthickness=0)
        c.probs = None
        c.pack(fill="x", pady=(0, 6))
        c.bind("<Configure>", lambda e, cv=c: self._paint_bar(cv))
        return c

    def _build_elo_panel(self, parent, column: int) -> None:
        f = tk.LabelFrame(parent, text="Elo model — Win / Draw / Loss",
                          bg="white", font=("Helvetica", 12, "bold"),
                          fg="#1a5276", padx=10, pady=8)
        f.grid(row=0, column=column, sticky="nsew", padx=(0, 8))
        self.elo_headline = tk.Label(f, text="", bg="white",
                                     font=("Helvetica", 13, "bold"),
                                     fg="#2c3e50")
        self.elo_headline.pack(anchor="w")
        self.elo_ratings = tk.Label(f, text="", bg="white",
                                    font=("Helvetica", 11), justify="left")
        self.elo_ratings.pack(anchor="w", pady=(2, 8))
        self.elo_bar = self._make_prob_bar(f)
        self.elo_probs = tk.Label(f, text="", bg="white",
                                  font=("Helvetica", 11), justify="left")
        self.elo_probs.pack(anchor="w")

    def _build_nn_panel(self, parent, column: int) -> None:
        f = tk.LabelFrame(parent, text="Neural net — Win / Draw / Loss",
                          bg="white", font=("Helvetica", 12, "bold"),
                          fg="#1a5276", padx=10, pady=8)
        f.grid(row=0, column=column, sticky="nsew", padx=8)
        self.nn_headline = tk.Label(f, text="", bg="white",
                                    font=("Helvetica", 13, "bold"),
                                    fg="#2c3e50")
        self.nn_headline.pack(anchor="w")
        self.nn_info = tk.Label(f, text="", bg="white",
                                font=("Helvetica", 11), justify="left")
        self.nn_info.pack(anchor="w", pady=(2, 8))
        self.nn_bar = self._make_prob_bar(f)
        self.nn_probs = tk.Label(f, text="", bg="white",
                                 font=("Helvetica", 11), justify="left")
        self.nn_probs.pack(anchor="w")

    def _build_graph_panel(self, parent, column: int) -> None:
        f = tk.LabelFrame(parent, text="Logic graph — transitive dominance",
                          bg="white", font=("Helvetica", 12, "bold"),
                          fg="#1a5276", padx=10, pady=8)
        f.grid(row=0, column=column, sticky="nsew", padx=(8, 0))
        self.graph_headline = tk.Label(f, text="", bg="white",
                                       font=("Helvetica", 13, "bold"),
                                       fg="#2c3e50")
        self.graph_headline.pack(anchor="w")
        self.graph_method = tk.Label(f, text="", bg="white",
                                     font=("Helvetica", 10, "italic"),
                                     fg="#566573")
        self.graph_method.pack(anchor="w", pady=(0, 4))
        self.graph_reason = tk.Label(f, text="", bg="white", wraplength=460,
                                     font=("Helvetica", 11), justify="left")
        self.graph_reason.pack(anchor="w", pady=(2, 6))
        self.graph_h2h = tk.Label(f, text="", bg="white",
                                  font=("Helvetica", 10), fg="#566573",
                                  justify="left")
        self.graph_h2h.pack(anchor="w")

    # ----------------------------------------------------------- behaviour
    def _on_combo_type(self, event) -> None:
        # filter the dropdown as the user types (skip navigation keys)
        if event.keysym in ("Up", "Down", "Return", "Left", "Right",
                             "Escape", "Tab"):
            return
        cb = event.widget
        typed = cb.get().lower()
        if not typed:
            cb["values"] = self.teams
            return
        matches = [t for t in self.teams if typed in t.lower()]
        cb["values"] = matches or self.teams

    def _swap(self) -> None:
        a, b = self.cb_a.get(), self.cb_b.get()
        self.cb_a.set(b)
        self.cb_b.set(a)

    def _resolve(self, text: str) -> str | None:
        """Map typed text to an actual team name, tolerantly."""
        text = text.strip()
        if not text:
            return None
        if text in self.teams:
            return text
        low = text.lower()
        exact = [t for t in self.teams if t.lower() == low]
        if exact:
            return exact[0]
        contains = [t for t in self.teams if low in t.lower()]
        if len(contains) == 1:
            return contains[0]
        if contains:
            # prefer a startswith match if unambiguous
            starts = [t for t in contains if t.lower().startswith(low)]
            if len(starts) == 1:
                return starts[0]
        return None

    def predict(self) -> None:
        a = self._resolve(self.cb_a.get())
        b = self._resolve(self.cb_b.get())
        if a is None or b is None:
            bad = self.cb_a.get() if a is None else self.cb_b.get()
            messagebox.showwarning(
                "Unknown team",
                f"Couldn't match '{bad}' to a team.\n\n"
                "Start typing to filter the list and pick an exact name.")
            return
        if a == b:
            messagebox.showinfo("Same team", "Pick two different teams.")
            return
        # normalise the boxes to the resolved names
        self.cb_a.set(a)
        self.cb_b.set(b)

        neutral = self.neutral_var.get()
        self._update_elo(a, b, neutral)
        if self.nn:
            self._update_nn(a, b, neutral)
        self._update_graph(a, b)

    # -------------------------------------------------------- elo display
    def _update_elo(self, a: str, b: str, neutral: bool) -> None:
        p = self.elo.predict(a, b, neutral=neutral)
        ra, rb = self._ranks.get(a, "?"), self._ranks.get(b, "?")
        pick = p["pick"]
        self._last_elo_pick = pick
        self.elo_headline.config(text=f"Pick: {pick}")
        venue = "neutral venue" if neutral else f"{a} at home"
        self.elo_ratings.config(
            text=(f"{a}:  Elo {p['rating_home']:.0f}  (rank #{ra})\n"
                  f"{b}:  Elo {p['rating_away']:.0f}  (rank #{rb})\n"
                  f"venue: {venue}   ·   edge {p['rating_diff']:+.0f}"))
        self._paint_bar(self.elo_bar,
                        (p["p_home_win"], p["p_draw"], p["p_away_win"]))
        self.elo_probs.config(
            text=(f"{a} win   {p['p_home_win']*100:5.1f}%\n"
                  f"draw       {p['p_draw']*100:5.1f}%\n"
                  f"{b} win   {p['p_away_win']*100:5.1f}%"))

    # --------------------------------------------------- neural-net display
    def _update_nn(self, a: str, b: str, neutral: bool) -> None:
        if not (self.nn.has_team(a) and self.nn.has_team(b)):
            self.nn_headline.config(text="Pick: n/a")
            self.nn_info.config(text="team not seen during training")
            self._paint_bar(self.nn_bar, (0.0, 0.0, 0.0))
            self.nn_probs.config(text="")
            return
        p = self.nn.predict(a, b, neutral=neutral)
        self.nn_headline.config(text=f"Pick: {p['pick']}")
        agree = ("agrees with Elo"
                 if p["pick"] == getattr(self, "_last_elo_pick", None)
                 else "differs from Elo")
        self.nn_info.config(
            text=f"entity-embedding network\nuses current form · {agree}")
        self._paint_bar(self.nn_bar,
                        (p["p_home_win"], p["p_draw"], p["p_away_win"]))
        self.nn_probs.config(
            text=(f"{a} win   {p['p_home_win']*100:5.1f}%\n"
                  f"draw       {p['p_draw']*100:5.1f}%\n"
                  f"{b} win   {p['p_away_win']*100:5.1f}%"))

    # --------------------------------------------------------- shared bar
    def _paint_bar(self, c: tk.Canvas, probs=None) -> None:
        if probs is not None:
            c.probs = probs
        if getattr(c, "probs", None) is None:
            return
        ph, pd_, pa = c.probs
        c.delete("all")
        c.update_idletasks()
        w = max(c.winfo_width(), 200)
        h, x = 34, 0
        for frac, color in ((ph, HOME_C), (pd_, DRAW_C), (pa, AWAY_C)):
            seg = frac * w
            c.create_rectangle(x, 0, x + seg, h, fill=color, outline="white")
            if seg > 28:
                c.create_text(x + seg / 2, h / 2, text=f"{frac*100:.0f}%",
                              fill="white", font=("Helvetica", 10, "bold"))
            x += seg

    # ------------------------------------------------------ graph display
    def _update_graph(self, a: str, b: str) -> None:
        p = self.graph.predict(a, b)
        winner = p["winner"] or "Toss-up"
        self.graph_headline.config(text=f"Pick: {winner}")
        conf = p["confidence"]
        method = {
            "head-to-head": "direct head-to-head record",
            "transitive": "transitive dominance chain",
            "structural-fallback": "structural fallback (weak)",
            "none": "no evidence",
        }.get(p["method"], p["method"])
        self.graph_method.config(text=f"basis: {method}   ·   "
                                      f"confidence: {conf}")
        if p["path"]:
            chain = "  →  ".join(p["path"])
            self.graph_reason.config(
                text=f"{p['reading']}\n\npath (arrows point to winner):\n{chain}")
        else:
            self.graph_reason.config(text=p["explanation"])

        h2h = p.get("h2h")
        if h2h:
            last = h2h["last_meeting"]
            last_s = last.date().isoformat() if last is not None else "n/a"
            self.graph_h2h.config(
                text=(f"head-to-head ({h2h['meetings']} mtgs, last {last_s}):  "
                      f"{a} {h2h['record_a']}  |  goals "
                      f"{h2h['goals_a']}–{h2h['goals_b']}"))
        else:
            self.graph_h2h.config(text="head-to-head: never met in the window")

        gv.draw_prediction(self.graph, p, self.ax)
        self.fig.tight_layout()
        self.canvas.draw()

    def _show_overview(self) -> None:
        gv.draw_overview(self.graph, self.ax, top_n=24)
        self.fig.tight_layout()
        self.canvas.draw()


def main() -> None:
    root = tk.Tk()
    try:
        PredictApp(root)
    except Exception as exc:  # surface startup failures instead of a bare crash
        try:
            messagebox.showerror(
                "Startup failed",
                f"Could not load the data or train the models:\n\n{exc}\n\n"
                "Try:  python data.py   to (re)download the dataset.")
        except tk.TclError:
            pass
        root.destroy()
        raise
    root.mainloop()


if __name__ == "__main__":
    main()
