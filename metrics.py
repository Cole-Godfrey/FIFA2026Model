"""
metrics.py
==========
Scoring metrics for 3-class (Home / Draw / Away) match-outcome forecasts.

Probabilities are always ordered ``[P(home win), P(draw), P(away win)]`` and
labels are ``0 = home win, 1 = draw, 2 = away win``.

Metrics reported
----------------
* accuracy            — fraction of matches whose most-likely class was correct.
* log loss            — multiclass cross-entropy (the proper score we optimise).
* RPS                 — Ranked Probability Score: the standard football metric;
                        it rewards probability mass placed *near* the true
                        ordered outcome (a draw forecast is "less wrong" for a
                        narrow home win than an away-win forecast). Lower better.
* Brier (multiclass)  — mean squared error of the probability vector.
* per-class precision / recall / F1 (draws are the hard class).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import log_loss, precision_recall_fscore_support

CLASSES = ["home", "draw", "away"]
_EPS = 1e-15


def _onehot(y: np.ndarray, k: int = 3) -> np.ndarray:
    oh = np.zeros((len(y), k))
    oh[np.arange(len(y)), y] = 1.0
    return oh


def ranked_probability_score(probs: np.ndarray, y: np.ndarray) -> float:
    """Mean RPS over ordered categories [home, draw, away]. Lower is better."""
    probs = np.clip(np.asarray(probs, float), 0, 1)
    probs = probs / probs.sum(axis=1, keepdims=True)   # ensure rows sum to 1
    oh = _onehot(np.asarray(y, int))
    cum_p = np.cumsum(probs, axis=1)[:, :-1]   # first r-1 cumulative probs
    cum_o = np.cumsum(oh, axis=1)[:, :-1]
    return float(np.mean(np.sum((cum_p - cum_o) ** 2, axis=1) / (probs.shape[1] - 1)))


def multiclass_brier(probs: np.ndarray, y: np.ndarray) -> float:
    probs = np.asarray(probs, float)
    oh = _onehot(np.asarray(y, int))
    return float(np.mean(np.sum((probs - oh) ** 2, axis=1)))


def accuracy(probs: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.argmax(probs, axis=1) == np.asarray(y, int)))


def evaluate(probs: np.ndarray, y: np.ndarray) -> dict:
    """Compute the full metric bundle for one set of forecasts."""
    probs = np.clip(np.asarray(probs, float), _EPS, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    y = np.asarray(y, int)
    pred = np.argmax(probs, axis=1)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y, pred, labels=[0, 1, 2], average=None, zero_division=0)
    return {
        "n": int(len(y)),
        "accuracy": accuracy(probs, y),
        "log_loss": float(log_loss(y, probs, labels=[0, 1, 2])),
        "rps": ranked_probability_score(probs, y),
        "brier": multiclass_brier(probs, y),
        "f1": {c: float(f1[i]) for i, c in enumerate(CLASSES)},
        "precision": {c: float(prec[i]) for i, c in enumerate(CLASSES)},
        "recall": {c: float(rec[i]) for i, c in enumerate(CLASSES)},
        "macro_f1": float(np.mean(f1)),
    }


def format_comparison(named_metrics: dict[str, dict]) -> str:
    """Render a side-by-side table of {model_name: metrics dict}."""
    names = list(named_metrics)
    lines = []
    w = 22
    head = "metric".ljust(14) + "".join(n.ljust(w) for n in names)
    lines.append(head)
    lines.append("-" * len(head))

    def row(label, key, fmt, better=None):
        vals = [named_metrics[n][key] for n in names]
        cells = []
        best = None
        if better == "max":
            best = max(vals)
        elif better == "min":
            best = min(vals)
        for v in vals:
            s = fmt.format(v)
            if best is not None and abs(v - best) < 1e-9:
                s += " *"
            cells.append(s.ljust(w))
        lines.append(label.ljust(14) + "".join(cells))

    row("accuracy", "accuracy", "{:.3f}", "max")
    row("log loss", "log_loss", "{:.4f}", "min")
    row("RPS", "rps", "{:.4f}", "min")
    row("Brier", "brier", "{:.4f}", "min")
    row("macro F1", "macro_f1", "{:.3f}", "max")
    lines.append("-" * len(head))
    for c in CLASSES:
        vals = [named_metrics[n]["f1"][c] for n in names]
        cells = [f"{v:.3f}".ljust(w) for v in vals]
        lines.append(f"F1 {c}".ljust(14) + "".join(cells))
    lines.append("\n(* = best column; lower is better for log loss / RPS / Brier)")
    return "\n".join(lines)
