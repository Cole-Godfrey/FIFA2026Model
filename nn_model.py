"""
nn_model.py
===========
A neural match-outcome model for international football.

Architecture (entity-embedding network)
----------------------------------------
The natural structure of this problem is "two categorical entities (the teams)
plus a little context → one of three ordered outcomes". The most suitable
architecture for that is an **entity-embedding network**:

    home team ─┐                                          ┌─► P(home win)
               ├─ Embedding(team) ─┐                      │
    away team ─┘                   ├─ concat ─► MLP ─────► ├─► P(draw)
    context  ───────────────────► ┘   (ReLU, dropout)     └─► P(away win)

* A shared **team embedding** learns a dense latent vector per team. Unlike a
  single Elo scalar, a vector can capture *non-transitive* matchup effects
  (style A troubles style B) once the MLP mixes the two teams' vectors.
* **Context features** (all known before kick-off, so no leakage):
  neutral venue, match importance, and each team's recent form (points-per-game
  and goal-difference-per-game over its last 10 matches).
* The MLP outputs 3 logits → softmax. Trained with cross-entropy (which is
  exactly the log-loss we report), so the probabilities stay calibrated.

The model is deliberately small (~10k params) and regularised (dropout + weight
decay + early stopping) because the dataset is only ~9k matches.
"""
from __future__ import annotations

from collections import deque, defaultdict

import numpy as np
import torch
import torch.nn as nn

from elo_model import tournament_weight

K_FORM = 10                     # rolling-form window (matches)
CONT_FEATURES = ["neutral", "importance",
                 "home_ppg", "home_gd", "away_ppg", "away_gd"]
CONT_DIM = len(CONT_FEATURES)
DEFAULT_IMPORTANCE = 30.0       # used for hypothetical GUI matchups (qualifier-ish)


# --------------------------------------------------------------------------- #
# Feature engineering (causal: each row uses only earlier matches)
# --------------------------------------------------------------------------- #
def _form_stats(hist: deque) -> tuple[float, float]:
    if not hist:
        return 1.0, 0.0          # neutral prior before any games are seen
    pts = np.mean([p for p, _ in hist])
    gd = np.mean([g for _, g in hist])
    return float(pts), float(gd)


def build_features(df) -> dict:
    """Build a causal feature table from chronologically-sorted matches.

    Returns a dict with arrays plus the metadata needed to score new matchups:
    team_index, the final rolling-form of every team, and the raw (unscaled)
    continuous features.
    """
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    team_index = {t: i for i, t in enumerate(teams)}

    hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=K_FORM))
    home_ids, away_ids, cont, y, dates = [], [], [], [], []

    for row in df.itertuples(index=False):
        h, a = row.home_team, row.away_team
        h_ppg, h_gd = _form_stats(hist[h])
        a_ppg, a_gd = _form_stats(hist[a])
        imp = tournament_weight(row.tournament)

        home_ids.append(team_index[h])
        away_ids.append(team_index[a])
        cont.append([1.0 if bool(row.neutral) else 0.0, imp,
                     h_ppg, h_gd, a_ppg, a_gd])

        hs, as_ = int(row.home_score), int(row.away_score)
        if hs > as_:
            y.append(0)            # home win
        elif hs == as_:
            y.append(1)            # draw
        else:
            y.append(2)            # away win
        dates.append(row.date)

        # update rolling form AFTER recording features (no leakage)
        gd = hs - as_
        hp = 3 if hs > as_ else (1 if hs == as_ else 0)
        ap = 3 if as_ > hs else (1 if hs == as_ else 0)
        hist[h].append((hp, gd))
        hist[a].append((ap, -gd))

    latest_form = {t: _form_stats(hist[t]) for t in teams}
    return {
        "team_index": team_index,
        "home_ids": np.asarray(home_ids, np.int64),
        "away_ids": np.asarray(away_ids, np.int64),
        "cont": np.asarray(cont, np.float32),
        "y": np.asarray(y, np.int64),
        "dates": np.asarray(dates),
        "latest_form": latest_form,
    }


# --------------------------------------------------------------------------- #
# The network
# --------------------------------------------------------------------------- #
class OutcomeNet(nn.Module):
    def __init__(self, n_teams: int, emb_dim: int = 16,
                 hidden=(64, 32), p_drop: float = 0.3):
        super().__init__()
        self.emb = nn.Embedding(n_teams, emb_dim)
        dims = [2 * emb_dim + CONT_DIM, *hidden]
        layers = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(d_in, d_out), nn.ReLU(), nn.Dropout(p_drop)]
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(dims[-1], 3)
        nn.init.normal_(self.emb.weight, std=0.1)

    def forward(self, home_ids, away_ids, cont):
        eh = self.emb(home_ids)
        ea = self.emb(away_ids)
        x = torch.cat([eh, ea, cont], dim=1)
        return self.head(self.mlp(x))


# --------------------------------------------------------------------------- #
# Predictor wrapper (training-agnostic; holds everything needed to score)
# --------------------------------------------------------------------------- #
class NNPredictor:
    def __init__(self, model: OutcomeNet, team_index: dict,
                 cont_mean: np.ndarray, cont_std: np.ndarray,
                 latest_form: dict, config: dict):
        self.model = model.eval()
        self.team_index = team_index
        self.cont_mean = np.asarray(cont_mean, np.float32)
        self.cont_std = np.asarray(cont_std, np.float32)
        self.latest_form = latest_form
        self.config = config

    # -- scoring helpers ---------------------------------------------------- #
    def _scale(self, cont: np.ndarray) -> np.ndarray:
        return (cont - self.cont_mean) / self.cont_std

    def proba_batch(self, home_ids, away_ids, cont_raw) -> np.ndarray:
        cont = self._scale(np.asarray(cont_raw, np.float32))
        with torch.no_grad():
            logits = self.model(
                torch.as_tensor(np.asarray(home_ids, np.int64)),
                torch.as_tensor(np.asarray(away_ids, np.int64)),
                torch.as_tensor(cont, dtype=torch.float32),
            )
            return torch.softmax(logits, dim=1).numpy()

    def has_team(self, team: str) -> bool:
        return team in self.team_index

    def predict(self, home: str, away: str, neutral: bool = True,
                importance: float = DEFAULT_IMPORTANCE) -> dict:
        """W/D/L for a hypothetical matchup, using each team's current form."""
        if home not in self.team_index or away not in self.team_index:
            raise KeyError("unknown team")
        h_ppg, h_gd = self.latest_form.get(home, (1.0, 0.0))
        a_ppg, a_gd = self.latest_form.get(away, (1.0, 0.0))
        cont = np.array([[1.0 if neutral else 0.0, importance,
                          h_ppg, h_gd, a_ppg, a_gd]], np.float32)
        p = self.proba_batch([self.team_index[home]],
                             [self.team_index[away]], cont)[0]
        pick = ["home", "draw", "away"][int(np.argmax(p))]
        pick_team = {"home": home, "draw": "Draw", "away": away}[pick]
        return {
            "home": home, "away": away, "neutral": neutral,
            "p_home_win": float(p[0]), "p_draw": float(p[1]),
            "p_away_win": float(p[2]), "pick": pick_team,
        }

    # -- persistence -------------------------------------------------------- #
    def save(self, path: str) -> None:
        torch.save({
            "state_dict": self.model.state_dict(),
            "team_index": self.team_index,
            "cont_mean": self.cont_mean,
            "cont_std": self.cont_std,
            "latest_form": self.latest_form,
            "config": self.config,
        }, path)

    @staticmethod
    def load(path: str) -> "NNPredictor":
        blob = torch.load(path, weights_only=False, map_location="cpu")
        cfg = blob["config"]
        model = OutcomeNet(n_teams=len(blob["team_index"]),
                           emb_dim=cfg["emb_dim"], hidden=tuple(cfg["hidden"]),
                           p_drop=cfg["p_drop"])
        model.load_state_dict(blob["state_dict"])
        return NNPredictor(model, blob["team_index"], blob["cont_mean"],
                           blob["cont_std"], blob["latest_form"], cfg)
