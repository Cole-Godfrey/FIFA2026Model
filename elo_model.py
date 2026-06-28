"""
elo_model.py
============
An Elo rating model for international football, in the style of the
World Football Elo Ratings (eloratings.net).

Design
------
* Every team starts at ``BASE_RATING`` (1500).
* Matches are processed in chronological order. After each match both teams'
  ratings are updated:

      R' = R + K * G * (S - E)

  where
      E  expected score from the logistic Elo curve (includes home advantage),
      S  actual score (1 win / 0.5 draw / 0 loss),
      K  match-importance weight (World-Cup finals matter more than friendlies),
      G  goal-difference multiplier (bigger wins move ratings more).

* The update is zero-sum: the winner gains exactly what the loser drops.

Turning a rating difference into W/D/L
--------------------------------------
Elo's expected score ``E`` mixes wins and draws (a draw counts 0.5), so Elo
alone does not tell you how often a match is drawn. We therefore *learn* a
draw model from history: for every processed match we record the pre-match
rating difference and whether it was drawn, then fit

      P(draw) = a * exp(-(dr / s)**2)

to the empirical draw rate. Win/loss probabilities then follow from the Elo
identity ``E = P(win) + 0.5 * P(draw)``:

      P(home win) = E - P(draw)/2
      P(away win) = 1 - E - P(draw)/2

which keeps the probabilities exactly consistent with the rating difference.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

BASE_RATING = 1500.0
HOME_ADVANTAGE = 65.0  # Elo points added to the home side on non-neutral games


def tournament_weight(tournament: str) -> float:
    """Map a tournament name to a K-factor (match importance).

    Mirrors the World Football Elo weighting: World Cup finals carry the most
    weight, friendlies the least.
    """
    t = (tournament or "").lower()
    if "world cup" in t and "qualif" not in t:
        return 60.0
    if "world cup" in t and "qualif" in t:
        return 40.0
    if "confederations cup" in t:
        return 50.0
    # Continental finals
    finals_keys = ("uefa euro", "copa américa", "copa america",
                   "african cup of nations", "afc asian cup", "gold cup")
    if any(k in t for k in finals_keys) and "qualif" not in t:
        return 50.0
    if "nations league" in t:
        return 40.0
    if "qualif" in t:
        return 40.0
    if "friendly" in t:
        return 20.0
    return 30.0


def goal_diff_multiplier(goal_diff: int) -> float:
    """World Football Elo goal-difference index G."""
    gd = abs(int(goal_diff))
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    if gd == 3:
        return 1.75
    return 1.75 + (gd - 3) / 8.0


def expected_score(rating_diff: float) -> float:
    """Logistic Elo expectation for the side whose (effective) rating is higher
    by ``rating_diff`` points."""
    return 1.0 / (1.0 + 10.0 ** (-rating_diff / 400.0))


@dataclass
class TeamStat:
    rating: float = BASE_RATING
    games: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    last_played: pd.Timestamp | None = None
    peak: float = BASE_RATING

    @property
    def record(self) -> str:
        return f"{self.wins}W-{self.draws}D-{self.losses}L"


@dataclass
class EloModel:
    base_rating: float = BASE_RATING
    home_advantage: float = HOME_ADVANTAGE
    # Learned draw-model parameters: P(draw) = draw_a * exp(-(dr/draw_s)**2)
    draw_a: float = 0.28
    draw_s: float = 300.0
    teams: dict[str, TeamStat] = field(default_factory=dict)
    _fitted: bool = False

    # ------------------------------------------------------------------ fit
    def _stat(self, team: str) -> TeamStat:
        st = self.teams.get(team)
        if st is None:
            st = TeamStat(rating=self.base_rating, peak=self.base_rating)
            self.teams[team] = st
        return st

    def fit(self, matches: pd.DataFrame) -> "EloModel":
        """Process every match chronologically and learn ratings + draw model."""
        self.teams.clear()
        dr_hist: list[float] = []   # pre-match effective rating difference
        drawn_hist: list[int] = []  # 1 if the match was a draw

        # itertuples is ~10x faster than iterrows for this many rows.
        for row in matches.itertuples(index=False):
            home, away = row.home_team, row.away_team
            hs, as_ = int(row.home_score), int(row.away_score)
            neutral = bool(row.neutral)

            h = self._stat(home)
            a = self._stat(away)

            hfa = 0.0 if neutral else self.home_advantage
            dr = (h.rating + hfa) - a.rating          # home perspective
            e_home = expected_score(dr)

            if hs > as_:
                s_home = 1.0
            elif hs < as_:
                s_home = 0.0
            else:
                s_home = 0.5

            k = tournament_weight(row.tournament)
            g = goal_diff_multiplier(hs - as_)
            delta = k * g * (s_home - e_home)

            h.rating += delta
            a.rating -= delta

            # Bookkeeping
            for st in (h, a):
                st.games += 1
                st.last_played = row.date
                st.peak = max(st.peak, st.rating)
            if s_home == 1.0:
                h.wins += 1; a.losses += 1
            elif s_home == 0.0:
                h.losses += 1; a.wins += 1
            else:
                h.draws += 1; a.draws += 1

            dr_hist.append(dr)
            drawn_hist.append(1 if s_home == 0.5 else 0)

        self._fit_draw_model(np.asarray(dr_hist), np.asarray(drawn_hist))
        self._fitted = True
        return self

    def _fit_draw_model(self, dr: np.ndarray, drawn: np.ndarray) -> None:
        """Fit P(draw) = a * exp(-(dr/s)**2) to empirical draw rates.

        We bin matches by their pre-match rating difference, compute the draw
        rate per bin, and do a count-weighted least-squares fit in log space.
        Falls back to sensible defaults if the data is too thin.
        """
        if len(dr) < 200:
            return  # keep defaults
        edges = np.arange(-500, 501, 50.0)
        centers, rates, weights = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (dr >= lo) & (dr < hi)
            n = int(mask.sum())
            if n < 25:
                continue
            rate = float(drawn[mask].mean())
            centers.append((lo + hi) / 2.0)
            rates.append(rate)
            weights.append(n)
        centers = np.asarray(centers)
        rates = np.asarray(rates)
        weights = np.asarray(weights, dtype=float)
        # Avoid log(0); clamp the empirical rates into (eps, peak).
        rates = np.clip(rates, 1e-3, 0.6)
        # ln(rate) = ln(a) - (1/s^2) * dr^2  ->  linear in x = dr^2.
        x = centers ** 2
        y = np.log(rates)
        w = weights
        # Weighted linear regression y = b0 + b1*x.
        sw = w.sum()
        xm = (w * x).sum() / sw
        ym = (w * y).sum() / sw
        sxx = (w * (x - xm) ** 2).sum()
        sxy = (w * (x - xm) * (y - ym)).sum()
        if sxx <= 0:
            return
        b1 = sxy / sxx
        b0 = ym - b1 * xm
        a = float(np.exp(b0))
        # b1 should be negative; s = sqrt(-1/b1).
        if b1 >= 0:
            return
        s = float(np.sqrt(-1.0 / b1))
        # Guardrails against pathological fits.
        self.draw_a = float(np.clip(a, 0.10, 0.45))
        self.draw_s = float(np.clip(s, 120.0, 800.0))

    # -------------------------------------------------------------- predict
    def rating(self, team: str) -> float:
        st = self.teams.get(team)
        return st.rating if st else self.base_rating

    def p_draw(self, rating_diff: float) -> float:
        return float(self.draw_a * np.exp(-((rating_diff / self.draw_s) ** 2)))

    def predict(self, home: str, away: str, neutral: bool = True) -> dict:
        """Predict W/D/L for ``home`` vs ``away``.

        With ``neutral=True`` (the default) no home-field bonus is applied, so
        the result is a pure team-vs-team comparison. Set ``neutral=False`` to
        treat ``home`` as playing at home.
        """
        rh = self.rating(home)
        ra = self.rating(away)
        hfa = 0.0 if neutral else self.home_advantage
        dr = (rh + hfa) - ra
        e_home = expected_score(dr)
        pd_ = self.p_draw(dr)
        # Keep draw mass feasible given the expectation.
        pd_ = min(pd_, 2.0 * e_home, 2.0 * (1.0 - e_home))
        p_home = e_home - pd_ / 2.0
        p_away = 1.0 - e_home - pd_ / 2.0
        # Clamp tiny negatives from numerical edges and renormalise.
        p_home = max(p_home, 0.0)
        p_away = max(p_away, 0.0)
        total = p_home + pd_ + p_away
        p_home, pd_, p_away = p_home / total, pd_ / total, p_away / total

        if p_home >= p_away and p_home >= pd_:
            pick = home
        elif p_away >= p_home and p_away >= pd_:
            pick = away
        else:
            pick = "Draw"

        return {
            "home": home,
            "away": away,
            "neutral": neutral,
            "rating_home": rh,
            "rating_away": ra,
            "rating_diff": dr,
            "expected_home": e_home,
            "p_home_win": p_home,
            "p_draw": pd_,
            "p_away_win": p_away,
            "pick": pick,
        }

    # ------------------------------------------------------------- ranking
    def ranking(self, min_games: int = 1) -> list[tuple[str, TeamStat]]:
        items = [
            (name, st)
            for name, st in self.teams.items()
            if st.games >= min_games
        ]
        items.sort(key=lambda kv: kv[1].rating, reverse=True)
        return items

    def ranking_frame(self, min_games: int = 1) -> pd.DataFrame:
        """Ranking table. ``rank`` is always the team's *global* Elo rank;
        ``min_games`` only filters which rows are returned, it does not
        renumber them, so ranks stay consistent across every view."""
        rows = []
        for rank, (name, st) in enumerate(self.ranking(min_games=1), start=1):
            if st.games < min_games:
                continue
            rows.append({
                "rank": rank,
                "team": name,
                "elo": round(st.rating, 1),
                "games": st.games,
                "record": st.record,
                "peak_elo": round(st.peak, 1),
            })
        return pd.DataFrame(rows)


def build_elo(matches: pd.DataFrame) -> EloModel:
    """Convenience constructor: fit an EloModel on the given matches."""
    return EloModel().fit(matches)


if __name__ == "__main__":
    from data import load_matches

    df = load_matches()
    model = build_elo(df)
    print(f"Fitted Elo on {len(df):,} matches.")
    print(f"Learned draw model: P(draw)={model.draw_a:.3f}*"
          f"exp(-(dr/{model.draw_s:.0f})^2)")
    print("\nTop 15 by Elo:")
    print(model.ranking_frame(min_games=15).head(15).to_string(index=False))
    print("\nExample (neutral): Brazil vs France")
    p = model.predict("Brazil", "France", neutral=True)
    print(f"  ratings {p['rating_home']:.0f} vs {p['rating_away']:.0f}")
    print(f"  Brazil win {p['p_home_win']*100:4.1f}% | "
          f"draw {p['p_draw']*100:4.1f}% | "
          f"France win {p['p_away_win']*100:4.1f}%  -> pick {p['pick']}")
