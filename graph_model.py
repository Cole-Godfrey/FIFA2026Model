"""
graph_model.py
==============
A "logical dominance graph" model for predicting international football results.

The idea
--------
Build a directed graph of teams. Whenever one team has the upper hand over
another, draw an **arrow pointing to the team that won**:

        loser  ───►  winner          (arrow points to the winner)

Dominance is transitive. If

        team1  ───►  team3           (team3 beat team1)
        team2  ───►  team1           (team1 beat team2)

then by following the arrows  team2 ──► team1 ──► team3  we conclude that
**team3 beats team2** even if they never actually met.

To predict A vs B we look for a directed path between them:

  * a path  A ──► … ──► B   means every arrow leads toward B, so **B wins**;
  * a path  B ──► … ──► A   means **A wins**.

The path itself is the justification, e.g.  ``Panama → Mexico → Argentina``
("Mexico beat Panama, Argentina beat Mexico ⟹ Argentina beats Panama").

Edges are not built from single games (international football is full of
upsets). For each pair of teams we aggregate every meeting in the window into
a single **net dominance score** — weighted by recency (recent games matter
more) and margin of victory — and draw one edge in the direction of whoever
comes out on top. Pairs that are dead even get no edge.

Path search prefers the **fewest hops** (the most direct dominance argument);
among equally short paths it prefers the one built from the strongest links.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import pandas as pd

from elo_model import goal_diff_multiplier

RECENCY_HALFLIFE_YEARS = 3.0   # a win this many years ago counts half as much
# Dijkstra trick: cost = 1 - EPS * normalised_strength. Because EPS is tiny the
# hop count always dominates, and strength only breaks ties between paths of
# equal length. Keep EPS < 1/(max realistic path length).
_EPS = 0.01


@dataclass
class PairRecord:
    """Aggregated head-to-head info for an unordered pair (a, b), a < b."""
    a: str
    b: str
    a_wins: int = 0
    b_wins: int = 0
    draws: int = 0
    a_goals: int = 0
    b_goals: int = 0
    score: float = 0.0          # net dominance, from a's perspective (+ => a)
    last_meeting: pd.Timestamp | None = None

    @property
    def meetings(self) -> int:
        return self.a_wins + self.b_wins + self.draws

    def record_from(self, team: str) -> str:
        """W-D-L string from ``team``'s perspective."""
        if team == self.a:
            return f"{self.a_wins}W-{self.draws}D-{self.b_wins}L"
        return f"{self.b_wins}W-{self.draws}D-{self.a_wins}L"


@dataclass
class GraphModel:
    halflife_years: float = RECENCY_HALFLIFE_YEARS
    graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    pairs: dict[tuple[str, str], PairRecord] = field(default_factory=dict)
    _max_strength: float = 1.0

    # --------------------------------------------------------------- build
    def fit(self, matches: pd.DataFrame) -> "GraphModel":
        self.graph = nx.DiGraph()
        self.pairs.clear()

        window_end = matches.attrs.get("window_end")
        if window_end is None:
            window_end = matches["date"].max()

        for row in matches.itertuples(index=False):
            home, away = row.home_team, row.away_team
            a, b = (home, away) if home < away else (away, home)
            key = (a, b)
            rec = self.pairs.get(key)
            if rec is None:
                rec = PairRecord(a=a, b=b)
                self.pairs[key] = rec

            hs, as_ = int(row.home_score), int(row.away_score)
            # goals from a's / b's perspective
            if a == home:
                a_g, b_g = hs, as_
            else:
                a_g, b_g = as_, hs
            rec.a_goals += a_g
            rec.b_goals += b_g
            rec.last_meeting = (
                row.date if rec.last_meeting is None
                else max(rec.last_meeting, row.date)
            )

            # recency + margin weighting
            age_years = max((window_end - row.date).days, 0) / 365.25
            recency = 0.5 ** (age_years / self.halflife_years)
            margin = goal_diff_multiplier(abs(hs - as_))
            contribution = recency * margin

            if hs == as_:
                rec.draws += 1
                # draws carry no dominance signal
            elif a_g > b_g:
                rec.a_wins += 1
                rec.score += contribution
            else:
                rec.b_wins += 1
                rec.score -= contribution

        self._build_edges()
        return self

    def _build_edges(self) -> None:
        g = self.graph
        # ensure every team is a node even if all its pairs are even
        for a, b in self.pairs:
            g.add_node(a)
            g.add_node(b)

        strengths = []
        for (a, b), rec in self.pairs.items():
            if rec.score > 0:            # a dominates b -> arrow b -> a
                loser, winner, strength = b, a, rec.score
            elif rec.score < 0:          # b dominates a -> arrow a -> b
                loser, winner, strength = a, b, -rec.score
            else:
                continue                 # dead even: no edge
            g.add_edge(loser, winner, strength=strength,
                       meetings=rec.meetings)
            strengths.append(strength)

        self._max_strength = max(strengths) if strengths else 1.0
        # assign Dijkstra costs (cheaper = stronger, but ~1 per hop)
        for _, _, d in g.edges(data=True):
            norm = d["strength"] / self._max_strength
            d["cost"] = 1.0 - _EPS * norm

    # --------------------------------------------------------------- query
    def has_team(self, team: str) -> bool:
        return team in self.graph

    def pair_record(self, x: str, y: str) -> PairRecord | None:
        a, b = (x, y) if x < y else (y, x)
        return self.pairs.get((a, b))

    def _best_path(self, src: str, dst: str):
        """Shortest (fewest-hop, then strongest) directed path src -> dst.

        Returns (path, hops, total_strength, min_strength) or None.
        """
        if src not in self.graph or dst not in self.graph:
            return None
        try:
            path = nx.shortest_path(self.graph, src, dst, weight="cost")
        except nx.NetworkXNoPath:
            return None
        except nx.NodeNotFound:
            return None
        strengths = [
            self.graph[u][v]["strength"]
            for u, v in zip(path[:-1], path[1:])
        ]
        return (path, len(path) - 1, sum(strengths),
                min(strengths) if strengths else 0.0)

    def predict(self, team_a: str, team_b: str) -> dict:
        """Predict the winner of ``team_a`` vs ``team_b`` from the graph.

        Returns a dict with keys: winner, loser, method, confidence, path
        (justification chain, winner last), reading (plain-English), h2h,
        and explanation.
        """
        out = {
            "team_a": team_a,
            "team_b": team_b,
            "winner": None,
            "loser": None,
            "method": "none",
            "confidence": "none",
            "path": None,
            "reading": None,
            "h2h": None,
            "explanation": "",
        }

        if not self.has_team(team_a) or not self.has_team(team_b):
            missing = [t for t in (team_a, team_b) if not self.has_team(t)]
            out["explanation"] = (
                f"No matches on record for: {', '.join(missing)}."
            )
            return out

        rec = self.pair_record(team_a, team_b)
        if rec is not None and rec.meetings:
            out["h2h"] = {
                "meetings": rec.meetings,
                "record_a": rec.record_from(team_a),
                "record_b": rec.record_from(team_b),
                "goals_a": rec.a_goals if team_a == rec.a else rec.b_goals,
                "goals_b": rec.b_goals if team_a == rec.a else rec.a_goals,
                "last_meeting": rec.last_meeting,
            }

        # A path a -> ... -> b means b wins; b -> ... -> a means a wins.
        path_b_wins = self._best_path(team_a, team_b)   # endpoint b wins
        path_a_wins = self._best_path(team_b, team_a)   # endpoint a wins

        chosen, winner = self._choose(path_a_wins, path_b_wins,
                                      team_a, team_b)
        if chosen is None:
            return self._structural_fallback(out, team_a, team_b)

        path, hops, _total, _mn = chosen
        loser = path[0]
        out["winner"] = winner
        out["loser"] = loser
        out["path"] = path
        out["method"] = "head-to-head" if hops == 1 else "transitive"
        out["confidence"] = (
            "high" if hops == 1 else "medium" if hops == 2 else "low"
        )
        out["reading"] = self._reading(path)
        out["explanation"] = self._explain(path, winner, hops)
        return out

    @staticmethod
    def _choose(path_a_wins, path_b_wins, team_a, team_b):
        """Pick the more convincing of the two directional paths."""
        if path_a_wins is None and path_b_wins is None:
            return None, None
        if path_a_wins is None:
            return path_b_wins, team_b
        if path_b_wins is None:
            return path_a_wins, team_a
        # Both directions reachable (the relation has a cycle). Prefer fewer
        # hops; tie-break on stronger aggregate, then stronger weakest link.
        a_key = (path_a_wins[1], -path_a_wins[2], -path_a_wins[3])
        b_key = (path_b_wins[1], -path_b_wins[2], -path_b_wins[3])
        if a_key <= b_key:
            return path_a_wins, team_a
        return path_b_wins, team_b

    def _structural_fallback(self, out: dict, team_a: str, team_b: str) -> dict:
        """No connecting path: compare each side's net direct dominance.

        Transitive reach is useless here (the graph is one big cycle-rich
        component, so almost everyone reaches everyone). Instead we compare net
        direct dominance = (teams directly beaten) − (teams directly lost to).
        """
        dom_a = self.dominance_score(team_a)
        dom_b = self.dominance_score(team_b)
        out["method"] = "structural-fallback"
        out["confidence"] = "very-low"
        if dom_a == dom_b:
            out["explanation"] = (
                f"No dominance path links {team_a} and {team_b}, and both have "
                f"a net direct dominance of {dom_a}. The graph cannot separate "
                f"them — treat as a toss-up."
            )
            return out
        winner = team_a if dom_a > dom_b else team_b
        out["winner"] = winner
        out["loser"] = team_b if winner == team_a else team_a
        out["explanation"] = (
            f"No dominance path links {team_a} and {team_b}. Falling back to "
            f"net direct dominance: {team_a}={dom_a:+d}, {team_b}={dom_b:+d}. "
            f"Edge to {winner} (weak, structural evidence)."
        )
        return out

    @staticmethod
    def _reading(path: list[str]) -> str:
        """Plain-English winner-first chain, e.g. 'C beat B, B beat A'."""
        # path is loser ... winner; each step u->v means v beat u.
        steps = []
        for u, v in zip(path[:-1], path[1:]):
            steps.append(f"{v} beat {u}")
        steps.reverse()  # lead with the ultimate winner
        return ", ".join(steps)

    def _explain(self, path: list[str], winner: str, hops: int) -> str:
        arrow_chain = " → ".join(path)
        if hops == 1:
            return (f"{winner} has the head-to-head edge over {path[0]}.  "
                    f"[{arrow_chain}]")
        chain = self._reading(path)
        return (f"{chain}  ⟹  {winner} transitively dominates {path[0]} "
                f"in {hops} steps.  [arrows point to winners: {arrow_chain}]")

    # ------------------------------------------------------- introspection
    def dominance_score(self, team: str) -> int:
        """Net direct dominance: teams directly beaten − teams directly lost to.

        Edges point loser → winner, so a team's in-degree is the number of
        opponents it has a net winning record against, and its out-degree is
        the number it has a net losing record against.
        """
        if team not in self.graph:
            return 0
        return self.graph.in_degree(team) - self.graph.out_degree(team)

    def top_dominators(self, n: int = 20) -> list[tuple[str, int]]:
        """Teams ranked by net direct dominance (a bounded, honest metric).

        Note: this is *not* a power ranking — for that use the Elo model. It
        just counts directly-dominated opponents minus directly-dominating
        ones, which is meaningful even though transitive reach is not.
        """
        scored = [(t, self.dominance_score(t)) for t in self.graph.nodes]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored[:n]

    def stats(self) -> dict:
        return {
            "teams": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "pairs": len(self.pairs),
        }


def build_graph(matches: pd.DataFrame) -> GraphModel:
    return GraphModel().fit(matches)


if __name__ == "__main__":
    from data import load_matches

    df = load_matches()
    gm = build_graph(df)
    print("Graph:", gm.stats())
    print("\nMost dominant (net direct dominance = beaten − lost-to):")
    for team, score in gm.top_dominators(10):
        print(f"  {team:<18} {score:+d}")

    for a, b in [("Brazil", "France"), ("Panama", "Argentina"),
                 ("Japan", "Germany")]:
        print(f"\n=== {a} vs {b} ===")
        p = gm.predict(a, b)
        print(f"  winner : {p['winner']}  ({p['method']}, "
              f"conf={p['confidence']})")
        if p["path"]:
            print(f"  path   : {' → '.join(p['path'])}")
        print(f"  why    : {p['explanation']}")
