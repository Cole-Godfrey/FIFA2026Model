"""
elo_rankings.py
===============
Compute and report the Elo power ranking of international teams over the last
10 years, and (optionally) a specific W/D/L prediction.

Usage
-----
    python elo_rankings.py                      # print top rankings, save CSV
    python elo_rankings.py --all                # print every ranked team
    python elo_rankings.py --top 50             # print the top 50
    python elo_rankings.py --min-games 20       # only teams with >= 20 games
    python elo_rankings.py --predict Brazil France
    python elo_rankings.py --predict Brazil France --home   # Brazil at home

The full ranking is always written to ``elo_rankings.csv``.
"""
from __future__ import annotations

import argparse
import os

from data import load_matches, summary
from elo_model import build_elo

CSV_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "elo_rankings.csv")


def print_ranking(model, full_frame, top: int | None, min_games: int) -> None:
    frame = full_frame[full_frame["games"] >= min_games]
    shown = frame if top is None else frame.head(top)
    print(f"\nElo ranking  (draw model: P(draw)={model.draw_a:.3f}*"
          f"exp(-(dr/{model.draw_s:.0f})^2))")
    print("-" * 64)
    header = f"{'#':>3}  {'team':<24}{'elo':>7}  {'games':>5}  {'record':>12}"
    print(header)
    print("-" * 64)
    for r in shown.itertuples(index=False):
        print(f"{r.rank:>3}  {r.team:<24}{r.elo:>7.1f}  {r.games:>5}  "
              f"{r.record:>12}")


def print_prediction(model, home: str, away: str, neutral: bool) -> None:
    for t in (home, away):
        if t not in model.teams:
            print(f"\n[!] Unknown team: '{t}'. "
                  f"Check spelling against elo_rankings.csv.")
            return
    p = model.predict(home, away, neutral=neutral)
    venue = "neutral venue" if neutral else f"{home} at home"
    print(f"\nElo prediction — {home} vs {away}  ({venue})")
    print("-" * 56)
    print(f"  {home:<22} Elo {p['rating_home']:7.1f}")
    print(f"  {away:<22} Elo {p['rating_away']:7.1f}")
    print(f"  rating edge: {p['rating_diff']:+.1f} (home perspective)")
    print(f"\n  {home} win : {p['p_home_win']*100:5.1f}%")
    print(f"  draw      : {p['p_draw']*100:5.1f}%")
    print(f"  {away} win : {p['p_away_win']*100:5.1f}%")
    print(f"\n  => most likely: {p['pick']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top", type=int, default=25,
                    help="how many teams to print (default 25)")
    ap.add_argument("--all", action="store_true",
                    help="print every ranked team")
    ap.add_argument("--min-games", type=int, default=10,
                    help="exclude teams with fewer than this many games "
                         "(default 10)")
    ap.add_argument("--predict", nargs=2, metavar=("TEAM_A", "TEAM_B"),
                    help="also print a W/D/L prediction for two teams")
    ap.add_argument("--home", action="store_true",
                    help="treat the first predicted team as playing at home "
                         "(default: neutral venue)")
    args = ap.parse_args()

    df = load_matches()
    print(summary(df))
    model = build_elo(df)

    # Build the full (global-rank) table once: persist it, then filter for view.
    full = model.ranking_frame(min_games=1)
    full.to_csv(CSV_OUT, index=False)
    print(f"Full ranking ({len(full)} teams) saved -> {CSV_OUT}")

    top = None if args.all else args.top
    print_ranking(model, full, top=top, min_games=args.min_games)

    if args.predict:
        a, b = args.predict
        print_prediction(model, a, b, neutral=not args.home)


if __name__ == "__main__":
    main()
