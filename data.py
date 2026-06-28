"""
data.py
=======
Fetch and prepare international football (FIFA) match results.

Source: martj42/international_results
    https://github.com/martj42/international_results

The raw CSV has one row per international match with columns:
    date, home_team, away_team, home_score, away_score, tournament, city,
    country, neutral

This module downloads + caches the data and exposes a cleaned DataFrame of
*played* matches over the last N years (default 10) with convenience columns:
    outcome   'H' (home win) / 'D' (draw) / 'A' (away win)
    winner    team name, or None on a draw
    loser     team name, or None on a draw
    goal_diff absolute goal difference

Both models (Elo + dominance graph) consume the output of `load_matches()`.
"""
from __future__ import annotations

import os
import sys
import time
import urllib.request
from datetime import datetime

import pandas as pd

RAW_URL = (
    "https://raw.githubusercontent.com/"
    "martj42/international_results/master/results.csv"
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
CSV_PATH = os.path.join(DATA_DIR, "results.csv")

DEFAULT_YEARS = 10

# Substrings identifying FIFA / confederation competitions. Any team that has
# played in one of these is treated as a FIFA member association. The raw
# dataset also contains CONIFA sides, dependencies and regional teams (Yorkshire,
# Northern Cyprus, Jersey, Greenland, Zanzibar, ...) which we exclude so the
# models reason purely over FIFA teams.
FIFA_COMPETITION_KEYWORDS = (
    "fifa world cup",          # World Cup + qualification
    "uefa euro",               # Euros + qualification
    "uefa nations league",
    "copa américa", "copa america",
    "african cup of nations",  # AFCON + qualification
    "afc asian cup",           # Asian Cup + qualification
    "concacaf",                # Nations League / Championship
    "gold cup",
    "ofc nations cup",
    "confederations cup",
    "fifa series",
)

# Teams that play in the competitions above but are NOT FIFA members. These are
# CONCACAF associate members (French overseas departments / Dutch territories)
# that enter the Gold Cup / Nations League, so the keyword whitelist would
# otherwise let them through. Listed explicitly and removed after the fact.
NON_FIFA_TEAMS = frozenset({
    "French Guiana", "Guadeloupe", "Martinique",
    "Saint Martin", "Sint Maarten", "Bonaire",
})


def download(force: bool = False, retries: int = 4, timeout: int = 60) -> str:
    """Download the results CSV to the local cache (``data/results.csv``).

    Returns the path to the cached file. If a cached copy already exists and
    ``force`` is False, the download is skipped.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(CSV_PATH) and not force:
        return CSV_PATH

    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                RAW_URL, headers={"User-Agent": "FIFAmodel/1.0 (+data.py)"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = resp.read()
            tmp = CSV_PATH + ".tmp"
            with open(tmp, "wb") as fh:
                fh.write(payload)
            os.replace(tmp, CSV_PATH)
            return CSV_PATH
        except Exception as exc:  # network hiccups are common; retry a few times
            last_err = exc
            print(f"[data] download attempt {attempt + 1} failed: {exc}",
                  file=sys.stderr)
            time.sleep(2 + 2 * attempt)
    raise RuntimeError(
        f"Could not download dataset from {RAW_URL}: {last_err}"
    )


def _read_raw(refresh: bool = False) -> pd.DataFrame:
    if refresh or not os.path.exists(CSV_PATH):
        download(force=refresh)
    df = pd.read_csv(CSV_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def fifa_team_set(df: pd.DataFrame) -> set[str]:
    """Teams that appear in any FIFA/confederation competition within ``df``."""
    t = df["tournament"].str.lower()
    mask = pd.Series(False, index=df.index)
    for key in FIFA_COMPETITION_KEYWORDS:
        mask = mask | t.str.contains(key, regex=False, na=False)
    rows = df[mask]
    teams = set(rows["home_team"]) | set(rows["away_team"])
    return teams - NON_FIFA_TEAMS


# The 10 CONMEBOL nations have no separate continental qualifiers (they use the
# World Cup qualifying group), so they are listed explicitly. Every other
# confederation is detected from its own internal competitions, checked in
# priority order so guest invitees (e.g. an AFC side invited to the Gold Cup)
# are assigned to their true confederation.
_CONMEBOL = frozenset({
    "Argentina", "Bolivia", "Brazil", "Chile", "Colombia",
    "Ecuador", "Paraguay", "Peru", "Uruguay", "Venezuela",
})
# OFC members play their continental matches inside generic "World Cup
# qualification", which isn't distinguishable by name, so they are hard-listed.
_OFC = frozenset({
    "New Zealand", "New Caledonia", "Tahiti", "Fiji", "Solomon Islands",
    "Vanuatu", "Papua New Guinea", "Samoa", "American Samoa", "Tonga",
    "Cook Islands",
})
_CONFED_KEYWORDS = (
    ("UEFA", ("uefa euro", "uefa nations league")),
    ("CAF", ("african cup of nations",)),
    ("AFC", ("afc asian cup",)),
    ("CONCACAF", ("concacaf", "gold cup")),
    ("OFC", ("ofc nations cup",)),
)


def team_confederations(df: pd.DataFrame) -> dict[str, str]:
    """Map each team in ``df`` to its FIFA confederation (UEFA, CONMEBOL, …).

    Detected from the competitions a team plays in; CONMEBOL is hard-listed.
    """
    t = df["tournament"].str.lower()
    members: dict[str, set[str]] = {}
    for conf, keys in _CONFED_KEYWORDS:
        mask = pd.Series(False, index=df.index)
        for k in keys:
            mask = mask | t.str.contains(k, regex=False, na=False)
        rows = df[mask]
        members[conf] = set(rows["home_team"]) | set(rows["away_team"])

    out: dict[str, str] = {}
    for team in set(df["home_team"]) | set(df["away_team"]):
        if team in _CONMEBOL:
            out[team] = "CONMEBOL"
            continue
        if team in _OFC:
            out[team] = "OFC"
            continue
        for conf, _ in _CONFED_KEYWORDS:
            if team in members[conf]:
                out[team] = conf
                break
        else:
            out[team] = "Other"
    return out


def load_matches(
    years: int = DEFAULT_YEARS,
    today: datetime | str | None = None,
    refresh: bool = False,
    fifa_only: bool = True,
) -> pd.DataFrame:
    """Return cleaned, *played* matches from the last ``years`` years.

    Parameters
    ----------
    years     : size of the look-back window in years (default 10).
    today     : the "current" date that anchors the window. Defaults to the real
                current date. The window is ``[today - years, today]``.
    refresh   : re-download the dataset before loading.
    fifa_only : keep only matches between FIFA member associations (default
                True), dropping CONIFA/regional/dependency teams.
    """
    df = _read_raw(refresh=refresh)

    if today is None:
        today_ts = pd.Timestamp.today().normalize()
    else:
        today_ts = pd.Timestamp(today).normalize()
    cutoff = today_ts - pd.DateOffset(years=years)

    # Keep only matches that were actually played (scores present) and fall
    # inside the look-back window. Future fixtures carry NaN scores.
    scores_ok = df["home_score"].notna() & df["away_score"].notna()
    in_window = (df["date"] >= cutoff) & (df["date"] <= today_ts)
    df = df[scores_ok & in_window].copy()

    if fifa_only:
        fifa_teams = fifa_team_set(df)
        df = df[df["home_team"].isin(fifa_teams)
                & df["away_team"].isin(fifa_teams)].copy()

    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"] = df["neutral"].astype(bool)

    hs, as_ = df["home_score"], df["away_score"]
    df["outcome"] = "D"
    df.loc[hs > as_, "outcome"] = "H"
    df.loc[hs < as_, "outcome"] = "A"

    df["winner"] = pd.NA
    df.loc[df["outcome"] == "H", "winner"] = df["home_team"]
    df.loc[df["outcome"] == "A", "winner"] = df["away_team"]
    df["loser"] = pd.NA
    df.loc[df["outcome"] == "H", "loser"] = df["away_team"]
    df.loc[df["outcome"] == "A", "loser"] = df["home_team"]

    df["goal_diff"] = (hs - as_).abs()

    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    # Attach metadata other modules can introspect without re-deriving it.
    df.attrs["window_start"] = cutoff
    df.attrs["window_end"] = today_ts
    df.attrs["years"] = years
    df.attrs["fifa_only"] = fifa_only
    return df


def team_list(df: pd.DataFrame) -> list[str]:
    """Sorted list of every team that appears in ``df``."""
    teams = set(df["home_team"]) | set(df["away_team"])
    return sorted(teams)


def summary(df: pd.DataFrame) -> str:
    start = df.attrs.get("window_start")
    end = df.attrs.get("window_end")
    span = ""
    if start is not None and end is not None:
        span = f"  ({start.date()} -> {end.date()})"
    return (
        f"{len(df):,} played matches across {len(team_list(df))} teams"
        f"{span}"
    )


if __name__ == "__main__":
    # Smoke test: download (if needed) and describe the window.
    data = load_matches()
    print(summary(data))
    print("\nMost recent 5 matches:")
    cols = ["date", "home_team", "home_score", "away_score", "away_team",
            "tournament"]
    print(data[cols].tail(5).to_string(index=False))
