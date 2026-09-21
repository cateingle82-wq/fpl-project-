"""
Fetches and caches historical per-gameweek FPL data from the
vaastav/Fantasy-Premier-League GitHub archive, and converts it into the
standardized schema ml_features.py expects.

Why this repo and not the live FPL API for training data: the live API
(bootstrap-static/, element-summary/) only ever tells you about the
CURRENT season, and only ~6 gameweeks of it exist so far this season —
nowhere near enough to train a model on. This archive has full completed
seasons back to 2020-21, thousands of player-gameweeks each. The current,
in-progress season stays on the live API path (fpl_stage0.py / backtest.py)
because this archive updates on its own schedule and lagged behind by
several gameweeks when checked — fine for finished seasons, not something
to build a live prediction on.

Run standalone to pre-populate the cache:  python fpl_history.py
"""

import io
import os

import pandas as pd
import requests

RAW_BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
CACHE_DIR = "history_cache"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# Position codes changed slightly season to season (an "AM" — Assistant
# Manager, a newer FPL chip-related pseudo-position — shows up in recent
# seasons' raw data). We only ever optimise GK/DEF/MID/FWD squads, so any
# other value is dropped rather than guessed at.
VALID_POSITIONS = {"GK", "DEF", "MID", "FWD"}


def _cached_csv(url, cache_path):
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path)
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


def fetch_season_gws(season):
    """Raw per-player-per-gameweek rows for one season, e.g. '2023-24'."""
    url = f"{RAW_BASE}/{season}/gws/merged_gw.csv"
    path = os.path.join(CACHE_DIR, season, "merged_gw.csv")
    return _cached_csv(url, path)


def fetch_season_teams(season):
    """Team strength ratings for one season — same schema fpl_stage0 could
    read for the current season via bootstrap-static's 'teams' list."""
    url = f"{RAW_BASE}/{season}/teams.csv"
    path = os.path.join(CACHE_DIR, season, "teams.csv")
    return _cached_csv(url, path)


def team_strength_dict(teams_df):
    """team_id -> {attack_home, attack_away, defence_home, defence_away}."""
    out = {}
    for _, r in teams_df.iterrows():
        out[r["id"]] = {
            "attack_home": r["strength_attack_home"],
            "attack_away": r["strength_attack_away"],
            "defence_home": r["strength_defence_home"],
            "defence_away": r["strength_defence_away"],
        }
    return out


def standardize_season(season):
    """
    Returns (rows_df, team_strength) for one season, in the schema
    ml_features.build_features() expects. Rows with a non-standard
    position (AM) or with no minutes info at all are dropped up front —
    they were never eligible squad picks in our optimiser's sense.
    """
    gws = fetch_season_gws(season)
    teams = fetch_season_teams(season)
    strength = team_strength_dict(teams)

    gws = gws[gws["position"].isin(VALID_POSITIONS)].copy()

    df = pd.DataFrame({
        "player_id": gws["element"],
        "season": season,
        "gw": gws["GW"],
        "position": gws["position"],
        "team_id": None,          # not needed downstream; opponent_id is what matters
        "opponent_id": gws["opponent_team"],
        "was_home": gws["was_home"].astype(bool),
        "total_points": gws["total_points"],
        "minutes": gws["minutes"],
        "ict_index": gws["ict_index"],
        "influence": gws["influence"],
        "creativity": gws["creativity"],
        "threat": gws["threat"],
        "bps": gws["bps"],
        "bonus": gws["bonus"],
        # Older seasons (pre-2022-23) never tracked xG — becomes float NaN
        # (not pandas' NA scalar, which silently turns the whole column
        # into dtype=object and breaks .rolling()), which ml_features
        # handles by letting every derived rolling column be NaN too, not
        # by dropping the season.
        "expected_goal_involvements": pd.to_numeric(
            gws["expected_goal_involvements"], errors="coerce"
        ) if "expected_goal_involvements" in gws.columns
        else pd.Series(float("nan"), index=gws.index),
        # Split out from the combined xGI above so the model can tell a
        # pure poacher (high xG, low xA) from a creator (the reverse) —
        # same reasoning as influence/creativity/threat being rolled
        # separately instead of only as the combined ict_index. Same
        # older-seasons-never-tracked-this NaN fallback as xGI.
        "expected_goals": pd.to_numeric(
            gws["expected_goals"], errors="coerce"
        ) if "expected_goals" in gws.columns
        else pd.Series(float("nan"), index=gws.index),
        "expected_assists": pd.to_numeric(
            gws["expected_assists"], errors="coerce"
        ) if "expected_assists" in gws.columns
        else pd.Series(float("nan"), index=gws.index),
        "value": gws["value"],
        "starts": pd.to_numeric(gws["starts"], errors="coerce")
        if "starts" in gws.columns else pd.Series(float("nan"), index=gws.index),
        # A fixture's scheduled date, not a match outcome — known well
        # before kickoff, so using it is not leakage. Feeds ml_features'
        # rest_days (fixture-congestion / rotation-risk signal).
        "kickoff_time": gws["kickoff_time"],
    })

    return df, {season: strength}


def load_seasons(seasons):
    """Concatenate standardize_season() over several seasons into one long
    table plus a combined team-strength lookup keyed by season."""
    all_rows, all_strength = [], {}
    for season in seasons:
        rows, strength = standardize_season(season)
        all_rows.append(rows)
        all_strength.update(strength)
    return pd.concat(all_rows, ignore_index=True), all_strength


if __name__ == "__main__":
    SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
    for s in SEASONS:
        df, _ = standardize_season(s)
        print(f"{s}: {len(df)} rows, {df['player_id'].nunique()} players, "
              f"GW range {df['gw'].min()}-{df['gw'].max()}")
