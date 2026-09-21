"""
Shared feature engineering for the ML points-predictor.

The whole point of a separate module: training (on past seasons) and live
prediction (on the current season) must build features EXACTLY the same
way, or the model sees different inputs than it was trained on and its
accuracy claims are meaningless. Both ml_train.py and ml_predict.py call
build_features() below — neither computes a feature by hand.

Leakage rule (same discipline as backtest.py's `round < target_gw`, just
implemented as a rolling shift instead of an explicit date filter): every
feature describing "how has this player been playing" is computed from
STRICTLY EARLIER gameweeks only, via groupby(...).shift(1) before any
rolling window. Row N's features never see row N's own result.

Standard input schema (one row per player per gameweek played), whatever
the source:
    player_id, season, gw, position, team_id, opponent_id, was_home,
    total_points, minutes, ict_index, influence, creativity, threat,
    bps, bonus, expected_goal_involvements, expected_goals,
    expected_assists, value, starts

`expected_goal_involvements`/`expected_goals`/`expected_assists` may be
all-NaN (older seasons didn't track xG) — every rolling feature built
from them degrades to NaN too, and the
model (HistGradientBoostingRegressor) handles NaN inputs natively, so
older seasons still contribute their other features instead of being
dropped entirely.
"""

import numpy as np
import pandas as pd

# Rolling windows, in number of past gameweeks. Two short windows because a
# 3-game window reacts fast to a change in role/fitness, a 5-game window is
# steadier — the model gets both and can weigh them itself instead of us
# guessing which horizon matters more. A longer 10-game window is added
# separately, only for total_points/minutes/starts, as a "nailed-on starter
# vs rotation risk" signal that a 3-5 game window is too short-sighted to
# capture reliably (e.g. a player rested for one cup game shouldn't look
# like a rotation risk in a 3-game average).
ROLL_WINDOWS = (3, 5)
LONG_WINDOW = 10

# influence/creativity/threat are ict_index's own components (ict_index is
# just influence+creativity+threat scaled down) — rolling them separately
# instead of only their combined total lets a tree-based model split on
# "high threat, low creativity" (an out-and-out striker's profile) in a way
# a single blended number can't represent. bonus and starts are cheap,
# already-available extra signal: bonus captures "the sort of player who
# gets rewarded beyond raw output" (a proxy for the bonus-points system's
# quirks), starts is a much more direct "will they actually play" signal
# than minutes alone (a start capped by a 60th-minute substitution still
# counts as a start).
ROLLED_STATS = (
    "total_points", "minutes", "ict_index", "bps",
    "expected_goal_involvements", "expected_goals", "expected_assists",
    "influence", "creativity", "threat", "bonus", "starts",
)
LONG_ROLLED_STATS = ("total_points", "minutes", "starts")

FEATURE_COLS = [
    "position", "was_home", "value",
    "games_so_far",
    "opp_strength_attack", "opp_strength_defence",
] + [
    f"{stat}_roll{w}" for stat in ROLLED_STATS for w in ROLL_WINDOWS
] + [
    f"{stat}_roll{LONG_WINDOW}" for stat in LONG_ROLLED_STATS
]

TARGET_COL = "total_points"


def add_opponent_strength(df, team_strength_by_season):
    """
    For each row, look up the OPPONENT's strength in the venue the
    opponent is actually playing at (they're playing away if our player is
    home, and vice versa) — this is the fixture difficulty signal: a
    tougher opponent defence suppresses attacking returns, a tougher
    opponent attack suppresses defenders'/keepers' clean-sheet points.

    Built as a merge, not a per-row Python loop — the loop version took
    ~50s on 6 seasons; this is a couple of seconds. Same result either way,
    this is purely a speed fix (see test_opponent_strength_picks_the_right_
    venue, which still passes against this version).
    """
    lookup_rows = [
        {"season": season, "opp_team_id": team_id, **strength}
        for season, teams in team_strength_by_season.items()
        for team_id, strength in teams.items()
    ]
    lookup = pd.DataFrame(lookup_rows)

    df = df.merge(
        lookup, how="left",
        left_on=["season", "opponent_id"], right_on=["season", "opp_team_id"],
    )

    # Player home -> opponent away, and vice versa.
    df["opp_strength_attack"] = np.where(df["was_home"], df["attack_away"], df["attack_home"])
    df["opp_strength_defence"] = np.where(df["was_home"], df["defence_away"], df["defence_home"])

    return df.drop(columns=["opp_team_id", "attack_home", "attack_away",
                             "defence_home", "defence_away"])


def add_rolling_form(df):
    """
    Per player, per season (form resets each season — no carrying a hot
    streak across a summer and a squad move), sort by gameweek and compute
    rolling averages using only STRICTLY PRIOR rows. `.shift(1)` before
    `.rolling()` is what enforces that: the rolling window for row N is
    built from rows before N, never including N itself.
    """
    df = df.sort_values(["season", "player_id", "gw"]).copy()
    group = df.groupby(["season", "player_id"], group_keys=False)

    df["games_so_far"] = group.cumcount()

    for stat in ROLLED_STATS:
        for w in ROLL_WINDOWS:
            col = f"{stat}_roll{w}"
            # shift(1) first (drop row N's own value), THEN roll over what's
            # left — one transform, one group-by, no re-grouping a detached
            # Series (which is an easy way to silently misalign rows).
            df[col] = group[stat].transform(
                lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean()
            )
    for stat in LONG_ROLLED_STATS:
        col = f"{stat}_roll{LONG_WINDOW}"
        df[col] = group[stat].transform(
            lambda s: s.shift(1).rolling(LONG_WINDOW, min_periods=1).mean()
        )
    return df


def build_features(df, team_strength_by_season):
    """
    df: standardized long-format rows (see module docstring for schema).
    team_strength_by_season: {season: {team_id: {attack_home, attack_away,
    defence_home, defence_away}}}.

    Returns df with every column in FEATURE_COLS added, ready for
    model.fit(df[FEATURE_COLS], df[TARGET_COL]) or model.predict(...).
    Rows are NOT filtered here (e.g. games_so_far==0 rows are kept) —
    callers decide what to train/predict on; this only adds columns.
    """
    df = add_rolling_form(df)
    df = add_opponent_strength(df, team_strength_by_season)

    # position as a small integer code — HistGradientBoostingRegressor
    # handles categorical-as-integer fine, and it's one less encoder to
    # keep in sync between training and prediction.
    pos_code = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    df["position"] = df["position"].map(pos_code)
    df["was_home"] = df["was_home"].astype(float)

    return df
