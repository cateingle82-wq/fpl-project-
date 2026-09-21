"""
Live cold-start predictions: use the trained ML model ONLY for players with
very little data this season (new signings, promoted-team players, the
first couple of gameweeks) and leave everyone else to the existing
shrinkage heuristic in fpl_stage0.py.

Why only cold-start players, not everyone — see ml_train.py's holdout
results: across a full season the ML model roughly TIES a dead-simple
"player's own last-3-games average", but at GW1 (zero rolling history) it
scores 0.47-0.48 rank correlation vs the naive baseline's 0.02. That's the
one place it has a real, repeatable edge, because it falls back on price/
position/opponent strength exactly when the heuristic's own shrinkage
prior is doing the same job with less information. Using ML everywhere
would be adding model risk for no proven benefit; using it only here uses
it exactly where the evidence says it helps.

How a live prediction is built, per eligible player:
  1. Pull their real gameweek history THIS season from the FPL API
     (element-summary/{id}/, cached — reuses backtest.py's cache file so
     there's no extra ~700-call fetch if you've already run a backtest).
  2. Append ONE placeholder row for the next gameweek and run the exact
     same ml_features.build_features() used in training — this is what
     makes "games_so_far" and the rolling-form features come out correct
     (shift(1) drops the placeholder's own [nonexistent] result and rolls
     over their real past games, precisely mirroring what a genuine
     future gameweek row would look like).
  3. That placeholder row's rolling/form/price features don't change
     across the horizon (they only depend on games already played), so
     they're computed ONCE per player, then combined with each future
     week's actual fixture (opponent, venue) pulled from fixtures/ — a
     blank week contributes 0, a double gameweek sums two separate
     per-fixture predictions.

Run standalone to sanity-check what it would predict, without touching
fpl_stage0.py's pipeline:  python ml_predict.py
"""

import os

import joblib
import numpy as np
import pandas as pd

import ml_features as mf
from fpl_stage0 import get, next_gameweek

MODEL_PATH = "ml_model.joblib"
HISTORY_CACHE = "player_history_cache.json"

# A player counts as "cold start" — eligible for the ML prediction instead
# of the shrinkage heuristic — if they've played fewer than this many
# gameweeks THIS season. 3 matches where the holdout evidence actually
# showed an edge (GW1-2 were the clear wins; by GW3 roll3 was already
# competitive again).
COLD_START_GAMES = 3

POSITION_CODE = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}   # fpl_stage0 spells GK "GKP"


def load_model():
    """Returns None (not an exception) if no trained model exists yet —
    callers must treat that as "ML unavailable, fall back to the
    heuristic", not a crash. Keeps this whole feature strictly optional:
    nothing breaks for someone who hasn't run ml_train.py."""
    if not os.path.exists(MODEL_PATH):
        return None
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["feature_cols"]


def fetch_current_histories(element_ids, cache_path=HISTORY_CACHE, pause=0.05):
    """Same cache file and endpoint backtest.py already uses — if you've
    run a backtest recently this is nearly instant (only new players get
    fetched). Duplicated here rather than imported from backtest.py to
    avoid a circular import (backtest.py imports FROM fpl_stage0, and this
    module is imported BY fpl_stage0 — importing backtest here too would
    create a cycle)."""
    import json
    import time

    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)

    missing = [str(i) for i in element_ids if str(i) not in cache]
    if missing:
        for i, pid in enumerate(missing):
            data = get(f"element-summary/{pid}/")
            cache[pid] = data.get("history", [])
            if (i + 1) % 50 == 0:
                with open(cache_path, "w") as f:
                    json.dump(cache, f)
            time.sleep(pause)
        with open(cache_path, "w") as f:
            json.dump(cache, f)

    return cache


def team_strength_from_boot(boot):
    """Same schema as fpl_history.team_strength_dict — bootstrap-static's
    'teams' list carries the identical strength_attack/defence_home/away
    fields the vaastav archive's teams.csv does, which is exactly why
    ml_features.py could be written generically instead of per-source."""
    out = {}
    for t in boot["teams"]:
        out[t["id"]] = {
            "attack_home": t["strength_attack_home"],
            "attack_away": t["strength_attack_away"],
            "defence_home": t["strength_defence_home"],
            "defence_away": t["strength_defence_away"],
        }
    return out


def _standardize_history_rows(el, hist, position):
    """One player's real element-summary history rows -> ml_features'
    standard schema. Mirrors fpl_history.standardize_season's column
    construction exactly (same source endpoint, same field names)."""
    rows = []
    for h in hist:
        rows.append({
            "player_id": el["id"], "season": "current", "gw": h["round"],
            "position": position, "team_id": el["team"],
            "opponent_id": h["opponent_team"], "was_home": bool(h["was_home"]),
            "total_points": h["total_points"], "minutes": h["minutes"],
            "ict_index": h["ict_index"], "influence": h["influence"],
            "creativity": h["creativity"], "threat": h["threat"],
            "bps": h["bps"], "bonus": h["bonus"],
            "expected_goal_involvements": h.get("expected_goal_involvements"),
            "value": h["value"], "starts": h.get("starts"),
        })
    return rows


def player_base_features(boot, histories, next_gw):
    """
    Returns a DataFrame indexed by player_id with: team_id, position code,
    games_so_far, and every rolling-form feature — each computed ONCE from
    real history via the exact same build_features() training used, by
    appending a single placeholder row for next_gw and reading its
    computed columns back off. Doesn't touch opp_strength/was_home (those
    are fixture-specific, added later per future week).
    """
    positions = {p["id"]: p["singular_name_short"] for p in boot["element_types"]}
    real_rows = []
    placeholder_rows = []
    for el in boot["elements"]:
        pos = positions[el["element_type"]]
        if pos not in POSITION_CODE:
            continue
        hist = histories.get(str(el["id"]), [])
        real_rows += _standardize_history_rows(el, hist, pos)
        placeholder_rows.append({
            "player_id": el["id"], "season": "current", "gw": next_gw,
            "position": pos, "team_id": el["team"],
            # opponent/was_home are throwaway here — only this row's
            # games_so_far/rolling/value/position columns get used.
            "opponent_id": None, "was_home": True,
            "total_points": np.nan, "minutes": np.nan, "ict_index": np.nan,
            "influence": np.nan, "creativity": np.nan, "threat": np.nan,
            "bps": np.nan, "bonus": np.nan,
            "expected_goal_involvements": np.nan,
            "value": el["now_cost"], "starts": np.nan,
        })

    combined = pd.DataFrame(real_rows + placeholder_rows)
    strength = {"current": team_strength_from_boot(boot)}
    feat = mf.build_features(combined, strength)

    base = feat[feat["gw"] == next_gw].set_index("player_id")
    keep = ["team_id", "position", "value", "games_so_far"] + [
        c for c in mf.FEATURE_COLS if c.endswith(tuple(f"roll{w}" for w in
            list(mf.ROLL_WINDOWS) + [mf.LONG_WINDOW]))
    ]
    return base[keep]


def fixtures_by_team_week(fixtures, start_gw, horizon):
    """team_id -> {week_offset (0..horizon-1): [(opponent_id, was_home), ...]}.
    A team with two entries for one week is a double gameweek (both kept,
    predicted and summed separately); zero entries is a blank."""
    window = range(start_gw, start_gw + horizon)
    out = {}
    for f in fixtures:
        if f["event"] is None or f["event"] not in window or f["finished"]:
            continue
        w = f["event"] - start_gw
        out.setdefault(f["team_h"], {}).setdefault(w, []).append((f["team_a"], True))
        out.setdefault(f["team_a"], {}).setdefault(w, []).append((f["team_h"], False))
    return out


def predict_cold_start(boot, fixtures, histories, next_gw, horizon,
                        cold_start_games=COLD_START_GAMES):
    """
    Returns a DataFrame indexed by player_id (only players below the
    cold-start threshold are included) with columns xw0..xw{horizon-1} —
    same column naming as fpl_stage0's per-week columns, ready to be
    blended in directly. Returns an empty DataFrame (not None) if no
    trained model exists — callers treat "nothing to blend" the same way
    whether it's because everyone has enough history or because there's no
    model at all.
    """
    loaded = load_model()
    if loaded is None:
        return pd.DataFrame()
    model, feature_cols = loaded

    base = player_base_features(boot, histories, next_gw)
    eligible = base[base["games_so_far"] < cold_start_games]
    if eligible.empty:
        return pd.DataFrame()

    strength = team_strength_from_boot(boot)
    fixmap = fixtures_by_team_week(fixtures, next_gw, horizon)

    out = pd.DataFrame(0.0, index=eligible.index, columns=[f"xw{w}" for w in range(horizon)])
    for pid, row in eligible.iterrows():
        team_id = row["team_id"]
        team_fixtures = fixmap.get(team_id, {})
        for w in range(horizon):
            matches = team_fixtures.get(w, [])
            if not matches:
                continue   # blank gameweek for this team -> stays 0.0
            total = 0.0
            for opp_id, was_home in matches:
                opp = strength.get(opp_id, {})
                pred_row = {c: row.get(c, np.nan) for c in feature_cols}
                pred_row["was_home"] = float(was_home)
                pred_row["opp_strength_attack"] = opp.get("attack_away" if was_home else "attack_home", np.nan)
                pred_row["opp_strength_defence"] = opp.get("defence_away" if was_home else "defence_home", np.nan)
                x = pd.DataFrame([pred_row])[feature_cols]
                pred = float(model.predict(x)[0])
                total += max(0.0, pred)   # expected points shouldn't go negative
            out.loc[pid, f"xw{w}"] = total

    return out


if __name__ == "__main__":
    boot = get("bootstrap-static/")
    fixtures = get("fixtures/")
    gw = next_gameweek(boot["events"])
    element_ids = [e["id"] for e in boot["elements"]]

    print("Fetching current-season player histories (cached)...")
    histories = fetch_current_histories(element_ids)

    from fpl_stage0 import HORIZON
    preds = predict_cold_start(boot, fixtures, histories, gw, HORIZON)

    if preds.empty:
        print("No cold-start players to predict for (or no trained model found "
              f"at {MODEL_PATH} — run ml_train.py first).")
    else:
        names = {e["id"]: e["web_name"] for e in boot["elements"]}
        preds = preds.copy()
        preds.insert(0, "name", [names.get(p, "?") for p in preds.index])
        preds["xpts_total"] = preds[[c for c in preds.columns if c.startswith("xw")]].sum(axis=1)
        print(f"\n{len(preds)} cold-start player(s) (< {COLD_START_GAMES} games this season):")
        print(preds.sort_values("xpts_total", ascending=False).round(2).to_string())
