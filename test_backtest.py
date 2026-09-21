"""Tests for backtest.py's reconstruction logic, using synthetic history data
(shaped like the real element-summary API response) — no network needed."""

import pandas as pd
import backtest as bt

# --- synthetic season -------------------------------------------------------
# 3 players, 4 finished gameweeks (1-4), testing snapshot at target_gw=4:
# only rounds 1-3 should be visible, round 4 is what we're "predicting".

boot = {
    "elements": [
        {"id": 1, "element_type": 3, "team": 10},   # MID
        {"id": 2, "element_type": 4, "team": 11},   # FWD
        {"id": 3, "element_type": 2, "team": 10},   # DEF, joined the league late
    ],
    "element_types": [
        {"id": 2, "singular_name_short": "DEF"},
        {"id": 3, "singular_name_short": "MID"},
        {"id": 4, "singular_name_short": "FWD"},
    ],
}

histories = {
    "1": [   # consistently good, price rose from 6.0 to 6.3 over the period
        {"round": 1, "total_points": 8, "minutes": 90, "value": 60},
        {"round": 2, "total_points": 6, "minutes": 90, "value": 61},
        {"round": 3, "total_points": 10, "minutes": 90, "value": 63},
        {"round": 4, "total_points": 12, "minutes": 90, "value": 64},   # the "future" row
    ],
    "2": [   # rotation player, patchy minutes
        {"round": 1, "total_points": 1, "minutes": 20, "value": 55},
        {"round": 2, "total_points": 0, "minutes": 0, "value": 55},
        {"round": 3, "total_points": 2, "minutes": 30, "value": 54},
        {"round": 4, "total_points": 0, "minutes": 0, "value": 54},
    ],
    "3": [   # only has history from round 3 onward — a mid-season signing
        {"round": 3, "total_points": 5, "minutes": 90, "value": 45},
        {"round": 4, "total_points": 3, "minutes": 90, "value": 45},
    ],
}

fixtures = [
    # GW4: player 1 & 3's team (10) has an easy game, player 2's team (11) hard
    {"event": 4, "team_h": 10, "team_a": 99, "team_h_difficulty": 2, "team_a_difficulty": 4},
    {"event": 4, "team_h": 88, "team_a": 11, "team_h_difficulty": 3, "team_a_difficulty": 5},
    # other gameweeks present but irrelevant to a target_gw=4 snapshot
    {"event": 1, "team_h": 10, "team_a": 11, "team_h_difficulty": 3, "team_a_difficulty": 3},
]

# --- 1. only prior rounds are visible ---------------------------------------
df = bt.snapshot_as_of(boot, fixtures, histories, target_gw=4, min_players=1)
assert set(df["id"]) == {1, 2, 3}, "all three have both prior history and a GW4 row"

row1 = df.set_index("id").loc[1]
assert row1["games_asof"] == 3, "player 1: 3 prior games (rounds 1-3), not 4"
assert abs(row1["ppg"] - (8 + 6 + 10) / 3) < 1e-9, "ppg must average ONLY rounds < target_gw"
assert row1["minutes"] == 270, "minutes summed over rounds 1-3 only"
assert row1["actual_points"] == 12, "actual_points is round 4's real score — the answer, not an input"
print("1. only prior rounds visible   OK")

# --- 2. price is the LAST KNOWN value before target_gw, not today's ---------
assert abs(row1["price"] - 6.3) < 1e-9, "price must be round 3's value (6.3), not round 4's (6.4)"
print("2. price uses last-known value OK")

# --- 3. a mid-season signing with a short history still gets included -------
row3 = df.set_index("id").loc[3]
assert row3["games_asof"] == 1, "player 3 only has round 3 as history"
assert row3["actual_points"] == 3
print("3. short-history player included OK")

# --- 4. no leakage: round 4's own total_points never touches ppg/minutes ----
for pid in [1, 2, 3]:
    r = df.set_index("id").loc[pid]
    future_row = next(h for h in histories[str(pid)] if h["round"] == 4)
    # minutes for round 4 must not be baked into the "minutes" (prior) column
    prior_minutes = sum(h["minutes"] for h in histories[str(pid)] if h["round"] < 4)
    assert r["minutes"] == prior_minutes, f"player {pid}: round 4 data leaked into features"
print("4. no leakage from the target gameweek OK")

# --- 5. fixture score reflects target_gw specifically, not some other week --
# team 10 (players 1 & 3) has FDR 2 in GW4 -> multiplier 1.12 (see FDR_MULTIPLIER)
# team 11 (player 2) has FDR 5 in GW4 -> multiplier 0.75
assert abs(row1["fixture_score_gw1"] - 1.12) < 1e-9
row2 = df.set_index("id").loc[2]
assert abs(row2["fixture_score_gw1"] - 0.75) < 1e-9
print("5. fixture score matches target_gw OK")

# --- 6. a player who didn't feature that gameweek is excluded ---------------
histories_gap = dict(histories)
histories_gap["2"] = [h for h in histories["2"] if h["round"] != 4]   # no round-4 row at all
df_gap = bt.snapshot_as_of(boot, fixtures, histories_gap, target_gw=4, min_players=1)
assert 2 not in set(df_gap["id"]), "no round-4 row means no ground truth — must be excluded, not filled with 0"
print("6. missing target-gw row excluded, not zero-filled OK")

# --- 7. evaluate(): perfect predictor gets correlation 1.0 -------------------
perfect = pd.DataFrame({
    "xpts_gw1": [5, 3, 8, 1, 9, 2, 7, 4, 6, 0, 10, 11],
    "actual_points": [5, 3, 8, 1, 9, 2, 7, 4, 6, 0, 10, 11],
    "xpts_gw1_raw": [5, 3, 8, 1, 9, 2, 7, 4, 6, 0, 10, 11],
    "ppg": [5, 3, 8, 1, 9, 2, 7, 4, 6, 0, 10, 11],
})
r = bt.evaluate(perfect)
assert abs(r["corr_shrunk"] - 1.0) < 1e-9, "identical predicted/actual ranking must correlate at 1.0"
assert abs(r["top11_shrunk"] - r["best_possible"]) < 1e-9, "a perfect predictor's top-11 IS the best possible top-11"
print("7. perfect predictor scores correlation 1.0 OK")

# --- 8. evaluate(): a predictor with no signal shouldn't beat random by much
import numpy as np
rng = np.random.default_rng(0)
noise = pd.DataFrame({
    "xpts_gw1": rng.random(200),
    "xpts_gw1_raw": rng.random(200),
    "ppg": rng.random(200),
    "actual_points": rng.random(200),   # unrelated to the "predictions" above
})
r2 = bt.evaluate(noise)
assert abs(r2["corr_shrunk"]) < 0.3, "unrelated columns shouldn't show strong correlation"
assert -1.0 <= r2["corr_shrunk"] <= 1.0
print("8. correlation sane on unrelated data OK")

# --- 9. evaluate(): a perfect predictor's team-total calibration is exact ---
# predicted_team_total must equal actual_team_total when xpts_gw1 ==
# actual_points everywhere (same fixture used in check 7).
r_perfect = bt.evaluate(perfect)
assert abs(r_perfect["predicted_team_total"] - r_perfect["actual_team_total"]) < 1e-9, \
    "a perfect predictor's predicted team total must exactly match its actual team total"
print("9. perfect predictor's team-total calibration is exact OK")

# --- 10. evaluate(): a systematically inflated predictor shows up as bias ---
inflated = pd.DataFrame({
    "xpts_gw1": [x * 1.5 for x in [5, 3, 8, 1, 9, 2, 7, 4, 6, 0, 10, 11]],  # same ranking, 50% too high
    "actual_points": [5, 3, 8, 1, 9, 2, 7, 4, 6, 0, 10, 11],
    "xpts_gw1_raw": [5, 3, 8, 1, 9, 2, 7, 4, 6, 0, 10, 11],
    "ppg": [5, 3, 8, 1, 9, 2, 7, 4, 6, 0, 10, 11],
})
r_inflated = bt.evaluate(inflated)
# ranking quality is untouched (scaling doesn't change order)...
assert abs(r_inflated["corr_shrunk"] - 1.0) < 1e-9
# ...but the team-total calibration correctly shows the 50% inflation
ratio = r_inflated["predicted_team_total"] / r_inflated["actual_team_total"]
assert abs(ratio - 1.5) < 1e-9, f"expected a 1.5x calibration gap, got {ratio}"
print("10. a scaled-up (but correctly-ranked) predictor shows up as calibration bias, not correlation OK")

print("\nAll checks passed.")
