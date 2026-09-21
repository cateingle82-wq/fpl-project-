"""Synthetic tests for ml_features.py — checked BEFORE trusting it on real
historical data, same discipline as every other file in this project."""

import numpy as np
import pandas as pd

import ml_features as mf


def make_toy_rows():
    """One player, one season, 6 gameweeks, hand-picked scores so the
    correct rolling average is easy to compute by hand and check against."""
    pts = [2, 10, 4, 6, 8, 0]   # gw1..gw6
    rows = []
    for gw, p in enumerate(pts, start=1):
        rows.append({
            "player_id": 1, "season": "toy", "gw": gw, "position": "MID",
            "team_id": 1, "opponent_id": 2, "was_home": gw % 2 == 0,
            "total_points": p, "minutes": 90, "ict_index": p * 1.5,
            "influence": 0, "creativity": 0, "threat": 0,
            "bps": p * 3, "bonus": 0,
            "expected_goal_involvements": 0.1 * p,
            "value": 55, "starts": 1,
        })
    return pd.DataFrame(rows)


TEAM_STRENGTH = {
    "toy": {
        1: {"attack_home": 1300, "attack_away": 1250, "defence_home": 1200, "defence_away": 1150},
        2: {"attack_home": 1100, "attack_away": 1050, "defence_home": 1400, "defence_away": 1350},
    }
}


def test_rolling_never_sees_its_own_row():
    """GW4's roll3 feature must be the mean of GW1-3 only (4,10,2 -> using
    pts[0:3] = [2,10,4], mean=5.33), never including GW4's own value (6)."""
    df = mf.build_features(make_toy_rows(), TEAM_STRENGTH)
    row_gw4 = df[df["gw"] == 4].iloc[0]
    expected = np.mean([2, 10, 4])
    assert abs(row_gw4["total_points_roll3"] - expected) < 1e-9, row_gw4["total_points_roll3"]
    print(f"1. roll3 at GW4 uses only GW1-3 ({expected:.2f})              OK")


def test_roll5_matches_all_prior_when_fewer_than_5_games():
    """GW3's roll5 (min_periods=1) should be mean of GW1-2 only (2,10 -> 6),
    not padded with anything from GW3 itself or fabricated zeros."""
    df = mf.build_features(make_toy_rows(), TEAM_STRENGTH)
    row_gw3 = df[df["gw"] == 3].iloc[0]
    expected = np.mean([2, 10])
    assert abs(row_gw3["total_points_roll5"] - expected) < 1e-9, row_gw3["total_points_roll5"]
    print(f"2. roll5 at GW3 uses only the 2 prior games available ({expected:.2f})  OK")


def test_first_gameweek_has_no_history():
    """GW1 has zero prior games — every rolling feature must be NaN (no
    fabricated 'prior form' for a player's very first appearance), and
    games_so_far must be exactly 0."""
    df = mf.build_features(make_toy_rows(), TEAM_STRENGTH)
    row_gw1 = df[df["gw"] == 1].iloc[0]
    assert row_gw1["games_so_far"] == 0
    assert pd.isna(row_gw1["total_points_roll3"])
    assert pd.isna(row_gw1["minutes_roll3"])
    print("3. GW1 has games_so_far=0 and NaN rolling features             OK")


def test_games_so_far_increments_correctly():
    df = mf.build_features(make_toy_rows(), TEAM_STRENGTH)
    got = df.sort_values("gw")["games_so_far"].tolist()
    assert got == [0, 1, 2, 3, 4, 5], got
    print("4. games_so_far counts prior appearances, not including today   OK")


def test_form_resets_across_seasons():
    """The same player_id appearing in two different 'seasons' must not
    let one season's form leak into the other's rolling window."""
    toy = make_toy_rows()
    toy2 = toy.copy()
    toy2["season"] = "toy2"
    toy2["total_points"] = [100, 100, 100, 100, 100, 100]   # wildly different
    both = pd.concat([toy, toy2], ignore_index=True)

    strength2 = dict(TEAM_STRENGTH)
    strength2["toy2"] = TEAM_STRENGTH["toy"]

    df = mf.build_features(both, strength2)
    gw1_toy2 = df[(df["season"] == "toy2") & (df["gw"] == 1)].iloc[0]
    assert gw1_toy2["games_so_far"] == 0
    assert pd.isna(gw1_toy2["total_points_roll3"])
    print("5. rolling form resets at a season boundary, doesn't leak across  OK")


def test_opponent_strength_picks_the_right_venue():
    """A player at HOME faces an opponent playing AWAY, so the opponent's
    AWAY attack/defence numbers must be the ones attached — not the
    opponent's home numbers, and not the player's own team's numbers."""
    df = mf.build_features(make_toy_rows(), TEAM_STRENGTH)
    home_row = df[(df["gw"] == 2)].iloc[0]     # gw2 is_home=True in make_toy_rows
    away_row = df[(df["gw"] == 1)].iloc[0]     # gw1 is_home=False

    assert home_row["opp_strength_attack"] == TEAM_STRENGTH["toy"][2]["attack_away"]
    assert home_row["opp_strength_defence"] == TEAM_STRENGTH["toy"][2]["defence_away"]
    assert away_row["opp_strength_attack"] == TEAM_STRENGTH["toy"][2]["attack_home"]
    assert away_row["opp_strength_defence"] == TEAM_STRENGTH["toy"][2]["defence_home"]
    print("6. opponent strength uses the opponent's actual venue           OK")


def test_position_and_was_home_encoded_numerically():
    df = mf.build_features(make_toy_rows(), TEAM_STRENGTH)
    assert set(df["position"].unique()) == {2}   # MID -> 2
    assert set(df["was_home"].unique()) <= {0.0, 1.0}
    print("7. position/was_home encoded as plain numbers for the model     OK")


def test_missing_xg_column_degrades_gracefully():
    """Older seasons (pre-2022-23) never tracked expected_goal_involvements
    at all. The feature pipeline must still run (NaNs flow through, not a
    crash) rather than needing that column to exist with real values."""
    toy = make_toy_rows()
    toy["expected_goal_involvements"] = np.nan
    df = mf.build_features(toy, TEAM_STRENGTH)
    assert df["expected_goal_involvements_roll3"].isna().all()
    # every OTHER feature must still be computed normally
    row_gw4 = df[df["gw"] == 4].iloc[0]
    assert not pd.isna(row_gw4["total_points_roll3"])
    print("8. an all-NaN xG column doesn't break the other features        OK")


def test_all_feature_columns_present():
    df = mf.build_features(make_toy_rows(), TEAM_STRENGTH)
    missing = [c for c in mf.FEATURE_COLS if c not in df.columns]
    assert not missing, f"missing columns: {missing}"
    print("9. every declared FEATURE_COLS column is actually produced      OK")


if __name__ == "__main__":
    test_rolling_never_sees_its_own_row()
    test_roll5_matches_all_prior_when_fewer_than_5_games()
    test_first_gameweek_has_no_history()
    test_games_so_far_increments_correctly()
    test_form_resets_across_seasons()
    test_opponent_strength_picks_the_right_venue()
    test_position_and_was_home_encoded_numerically()
    test_missing_xg_column_degrades_gracefully()
    test_all_feature_columns_present()
    print("\nAll checks passed.")
