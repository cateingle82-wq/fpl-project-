"""
Synthetic tests for ml_predict.py — no network, no trained model file
required. A hand-built fake model (a plain function wrapped to look like
a fitted sklearn estimator) makes the tests deterministic and fast, and
keeps them from silently depending on whatever ml_model.joblib happens to
contain on this machine.
"""

import numpy as np
import pandas as pd

import ml_features as mf
import ml_predict as mp


class FakeModel:
    """predict() returns a score built from a FIXED, hand-known formula of
    the input row, so every assertion below can be checked by hand rather
    than trusting the model's own arithmetic."""

    def predict(self, X):
        # score = 2 * opp attack weakness proxy + 1 if home, using columns
        # guaranteed present in FEATURE_COLS. Deliberately simple and
        # order-independent (reads by column name, not position).
        row = X.iloc[0]
        base = 3.0
        home_bonus = 1.0 if row["was_home"] else 0.0
        # weaker opponent defence (lower number) -> higher score
        defence = row["opp_strength_defence"]
        defence_term = 0.0 if pd.isna(defence) else max(0.0, (1300 - defence) / 100)
        return np.array([base + home_bonus + defence_term])


class RecordingModel:
    """Records every row it's asked to predict on, so a test can inspect
    exactly what rest_days (or any other feature) the pipeline computed
    for a given fixture, not just the final predicted score."""

    def __init__(self):
        self.seen = []

    def predict(self, X):
        self.seen.append(X.iloc[0].to_dict())
        return np.array([1.0])


def make_boot(n_teams=4):
    teams = []
    for i in range(1, n_teams + 1):
        teams.append({
            "id": i, "short_name": f"T{i}",
            "strength_attack_home": 1300 + i * 5, "strength_attack_away": 1250 + i * 5,
            "strength_defence_home": 1200 + i * 5, "strength_defence_away": 1150 + i * 5,
        })
    element_types = [
        {"id": 1, "singular_name_short": "GKP"}, {"id": 2, "singular_name_short": "DEF"},
        {"id": 3, "singular_name_short": "MID"}, {"id": 4, "singular_name_short": "FWD"},
    ]
    elements = [
        # a cold-start player: 1 game played this season
        {"id": 101, "web_name": "NewSigning", "team": 1, "element_type": 3, "now_cost": 60},
        # a veteran with plenty of history: NOT cold-start
        {"id": 102, "web_name": "Veteran", "team": 2, "element_type": 4, "now_cost": 90},
        # a player with zero games this season at all (brand new, no history rows)
        {"id": 103, "web_name": "ZeroGames", "team": 3, "element_type": 2, "now_cost": 45},
    ]
    return {"teams": teams, "element_types": element_types, "elements": elements}


def make_fixtures():
    """GW10: team1 (home) vs team2. GW11: team1 has NO fixture (blank).
    GW12: team1 has TWO fixtures (double, vs team3 away and team4 home,
    3 days apart — a midweek/weekend double)."""
    return [
        {"event": 10, "team_h": 1, "team_a": 2, "finished": False,
         "team_h_difficulty": 3, "team_a_difficulty": 3,
         "kickoff_time": "2024-11-02T15:00:00Z"},
        {"event": 12, "team_h": 3, "team_a": 1, "finished": False,
         "team_h_difficulty": 2, "team_a_difficulty": 4,
         "kickoff_time": "2024-11-16T19:45:00Z"},
        {"event": 12, "team_h": 1, "team_a": 4, "finished": False,
         "team_h_difficulty": 3, "team_a_difficulty": 3,
         "kickoff_time": "2024-11-19T20:00:00Z"},
        # team2/3/4's own other fixtures, irrelevant to team1's test rows
        {"event": 10, "team_h": 3, "team_a": 4, "finished": False,
         "team_h_difficulty": 3, "team_a_difficulty": 3,
         "kickoff_time": "2024-11-02T15:00:00Z"},
    ]


def make_histories():
    """player 101: one game played (round 9, GW10's kickoff is 2024-11-02).
    player 102: five games (established). player 103: no history at all
    (brand new)."""
    def gw_row(round_, pts, mins, kickoff_time="2024-10-26T15:00:00Z"):
        return {
            "round": round_, "opponent_team": 2, "was_home": True,
            "total_points": pts, "minutes": mins, "ict_index": pts * 1.2,
            "influence": pts, "creativity": pts, "threat": pts,
            "bps": pts * 3, "bonus": 0, "expected_goal_involvements": 0.1,
            "value": 55, "starts": 1, "kickoff_time": kickoff_time,
        }
    return {
        "101": [gw_row(9, 6, 90)],
        "102": [gw_row(r, 4, 90) for r in range(5, 10)],
        "103": [],
    }


def test_fixtures_by_team_week_handles_blank_and_double():
    fixmap = mp.fixtures_by_team_week(make_fixtures(), start_gw=10, horizon=3)
    team1 = fixmap[1]
    assert [(o, h) for o, h, _ in team1[0]] == [(2, True)], team1[0]   # GW10: home vs team2
    assert 1 not in team1, "GW11 (week offset 1) should be a blank — no entry"
    assert {(o, h) for o, h, _ in team1[2]} == {(3, False), (4, True)}, team1[2]  # GW12: double
    print("1. fixtures_by_team_week: blank week has no entry, double has two   OK")


def test_fixtures_by_team_week_carries_kickoff_time():
    """kickoff_time must be attached to each fixture tuple (not just used
    internally), parsed as a real timestamp — rest_days depends on it."""
    fixmap = mp.fixtures_by_team_week(make_fixtures(), start_gw=10, horizon=3)
    _, _, kt = fixmap[1][0][0]
    assert kt == pd.Timestamp("2024-11-02T15:00:00Z")
    print("2. fixtures_by_team_week attaches a real parsed kickoff_time      OK")


def test_base_features_games_so_far_correct():
    boot = make_boot()
    histories = make_histories()
    base = mp.player_base_features(boot, histories, next_gw=10)

    assert base.loc[101, "games_so_far"] == 1     # one game played
    assert base.loc[102, "games_so_far"] == 5      # five games played
    assert base.loc[103, "games_so_far"] == 0      # brand new, zero games
    print("3. player_base_features: games_so_far matches real history count  OK")


def test_zero_history_player_has_nan_rolling_features():
    boot = make_boot()
    base = mp.player_base_features(boot, make_histories(), next_gw=10)
    row = base.loc[103]
    assert pd.isna(row["total_points_roll3"])
    print("4. a player with zero games has NaN rolling features (not 0)      OK")


def test_predict_cold_start_only_includes_cold_start_players(monkeypatch):
    monkeypatch.setattr(mp, "load_model", lambda: (FakeModel(), mf.FEATURE_COLS))
    boot = make_boot()
    preds = mp.predict_cold_start(boot, make_fixtures(), make_histories(),
                                   next_gw=10, horizon=3, cold_start_games=3)
    assert set(preds.index) == {101, 103}, preds.index.tolist()
    assert 102 not in preds.index, "veteran (5 games) must NOT be in cold-start output"
    print("5. predict_cold_start only returns players under the games threshold  OK")


def test_predict_cold_start_blank_week_is_zero():
    """Player 101 is on team 1, which has no fixture at week offset 1 (GW11)
    in make_fixtures() — the ML prediction for that week must be exactly 0,
    not some extrapolated guess."""
    import ml_predict as mp2
    mp2.load_model = lambda: (FakeModel(), mf.FEATURE_COLS)
    boot = make_boot()
    preds = mp2.predict_cold_start(boot, make_fixtures(), make_histories(),
                                    next_gw=10, horizon=3, cold_start_games=3)
    assert preds.loc[101, "xw1"] == 0.0
    print("6. a blank gameweek predicts exactly 0, not a guess                OK")


def test_predict_cold_start_double_week_sums_both_fixtures():
    """Week offset 2 (GW12) is a double for team 1 — the prediction must be
    the SUM of two separate per-fixture predictions, not a single one."""
    import ml_predict as mp2
    mp2.load_model = lambda: (FakeModel(), mf.FEATURE_COLS)
    boot = make_boot()
    preds = mp2.predict_cold_start(boot, make_fixtures(), make_histories(),
                                    next_gw=10, horizon=3, cold_start_games=3)

    # Manually compute what each of the two fixtures should score with
    # FakeModel, using the same opponent-venue logic predict_cold_start uses.
    strength = mp2.team_strength_from_boot(boot)
    base = mp2.player_base_features(boot, make_histories(), next_gw=10)
    row = base.loc[101]

    def fake_score(opp_id, was_home):
        opp = strength[opp_id]
        defence = opp["defence_away" if was_home else "defence_home"]
        return 3.0 + (1.0 if was_home else 0.0) + max(0.0, (1300 - defence) / 100)

    expected = fake_score(3, False) + fake_score(4, True)   # away @ T3, home vs T4
    assert abs(preds.loc[101, "xw2"] - expected) < 1e-6, (preds.loc[101, "xw2"], expected)
    print("7. a double gameweek sums two independent per-fixture predictions  OK")


def test_predict_cold_start_computes_rest_days_from_last_real_match():
    """Player 101's last REAL match kicked off 2024-10-26T15:00; GW10's
    fixture kicks off 2024-11-02T15:00 — exactly 7 days later."""
    rec = RecordingModel()
    import ml_predict as mp2
    mp2.load_model = lambda: (rec, mf.FEATURE_COLS)
    boot = make_boot()
    mp2.predict_cold_start(boot, make_fixtures(), make_histories(),
                            next_gw=10, horizon=3, cold_start_games=3)
    gw10_row = rec.seen[0]   # player 101's single GW10 fixture, predicted first
    assert gw10_row["rest_days"] == 7.0, gw10_row["rest_days"]
    print("8. rest_days for a single fixture matches the gap since the last real match  OK")


def test_predict_cold_start_chains_rest_days_within_a_double_gameweek():
    """GW12 is a double for team 1, 3 days apart (Nov 16 -> Nov 19). The
    SECOND fixture's rest_days must be measured from the FIRST fixture of
    the double, not from GW10 — a much shorter rest than the first fixture
    of that double sees."""
    rec = RecordingModel()
    import ml_predict as mp2
    mp2.load_model = lambda: (rec, mf.FEATURE_COLS)
    boot = make_boot()
    mp2.predict_cold_start(boot, make_fixtures(), make_histories(),
                            next_gw=10, horizon=3, cold_start_games=3)
    # rec.seen[0] = GW10 (week 0); [1] and [2] = the GW12 double (week 2),
    # in kickoff order (vs team3 on the 16th, then vs team4 on the 19th).
    first_of_double, second_of_double = rec.seen[1], rec.seen[2]
    assert abs(first_of_double["rest_days"] - 14.198) < 0.01, first_of_double["rest_days"]
    assert abs(second_of_double["rest_days"] - 3.010) < 0.01, second_of_double["rest_days"]
    print("9. a double gameweek's 2nd fixture rests from the 1st, not the prior week  OK")


def test_no_model_returns_empty_dataframe(monkeypatch):
    """If ml_model.joblib doesn't exist, predict_cold_start must return an
    empty DataFrame (a clean 'nothing to blend' signal), never raise."""
    monkeypatch.setattr(mp, "load_model", lambda: None)
    boot = make_boot()
    preds = mp.predict_cold_start(boot, make_fixtures(), make_histories(),
                                   next_gw=10, horizon=3)
    assert preds.empty
    print("10. no trained model -> empty DataFrame, not a crash               OK")


if __name__ == "__main__":
    class _Ctx:
        """Tiny monkeypatch stand-in so this file runs with plain
        `python test_ml_predict.py`, matching every other test file in this
        project, without requiring pytest to be installed."""
        def setattr(self, obj, name, value):
            setattr(obj, name, value)

    ctx = _Ctx()
    test_fixtures_by_team_week_handles_blank_and_double()
    test_fixtures_by_team_week_carries_kickoff_time()
    test_base_features_games_so_far_correct()
    test_zero_history_player_has_nan_rolling_features()
    test_predict_cold_start_only_includes_cold_start_players(ctx)
    test_predict_cold_start_blank_week_is_zero()
    test_predict_cold_start_double_week_sums_both_fixtures()
    test_predict_cold_start_computes_rest_days_from_last_real_match()
    test_predict_cold_start_chains_rest_days_within_a_double_gameweek()
    test_no_model_returns_empty_dataframe(ctx)
    print("\nAll checks passed.")
