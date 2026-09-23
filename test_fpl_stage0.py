"""
Tests for fpl_stage0.py's ML cold-start wiring — the highest-risk part of
today's change, since build_table() is what every other script in this
project imports. No network calls: fpl_stage0.get() is monkeypatched to
return canned data, and ml_predict's functions are monkeypatched too, so
these tests check the WIRING (does a blend actually happen, does a failure
degrade gracefully) rather than re-testing ml_predict's own internals
(already covered in test_ml_predict.py) or the heuristic (unchanged,
pre-existing code).
"""

from datetime import date as _date, timedelta as _timedelta

import pandas as pd

import fpl_stage0 as fs0
import ml_predict as mp

# gw10's deadline is exactly 2026-10-10, one week apart either side —
# matches the real-data example the confirmed-comeback restore logic was
# verified against (an "Expected back 10 Oct" case).
_GW10_DEADLINE = _date(2026, 10, 10)


def make_boot():
    events = [{"id": i, "is_next": i == 10, "finished": i < 10,
               "deadline_time": (_GW10_DEADLINE + _timedelta(weeks=(i - 10))).isoformat() + "T18:00:00Z"}
              for i in range(1, 15)]
    teams = [{"id": i, "short_name": f"T{i}",
              "strength_attack_home": 1300, "strength_attack_away": 1250,
              "strength_defence_home": 1200, "strength_defence_away": 1150}
             for i in range(1, 5)]
    element_types = [
        {"id": 1, "singular_name_short": "GKP"}, {"id": 2, "singular_name_short": "DEF"},
        {"id": 3, "singular_name_short": "MID"}, {"id": 4, "singular_name_short": "FWD"},
    ]
    elements = []
    pid = 1
    for et, n in [(1, 2), (2, 5), (3, 5), (4, 3)]:
        for i in range(n):
            elements.append({
                "id": pid, "web_name": f"P{pid}", "team": (pid % 4) + 1,
                "element_type": et, "now_cost": 50 + (pid % 10) * 5,
                "points_per_game": "4.0", "form": "3.5", "minutes": 450,
                "chance_of_playing_next_round": None, "status": "a",
            })
            pid += 1
    return {"events": events, "teams": teams, "element_types": element_types, "elements": elements}


def make_fixtures():
    out = []
    for gw in range(10, 15):
        for h, a in [(1, 2), (3, 4)]:
            out.append({"event": gw, "team_h": h, "team_a": a, "finished": False,
                        "team_h_difficulty": 3, "team_a_difficulty": 3})
    return out


def fake_get(endpoint):
    if endpoint == "bootstrap-static/":
        return make_boot()
    if endpoint == "fixtures/":
        return make_fixtures()
    raise ValueError(f"unexpected endpoint in test: {endpoint}")


def test_ml_blend_overwrites_only_targeted_players(monkeypatch):
    monkeypatch.setattr(fs0, "get", fake_get)
    # Isolate from the REAL backtest_results.csv on disk — these tests
    # check ML-blend wiring specifically, not calibration (see
    # test_calibration_factor_* below for that).
    monkeypatch.setattr(fs0, "USE_CALIBRATION", False)
    monkeypatch.setattr(fs0, "USE_ML_COLD_START", True)

    monkeypatch.setattr(mp, "fetch_current_histories", lambda ids, **kw: {})

    def fake_predict(boot, fixtures, histories, gw, horizon, **kw):
        cols = [f"xw{w}" for w in range(horizon)]
        # only player id 1 gets an ML prediction, with an obviously
        # distinctive value (99.0) that could never come from the heuristic
        return pd.DataFrame({c: [99.0] for c in cols}, index=[1])
    monkeypatch.setattr(mp, "predict_cold_start", fake_predict)

    df, gw = fs0.build_table()

    row1 = df[df["id"] == 1].iloc[0]
    assert row1["xw0"] == 99.0
    assert row1["xpts_source"] == "ml"
    assert row1["xpts_gw1"] == 99.0            # derived column must reflect the blend
    assert row1["xpts"] == 99.0 * fs0.HORIZON  # summed across the whole horizon

    other = df[df["id"] == 2].iloc[0]
    assert other["xpts_source"] == "heuristic"
    assert other["xw0"] != 99.0
    print("1. ML blend overwrites only the targeted player's xw{w}/source     OK")


def test_ml_unavailable_falls_back_cleanly(monkeypatch, capsys):
    """predict_cold_start raising (e.g. no trained model file, a bug, a
    network hiccup mid-fetch) must not take down build_table() at all."""
    monkeypatch.setattr(fs0, "get", fake_get)
    # Isolate from the REAL backtest_results.csv on disk — these tests
    # check ML-blend wiring specifically, not calibration (see
    # test_calibration_factor_* below for that).
    monkeypatch.setattr(fs0, "USE_CALIBRATION", False)
    monkeypatch.setattr(fs0, "USE_ML_COLD_START", True)
    monkeypatch.setattr(mp, "fetch_current_histories", lambda ids, **kw: {})

    def boom(*a, **kw):
        raise RuntimeError("simulated failure (e.g. missing ml_model.joblib)")
    monkeypatch.setattr(mp, "predict_cold_start", boom)

    df, gw = fs0.build_table()   # must not raise
    assert (df["xpts_source"] == "heuristic").all()
    print("2. ML failure falls back to pure heuristic, doesn't crash          OK")


def test_ml_disabled_flag_skips_entirely(monkeypatch):
    monkeypatch.setattr(fs0, "get", fake_get)
    # Isolate from the REAL backtest_results.csv on disk — these tests
    # check ML-blend wiring specifically, not calibration (see
    # test_calibration_factor_* below for that).
    monkeypatch.setattr(fs0, "USE_CALIBRATION", False)
    monkeypatch.setattr(fs0, "USE_ML_COLD_START", False)
    # USE_RECENT_MINUTES fetches histories independently of the ML flag
    # (see fpl_stage0.py), so this still needs mocking here to avoid a
    # real network call in this test.
    monkeypatch.setattr(mp, "fetch_current_histories", lambda ids, **kw: {})

    called = {"predict": False}
    def spy(*a, **kw):
        called["predict"] = True
        return pd.DataFrame()
    monkeypatch.setattr(mp, "predict_cold_start", spy)

    df, gw = fs0.build_table()
    assert not called["predict"], "predict_cold_start must not be called when the flag is off"
    assert (df["xpts_source"] == "heuristic").all()
    print("3. USE_ML_COLD_START=False skips the ML path entirely              OK")


def test_empty_ml_predictions_leaves_everyone_heuristic(monkeypatch):
    """No cold-start players this run (everyone has enough history) is a
    normal, common case — must not be treated as a failure."""
    monkeypatch.setattr(fs0, "get", fake_get)
    # Isolate from the REAL backtest_results.csv on disk — these tests
    # check ML-blend wiring specifically, not calibration (see
    # test_calibration_factor_* below for that).
    monkeypatch.setattr(fs0, "USE_CALIBRATION", False)
    monkeypatch.setattr(fs0, "USE_ML_COLD_START", True)
    monkeypatch.setattr(mp, "fetch_current_histories", lambda ids, **kw: {})
    monkeypatch.setattr(mp, "predict_cold_start", lambda *a, **kw: pd.DataFrame())

    df, gw = fs0.build_table()
    assert (df["xpts_source"] == "heuristic").all()
    print("4. no cold-start players -> everyone stays heuristic, no error     OK")


def test_calibration_factor_missing_file_returns_neutral():
    assert fs0.calibration_factor(path="definitely_does_not_exist.csv") == 1.0
    print("5. calibration_factor: missing file -> no correction (1.0)          OK")


def test_calibration_factor_too_few_gameweeks_returns_neutral(tmp_path):
    p = tmp_path / "backtest_results.csv"
    pd.DataFrame({
        "predicted_team_total": [100.0, 100.0],
        "actual_team_total": [50.0, 50.0],
    }).to_csv(p, index=False)
    factor = fs0.calibration_factor(path=str(p), min_gameweeks=3)
    assert factor == 1.0, "only 2 rows, below the min_gameweeks floor -- must not correct"
    print("6. calibration_factor: below min_gameweeks -> no correction (1.0)   OK")


def test_calibration_factor_computes_actual_over_predicted_ratio(tmp_path):
    """Sums both totals first, THEN divides — a gameweek with more players
    fielding shouldn't count the same as a thin one, so this must NOT be
    an average of each gameweek's own ratio."""
    p = tmp_path / "backtest_results.csv"
    pd.DataFrame({
        "predicted_team_total": [100.0, 100.0, 100.0],
        "actual_team_total": [50.0, 60.0, 70.0],
    }).to_csv(p, index=False)
    factor = fs0.calibration_factor(path=str(p), min_gameweeks=3)
    expected = (50 + 60 + 70) / (100 + 100 + 100)
    assert abs(factor - expected) < 1e-9
    print("7. calibration_factor: sums both totals THEN divides                OK")


def test_build_table_applies_calibration_when_enabled(monkeypatch, tmp_path):
    """A known, non-1.0 factor must actually reach the xw{w} columns —
    proves the flag is really wired into build_table, not just present."""
    monkeypatch.setattr(fs0, "get", fake_get)
    monkeypatch.setattr(fs0, "USE_ML_COLD_START", False)

    p = tmp_path / "backtest_results.csv"
    pd.DataFrame({
        "predicted_team_total": [100.0, 100.0, 100.0],
        "actual_team_total": [50.0, 50.0, 50.0],   # -> factor 0.5
    }).to_csv(p, index=False)
    monkeypatch.setattr(fs0, "CALIBRATION_RESULTS_PATH", str(p))

    monkeypatch.setattr(fs0, "USE_CALIBRATION", True)
    df_cal, _ = fs0.build_table()
    monkeypatch.setattr(fs0, "USE_CALIBRATION", False)
    df_uncal, _ = fs0.build_table()

    row_cal = df_cal[df_cal["id"] == 2].iloc[0]
    row_uncal = df_uncal[df_uncal["id"] == 2].iloc[0]
    assert abs(row_cal["xw0"] - row_uncal["xw0"] * 0.5) < 1e-6
    assert abs(row_cal["calibration_factor"] - 0.5) < 1e-9
    print("8. build_table actually applies the calibration factor to xw{w}     OK")


def test_recent_start_share_computes_fraction_of_recent_starts():
    histories = {
        "1": [{"starts": 1}, {"starts": 0}, {"starts": 1}, {"starts": 1}],  # 3/4 started
        "2": [{"starts": 0}, {"starts": 0}],  # 0/2 started, fewer than the full window
    }
    result = fs0.recent_start_share([1, 2, 3], histories, n=4)
    assert abs(result[1] - 0.75) < 1e-9
    assert abs(result[2] - 0.0) < 1e-9
    assert 3 not in result, "no history at all -> left out, not zero-filled"
    print("9. recent_start_share computes fraction of recent games started    OK")


def test_build_table_discounts_impact_substitute_pattern(monkeypatch):
    """A player with decent recent minutes but ZERO recent starts (a pure
    impact substitute) must get a lower mins_share than the exact same
    recent minutes would give a nailed starter — the discount this
    feature adds on top of recent_mins_share, which can't tell them
    apart on minutes alone."""
    monkeypatch.setattr(fs0, "get", fake_get)
    monkeypatch.setattr(fs0, "USE_ML_COLD_START", False)
    monkeypatch.setattr(fs0, "USE_CALIBRATION", False)

    # id 1: 40 min/game every game, always as a substitute (0 starts).
    # id 2: the SAME 40 min/game every game, always as a starter.
    sub_hist = [{"round": r, "minutes": 40, "starts": 0} for r in range(10, 14)]
    starter_hist = [{"round": r, "minutes": 40, "starts": 1} for r in range(10, 14)]
    histories = {"1": sub_hist, "2": starter_hist}
    monkeypatch.setattr(mp, "fetch_current_histories", lambda ids, **kw: histories)

    df, gw = fs0.build_table()
    row_sub = df[df["id"] == 1].iloc[0]
    row_starter = df[df["id"] == 2].iloc[0]
    assert row_sub["mins_share"] < row_starter["mins_share"], (
        "identical recent minutes, but the impact-substitute pattern "
        "must be discounted below the equal-minutes nailed starter"
    )
    print("10. build_table discounts an impact-substitute's mins_share vs an equal-minutes starter  OK")


def test_parse_expected_return_extracts_a_real_date():
    got = fs0.parse_expected_return("Hamstring injury - Expected back 10 Oct", "2026-09-12T18:00:08Z")
    assert got == fs0.date(2026, 10, 10)
    print("11. parse_expected_return extracts a real date from injury news     OK")


def test_parse_expected_return_ignores_non_dates():
    assert fs0.parse_expected_return("Back injury - Unknown return date", "2026-07-23T12:01:23Z") is None
    assert fs0.parse_expected_return("Has joined Al Hilal permanently", "2026-09-04T12:34:46Z") is None
    assert fs0.parse_expected_return(None, "2026-09-04T12:34:46Z") is None
    print("12. parse_expected_return returns None for unparseable/non-injury news  OK")


def test_parse_expected_return_rolls_over_to_next_year():
    """News posted in November about a return '05 Jan' must resolve to
    January of the FOLLOWING year, not a date already in the past."""
    got = fs0.parse_expected_return("Knee injury - Expected back 05 Jan", "2026-11-20T18:00:08Z")
    assert got == fs0.date(2027, 1, 5)
    print("13. parse_expected_return rolls the year over correctly             OK")


def test_build_table_restores_availability_after_confirmed_return(monkeypatch):
    """A hard-out injured player (avail=0.0 for every week via the
    existing status check) with a real 'Expected back' date within the
    horizon must have their score RESTORED from the week they're
    confirmed back, not stuck at 0.0 for the whole horizon."""
    monkeypatch.setattr(fs0, "USE_ML_COLD_START", False)
    monkeypatch.setattr(fs0, "USE_CALIBRATION", False)
    monkeypatch.setattr(mp, "fetch_current_histories", lambda ids, **kw: {})

    boot = make_boot()
    boot["elements"][0]["status"] = "i"
    boot["elements"][0]["chance_of_playing_next_round"] = 0
    boot["elements"][0]["news"] = "Hamstring injury - Expected back 24 Oct"
    boot["elements"][0]["news_added"] = "2026-09-12T18:00:08Z"

    def get_with_injury(endpoint):
        if endpoint == "bootstrap-static/":
            return boot
        return fake_get(endpoint)
    monkeypatch.setattr(fs0, "get", get_with_injury)

    df, gw = fs0.build_table()
    row = df[df["id"] == 1].iloc[0]
    assert row["expected_return"] == fs0.date(2026, 10, 24)
    assert row["xw0"] == 0.0, "gw10 deadline (10 Oct) is before the 24 Oct return -- still out"
    assert row["xw1"] == 0.0, "gw11 deadline (17 Oct) is before the 24 Oct return -- still out"
    assert row["xw2"] > 0.0, "gw12 deadline (24 Oct) is ON the return date -- must be restored"
    print("14. build_table restores availability from the confirmed return week onward  OK")


if __name__ == "__main__":
    class _MP:
        def setattr(self, obj, name, value):
            setattr(obj, name, value)

    mpatch = _MP()
    test_ml_blend_overwrites_only_targeted_players(mpatch)
    test_ml_unavailable_falls_back_cleanly(mpatch, None)
    test_ml_disabled_flag_skips_entirely(mpatch)
    test_empty_ml_predictions_leaves_everyone_heuristic(mpatch)
    print("\nAll checks passed.")
