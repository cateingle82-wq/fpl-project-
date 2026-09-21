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

import pandas as pd

import fpl_stage0 as fs0
import ml_predict as mp


def make_boot():
    events = [{"id": i, "is_next": i == 10, "finished": i < 10} for i in range(1, 15)]
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
    monkeypatch.setattr(fs0, "USE_ML_COLD_START", False)

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
    monkeypatch.setattr(fs0, "USE_ML_COLD_START", True)
    monkeypatch.setattr(mp, "fetch_current_histories", lambda ids, **kw: {})
    monkeypatch.setattr(mp, "predict_cold_start", lambda *a, **kw: pd.DataFrame())

    df, gw = fs0.build_table()
    assert (df["xpts_source"] == "heuristic").all()
    print("4. no cold-start players -> everyone stays heuristic, no error     OK")


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
