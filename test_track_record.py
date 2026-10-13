"""Tests for track_record.py — no network calls: fpl_stage0.get() is
monkeypatched to return canned data."""

import os

import pandas as pd

import track_record as tr


POS_OF = {1: "GKP", 2: "GKP", 3: "DEF", 4: "DEF", 5: "MID", 6: "MID", 7: "FWD"}


def test_resolve_real_autosubs_no_blanks_is_a_plain_sum():
    xi = [1, 3, 4, 5, 6, 7]
    bench = [2]
    real_points = {1: 5, 3: 2, 4: 6, 5: 3, 6: 4, 7: 8}
    real_minutes = {p: 90 for p in xi}
    score = tr._resolve_real_autosubs(xi, bench, captain_id=7, pos_of=POS_OF,
                                        real_points=real_points, real_minutes=real_minutes)
    assert score == (5 + 2 + 6 + 3 + 4 + 8) + 8   # captain 7 doubled
    print("1. no blanks -> plain XI sum + captain double                     OK")


def test_resolve_real_autosubs_covers_a_blanked_starter():
    """Player 4 (a DEF starter) blanks (0 minutes); the only outfield
    bench player (6, a MID) must be admitted in their place."""
    xi = [1, 3, 4, 5]        # GK + 3 outfield starters
    bench = [2, 6]           # bench GK + one outfield bench
    real_points = {1: 5, 3: 2, 4: 0, 5: 3, 2: 1, 6: 7}
    real_minutes = {1: 90, 3: 90, 4: 0, 5: 90, 2: 0, 6: 90}
    score = tr._resolve_real_autosubs(xi, bench, captain_id=5, pos_of=POS_OF,
                                        real_points=real_points, real_minutes=real_minutes)
    # GK(5) + DEF3(2) + [player 4 blanked, skipped] + MID5(3) + bench MID6(7) + captain(5) double
    assert score == 5 + 2 + 3 + 7 + 3
    print("2. a blanked starter is covered by the right bench player        OK")


def test_resolve_real_autosubs_gk_substitution():
    """Starting GK blanks -> bench GK covers, 1-for-1."""
    xi = [1, 3, 4]
    bench = [2]
    real_points = {1: 0, 2: 6, 3: 2, 4: 3}
    real_minutes = {1: 0, 2: 90, 3: 90, 4: 90}
    score = tr._resolve_real_autosubs(xi, bench, captain_id=3, pos_of=POS_OF,
                                        real_points=real_points, real_minutes=real_minutes)
    assert score == 6 + 2 + 3 + 2   # bench GK's real points, not the blanked starter's
    print("3. a blanked starting GK is covered by the bench GK              OK")


def test_resolve_real_autosubs_blanked_captain_not_doubled_if_also_out():
    """If the captain themselves blanks, their real (zero) score is what
    gets doubled — no vice-captain modelled, matching fpl_optimise's own
    simulate_autosub_expected_points and its stated simplification."""
    xi = [1, 3, 4]
    bench = [2]
    real_points = {1: 5, 3: 0, 4: 3, 2: 1}
    real_minutes = {1: 90, 3: 0, 4: 90, 2: 0}
    score = tr._resolve_real_autosubs(xi, bench, captain_id=3, pos_of=POS_OF,
                                        real_points=real_points, real_minutes=real_minutes)
    # player 3 blanked (0 min) -> not counted in the XI sum AND not doubled
    # (captain's own double only applies if THEY played)
    assert score == 5 + 3
    print("4. a blanked captain contributes nothing, including no double    OK")


def test_log_recommendation_appends(tmp_path):
    path = str(tmp_path / "rec_log_test.csv")
    tr.log_recommendation(6, [1, 2, 3], 1, [4, 5], 55.5, path=path)
    tr.log_recommendation(7, [1, 2, 3], 2, [4, 5], 60.0, path=path)
    with open(path) as f:
        lines = f.read().strip().splitlines()
    assert len(lines) == 3            # header + 2 rows
    assert lines[0].startswith("date,gw,xi_ids,captain_id,bench_ids,predicted_score")
    assert ",6," in lines[1] and ",7," in lines[2]
    print("5. log_recommendation appends rows, writes header once            OK")


def test_track_record_leaves_unfinished_gw_as_nan(monkeypatch, tmp_path):
    path = str(tmp_path / "rec_log_test2.csv")
    tr.log_recommendation(6, [1, 2, 3], 1, [4, 5], 55.5, path=path)

    def fake_get(endpoint):
        assert endpoint == "bootstrap-static/"
        return {
            "events": [{"id": 6, "finished": False}],
            "element_types": [{"id": 1, "singular_name_short": "GKP"}],
            "elements": [],
        }
    monkeypatch.setattr(tr.fs0, "get", fake_get)

    rec = tr.track_record(path=path)
    assert pd.isna(rec.iloc[0]["actual_score"])
    print("6. an unfinished gameweek's actual_score stays NaN, not a guess   OK")


def test_track_record_computes_actual_for_a_finished_gw(monkeypatch, tmp_path):
    path = str(tmp_path / "rec_log_test3.csv")
    tr.log_recommendation(6, [1, 3], 3, [2], 20.0, path=path)   # GK1+DEF3, captain=3(DEF), bench GK2

    def fake_get(endpoint):
        if endpoint == "bootstrap-static/":
            return {
                "events": [{"id": 6, "finished": True}],
                "element_types": [
                    {"id": 1, "singular_name_short": "GKP"}, {"id": 2, "singular_name_short": "DEF"},
                ],
                "elements": [
                    {"id": 1, "element_type": 1}, {"id": 2, "element_type": 1}, {"id": 3, "element_type": 2},
                ],
            }
        if endpoint == "element-summary/1/":
            return {"history": [{"round": 6, "total_points": 4, "minutes": 90}]}
        if endpoint == "element-summary/2/":
            return {"history": [{"round": 6, "total_points": 0, "minutes": 0}]}
        if endpoint == "element-summary/3/":
            return {"history": [{"round": 6, "total_points": 9, "minutes": 90}]}
        raise ValueError(endpoint)
    monkeypatch.setattr(tr.fs0, "get", fake_get)

    rec = tr.track_record(path=path)
    assert rec.iloc[0]["actual_score"] == 4 + 9 + 9   # GK1 + DEF3 + captain(3) double
    print("7. a finished gameweek gets a real actual_score computed          OK")


if __name__ == "__main__":
    class _MP:
        def setattr(self, obj, name, value):
            setattr(obj, name, value)

    import pandas as _pd
    import tempfile

    mpatch = _MP()
    test_resolve_real_autosubs_no_blanks_is_a_plain_sum()
    test_resolve_real_autosubs_covers_a_blanked_starter()
    test_resolve_real_autosubs_gk_substitution()
    test_resolve_real_autosubs_blanked_captain_not_doubled_if_also_out()
    with tempfile.TemporaryDirectory() as d:
        import pathlib
        test_log_recommendation_appends(pathlib.Path(d))
    with tempfile.TemporaryDirectory() as d:
        import pathlib
        test_track_record_leaves_unfinished_gw_as_nan(mpatch, pathlib.Path(d))
    with tempfile.TemporaryDirectory() as d:
        import pathlib
        test_track_record_computes_actual_for_a_finished_gw(mpatch, pathlib.Path(d))
    print("\nAll checks passed.")
