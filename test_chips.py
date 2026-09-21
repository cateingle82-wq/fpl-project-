"""Smoke tests for chips.py using synthetic data (no API, no device needed)."""

import random
import pulp

import fpl_optimise as opt
import chips
from fpl_stage0 import HORIZON
from test_optimise import make_players, solve

random.seed(11)


def cold_start_squad(df):
    """Use build_problem itself (unlimited budget/transfers) to hand back a
    legal 15-man squad to build chip tests on top of — same trick
    test_optimise.py uses for its own cold-start test."""
    orig = (opt.MAX_TRANSFERS, opt.HIT_COST, opt.FREE_TRANSFERS)
    opt.MAX_TRANSFERS, opt.HIT_COST, opt.FREE_TRANSFERS = 15, 0.0, 15
    try:
        r = solve(df, [], 10_000)   # huge bank, empty current squad
    finally:
        opt.MAX_TRANSFERS, opt.HIT_COST, opt.FREE_TRANSFERS = orig
    return r["chosen"]


def test_bench_boost_matches_manual_bench_sum():
    df = make_players()
    squad = cold_start_squad(df)
    bb = chips.bench_boost_value(df, squad)
    assert set(bb.keys()) == set(range(HORIZON))

    info = df.set_index("id")
    for w in range(HORIZON):
        xi, _ = opt.choose_lineup_for_week(df, squad, f"xw{w}")
        bench = [p for p in squad if p not in xi]
        manual = sum(info.loc[p, f"xw{w}"] for p in bench)
        assert abs(bb[w] - manual) < 1e-6, f"week {w}: {bb[w]} != {manual}"
        # every player is on the pitch or the bench, nothing double counted
        assert len(xi) + len(bench) == 15
    print("1. bench boost value matches manual bench sum, every week   OK")


def test_bench_boost_never_negative():
    df = make_players()
    squad = cold_start_squad(df)
    bb = chips.bench_boost_value(df, squad)
    assert all(v >= 0 for v in bb.values()), bb
    print("2. bench boost value is never negative (xw >= 0 always)     OK")


def test_triple_captain_matches_manual_captain_score():
    df = make_players()
    squad = cold_start_squad(df)
    tc = chips.triple_captain_value(df, squad)
    info = df.set_index("id")

    for w in range(HORIZON):
        _, cap = opt.choose_lineup_for_week(df, squad, f"xw{w}")
        assert abs(tc[w] - info.loc[cap, f"xw{w}"]) < 1e-6
    print("3. triple captain value matches optimal captain's own score  OK")


def test_triple_captain_picks_highest_scorer_in_double():
    """Give one player a huge score in week 2 (simulating a double gameweek)
    and confirm triple captain value for that week equals THEIR score, not
    someone else's — i.e. it's tracking the actual optimal captain, not a
    fixed player."""
    df = make_players()
    squad = cold_start_squad(df)
    info = df.set_index("id")
    boosted = squad[0]
    df.loc[df["id"] == boosted, "xw2"] = info["xw2"].max() + 50.0

    tc = chips.triple_captain_value(df, squad)
    assert abs(tc[2] - (info.loc[boosted, "xw2"] if boosted != squad[0] else
                         df.loc[df["id"] == boosted, "xw2"].iloc[0])) < 1e-6
    _, cap = opt.choose_lineup_for_week(df, squad, "xw2")
    assert cap == boosted
    print("4. triple captain tracks a newly-boosted double-GW player    OK")


def test_wildcard_never_worse_than_frozen():
    """Freely rebuilding can never score less than being frozen — the
    frozen squad is always a feasible (if suboptimal) answer to the free
    rebuild problem, so wildcard_value must be >= 0."""
    df = make_players()
    squad = cold_start_squad(df)
    wc = chips.wildcard_value(df, squad, bank_t=0)
    assert wc >= -1e-6, wc
    print(f"5. wildcard value is never negative ({wc:.2f})              OK")


def test_wildcard_config_restored_after_call():
    df = make_players()
    squad = cold_start_squad(df)
    before = (opt.MAX_TRANSFERS, opt.HIT_COST)
    chips.wildcard_value(df, squad, bank_t=0)
    after = (opt.MAX_TRANSFERS, opt.HIT_COST)
    assert before == after, f"config leaked: {before} -> {after}"
    print("6. wildcard_value restores MAX_TRANSFERS/HIT_COST             OK")


def test_wildcard_large_for_deliberately_bad_squad():
    """A squad stuffed with cheap, low-scoring players should show a large
    wildcard value — there's obvious room to improve."""
    df = make_players()
    info = df.set_index("id")
    # Build a deliberately bad but legal squad: cheapest player per slot.
    bad_squad = []
    for pos, n in [("GKP", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)]:
        pool = info[info["pos"] == pos].sort_values("price").index.tolist()
        picked, used_teams = [], {}
        for p in pool:
            t = info.loc[p, "team"]
            if used_teams.get(t, 0) >= 3:
                continue
            picked.append(p)
            used_teams[t] = used_teams.get(t, 0) + 1
            if len(picked) == n:
                break
        bad_squad += picked
    assert len(bad_squad) == 15

    wc = chips.wildcard_value(df, bad_squad, bank_t=0)
    assert wc > 0, f"expected clear upside from wildcarding a bad squad, got {wc}"
    print(f"7. wildcard value is clearly positive for a bad squad ({wc:.2f})  OK")


def test_free_hit_never_worse_than_playing_current():
    df = make_players()
    squad = cold_start_squad(df)
    fh = chips.free_hit_value(df, squad, bank_t=0)
    assert fh >= -1e-6, fh
    print(f"8. free hit value is never negative ({fh:.2f})               OK")


def test_free_hit_only_cares_about_week_zero():
    """Tank every player's week-0 score to zero (a total blank gameweek) and
    boost week-1 scores instead. Free hit value should collapse toward zero
    even though there's huge value sitting in week 1 — a free-hit squad
    doesn't carry over, so week 1's numbers must not leak into this."""
    df = make_players()
    squad = cold_start_squad(df)

    df["xw0"] = 0.0
    df["xw1"] = df["xw1"] + 20.0   # inflate week 1 only

    fh = chips.free_hit_value(df, squad, bank_t=0)
    assert abs(fh) < 1e-6, f"free hit leaked future-week value: {fh}"
    print("9. free hit value ignores inflated future weeks (week-0 only) OK")


def test_free_hit_detail_is_internally_consistent():
    """The detail dict backing print_free_hit_xi must actually describe the
    squad it claims to: XI + bench = the 15, XI is 11, captain is in the
    XI, and the printed score matches free_hit_value's own number."""
    df = make_players()
    squad = cold_start_squad(df)
    detail = chips.free_hit_detail(df, squad, bank_t=0)

    assert len(detail["squad"]) == 15
    assert len(detail["xi"]) == 11
    assert detail["captain"] in detail["xi"]
    bench = [p for p in detail["squad"] if p not in detail["xi"]]
    assert len(bench) == 4
    assert set(detail["xi"]) | set(bench) == set(detail["squad"])

    # gain from the detail dict must match the plain free_hit_value() figure
    fh = chips.free_hit_value(df, squad, bank_t=0)
    assert abs(detail["gain"] - fh) < 1e-6

    # print_free_hit_xi must run without error on real output (not just
    # synthetic sanity — this is the function the user actually looks at)
    chips.print_free_hit_xi(df, detail)
    print("11. free hit detail dict is internally consistent            OK")


def test_log_row_appends(tmp_path_str="chip_log_test.csv"):
    """One run now logs one row PER WEEK of the horizon (not just week 0),
    with wildcard/free_hit only on the target_gw == run_gw row — they're
    a horizon-wide total and a this-week-only number, not real per-week
    trajectories the way bench_boost/triple_captain are."""
    import os
    chips.LOG_PATH = tmp_path_str
    if os.path.exists(tmp_path_str):
        os.remove(tmp_path_str)
    bb = {0: 1.23, 1: 1.50, 2: 1.80}
    tc = {0: 2.34, 1: 2.50, 2: 2.60}
    chips.log_row(6, bb, tc, 3.45, 4.56, horizon=3)
    with open(tmp_path_str) as f:
        lines = f.read().strip().splitlines()
    assert len(lines) == 4, lines            # header + 3 weeks
    assert lines[0].startswith("date,run_gw,target_gw,")
    assert ",6,6," in lines[1]               # week offset 0: target == run
    assert ",6,7," in lines[2]               # week offset 1
    assert ",6,8," in lines[3]               # week offset 2
    assert lines[1].endswith("3.45,4.56")    # wildcard/free_hit on week 0
    assert lines[2].endswith(",")            # ...but blank on later weeks
    os.remove(tmp_path_str)
    print("10. log_row logs one row per horizon week, wc/fh only on week 0  OK")


def test_chip_scores_horizon_picks_the_best_visible_week():
    """Bench Boost/Triple Captain are scored against the horizon itself —
    this week's value should score 10/10 when it's the best week visible,
    and something lower (naming the better week) when it isn't."""
    bb = {0: 10.0, 1: 4.0, 2: 2.0}          # this week (0) IS the best
    tc = {0: 3.0, 1: 3.0, 2: 9.0}           # week 2 is clearly better
    scores = chips.chip_scores(bb, tc, wc=0.0, fh=0.0, log_path="nonexistent_log.csv")
    assert scores["bench_boost"]["score"] == 10.0, scores["bench_boost"]
    assert "use it now" in scores["bench_boost"]["verdict"].lower()
    assert scores["triple_captain"]["score"] == 0.0, scores["triple_captain"]
    assert "GW+2" in scores["triple_captain"]["verdict"]
    print("12. chip_scores scores bb/tc against the visible horizon           OK")


def test_chip_scores_history_needs_a_minimum_and_then_ranks():
    """Wildcard/Free Hit have no per-week trajectory — scored against
    logged history instead, with a graceful 'not enough yet' below the
    minimum, then a real percentile once there's enough."""
    import os
    log_path = "chip_score_test_log.csv"
    if os.path.exists(log_path):
        os.remove(log_path)

    bb, tc = {0: 1.0}, {0: 1.0}   # irrelevant to this test

    # Below MIN_HISTORY_FOR_SCORE: no file at all yet.
    scores = chips.chip_scores(bb, tc, wc=50.0, fh=10.0, log_path=log_path)
    assert scores["wildcard"]["score"] is None
    assert "not enough" in scores["wildcard"]["verdict"].lower()

    # Log enough prior readings that wc=50 is clearly the best seen.
    # log_row always writes to the module-level LOG_PATH, not our tmp
    # path — point it there for this call, then restore it.
    orig_log_path = chips.LOG_PATH
    try:
        chips.LOG_PATH = log_path
        for wc_val in [10.0, 20.0, 30.0]:
            chips.log_row(6, bb, tc, wc_val, 5.0, horizon=1)
    finally:
        chips.LOG_PATH = orig_log_path

    scores = chips.chip_scores(bb, tc, wc=50.0, fh=5.0, log_path=log_path)
    assert scores["wildcard"]["score"] == 10.0, scores["wildcard"]
    assert "best" in scores["wildcard"]["verdict"].lower()
    os.remove(log_path)
    print("13. chip_scores needs a history minimum, then ranks correctly      OK")


def test_season_outlook_finds_a_double_beyond_the_horizon():
    """A squad split across two teams: this week (gw6) both teams have one
    fixture each (2 total); a later week (gw10) team1 alone has three
    fixtures (a synthetic stand-in for a double/treble) — outside a
    3-week horizon (gw6-8), so within_horizon must be False and best_gw
    must correctly find gw10, not just whatever the visible horizon shows."""
    import pandas as pd
    df = pd.DataFrame([
        {"id": 1, "team": 1}, {"id": 2, "team": 1},
        {"id": 3, "team": 2}, {"id": 4, "team": 2},
    ])
    squad_ids = [1, 2, 3, 4]
    fixtures = [
        {"event": 6, "team_h": 1, "team_a": 5},
        {"event": 6, "team_h": 2, "team_a": 6},
        {"event": 10, "team_h": 1, "team_a": 7},
        {"event": 10, "team_h": 1, "team_a": 8},
        {"event": 10, "team_h": 1, "team_a": 9},
    ]
    outlook = chips.season_outlook(df, squad_ids, fixtures, gw=6, horizon=3, end_gw=12)
    assert outlook is not None
    assert outlook["this_week_fixtures"] == 2
    assert outlook["best_gw"] == 10
    assert outlook["best_gw_fixtures"] == 3
    assert outlook["within_horizon"] is False
    print("14. season_outlook finds a stronger week beyond the horizon        OK")


def test_season_outlook_returns_none_with_nothing_in_range():
    import pandas as pd
    df = pd.DataFrame([{"id": 1, "team": 1}])
    outlook = chips.season_outlook(df, [1], fixtures=[], gw=6, horizon=3, end_gw=12)
    assert outlook is None
    print("15. season_outlook returns None when there's nothing to scan       OK")


if __name__ == "__main__":
    test_bench_boost_matches_manual_bench_sum()
    test_bench_boost_never_negative()
    test_triple_captain_matches_manual_captain_score()
    test_triple_captain_picks_highest_scorer_in_double()
    test_wildcard_never_worse_than_frozen()
    test_wildcard_config_restored_after_call()
    test_wildcard_large_for_deliberately_bad_squad()
    test_free_hit_never_worse_than_playing_current()
    test_free_hit_only_cares_about_week_zero()
    test_free_hit_detail_is_internally_consistent()
    test_log_row_appends()
    test_chip_scores_horizon_picks_the_best_visible_week()
    test_chip_scores_history_needs_a_minimum_and_then_ranks()
    test_season_outlook_finds_a_double_beyond_the_horizon()
    test_season_outlook_returns_none_with_nothing_in_range()
    print("\nAll checks passed.")
