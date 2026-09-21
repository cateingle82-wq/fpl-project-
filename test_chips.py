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
    import os
    chips.LOG_PATH = tmp_path_str
    if os.path.exists(tmp_path_str):
        os.remove(tmp_path_str)
    chips.log_row(6, 1.23, 2.34, 3.45, 4.56)
    chips.log_row(7, 1.11, 2.22, 3.33, 4.44)
    with open(tmp_path_str) as f:
        lines = f.read().strip().splitlines()
    assert len(lines) == 3, lines            # header + 2 rows
    assert lines[0].startswith("date,gw,")
    assert ",6," in lines[1] and ",7," in lines[2]
    os.remove(tmp_path_str)
    print("10. log_row appends rows, writes header once                 OK")


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
    print("\nAll checks passed.")
