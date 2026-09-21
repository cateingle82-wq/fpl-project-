"""
FPL chip strategy — should you play a chip this week, and if so which one.

The four chips, and how each is valued here:

  Bench Boost:    your bench players' points count too, for one week.
                  Value = sum of the bench's own xw{w} scores for that week,
                  using an HONEST per-week optimal lineup (choose_lineup_for_
                  week), not just "whoever's on the bench right now" — the
                  bench composition itself should be this week's, since a
                  double-gameweek defender you'd normally bench is exactly
                  who you want warming the bench on Bench Boost week.

  Triple Captain: your captain scores x3 instead of x2 for one week.
                  Value = the marginal week, i.e. ONE extra copy of that
                  week's optimal captain's own xw{w} score (the same player
                  Stage B would already captain that week — triple captain
                  doesn't change WHO you captain, only the multiplier).

  Wildcard:       rebuild your whole 15 for free (no hit, no transfer cap).
                  Value = (best possible squad's HORIZON objective) minus
                  (your current squad's HORIZON objective, frozen, i.e. no
                  transfers at all). This is "how much are you leaving on
                  the table by not being able to rebuild freely right now."

  Free Hit:       rebuild your whole 15 for ONE week only, then it reverts.
                  Value = (best possible squad's week-0 lineup score) minus
                  (your current squad's week-0 lineup score). Only week 0
                  matters because the squad snaps back afterwards — there's
                  no point building a free-hit squad around a fixture three
                  weeks away you won't have the players for.

None of this decides FOR you when to play a chip — a single week's reading
is noisy (see backtest.py's whole reason for existing). This just answers
"how much is each chip worth THIS week", so you can log it over time
(chip_log.csv) and play a chip when its reading is clearly, repeatedly
ahead of the others — not off one snapshot.

Run:  python chips.py
"""

import csv
import os
from datetime import date, datetime

import pulp

import fpl_optimise as opt
from fpl_stage0 import HORIZON, build_table, horizon_of, season_fixture_counts

LOG_PATH = "chip_log.csv"


def bench_boost_value(df, squad_ids):
    """
    Points gained by playing Bench Boost, for each week in the horizon.

    Uses choose_lineup_for_week per week (not the frozen week-0 XI) because
    the bench itself changes week to week — a Bench Boost played in a week
    with a double gameweek defender on your bench is worth more than one
    played when your bench is all blanking, and that only shows up if you
    re-solve the lineup fresh for that week.

    Returns {week: value}.
    """
    values = {}
    for w in range(horizon_of(df)):
        xpts_col = f"xw{w}"
        xi, _ = opt.choose_lineup_for_week(df, squad_ids, xpts_col)
        bench = [p for p in squad_ids if p not in xi]
        info = df.set_index("id")
        values[w] = sum(info.loc[p, xpts_col] for p in bench)
    return values


def triple_captain_value(df, squad_ids):
    """
    Marginal points gained by tripling (vs the normal doubling) your captain,
    for each week. This is just the optimal captain's own score that week —
    triple captain doesn't change the lineup or who's captained, only the
    multiplier on a decision Stage B already makes.

    Returns {week: value}.
    """
    values = {}
    info = df.set_index("id")
    for w in range(horizon_of(df)):
        xpts_col = f"xw{w}"
        _, captain = opt.choose_lineup_for_week(df, squad_ids, xpts_col)
        values[w] = info.loc[captain, xpts_col]
    return values


def season_outlook(df, squad_ids, fixtures, gw, horizon, end_gw=38):
    """
    Scans the FULL REST OF THE SEASON (not just the visible horizon) for
    a gameweek where your squad's teams collectively have notably more
    fixtures than anything currently visible — i.e. a double gameweek the
    horizon slider is too short to see yet. Uses REAL, currently
    confirmed fixtures (fpl_stage0.season_fixture_counts), deliberately
    NOT a guess extrapolated from other seasons' gameweek numbers — see
    chip_scores' docstring for why cross-season pattern-matching isn't
    reliable here (cup replay rules, European scheduling and
    international breaks have all changed structurally season to
    season), while this season's own confirmed fixture list is exactly
    the reliable signal that replaces it.

    Two real limits, both honestly reflected in the output rather than
    faked:
      - This is a rough proxy, not a solve: it uses your CURRENT squad's
        teams as a stand-in for who you'll actually own that far out —
        you'll likely have transferred by then. Treat a flagged week as
        "something worth watching for", not a committed plan.
      - It can only ever see fixtures FPL has already scheduled. Most
        doubles/blanks aren't confirmed until partway through the season
        (cup replays, European competition reshuffling get locked in
        gradually) — early in the season this will often correctly find
        nothing, which is a real absence of data, not a bug.

    Returns None if there are no fixtures in range at all. Otherwise a
    dict: this_week_fixtures, best_gw/best_gw_fixtures (the strongest
    week found across the WHOLE rest of the season), and within_horizon
    (whether that best week is already visible in the current horizon —
    i.e. there's nothing further out worth flagging).
    """
    info = df.set_index("id")
    squad_teams = set(info.loc[squad_ids, "team"])
    counts = season_fixture_counts(fixtures, start_gw=gw, end_gw=end_gw)

    week_totals = {
        w: sum(counts.get(t, {}).get(w, 0) for t in squad_teams)
        for w in range(gw, end_gw + 1)
    }
    if not any(week_totals.values()):
        return None

    best_gw = max(week_totals, key=week_totals.get)
    return {
        "this_week_fixtures": week_totals.get(gw, 0),
        "best_gw": best_gw,
        "best_gw_fixtures": week_totals[best_gw],
        "within_horizon": best_gw < gw + horizon,
        "week_totals": week_totals,
    }


def _solve_stage_a(df, current_ids, bank_t):
    prob, squad, start, cap, hits, ft, tin, tout, cost_t, budget_t = opt.build_problem(
        df, current_ids, bank_t
    )
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        raise SystemExit(f"Stage A solver returned {status} in chip evaluation.")
    # Week 0's squad is the only one that's "real" for this hypothetical —
    # see build_problem's docstring. Chip valuations only ever compare
    # week-0 (or horizon-total) outcomes, never a future planned squad.
    chosen = [p for p in df["id"] if squad[0][p].value() > 0.5]
    return chosen, pulp.value(prob.objective)


def wildcard_detail(df, current_ids, bank_t):
    """
    Full working behind the wildcard number, not just the gap — in
    particular `objective`, the raw horizon-total value of a free full
    rebuild right now, which is what actually needs comparing against
    fpl_optimise's own real (hit-taking) recommendation to answer "is it
    better to take these hits, or just play Wildcard instead" — the gap
    alone (vs a FROZEN squad baseline) can't answer that, since the real
    recommended plan already isn't frozen; it's paying for transfers.

    Temporarily loosens the transfer constraints to "anything goes, no
    cost", solves, then restores the real config — a wildcard is exactly
    that hypothetical, made real for one solve. Config is restored in a
    finally block so a crash mid-evaluation can't leave your real settings
    silently changed for a later run in the same process.

    Returns a dict: squad (the wildcard rebuild), objective (its
    horizon-total value), frozen_objective (your current 15, untouched,
    valued the same honest way), and gain = objective - frozen_objective.
    """
    orig = (opt.MAX_TRANSFERS, opt.HIT_COST)
    try:
        # Best possible squad if you could freely rebuild, no penalty at all.
        opt.MAX_TRANSFERS = 15
        opt.HIT_COST = 0.0
        wildcard_squad, wildcard_obj = _solve_stage_a(df, current_ids, bank_t)

        # Baseline: your current 15, untouched, valued the same honest way
        # (best per-week lineup + captain for that fixed squad).
        opt.MAX_TRANSFERS = 0
        _, frozen_obj = _solve_stage_a(df, current_ids, bank_t)
    finally:
        opt.MAX_TRANSFERS, opt.HIT_COST = orig

    return {
        "squad": wildcard_squad,
        "objective": wildcard_obj,
        "frozen_objective": frozen_obj,
        "gain": wildcard_obj - frozen_obj,
    }


def wildcard_value(df, current_ids, bank_t):
    """Points gained, over the full horizon, by rebuilding the whole squad
    for free right now versus keeping your current 15 untouched. Thin
    wrapper over wildcard_detail — see there for the full working,
    including the raw `objective` a real (hit-taking) recommendation
    should actually be compared against."""
    return wildcard_detail(df, current_ids, bank_t)["gain"]


def free_hit_detail(df, current_ids, bank_t):
    """
    Full working behind the free hit number, not just the gap: the actual
    squad/XI/captain Stage A would build for a one-week-only rebuild, plus
    your current squad's own week-0 XI/captain for comparison. Lets you
    actually LOOK at the free-hit squad rather than trust a bare number —
    the same "inspect what the solver chose, not just its objective value"
    habit that matters for checking any LP result by hand.

    Returns a dict: squad, xi, captain, score (free-hit side) and
    current_xi, current_captain, current_score (your real squad, week 0
    only), plus gain = score - current_score.
    """
    orig = (opt.MAX_TRANSFERS, opt.HIT_COST)
    try:
        opt.MAX_TRANSFERS = 15
        opt.HIT_COST = 0.0
        free_hit_squad, _ = _solve_stage_a(df, current_ids, bank_t)
    finally:
        opt.MAX_TRANSFERS, opt.HIT_COST = orig

    fh_xi, fh_cap = opt.choose_lineup_for_week(df, free_hit_squad, "xw0")
    cur_xi, cur_cap = opt.choose_lineup_for_week(df, current_ids, "xw0")

    info = df.set_index("id")
    # captain doubles, so add their score a second time for both sides
    fh_score = sum(info.loc[p, "xw0"] for p in fh_xi) + info.loc[fh_cap, "xw0"]
    cur_score = sum(info.loc[p, "xw0"] for p in cur_xi) + info.loc[cur_cap, "xw0"]

    return {
        "squad": free_hit_squad,
        "xi": fh_xi,
        "captain": fh_cap,
        "score": fh_score,
        "current_xi": cur_xi,
        "current_captain": cur_cap,
        "current_score": cur_score,
        "gain": fh_score - cur_score,
    }


def free_hit_value(df, current_ids, bank_t):
    """
    Points gained THIS WEEK ONLY by rebuilding the whole squad for one week,
    versus playing your current squad as-is. A free hit squad reverts after
    the gameweek, so only week 0's lineup score is comparable — anything a
    free-hit squad is "worth" beyond week 0 is fictional, you won't own
    those players next week.
    """
    return free_hit_detail(df, current_ids, bank_t)["gain"]


def print_free_hit_xi(df, detail):
    """Print the actual free-hit squad Stage A built, so you can eyeball it
    rather than just trust the value gap — same layout as fpl_optimise's
    report() STARTING XI/BENCH section, scored on xw0 only since that's all
    that matters for a squad you're reverting away from next week."""
    info = df.set_index("id")
    xi, cap, bench = detail["xi"], detail["captain"], [
        p for p in detail["squad"] if p not in detail["xi"]
    ]

    def line(p):
        r = info.loc[p]
        tag = "(C)" if p == cap else ""
        return (f"  {r['name']:<16} {r['pos']:<4} {r['team_name']:<4} "
                f"£{r['price']:>4.1f}  xPts(this GW) {r['xw0']:>5.2f} {tag}")

    order = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
    print("\nFREE HIT XI — the squad Stage A would build if you played it this week")
    for p in sorted(xi, key=lambda p: (order[info.loc[p, "pos"]], -info.loc[p, "xw0"])):
        print(line(p))
    print("\nFREE HIT BENCH")
    for p in sorted(bench, key=lambda p: -info.loc[p, "xw0"]):
        print(line(p))
    spend = sum(info.loc[p, "now_cost"] for p in detail["squad"])
    print(f"\nFree hit squad cost £{spend / 10:.1f}m "
          f"(full budget, no sell-price penalty applied — it's hypothetical)")


def log_row(gw, bb, tc, wc, fh, horizon):
    """Append one row PER WEEK of the horizon to chip_log.csv, not just
    this week's reading — bb/tc are already per-week dicts (see
    bench_boost_value/triple_captain_value), so a single run now leaves a
    full week-by-week trace of what it saw, e.g. letting you compare what
    GW9's bench boost value looked like from three different runs weeks
    apart, not just track a single collapsed number. Never overwrite —
    track over time.

    wildcard/free_hit are a horizon-wide total and a this-week-only
    number respectively, not real per-week trajectories the way bb/tc
    are — written only on the target_gw == run_gw row (blank elsewhere)
    so the log doesn't imply they vary week to week when they don't."""
    is_new = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["date", "run_gw", "target_gw", "bench_boost",
                        "triple_captain", "wildcard", "free_hit"])
        today = date.today().isoformat()
        for offset in range(horizon):
            wc_cell = f"{wc:.2f}" if offset == 0 else ""
            fh_cell = f"{fh:.2f}" if offset == 0 else ""
            w.writerow([today, gw, gw + offset, f"{bb[offset]:.2f}",
                        f"{tc[offset]:.2f}", wc_cell, fh_cell])


# How many past logged readings a chip needs before scoring it against
# its own history means anything — below this, "above/below average" is
# just noise from a tiny sample, same reasoning as backtest.py/ml_train.py
# both refusing to draw conclusions from a handful of gameweeks.
MIN_HISTORY_FOR_SCORE = 3


def _percentile_score(value, values):
    """0-10: where `value` ranks among `values` (10 = highest seen,
    0 = lowest). Returns 5.0 (neutral) if every value is identical —
    nothing to rank against, not a real signal either way."""
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return 5.0
    return 10 * (value - lo) / (hi - lo)


def load_chip_history(log_path=None):
    """Past wildcard/free_hit readings from chip_log.csv (the ones logged
    on their own target_gw == run_gw row — see log_row) — this chip's own
    real track record this season, to score a new reading against.
    Returns ([], []) if the file doesn't exist yet or has no data rows."""
    log_path = log_path or LOG_PATH
    wc_hist, fh_hist = [], []
    if not os.path.exists(log_path):
        return wc_hist, fh_hist
    with open(log_path) as f:
        for row in csv.DictReader(f):
            if row.get("wildcard"):
                wc_hist.append(float(row["wildcard"]))
            if row.get("free_hit"):
                fh_hist.append(float(row["free_hit"]))
    return wc_hist, fh_hist


def _horizon_verdict(name, score, best_offset):
    """Bench Boost / Triple Captain verdict — scored against the visible
    horizon, so it can point at exactly which future week looks better."""
    if best_offset == 0:
        if score >= 8:
            return f"Best week visible for {name} — use it now."
        return (f"Best week visible for {name}, but only just ahead of "
                f"the rest — worth a second look before committing.")
    if score >= 5:
        return f"Reasonable week for {name}, but GW+{best_offset} looks stronger."
    return f"Weak week for {name} — GW+{best_offset} looks meaningfully stronger, wait if you can."


def _history_verdict(name, score, n_have):
    """Wildcard / Free Hit verdict — scored against logged history, since
    there's no cheap per-week trajectory for these two (see chip_scores)."""
    if score is None:
        return (f"Not enough logged history for {name} yet "
                f"({n_have}/{MIN_HISTORY_FOR_SCORE} readings) — check back "
                f"after a few more runs.")
    if score >= 8:
        return f"One of the best {name} readings you've logged this season — strong case to use it."
    if score >= 5:
        return f"Above-average {name} reading vs your own history — reasonable, not a standout."
    return f"Below your own recent average for {name} — may be worth waiting for a better reading."


def chip_scores(bb, tc, wc, fh, log_path=None):
    """
    0-10 "how good is it to use THIS chip THIS week", plus a plain-English
    verdict, for all four chips. Two different reference points, because
    the data available genuinely differs between them:

      Bench Boost / Triple Captain: scored against the VISIBLE HORIZON —
      bb[w]/tc[w] are already computed for every week, so this week's
      value is just its percentile against the best/worst weeks already
      in view. Free (no extra solving), and can name exactly which future
      week looks better.

      Wildcard / Free Hit: only ever evaluated for "right now" (see their
      docstrings — a proper per-week "should I wait" answer needs a full
      Stage A re-solve per candidate week, not implemented, expensive).
      Scored instead against this chip's own LOGGED HISTORY this season
      (chip_log.csv) — a genuine trend from your real fixtures, not a
      guess extrapolated from past seasons' gameweek numbers. That
      extrapolation deliberately isn't attempted: which gameweek has a
      blank or double is driven by cup replay rules, European competition
      scheduling and international breaks, all of which have changed
      structurally between seasons — a past season's GW28 blank says
      close to nothing about whether THIS season's GW28 will have one.
      Needs a few weeks of your own logged history first (see
      MIN_HISTORY_FOR_SCORE); returns score=None with a "not enough
      history yet" verdict until then, rather than a fake number.
    """
    best_bb_w = max(bb, key=bb.get)
    bb_score = _percentile_score(bb[0], list(bb.values()))

    best_tc_w = max(tc, key=tc.get)
    tc_score = _percentile_score(tc[0], list(tc.values()))

    wc_hist, fh_hist = load_chip_history(log_path)
    wc_score = (_percentile_score(wc, wc_hist + [wc])
                if len(wc_hist) >= MIN_HISTORY_FOR_SCORE else None)
    fh_score = (_percentile_score(fh, fh_hist + [fh])
                if len(fh_hist) >= MIN_HISTORY_FOR_SCORE else None)

    return {
        "bench_boost": {"score": round(bb_score, 1),
                         "verdict": _horizon_verdict("Bench Boost", bb_score, best_bb_w)},
        "triple_captain": {"score": round(tc_score, 1),
                            "verdict": _horizon_verdict("Triple Captain", tc_score, best_tc_w)},
        "wildcard": {"score": round(wc_score, 1) if wc_score is not None else None,
                     "verdict": _history_verdict("Wildcard", wc_score, len(wc_hist))},
        "free_hit": {"score": round(fh_score, 1) if fh_score is not None else None,
                     "verdict": _history_verdict("Free Hit", fh_score, len(fh_hist))},
    }


def main():
    df, gw = build_table(horizon=opt.OPTIMISE_HORIZON) if opt.OPTIMISE_HORIZON else build_table()
    horizon = horizon_of(df)

    if opt.MANUAL_SQUAD:
        current_ids = opt.MANUAL_SQUAD
    elif opt.TEAM_ID:
        current_ids = opt.fetch_squad(opt.TEAM_ID, gw)
    else:
        raise SystemExit("Set TEAM_ID or MANUAL_SQUAD in fpl_optimise.py.")

    if len(current_ids) != 15:
        raise SystemExit(f"Expected 15 players, got {len(current_ids)}.")

    bank_t = int(round(opt.BANK * 10))

    bb = bench_boost_value(df, current_ids)
    tc = triple_captain_value(df, current_ids)
    wc = wildcard_value(df, current_ids, bank_t)
    fh_detail = free_hit_detail(df, current_ids, bank_t)
    fh = fh_detail["gain"]

    print(f"\nChip values for GW{gw} onward (your CURRENT squad, no transfers "
          f"applied first — run fpl_optimise.py separately for that decision)\n")

    print("BENCH BOOST — bench points you'd gain, by week:")
    for w in range(horizon):
        print(f"  GW{gw + w}: +{bb[w]:.2f} pts")

    print("\nTRIPLE CAPTAIN — extra points from x3 instead of x2, by week:")
    for w in range(horizon):
        print(f"  GW{gw + w}: +{tc[w]:.2f} pts")

    print(f"\nWILDCARD — horizon points gained by freely rebuilding now "
          f"vs keeping this squad untouched:\n  +{wc:.2f} pts over {horizon} GWs")

    print(f"\nFREE HIT — points gained THIS WEEK ONLY by a one-week rebuild "
          f"vs playing your current squad:\n  +{fh:.2f} pts for GW{gw}")
    print_free_hit_xi(df, fh_detail)

    print(f"\nBest week to play Bench Boost (of the next {horizon}): "
          f"GW{gw + max(bb, key=bb.get)} (+{max(bb.values()):.2f} pts)")
    print(f"Best week to play Triple Captain (of the next {horizon}): "
          f"GW{gw + max(tc, key=tc.get)} (+{max(tc.values()):.2f} pts)")

    # Scored against PRIOR log history — must happen before log_row below
    # writes this run's own reading, or a chip would be scored partly
    # against itself.
    scores = chip_scores(bb, tc, wc, fh)
    print("\n=== How good is it to use each chip THIS WEEK? (0-10) ===")
    for key, label in [("bench_boost", "Bench Boost"), ("triple_captain", "Triple Captain"),
                        ("wildcard", "Wildcard"), ("free_hit", "Free Hit")]:
        s = scores[key]["score"]
        score_str = f"{s}/10" if s is not None else "n/a"
        print(f"  {label:<15} {score_str:<6} {scores[key]['verdict']}")

    log_row(gw, bb, tc, wc, fh, horizon)
    print(f"\nLogged GW{gw}-GW{gw + horizon - 1} readings to {LOG_PATH}. Don't "
          f"act on one week's numbers alone — check back after a few "
          f"gameweeks for a trend, same as backtest.py.")


if __name__ == "__main__":
    main()
