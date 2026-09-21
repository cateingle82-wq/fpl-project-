"""Smoke test for fpl_optimise.build_problem using synthetic data (no API).

Rewritten for the multi-period model: squad[w] can now genuinely differ
week to week (not one squad valued against a lookahead), with transfers
and free-transfer banking as real per-week decisions. legal() now checks
EVERY week's squad/lineup independently, not just one shared shape.
"""

import random
import pandas as pd
import pulp
import fpl_optimise as opt
from fpl_stage0 import HORIZON, horizon_of

random.seed(7)

POS_COUNTS = {"GKP": 40, "DEF": 80, "MID": 90, "FWD": 50}
N_TEAMS = 20


def make_players():
    """Every player gets one xw{w} value per horizon week (loosely correlated
    with price, some noise), plus 'xpts' as their true sum — matching how
    fpl_stage0.build_table actually derives it now."""
    rows, pid = [], 1
    for pos, n in POS_COUNTS.items():
        for i in range(n):
            price_t = random.choice([40, 45, 50, 55, 60, 70, 80, 95, 120, 145])
            team = (pid % N_TEAMS) + 1
            base_per_week = max(0.0, price_t / 18 + random.gauss(0, 1.2)) / HORIZON
            row = {
                "id": pid,
                "name": f"{pos}{i}",
                "pos": pos,
                "team": team,
                "team_name": f"T{team}",
                "now_cost": price_t,
                "price": price_t / 10,
                "avail": 1.0,
                "mins_share": 1.0,   # so print_lookahead's blank-flagging logic has a clean signal
            }
            for w in range(HORIZON):
                row[f"xw{w}"] = round(max(0.0, base_per_week + random.gauss(0, 0.3)), 2)
            row["xpts"] = round(sum(row[f"xw{w}"] for w in range(HORIZON)), 2)
            rows.append(row)
            pid += 1
    return pd.DataFrame(rows)


def set_flat_xpts(df, player_id, total):
    """Overwrite one player's total horizon value, spread evenly across every
    week."""
    per_week = total / HORIZON
    for w in range(HORIZON):
        df.loc[df["id"] == player_id, f"xw{w}"] = per_week
    df.loc[df["id"] == player_id, "xpts"] = total


def make_flat_squad_df(horizon, n_extra_targets=3, base_val=2.0, price=50):
    """A hand-built, easy-to-reason-about 15-man squad (2 GKP/5 DEF/5 MID/
    3 FWD, all flat base_val points every week, all the same price) plus
    n_extra_targets unowned MID 'transfer target' players, same baseline —
    a clean slate the transfer-banking tests below overwrite specific
    week/player values on, so the arithmetic behind each assertion can be
    checked by hand rather than trusted blindly."""
    rows, pid, team = [], 1, 1
    squad_ids = []
    for pos, n in [("GKP", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)]:
        for i in range(n):
            row = {
                "id": pid, "name": f"{pos}{i}", "pos": pos, "team": team,
                "team_name": f"T{team}", "now_cost": price, "price": price / 10,
                "avail": 1.0, "mins_share": 1.0,
            }
            for w in range(horizon):
                row[f"xw{w}"] = base_val
            row["xpts"] = base_val * horizon
            rows.append(row)
            squad_ids.append(pid)
            pid += 1
            team = team % N_TEAMS + 1

    target_ids = []
    for i in range(n_extra_targets):
        row = {
            "id": pid, "name": f"TARGET{i}", "pos": "MID", "team": team,
            "team_name": f"T{team}", "now_cost": price, "price": price / 10,
            "avail": 1.0, "mins_share": 1.0,
        }
        for w in range(horizon):
            row[f"xw{w}"] = base_val
        row["xpts"] = base_val * horizon
        rows.append(row)
        target_ids.append(pid)
        pid += 1
        team = team % N_TEAMS + 1

    return pd.DataFrame(rows), squad_ids, target_ids


def set_week(df, player_id, w, value):
    """Overwrite one player's SINGLE week value and recompute their xpts
    total from the actual per-week columns (not a flat spread) — used by
    the banking tests, where week 0 and week 1 need genuinely different
    values, not the same total smeared evenly."""
    df.loc[df["id"] == player_id, f"xw{w}"] = value
    wk_cols = [c for c in df.columns if c.startswith("xw") and c[2:].isdigit()]
    df.loc[df["id"] == player_id, "xpts"] = df.loc[df["id"] == player_id, wk_cols].sum(axis=1).iloc[0]


def solve(df, squad, bank_t):
    prob, sq, st, cap, hits, ft, tin, tout, cost_t, budget_t = opt.build_problem(df, squad, bank_t)
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    assert pulp.LpStatus[prob.status] == "Optimal", pulp.LpStatus[prob.status]
    horizon = horizon_of(df)
    squad_w = {w: [p for p in df["id"] if sq[w][p].value() > 0.5] for w in range(horizon)}
    xi = {w: [p for p in squad_w[w] if st[w][p].value() > 0.5] for w in range(horizon)}
    captain = {w: [p for p in squad_w[w] if cap[w][p].value() > 0.5] for w in range(horizon)}
    hits_w = {w: int(round(hits[w].value())) for w in range(horizon)}
    ft_w = {0: ft[0]}
    for w in range(1, horizon):
        ft_w[w] = int(round(ft[w].value()))
    return dict(
        chosen=squad_w[0], squad_w=squad_w, xi=xi, captain=captain,
        hits=hits_w, ft=ft_w, cost_t=cost_t, budget_t=budget_t,
        obj=pulp.value(prob.objective), df=df, horizon=horizon,
    )


def legal(r):
    """Squad shape, budget and lineup legality, checked independently for
    EVERY week — squad[w] can genuinely differ week to week now, so
    checking week 0 alone (the old approach) would miss a broken week 3."""
    info = r["df"].set_index("id")
    for w in range(r["horizon"]):
        chosen_w = r["squad_w"][w]
        assert len(chosen_w) == 15, f"week {w}: squad size"
        c = info.loc[chosen_w]
        assert c["pos"].value_counts().to_dict() == {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}, \
            f"week {w}: position counts"
        assert c["team"].value_counts().max() <= 3, f"week {w}: club limit"
        spend = sum(r["cost_t"][p] for p in chosen_w)
        assert spend <= r["budget_t"], f"week {w}: over budget {spend} > {r['budget_t']}"

        xi = r["xi"][w]
        capw = r["captain"][w]
        assert len(xi) == 11, f"week {w}: XI size"
        assert set(xi) <= set(chosen_w), f"week {w}: XI subset of squad"
        xc = info.loc[xi, "pos"].value_counts().to_dict()
        assert xc.get("GKP", 0) == 1 and xc.get("DEF", 0) >= 3, f"week {w}: formation"
        assert xc.get("MID", 0) >= 2 and xc.get("FWD", 0) >= 1, f"week {w}: formation"
        assert len(capw) == 1 and capw[0] in xi, f"week {w}: captain in that week's XI"



def run_all_checks():
    """Run every numbered check below in order. Kept as a function (not
    top-level module code) so other test files can import this module's
    helpers (make_players, solve, legal, ...) without paying the cost of
    re-running this whole multi-minute suite on every import — only
    `python test_optimise.py` itself runs it."""

    # --- 1. sell-price rule -----------------------------------------------------
    assert opt.sell_price_tenths(75, 78) == 76      # +0.3 rise -> keep 0.1
    assert opt.sell_price_tenths(75, 79) == 77      # +0.4 rise -> keep 0.2
    assert opt.sell_price_tenths(75, 75) == 75      # no change
    assert opt.sell_price_tenths(75, 72) == 72      # drop absorbed in full
    print("1. sell-price rule            OK")

    # --- 2. build a legal starting squad to own ---------------------------------
    df = make_players()
    opt.PURCHASE_PRICES = {}
    opt.BAN_UNAVAILABLE = True
    opt.MAX_TRANSFERS = 15
    opt.FREE_TRANSFERS = 15          # free rein: gives us a valid squad to start from
    r = solve(df, [], 1000)          # empty "current squad", £100.0m
    legal(r)
    squad = r["chosen"]
    info = df.set_index("id")
    print(f"2. cold-start squad legal     OK  (obj {r['obj']:.1f})")

    # --- 3. optimal squad proposes no transfer ----------------------------------
    opt.MAX_TRANSFERS, opt.FREE_TRANSFERS = 2, 1
    bank_t = 1000 - sum(r["cost_t"][p] for p in squad)
    r2 = solve(df, squad, bank_t)
    legal(r2)
    assert set(r2["chosen"]) == set(squad), "should not churn an already-optimal squad"
    assert r2["hits"][0] == 0
    print("3. no pointless churn         OK")

    # --- 4. a bad player gets transferred out -----------------------------------
    df4 = df.copy()
    victim = max(squad, key=lambda p: df4.set_index("id").loc[p, "now_cost"])
    set_flat_xpts(df4, victim, 0.0)          # e.g. he just got injured for the whole horizon
    r4 = solve(df4, squad, bank_t)
    legal(r4)
    assert victim not in r4["chosen"], "should sell the dud"
    assert len(set(squad) - set(r4["chosen"])) == 1, "one transfer only"
    assert r4["hits"][0] == 0
    print("4. sells a dud, one transfer  OK")

    # --- 5. a hit is taken only when it pays -------------------------------------
    # Zeroing just two players in a position is NOT enough to force a hit — the
    # per-week model has real bench depth, so losing a couple of starters can
    # often be absorbed for free by promoting existing bench cover via the
    # LINEUP decision. Wiping out an ENTIRE position group removes that
    # escape hatch, forcing a genuine, unavoidable need for outside replacements.
    mid_squad = [p for p in squad if info.loc[p, "pos"] == "MID"]
    assert len(mid_squad) == 5, "squad should own exactly 5 MIDs"

    df5 = df.copy()
    for p in mid_squad:
        set_flat_xpts(df5, p, 0.0)     # every MID you own is now worthless
    r5 = solve(df5, squad, bank_t)
    legal(r5)
    assert r5["hits"][0] >= 1, (
        "with the WHOLE position wiped out, no bench reshuffle can help — "
        "this MUST require outside replacements beyond the 1 free transfer"
    )
    print("5. takes a hit when worth it  OK")

    df6 = df.copy()
    for p in mid_squad:
        set_flat_xpts(df6, p, info.loc[p, "xpts"] - 0.05)   # all barely worse, not wiped out
    r6 = solve(df6, squad, bank_t)
    assert r6["hits"][0] == 0, "a tiny across-the-board dip shouldn't be worth a 4pt hit"
    print("6. refuses a bad hit          OK")

    # --- 7. won't buy a flagged player ------------------------------------------
    df7 = df.copy()
    set_flat_xpts(df7, victim, 0.0)
    star = df7[(~df7["id"].isin(squad))].nlargest(1, "xpts")["id"].iloc[0]
    df7.loc[df7["id"] == star, "avail"] = 0.0         # best available target is injured
    r7 = solve(df7, squad, bank_t)
    assert star not in r7["chosen"], "bought a flagged player"
    print("7. skips flagged targets      OK")

    # --- 8. budget actually binds ------------------------------------------------
    r8 = solve(df, [], 800)                           # only £80.0m
    legal(r8)
    assert sum(r8["cost_t"][p] for p in r8["chosen"]) <= 800
    assert r8["obj"] < r["obj"], "less money should mean fewer points"
    print("8. budget constraint binds    OK")

    # --- 9. per-week reactivity: bench a player during their OWN blank week ----
    # The whole point of the per-week model: a player scoring 0 in ONE specific
    # week (everything else about them unchanged) should be benched THAT week
    # only, with a bench player filling in — something a single flat 'xpts'
    # number could never represent.
    opt.MAX_TRANSFERS = 0        # freeze the squad in EVERY week; isolate lineup logic only
    BLANK_WEEK = 2

    blank_target = next(p for p in squad if info.loc[p, "pos"] != "GKP")
    df9 = df.copy()
    df9.loc[df9["id"] == blank_target, f"xw{BLANK_WEEK}"] = 0.0
    df9.loc[df9["id"] == blank_target, "xpts"] = (
        df9.loc[df9["id"] == blank_target, [f"xw{w}" for w in range(HORIZON)]].sum(axis=1).iloc[0]
    )

    r9 = solve(df9, squad, bank_t)
    legal(r9)
    assert all(set(r9["squad_w"][w]) == set(squad) for w in range(r9["horizon"])), \
        "MAX_TRANSFERS=0 must freeze the squad exactly, every week"
    assert blank_target not in r9["xi"][BLANK_WEEK], (
        "a player scoring 0 that specific week, with a legal bench alternative, "
        "shouldn't start it"
    )
    print("9. benches a player only during their own blank week   OK")

    # --- 10. per-week captaincy: a double gameweek changes the captain choice --
    # The other half of the same point: captaincy should react to WHICH week
    # you're looking at, not stay fixed to whoever's best "on average".
    DOUBLE_WEEK = 3
    dbl_target = min(squad, key=lambda p: info.loc[p, "price"])   # a cheap, normally-uninteresting squad member
    top_candidate = max(squad, key=lambda p: info.loc[p, "xpts"])
    assert dbl_target != top_candidate, "test setup needs these to be different players"

    df10 = df.copy()
    huge = info.loc[top_candidate, "xpts"] / HORIZON * 5   # comfortably the best score anyone has that week
    df10.loc[df10["id"] == dbl_target, f"xw{DOUBLE_WEEK}"] = huge
    df10.loc[df10["id"] == dbl_target, "xpts"] = (
        df10.loc[df10["id"] == dbl_target, [f"xw{w}" for w in range(HORIZON)]].sum(axis=1).iloc[0]
    )

    r10 = solve(df10, squad, bank_t)
    legal(r10)
    assert all(set(r10["squad_w"][w]) == set(squad) for w in range(r10["horizon"]))
    assert r10["captain"][DOUBLE_WEEK] == [dbl_target], "the double-gameweek player should be captained that week"
    other_week = 0 if DOUBLE_WEEK != 0 else 1
    assert r10["captain"][other_week] != [dbl_target], "but NOT captained in a week with no special boost"
    print("10. captaincy adapts per week for a double gameweek     OK")

    # --- 11. banks a free transfer when a bigger, later opportunity needs it ---
    # The actual feature this rewrite exists for. Hand-built 2-week scenario,
    # numbers chosen so the right answer can be checked by arithmetic:
    #
    #   Week 0: target_a is a small, immediately-available upgrade (+0.5 pts
    #           over a baseline MID) — spending the (only) free transfer on it
    #           now is legal and mildly profitable in isolation.
    #   Week 1: target_b AND target_c are each a big upgrade (+4.0 pts over a
    #           baseline MID) — but capturing BOTH needs 2 transfers that week.
    #           Both score BADLY (-100) if bought in week 0 instead — think "not
    #           worth owning yet" — so there's no way to sidestep the timing
    #           question by just acquiring them early while they're cheap to
    #           hold; they must genuinely be bought IN week 1 to be worth having.
    #
    #   Spend now:  +0.5 (week 0) + [8.0 - HIT_COST] (week 1, 1 free + 1 hit
    #               since the only free transfer was already used) = +4.5 total
    #   Bank now:   +0.0 (week 0, skip target_a) + 8.0 (week 1, both transfers
    #               free — 2 free transfers were preserved) = +8.0 total
    #
    # Banking strictly dominates (8.0 > 4.5), so an honestly-modelled solver
    # must choose it on its own — no TRANSFER_OPPORTUNITY_COST knob involved.
    df11, squad11, targets11 = make_flat_squad_df(horizon=2, n_extra_targets=3, base_val=2.0, price=50)
    target_a, target_b, target_c = targets11

    set_week(df11, target_a, 0, 2.5)      # week 0: +0.5 over baseline (2.0)
    set_week(df11, target_b, 0, -100.0)   # unbuyable early — must wait for week 1
    set_week(df11, target_b, 1, 6.0)      # week 1: +4.0 over baseline
    set_week(df11, target_c, 0, -100.0)
    set_week(df11, target_c, 1, 6.0)

    opt.FREE_TRANSFERS = 1
    opt.MAX_TRANSFERS = 2
    opt.HIT_COST = 4.0
    r11 = solve(df11, squad11, 100000)   # budget irrelevant here (everything's the same price)
    legal(r11)

    assert target_a not in r11["chosen"], (
        "took the small week-0 upgrade instead of banking for the bigger week-1 "
        "payoff — banking should have won on total horizon points"
    )
    assert r11["hits"][0] == 0
    assert r11["ft"][1] == 2, "should have preserved both free transfers for week 1"
    assert target_b in r11["squad_w"][1] and target_c in r11["squad_w"][1], \
        "should have captured BOTH week-1 upgrades"
    assert r11["hits"][1] == 0, "both week-1 transfers should have been free, not a hit"
    print("11. banks transfers for a bigger payoff later            OK")

    # --- 12. spends now when there's nothing worth banking for ------------------
    # Same shape, numbers reversed: week 0 has a genuinely big upgrade, week 1
    # only has small ones. There's no future opportunity worth protecting for,
    # so the solver should just take the good trade in front of it — proving
    # test 11 isn't just "the solver never transfers early", it specifically
    # recognises WHEN banking pays off and when it doesn't.
    df12, squad12, targets12 = make_flat_squad_df(horizon=2, n_extra_targets=3, base_val=2.0, price=50)
    target_a2, target_b2, target_c2 = targets12

    set_week(df12, target_a2, 0, 10.0)   # week 0: a big, clearly-worth-it upgrade
    set_week(df12, target_b2, 1, 2.5)    # week 1: only marginal upgrades available
    set_week(df12, target_c2, 1, 2.5)

    opt.FREE_TRANSFERS = 1
    opt.MAX_TRANSFERS = 2
    opt.HIT_COST = 4.0
    r12 = solve(df12, squad12, 100000)
    legal(r12)

    assert target_a2 in r12["chosen"], "should take a clearly-worthwhile transfer now, not hold it back"
    assert r12["hits"][0] == 0
    assert r12["hits"][1] == 0, "a 0.5pt upgrade should never be worth a 4pt hit"
    print("12. spends now when banking wouldn't pay off             OK")

    print("\nAll checks passed.")


if __name__ == "__main__":
    run_all_checks()
