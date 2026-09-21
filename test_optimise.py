"""Smoke test for fpl_optimise.build_problem using synthetic data (no API)."""

import random
import pandas as pd
import pulp
import fpl_optimise as opt
from fpl_stage0 import HORIZON

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
    week. Used by tests that only care about the TOTAL (they predate the
    per-week feature) — new per-week-specific tests (11, 12) edit xw{w}
    columns directly instead."""
    per_week = total / HORIZON
    for w in range(HORIZON):
        df.loc[df["id"] == player_id, f"xw{w}"] = per_week
    df.loc[df["id"] == player_id, "xpts"] = total


def solve(df, squad, bank_t):
    prob, s, st, cap, hits, cost_t, budget_t = opt.build_problem(df, squad, bank_t)
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    assert pulp.LpStatus[prob.status] == "Optimal", pulp.LpStatus[prob.status]
    chosen = [p for p in df["id"] if s[p].value() > 0.5]
    xi = {w: [p for p in chosen if st[w][p].value() > 0.5] for w in range(HORIZON)}
    captain = {w: [p for p in chosen if cap[w][p].value() > 0.5] for w in range(HORIZON)}
    return dict(chosen=chosen, xi=xi, captain=captain,
                hits=int(round(hits.value())), cost_t=cost_t,
                budget_t=budget_t, obj=pulp.value(prob.objective), df=df)


def legal(r):
    """Checks squad shape once, then checks EVERY week's lineup/captain
    independently — this is the part that changed: legality is no longer a
    single check, it's HORIZON separate checks that must all hold."""
    info = r["df"].set_index("id")
    c = info.loc[r["chosen"]]
    assert len(r["chosen"]) == 15, "squad size"
    assert c["pos"].value_counts().to_dict() == {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
    assert c["team"].value_counts().max() <= 3, "club limit"
    spend = sum(r["cost_t"][p] for p in r["chosen"])
    assert spend <= r["budget_t"], f"over budget {spend} > {r['budget_t']}"

    for w in range(HORIZON):
        xi = r["xi"][w]
        capw = r["captain"][w]
        assert len(xi) == 11, f"week {w}: XI size"
        assert set(xi) <= set(r["chosen"]), f"week {w}: XI subset of squad"
        xc = info.loc[xi, "pos"].value_counts().to_dict()
        assert xc.get("GKP", 0) == 1 and xc.get("DEF", 0) >= 3, f"week {w}: formation"
        assert xc.get("MID", 0) >= 2 and xc.get("FWD", 0) >= 1, f"week {w}: formation"
        assert len(capw) == 1 and capw[0] in xi, f"week {w}: captain in that week's XI"


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
opt.TRANSFER_OPPORTUNITY_COST = 0.0   # tests 1-10 assume the old "spend if profitable" behaviour
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
assert r2["hits"] == 0
print("3. no pointless churn         OK")

# --- 4. a bad player gets transferred out -----------------------------------
df4 = df.copy()
victim = max(squad, key=lambda p: df4.set_index("id").loc[p, "now_cost"])
set_flat_xpts(df4, victim, 0.0)          # e.g. he just got injured for the whole horizon
r4 = solve(df4, squad, bank_t)
legal(r4)
assert victim not in r4["chosen"], "should sell the dud"
assert len(set(squad) - set(r4["chosen"])) == 1, "one transfer only"
assert r4["hits"] == 0
print("4. sells a dud, one transfer  OK")

# --- 5. a hit is taken only when it pays -------------------------------------
# IMPORTANT: zeroing just two players in a position is NOT enough to force a
# hit anymore. The new per-week model has real bench depth — if you own 5
# MIDs and only need 2-5 starting, zeroing two of them can often be absorbed
# for free by benching them and promoting existing bench cover into the XI
# via the LINEUP decision, with the squad's one transfer spent on the single
# biggest upgrade instead. That's genuinely smarter than the old model, but
# it means this test must wipe out an ENTIRE position group — something no
# amount of bench reshuffling within that group can fix — to force a real,
# unavoidable need for outside replacements.
mid_squad = [p for p in squad if info.loc[p, "pos"] == "MID"]
assert len(mid_squad) == 5, "squad should own exactly 5 MIDs"

df5 = df.copy()
for p in mid_squad:
    set_flat_xpts(df5, p, 0.0)     # every MID you own is now worthless
r5 = solve(df5, squad, bank_t)
legal(r5)
assert r5["hits"] >= 1, (
    "with the WHOLE position wiped out, no bench reshuffle can help — "
    "this MUST require outside replacements beyond the 1 free transfer"
)
print("5. takes a hit when worth it  OK")

df6 = df.copy()
for p in mid_squad:
    set_flat_xpts(df6, p, info.loc[p, "xpts"] - 0.05)   # all barely worse, not wiped out
r6 = solve(df6, squad, bank_t)
assert r6["hits"] == 0, "a tiny across-the-board dip shouldn't be worth a 4pt hit"
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

# --- 9. opportunity cost blocks a marginal transfer --------------------------
# victim9 must be a near-certain STARTER, not just any squad member — a bench
# player only counts at BENCH_WEIGHT (0.1) in the objective, so a "+5 point"
# swap on a benched player is really worth just +0.5 to the solver, which
# proves nothing about the opportunity-cost threshold. The squad's single
# highest-value outfield player is about as close to "always starts" as this
# synthetic data can guarantee.
df9 = df.copy()
victim9 = max(
    (p for p in squad if info.loc[p, "pos"] != "GKP"),
    key=lambda p: info.loc[p, "xpts"],
)
same_pos = info.loc[victim9, "pos"]
target9 = df9[(df9["pos"] == same_pos) & (~df9["id"].isin(squad))].iloc[0]["id"]
set_flat_xpts(df9, target9, info.loc[victim9, "xpts"] + 0.3)   # tiny available upgrade
df9.loc[df9["id"] == target9, "now_cost"] = info.loc[victim9, "now_cost"]

opt.TRANSFER_OPPORTUNITY_COST = 1.5
opt.MAX_TRANSFERS, opt.FREE_TRANSFERS = 2, 2
r9 = solve(df9, squad, bank_t)
legal(r9)
assert set(r9["chosen"]) == set(squad), "a 0.3pt gain shouldn't clear a 1.5pt opportunity cost"
print("9. blocks a marginal transfer  OK")

# --- 10. but a clearly worthwhile transfer still goes through -----------------
df10 = df.copy()
set_flat_xpts(df10, target9, info.loc[victim9, "xpts"] + 5.0)
df10.loc[df10["id"] == target9, "now_cost"] = info.loc[victim9, "now_cost"]
r10 = solve(df10, squad, bank_t)
legal(r10)
assert target9 in r10["chosen"], "a 5pt gain should easily clear a 1.5pt opportunity cost"
# NOTE: this does NOT assert victim9 is dropped — target9 just needs to fit
# the position count. The solver is free to (and should) drop a WEAKER same-
# position squad member instead of victim9, keeping its best player and
# upgrading elsewhere. That's the better decision, not a bug.
print("10. allows a clear upgrade     OK")

# --- 11. per-week reactivity: bench a player during their OWN blank week ----
# The whole point of this build: a player scoring 0 in ONE specific week
# (everything else about them unchanged) should be benched THAT week only,
# with a bench player filling in — something a single flat 'xpts' number
# could never represent, since it can't distinguish "bad all season" from
# "fine except this one gameweek".
opt.TRANSFER_OPPORTUNITY_COST = 0.0
opt.MAX_TRANSFERS = 0        # freeze the squad; isolate per-week lineup logic only
BLANK_WEEK = 2

blank_target = next(p for p in squad if info.loc[p, "pos"] != "GKP")
df11 = df.copy()
df11.loc[df11["id"] == blank_target, f"xw{BLANK_WEEK}"] = 0.0
df11.loc[df11["id"] == blank_target, "xpts"] = (
    df11.loc[df11["id"] == blank_target, [f"xw{w}" for w in range(HORIZON)]].sum(axis=1).iloc[0]
)

r11 = solve(df11, squad, bank_t)
legal(r11)
assert set(r11["chosen"]) == set(squad), "MAX_TRANSFERS=0 must freeze the squad exactly"
assert blank_target not in r11["xi"][BLANK_WEEK], (
    "a player scoring 0 that specific week, with a legal bench alternative, "
    "shouldn't start it — the OLD single-number model had no way to even ask this"
)
print("11. benches a player only during their own blank week   OK")

# --- 12. per-week captaincy: a double gameweek changes the captain choice --
# The other half of the same point: captaincy should react to WHICH week
# you're looking at, not stay fixed to whoever's best "on average".
DOUBLE_WEEK = 3
dbl_target = min(squad, key=lambda p: info.loc[p, "price"])   # a cheap, normally-uninteresting squad member
top_candidate = max(squad, key=lambda p: info.loc[p, "xpts"])
assert dbl_target != top_candidate, "test setup needs these to be different players"

df12 = df.copy()
huge = info.loc[top_candidate, "xpts"] / HORIZON * 5   # comfortably the best score anyone has that week
df12.loc[df12["id"] == dbl_target, f"xw{DOUBLE_WEEK}"] = huge
df12.loc[df12["id"] == dbl_target, "xpts"] = (
    df12.loc[df12["id"] == dbl_target, [f"xw{w}" for w in range(HORIZON)]].sum(axis=1).iloc[0]
)

r12 = solve(df12, squad, bank_t)
legal(r12)
assert set(r12["chosen"]) == set(squad)
assert r12["captain"][DOUBLE_WEEK] == [dbl_target], "the double-gameweek player should be captained that week"
other_week = 0 if DOUBLE_WEEK != 0 else 1
assert r12["captain"][other_week] != [dbl_target], "but NOT captained in a week with no special boost"
print("12. captaincy adapts per week for a double gameweek     OK")

print("\nAll checks passed.")
