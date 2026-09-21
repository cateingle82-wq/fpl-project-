"""
FPL Stage 1 — constrained transfer optimiser.

Two separate optimisation problems, solved in sequence, not one:

  Stage A (build_problem): which 15 players should you OWN? Uses the
  multi-gameweek horizon xpts, because a transfer is a bet you can't undo
  next week for free — it should weigh several weeks of fixtures.

  Stage B (choose_lineup): given that fixed 15, which 11 do you START and
  who's CAPTAIN this week? Uses xpts_gw1 only. Starting XI and captaincy
  are free to change every gameweek, so a good fixture three weeks out is a
  reason to buy someone, never a reason to bench your best player now.

Both are integer linear programs, not machine learning: each solver searches
every legal combination exactly and returns the provable optimum for its
own, narrower question.

Run:  pip install pulp
      python fpl_optimise.py
"""

import pulp
from fpl_stage0 import HORIZON, build_table, get, horizon_of

# ----------------------------------------------------------------------------
# CONFIG — edit this block
# ----------------------------------------------------------------------------

TEAM_ID = 7362936           # your FPL entry id (the number in your team's URL)
MANUAL_SQUAD = []         # or hardcode 15 element ids if you'd rather

# How many gameweeks ahead to plan over. None = use fpl_stage0's own
# default (HORIZON, currently 5). This is the REAL lever for how much
# squad selection values bench depth/rotation — see build_problem's
# docstring. A longer horizon rewards a squad that can field a strong XI
# in many different weeks; horizon=1 has no rotation value at all, by
# construction, whatever BENCH_WEIGHT is set to.
OPTIMISE_HORIZON = None

BANK = 0.0                # money in the bank, in millions
FREE_TRANSFERS = 5
MAX_TRANSFERS = 5        # hard cap, stops it proposing a wildcard
HIT_COST = 4.0            # points docked per transfer beyond the free ones

# Points-value of KEEPING a free transfer in reserve rather than spending it,
# even when it's free. This is the model's only way to represent "maybe I
# should hold this in case something better comes up" — see the long comment
# in build_problem for why a one-shot LP needs this spelled out explicitly.
# 0.0 = spend every profitable transfer, however marginal (the old behaviour).
# Higher = more conservative; only clearly-worthwhile transfers get made.
# There's no "correct" value — it's your own risk tolerance, expressed as a
# points threshold. Try 1.0-2.0 as a starting point and see if you agree with
# what it holds back.
TRANSFER_OPPORTUNITY_COST = 9

BENCH_WEIGHT = 0.1        # how much a bench player's points are worth to us
BAN_UNAVAILABLE = True    # don't BUY flagged players (keeping one is fine)

# element_id -> price you actually paid, in millions.
# Leave a player out and we assume you paid today's price (i.e. no profit).
PURCHASE_PRICES = {}

# ----------------------------------------------------------------------------


def fetch_squad(team_id, gw):
    """Public endpoint — picks are published once a gameweek is underway."""
    last_gw = max(gw - 1, 1)
    data = get(f"entry/{team_id}/event/{last_gw}/picks/")
    if "picks" not in data:
        raise SystemExit(
            f"No 'picks' in response for team {team_id}, GW{last_gw}. "
            f"Raw response was: {data}\n"
            "Usual causes: wrong TEAM_ID, or that gameweek's picks aren't "
            "published yet (try an earlier GW, or wait until after the deadline)."
        )
    picks = [p["element"] for p in data["picks"]]
    if len(picks) != 15:
        raise SystemExit(
            f"Team {team_id}, GW{last_gw} returned {len(picks)} picks, not 15: "
            f"{picks}\nRaw picks entries: {data['picks']}"
        )
    return picks


def sell_price_tenths(purchase_t, current_t):
    """
    FPL's selling rule: you keep your purchase price plus HALF the rise,
    rounded down to the nearest 0.1. Price drops are absorbed in full.

    Getting this wrong makes the optimiser think it has money it doesn't.
    """
    if current_t <= purchase_t:
        return current_t
    return purchase_t + (current_t - purchase_t) // 2


def build_problem(df, current_ids, bank_t):
    """
    Stage A. Which 15 players to own — valued by simulating your BEST possible
    lineup in EACH of the next `horizon` gameweeks separately, not one blended
    average. This is what lets a squad decision correctly reason "keep him
    even though he blanks GW9, because he doubles GW10" instead of averaging
    those two facts into a single number that hides both of them.

    How many weeks "the horizon" actually is comes from df itself (via
    horizon_of), not a fixed constant — build_table() can be called with
    any horizon (the app's sidebar lets you choose), and this just plans
    over however many weeks that df was actually built for. A LONGER
    horizon is the real lever for "value bench depth more": with more
    weeks in play, a squad gets rewarded for having players who can
    rotate into different weeks' best XIs, not just one strong XI —
    that reward doesn't exist at all with horizon=1.

    start[w][p] / cap[w][p]: does player p start / captain gameweek w
    (w=0 is next week, w=1 the one after, ...). One full XI-and-captain
    sub-decision per week, all sharing the same fixed squad[p].
    """
    horizon = horizon_of(df)
    ids = df["id"].tolist()
    weekly_xpts = [dict(zip(df["id"], df[f"xw{w}"])) for w in range(horizon)]
    pos = dict(zip(df["id"], df["pos"]))
    team = dict(zip(df["id"], df["team"]))
    now_t = dict(zip(df["id"], df["now_cost"]))          # already in tenths
    avail = dict(zip(df["id"], df["avail"]))

    current = set(current_ids)

    # What each player costs us against the budget. If we already own him and
    # keep him, that's his sell price. If we're buying, it's the market price.
    cost_t = {}
    for p in ids:
        if p in current:
            paid_t = int(round(PURCHASE_PRICES.get(p, now_t[p] / 10.0) * 10))
            cost_t[p] = sell_price_tenths(paid_t, now_t[p])
        else:
            cost_t[p] = now_t[p]

    budget_t = bank_t + sum(cost_t[p] for p in current)

    prob = pulp.LpProblem("fpl_transfers", pulp.LpMaximize)

    squad = pulp.LpVariable.dicts("squad", ids, cat="Binary")
    start = {w: pulp.LpVariable.dicts(f"start_w{w}", ids, cat="Binary") for w in range(horizon)}
    cap = {w: pulp.LpVariable.dicts(f"cap_w{w}", ids, cat="Binary") for w in range(horizon)}
    hits = pulp.LpVariable("hits", lowBound=0, cat="Integer")

    # Anyone we own but don't keep has been sold. Defined here, ahead of the
    # objective, because TRANSFER_OPPORTUNITY_COST below needs it.
    sold = pulp.lpSum(1 - squad[p] for p in current)

    # --- objective ----------------------------------------------------------
    # Sum the best-lineup value of EACH week separately, then apply the
    # transfer penalties once (transfers are a single, one-off decision, not
    # a per-week one). Starters score once per week, that week's captain
    # scores twice, bench players are worth a fraction so a squad's bench
    # depth still counts for something.
    #
    # TRANSFER_OPPORTUNITY_COST is a shadow price on every transfer used, not
    # just the ones over your free limit. This solver is a single, deterministic
    # snapshot — given only what it knows right now, holding a free transfer
    # back can never look better than spending it, because there's no way to
    # represent "better information might turn up next week" inside one LP.
    # This term is the workaround: it says a saved transfer is worth roughly
    # this many points of future flexibility, so a transfer only gets used
    # when its actual gain clears that bar, not for a marginal 0.3-point tweak.
    week_terms = [
        pulp.lpSum(
            weekly_xpts[w][p] * (start[w][p] + cap[w][p] + BENCH_WEIGHT * (squad[p] - start[w][p]))
            for p in ids
        )
        for w in range(horizon)
    ]
    prob += (
        pulp.lpSum(week_terms)
        - HIT_COST * hits
        - TRANSFER_OPPORTUNITY_COST * sold
    )

    # --- squad shape (one decision, shared across every week) ---------------
    prob += pulp.lpSum(squad[p] for p in ids) == 15
    for label, n in [("GKP", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)]:
        prob += pulp.lpSum(squad[p] for p in ids if pos[p] == label) == n

    # max 3 players from any one club
    for t in set(team.values()):
        prob += pulp.lpSum(squad[p] for p in ids if team[p] == t) <= 3

    # --- money ----------------------------------------------------------------
    prob += pulp.lpSum(cost_t[p] * squad[p] for p in ids) <= budget_t

    # --- starting XI + captain, repeated for EACH week independently --------
    for w in range(horizon):
        prob += pulp.lpSum(start[w][p] for p in ids) == 11
        for p in ids:
            prob += start[w][p] <= squad[p]     # can't start who you don't own

        prob += pulp.lpSum(start[w][p] for p in ids if pos[p] == "GKP") == 1
        prob += pulp.lpSum(start[w][p] for p in ids if pos[p] == "DEF") >= 3
        prob += pulp.lpSum(start[w][p] for p in ids if pos[p] == "MID") >= 2
        prob += pulp.lpSum(start[w][p] for p in ids if pos[p] == "FWD") >= 1

        prob += pulp.lpSum(cap[w][p] for p in ids) == 1
        for p in ids:
            prob += cap[w][p] <= start[w][p]

    # --- transfers (one-off, not per-week) -----------------------------------
    prob += sold <= MAX_TRANSFERS
    prob += hits >= sold - FREE_TRANSFERS
    # hits is only ever pushed DOWN by the objective, so it settles at
    # max(0, sold - free). No need for an upper bound.

    # --- don't buy injured players -----------------------------------------
    if BAN_UNAVAILABLE:
        for p in ids:
            if p not in current and avail[p] == 0:
                prob += squad[p] == 0

    return prob, squad, start, cap, hits, cost_t, budget_t


def choose_lineup_for_week(df, squad_ids, xpts_col, tiebreak_col="xpts"):
    """
    General version of Stage B: given a FIXED 15-man squad, pick the best
    starting XI and captain for ONE named week's score column. choose_lineup()
    below is just this called with xpts_col="xpts_gw1" — the general form is
    what chip evaluation (chips.py) needs, since bench boost and triple
    captain each need an honest, independently-solved best XI for EVERY
    future week, not just next week.

    No budget, club limit or transfer constraints here — the 15 are already
    decided. 15 binaries instead of 700+, solves instantly.
    """
    ids = list(squad_ids)
    xpts_w = dict(zip(df["id"], df[xpts_col]))
    xpts_tb = dict(zip(df["id"], df[tiebreak_col]))   # tie-break only, see eps below
    pos = dict(zip(df["id"], df["pos"]))

    prob = pulp.LpProblem("fpl_lineup", pulp.LpMaximize)
    start = pulp.LpVariable.dicts("lineup_start", ids, cat="Binary")
    cap = pulp.LpVariable.dicts("lineup_cap", ids, cat="Binary")

    # A blank gameweek can leave several players tied at exactly 0. This
    # epsilon breaks such ties using the tiebreak column (mildly prefer your
    # better long-term players when this week's number can't decide), but
    # it's far too small to ever override a real points difference.
    eps = 1e-4
    prob += pulp.lpSum(
        (xpts_w[p] + eps * xpts_tb[p]) * (start[p] + cap[p]) for p in ids
    )

    prob += pulp.lpSum(start[p] for p in ids) == 11
    prob += pulp.lpSum(start[p] for p in ids if pos[p] == "GKP") == 1
    prob += pulp.lpSum(start[p] for p in ids if pos[p] == "DEF") >= 3
    prob += pulp.lpSum(start[p] for p in ids if pos[p] == "MID") >= 2
    prob += pulp.lpSum(start[p] for p in ids if pos[p] == "FWD") >= 1

    prob += pulp.lpSum(cap[p] for p in ids) == 1
    for p in ids:
        prob += cap[p] <= start[p]

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[prob.status] != "Optimal":
        raise SystemExit(
            f"Lineup solver returned {pulp.LpStatus[prob.status]} — "
            "should never happen against an already-legal 15-man squad."
        )

    xi = [p for p in ids if start[p].value() > 0.5]
    captain = next(p for p in ids if cap[p].value() > 0.5)
    return xi, captain


def choose_lineup(df, squad_ids):
    """Stage B proper: next gameweek only. Thin wrapper over the general form."""
    return choose_lineup_for_week(df, squad_ids, "xpts_gw1")


def print_lookahead(df, gw, chosen, start, cap):
    """
    Stage A solves a full per-week plan internally just to VALUE candidate
    squads — this prints that plan for weeks 1..horizon-1 (week 0 is Stage
    B's job, reported properly in report() below; this is context for WHY
    the transfer decision above looks the way it does, not a second opinion
    on it — always defer to Stage B for what to actually do this week).
    """
    horizon = horizon_of(df)
    info = df.set_index("id")
    print("\nLOOK-AHEAD (why Stage A valued this squad the way it did — "
          "always re-run Stage B fresh each week for the real decision)")
    for w in range(1, horizon):
        captain_w = next((p for p in chosen if cap[w][p].value() > 0.5), None)
        starters_w = [p for p in chosen if start[w][p].value() > 0.5]
        # A starter scoring exactly 0 that week despite normally being a
        # regular (mins_share > 0.3) is very likely a blank, not just poor
        # form — flag it so a squad decision that looks odd out of context
        # ("why keep someone who blanks?") is explained right here.
        blanking = [
            p for p in starters_w
            if info.loc[p, f"xw{w}"] == 0 and info.loc[p, "mins_share"] > 0.3
        ]
        cap_name = info.loc[captain_w, "name"] if captain_w else "?"
        line = f"  GW{gw + w}: captain {cap_name}"
        if blanking:
            names = ", ".join(info.loc[p, "name"] for p in blanking)
            line += f"   [{names} started anyway despite no fixture — check formation legality]"
        print(line)


def report(df, current_ids, squad, hits, cost_t, budget_t, xi, captain):
    info = df.set_index("id")
    chosen = [p for p in info.index if squad[p].value() > 0.5]
    bench = [p for p in chosen if p not in xi]

    current = set(current_ids)
    # Transfer ranking still uses the horizon score — that's what a transfer
    # is actually buying (see module docstring).
    out = sorted(current - set(chosen), key=lambda p: -info.loc[p, "xpts"])
    inn = sorted(set(chosen) - current, key=lambda p: -info.loc[p, "xpts"])

    def line(p, tag="", show_gw1=False):
        r = info.loc[p]
        text = (f"  {r['name']:<16} {r['pos']:<4} {r['team_name']:<4} "
                f"£{r['price']:>4.1f}  xPts {r['xpts']:>5.2f}")
        if show_gw1:
            text += f"  xPts(next GW) {r['xpts_gw1']:>5.2f}"
        # Flags a player whose score came from the cold-start ML model
        # instead of the usual shrinkage heuristic (see fpl_stage0.py's
        # USE_ML_COLD_START) — worth knowing when you're eyeballing a
        # transfer suggestion, since it's a newer, less battle-tested
        # source of the number. Column may not exist on older cached data.
        if "xpts_source" in info.columns and r.get("xpts_source") == "ml":
            text += " [ML]"
        return text + f" {tag}"

    print("\n" + "=" * 78)
    if not inn:
        print("RECOMMENDATION: no transfer. Roll it.")
    else:
        n_hits = int(round(hits.value()))
        print(f"RECOMMENDATION: {len(inn)} transfer(s), {n_hits} hit(s) "
              f"= -{n_hits * int(HIT_COST)} pts")
        print("\nOUT")
        for p in out:
            print(line(p))
        print("\nIN")
        for p in inn:
            print(line(p))

    print("\nSTARTING XI — chosen for next gameweek's fixtures only")
    order = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
    for p in sorted(xi, key=lambda p: (order[info.loc[p, "pos"]], -info.loc[p, "xpts_gw1"])):
        print(line(p, "(C)" if p == captain else "", show_gw1=True))

    print("\nBENCH")
    for p in sorted(bench, key=lambda p: -info.loc[p, "xpts_gw1"]):
        print(line(p, show_gw1=True))

    spend = sum(cost_t[p] for p in chosen)
    print(f"\nSquad cost £{spend / 10:.1f}m of £{budget_t / 10:.1f}m available "
          f"(£{(budget_t - spend) / 10:.1f}m left in the bank)")
    print("=" * 78 + "\n")


def main():
    df, gw = build_table(horizon=OPTIMISE_HORIZON) if OPTIMISE_HORIZON else build_table()

    if MANUAL_SQUAD:
        current_ids = MANUAL_SQUAD
    elif TEAM_ID:
        current_ids = fetch_squad(TEAM_ID, gw)
    else:
        raise SystemExit("Set TEAM_ID or MANUAL_SQUAD at the top of the file.")

    if len(current_ids) != 15:
        raise SystemExit(f"Expected 15 players, got {len(current_ids)}.")

    # Stage A: which 15 to own, weighing the full horizon of fixtures
    # (df's own horizon_of(df) weeks — see OPTIMISE_HORIZON above).
    prob, squad, start, cap, hits, cost_t, budget_t = build_problem(
        df, current_ids, int(round(BANK * 10))
    )

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        raise SystemExit(f"Solver returned {status} — usually means your budget "
                         "or transfer cap makes a legal squad impossible.")

    chosen = [p for p in df["id"] if squad[p].value() > 0.5]

    # Stage B: which 11 to start and who captains, weighing ONLY next GW.
    xi, captain = choose_lineup(df, chosen)

    print(f"\nOptimising for GW{gw} onward. Squad objective: "
          f"{pulp.value(prob.objective):.2f} expected points over the horizon.")
    report(df, current_ids, squad, hits, cost_t, budget_t, xi, captain)
    print_lookahead(df, gw, chosen, start, cap)


if __name__ == "__main__":
    main()