"""
FPL Stage 1 — constrained transfer optimiser.

Two separate optimisation problems, solved in sequence, not one:

  Stage A (build_problem): a genuinely MULTI-PERIOD plan — which 15 players
  do you own in EACH week of the horizon, and how many of your free
  transfers do you spend in each of those weeks vs bank for later? One
  MILP, one solve, one global optimum over the whole horizon at once.

  Stage B (choose_lineup): given week 0's committed squad, which 11 do you
  START and who's CAPTAIN this week? Only week 0 is ever actually acted on
  — everything the model plans for future weeks is a hypothesis about what
  it would do GIVEN NO NEW INFORMATION, and gets thrown away and re-solved
  fresh next week once real data (injuries, price changes, new fixtures)
  comes in. See build_problem's docstring for why this still matters even
  though it can't be "acted on" yet.

Both are integer linear programs, not machine learning: each solver searches
every legal combination exactly and returns the provable optimum for its
own, narrower question.

Run:  pip install pulp
      python fpl_optimise.py
"""

import itertools

import pulp
from fpl_stage0 import HORIZON, build_table, get, horizon_of

# ----------------------------------------------------------------------------
# CONFIG — edit this block
# ----------------------------------------------------------------------------

TEAM_ID = 7362936           # your FPL entry id (the number in your team's URL)
MANUAL_SQUAD = []         # or hardcode 15 element ids if you'd rather

# How many gameweeks ahead to plan over. None = use fpl_stage0's own
# default (HORIZON, currently 5). This is what lets the model reason about
# WHEN to use a transfer, not just whether to — a 1-week horizon can never
# see a reason to bank one (there's no "later" to bank it for); an 8-week
# horizon can genuinely compare "swap someone in now for a small gain" vs
# "hold both free transfers so week 4's double-gameweek swap doesn't cost
# a hit". See build_problem's docstring.
OPTIMISE_HORIZON = None

BANK = 0.0                # money in the bank, in millions
FREE_TRANSFERS = 5        # free transfers you actually have banked right now
MAX_TRANSFERS = 5         # hard cap on transfers used in ANY SINGLE week —
                           # a solver-speed guard and a sanity ceiling, not
                           # the lever for "how big a rehaul": the model
                           # itself now decides how many of your available
                           # transfers to actually spend, and when.
HIT_COST = 4.0             # points docked per transfer beyond the free ones
BAN_UNAVAILABLE = True     # don't BUY flagged players (keeping one is fine)

# element_id -> price you actually paid, in millions.
# Leave a player out and we assume you paid today's price (i.e. no profit).
PURCHASE_PRICES = {}

# ----------------------------------------------------------------------------
# Internal constants — not meant to be hand-tuned per run.
# ----------------------------------------------------------------------------

# A tiny nudge so the solver doesn't leave your bench full of £4.0m zeros
# once BOTH squad and lineup are being chosen together (see the objective's
# comment below for why it needs to exist at all). Deliberately far too
# small to compete with any real points decision — this is bookkeeping,
# not a "how much do I value my bench" knob. That question used to be a
# user-set BENCH_WEIGHT; it isn't one any more because with a real
# multi-period model, a squad that can field a strong XI in more different
# weeks already scores higher on its own merits (see build_problem) —
# rewarding bench depth a second time via a hand-tuned weight was doing the
# same job worse.
BENCH_TIEBREAK = 0.02

# Max free transfers FPL lets you bank at once. A rule of the game, not a
# preference — change this only if FPL itself changes the cap.
MAX_BANKED_FREE_TRANSFERS = 5

# Big-M for the free-transfer-banking linearisation below. Must exceed the
# largest possible |ft[w] - transfers_used[w]|, which is bounded by
# MAX_TRANSFERS (the largest transfers_used can be) — 20 gives headroom.
_BIG_M = 20

# ----------------------------------------------------------------------------


def fetch_squad_and_bank(team_id, gw):
    """Public endpoint — picks are published once a gameweek is underway.
    Also returns the REAL bank balance FPL already tracks for this team
    (entry_history.bank, in tenths) — this reflects every past transfer
    and every real price rise/fall on your squad automatically, which is
    strictly better than predicting future prices: no need to guess where
    prices are headed when the actual current number is one field away in
    a response we're already fetching."""
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
    bank_t = data.get("entry_history", {}).get("bank")
    return picks, bank_t


def fetch_squad(team_id, gw):
    """Thin wrapper over fetch_squad_and_bank for callers that only need
    the 15 player ids (e.g. chips.py, which takes bank from its own
    caller instead)."""
    picks, _ = fetch_squad_and_bank(team_id, gw)
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
    Stage A. A single MILP that plans your squad for EVERY week of the
    horizon at once, including how many transfers to use in each week —
    not one fixed squad valued against a multi-week lookahead (the old
    design), but genuinely different squad[w] for each week w, linked
    week-to-week by transfer and free-transfer-banking constraints.

    WHY THIS MATTERS: the old model could only ever ask "what's the best
    squad for the whole horizon, at once?" — it had no way to represent
    spreading transfers OUT over time, because it never modelled a future
    week's squad as different from this week's. That meant it had no way
    to answer "should I bank a transfer now for a bigger swap in 3 weeks?"
    honestly — it could only fake caution with a hand-tuned penalty on
    every transfer (TRANSFER_OPPORTUNITY_COST, now removed). With squad[w]
    genuinely free to differ week to week, banking becomes a REAL trade-off
    the solver reasons about on its own: spending a transfer now can only
    ever cost you flexibility later (fewer transfers available when a
    bigger opportunity shows up, possibly forcing a hit then) — so the
    solver only spends one when the number in hand clears that real bar,
    not a hand-set one.

    Decision variables, each indexed by week w = 0..horizon-1 (w=0 is next
    gameweek):
      squad[w][p]  — do you own player p in week w
      start[w][p]  — do you start them that week (only meaningful if owned)
      cap[w][p]    — are they captain that week
      tin[w][p] / tout[w][p] — transferred in/out entering week w, relative
                     to week w-1's squad (week -1 is your real squad now)
      hits[w]      — transfers used beyond that week's free allowance
      ft[w]        — free transfers AVAILABLE going into week w (ft[0] is
                     given — your real FREE_TRANSFERS right now; ft[1:] are
                     solved for, following FPL's own banking rule: unused
                     free transfers carry over, +1 per week, capped at
                     MAX_BANKED_FREE_TRANSFERS)

    Only week 0 is ever real — everything from week 1 on is the model's own
    best guess at what it WOULD do, useful for understanding why it's
    making week 0's decision the way it is (see print_lookahead), not a
    commitment. Re-run fresh every week once real news comes in.
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
    # Simplification: this doesn't change week to week (no modelling of
    # in-season price rises/falls) — a reasonable approximation over a
    # handful of weeks, but worth knowing if you're staring at week 5's
    # hypothetical budget and it looks a little off from reality.
    cost_t = {}
    for p in ids:
        if p in current:
            paid_t = int(round(PURCHASE_PRICES.get(p, now_t[p] / 10.0) * 10))
            cost_t[p] = sell_price_tenths(paid_t, now_t[p])
        else:
            cost_t[p] = now_t[p]

    budget_t = bank_t + sum(cost_t[p] for p in current)

    prob = pulp.LpProblem("fpl_transfers", pulp.LpMaximize)

    squad = {w: pulp.LpVariable.dicts(f"squad_w{w}", ids, cat="Binary") for w in range(horizon)}
    start = {w: pulp.LpVariable.dicts(f"start_w{w}", ids, cat="Binary") for w in range(horizon)}
    cap = {w: pulp.LpVariable.dicts(f"cap_w{w}", ids, cat="Binary") for w in range(horizon)}
    tin = {w: pulp.LpVariable.dicts(f"in_w{w}", ids, cat="Binary") for w in range(horizon)}
    tout = {w: pulp.LpVariable.dicts(f"out_w{w}", ids, cat="Binary") for w in range(horizon)}
    hits = {w: pulp.LpVariable(f"hits_w{w}", lowBound=0, cat="Integer") for w in range(horizon)}

    # ft[0] is a KNOWN CONSTANT (your real transfer count right now), not a
    # decision — nothing before this week is up for negotiation. ft[1:] are
    # real variables, solved for via the banking recursion below.
    ft = {0: FREE_TRANSFERS}
    for w in range(1, horizon):
        ft[w] = pulp.LpVariable(f"ft_w{w}", lowBound=0, upBound=MAX_BANKED_FREE_TRANSFERS, cat="Integer")

    # --- objective ------------------------------------------------------------
    # Sum the best-lineup value of EACH week, minus hit costs summed across
    # every week they occur — a hit in week 3 costs exactly as much as a
    # hit in week 0, because both are real points lost, whenever they land.
    # BENCH_TIEBREAK exists purely so an unstarted bench player scoring 0 in
    # the objective doesn't make the solver indifferent between a useful
    # squad player and a throwaway one sitting there doing nothing — see
    # its own comment above for why it's not a "value my bench" knob.
    week_terms = [
        pulp.lpSum(
            weekly_xpts[w][p] * (start[w][p] + cap[w][p] + BENCH_TIEBREAK * (squad[w][p] - start[w][p]))
            for p in ids
        )
        for w in range(horizon)
    ]
    prob += (
        pulp.lpSum(week_terms)
        - HIT_COST * pulp.lpSum(hits[w] for w in range(horizon))
    )

    # --- squad shape, repeated for EACH week independently ---------------------
    for w in range(horizon):
        prob += pulp.lpSum(squad[w][p] for p in ids) == 15
        for label, n in [("GKP", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)]:
            prob += pulp.lpSum(squad[w][p] for p in ids if pos[p] == label) == n
        for t in set(team.values()):
            prob += pulp.lpSum(squad[w][p] for p in ids if team[p] == t) <= 3
        prob += pulp.lpSum(cost_t[p] * squad[w][p] for p in ids) <= budget_t

    # --- starting XI + captain, repeated for EACH week independently ----------
    for w in range(horizon):
        prob += pulp.lpSum(start[w][p] for p in ids) == 11
        for p in ids:
            prob += start[w][p] <= squad[w][p]     # can't start who you don't own

        prob += pulp.lpSum(start[w][p] for p in ids if pos[p] == "GKP") == 1
        prob += pulp.lpSum(start[w][p] for p in ids if pos[p] == "DEF") >= 3
        prob += pulp.lpSum(start[w][p] for p in ids if pos[p] == "MID") >= 2
        prob += pulp.lpSum(start[w][p] for p in ids if pos[p] == "FWD") >= 1

        prob += pulp.lpSum(cap[w][p] for p in ids) == 1
        for p in ids:
            prob += cap[w][p] <= start[w][p]

    # --- transfers: how squad[w] relates to squad[w-1] (or your real squad,
    # for w=0) ------------------------------------------------------------------
    transfers_used = {}
    for w in range(horizon):
        prev = {p: (1 if p in current else 0) for p in ids} if w == 0 else squad[w - 1]
        for p in ids:
            prob += squad[w][p] - prev[p] == tin[w][p] - tout[w][p]
        transfers_used[w] = pulp.lpSum(tout[w][p] for p in ids)
        prob += transfers_used[w] <= MAX_TRANSFERS
        prob += hits[w] >= transfers_used[w] - ft[w]
        # hits[w] is only ever pushed DOWN by the objective, so it settles
        # at max(0, transfers_used[w] - ft[w]). No upper bound needed.

    # --- free-transfer banking: ft[w+1] = min(5, max(0, ft[w] - used[w]) + 1) -
    # The max(0, ...) needs an explicit linearisation because ft[w] - used[w]
    # CAN go negative (you took a hit that week) and free transfers can't
    # carry a negative balance into next week — this is the standard
    # max(a, 0) trick: slack >= a, slack >= 0, and two upper bounds gated by
    # a binary z so slack is only ever pinned to a OR to 0, never in
    # between. It's forced to the correct one of those two values (not just
    # bounded) because the objective always wants slack as large as
    # possible (more banked free transfers can only ever help later), so at
    # any optimal solution the solver pushes it right up against whichever
    # upper bound is actually reachable.
    for w in range(horizon - 1):
        a_w = ft[w] - transfers_used[w]
        slack = pulp.LpVariable(f"ft_slack_w{w}", lowBound=0, cat="Integer")
        z = pulp.LpVariable(f"ft_hit_flag_w{w}", cat="Binary")
        prob += slack >= a_w
        prob += slack <= a_w + _BIG_M * (1 - z)
        prob += slack <= _BIG_M * z
        prob += ft[w + 1] <= slack + 1
        # ft[w+1] <= MAX_BANKED_FREE_TRANSFERS and >= 0 are already its
        # variable bounds — no separate constraint needed for those.

    # --- don't buy injured players (keeping an already-owned one is fine) -----
    if BAN_UNAVAILABLE:
        for w in range(horizon):
            for p in ids:
                if avail[p] == 0:
                    prob += tin[w][p] == 0

    return prob, squad, start, cap, hits, ft, tin, tout, cost_t, budget_t


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


def simulate_autosub_expected_points(df, xi, bench, captain, xpts_col):
    """
    Exact expected points for a starting XI + bench under FPL's real
    autosub rule: a blanked (0-minute) starter is covered by the highest-
    priority bench player who themselves actually played that week — not
    the naive "sum of the 11 starters' own expected points" used
    elsewhere, which silently treats a strong bench as worth nothing (see
    BENCH_TIEBREAK) and so understates a squad with good backups' true
    expected return.

    Computed as an EXACT expectation over every played/blanked
    realization of the 15 squad members (2^15 combinations), each
    independently using their own `avail` as the probability of playing
    that week — not a Monte Carlo sample, so there's no sampling noise.
    `pts_if_play` divides each player's own xw{w} back out by their avail
    to get "points GIVEN they play" (xw{w} = avail * mins_share * ... —
    see fpl_stage0.build_table), since a realization already fixes
    whether they played or not.

    Two known simplifications, both affecting only the rarer MULTI-blank
    case — a single blank, which dominates the probability mass for
    realistic avail values, is handled exactly:
      - Outfield bench players are admitted in bench order as long as
        doing so doesn't exceed a position's MAXIMUM allowed count (5
        DEF / 5 MID / 3 FWD); the resulting minimums (>=3 DEF, >=2 MID,
        >=1 FWD) aren't separately re-checked. This is fine for the
        POINTS total specifically because which blanked starter a given
        sub is "credited to" doesn't change it — only how many bench
        players get admitted does, and that's exactly what this checks.
      - No vice-captain: if the captain blanks, no one else is doubled
        (this project doesn't track a vice-captain) — a small,
        conservative simplification.
    """
    info = df.set_index("id")
    gk_starter = next(p for p in xi if info.loc[p, "pos"] == "GKP")
    outfield_starters = [p for p in xi if p != gk_starter]
    gk_bench = next((p for p in bench if info.loc[p, "pos"] == "GKP"), None)
    outfield_bench = [p for p in bench if p != gk_bench]

    squad15 = list(xi) + list(bench)
    avail = {p: float(info.loc[p, "avail"]) for p in squad15}
    pts_if_play = {
        p: (info.loc[p, xpts_col] / avail[p]) if avail[p] > 1e-6 else 0.0
        for p in squad15
    }
    pos = {p: info.loc[p, "pos"] for p in squad15}

    MAX_DEF, MAX_MID, MAX_FWD = 5, 5, 3

    def counts(ids):
        n_def = n_mid = n_fwd = 0
        for p in ids:
            if pos[p] == "DEF":
                n_def += 1
            elif pos[p] == "MID":
                n_mid += 1
            elif pos[p] == "FWD":
                n_fwd += 1
        return n_def, n_mid, n_fwd

    total = 0.0
    for bits in itertools.product((0, 1), repeat=len(squad15)):
        played = dict(zip(squad15, bits))
        prob = 1.0
        for p, b in played.items():
            prob *= avail[p] if b else (1 - avail[p])
        if prob <= 1e-12:
            continue

        realized = 0.0

        # GK: exact 1-for-1, always legal.
        if played[gk_starter]:
            realized += pts_if_play[gk_starter]
        elif gk_bench and played[gk_bench]:
            realized += pts_if_play[gk_bench]

        # Outfield: survivors + greedily admitted bench, capped by max counts.
        final_outfield = [p for p in outfield_starters if played[p]]
        n_open = sum(1 for p in outfield_starters if not played[p])
        admitted = 0
        for b in outfield_bench:
            if admitted >= n_open:
                break
            if not played[b]:
                continue
            trial = final_outfield + [b]
            n_def, n_mid, n_fwd = counts(trial)
            if n_def <= MAX_DEF and n_mid <= MAX_MID and n_fwd <= MAX_FWD:
                final_outfield = trial
                admitted += 1
        for p in final_outfield:
            realized += pts_if_play[p]

        if played[captain]:
            realized += pts_if_play[captain]   # captain's double

        total += prob * realized

    return total


def print_lookahead(df, gw, squad, start, cap):
    """
    Stage A solves a full per-week plan internally just to VALUE the week-0
    decision — this prints that plan for weeks 1..horizon-1 (week 0 is
    Stage B's job, reported properly in report() below; this is context for
    WHY the transfer decision above looks the way it does, not a second
    opinion on it — always defer to Stage B for what to actually do this
    week). Weeks 1+ can now include the model's OWN planned transfers, not
    just lineup/captain choices against a fixed squad — printed here so you
    can see when it's planning to use the rest of your transfers, and why.
    """
    horizon = horizon_of(df)
    ids = df["id"].tolist()
    info = df.set_index("id")
    print("\nLOOK-AHEAD (why Stage A valued this squad the way it did — "
          "always re-run Stage B fresh each week for the real decision)")
    prev_chosen = [p for p in ids if squad[0][p].value() > 0.5]
    for w in range(1, horizon):
        chosen_w = [p for p in ids if squad[w][p].value() > 0.5]
        captain_w = next((p for p in chosen_w if cap[w][p].value() > 0.5), None)
        starters_w = [p for p in chosen_w if start[w][p].value() > 0.5]

        transferred_in = set(chosen_w) - set(prev_chosen)
        transferred_out = set(prev_chosen) - set(chosen_w)

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
        if transferred_in or transferred_out:
            in_names = ", ".join(info.loc[p, "name"] for p in transferred_in) or "—"
            out_names = ", ".join(info.loc[p, "name"] for p in transferred_out) or "—"
            line += f"   [planned transfer: IN {in_names} / OUT {out_names}]"
        if blanking:
            names = ", ".join(info.loc[p, "name"] for p in blanking)
            line += f"   [{names} started anyway despite no fixture — check formation legality]"
        print(line)
        prev_chosen = chosen_w


def report(df, current_ids, squad0, hits0, cost_t, budget_t, xi, captain):
    """squad0/hits0 are week 0's squad dict and hits variable — the only
    week that's actually real; see build_problem's docstring."""
    info = df.set_index("id")
    chosen = [p for p in info.index if squad0[p].value() > 0.5]
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
        n_hits = int(round(hits0.value()))
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

    bank_t = int(round(BANK * 10))
    if MANUAL_SQUAD:
        current_ids = MANUAL_SQUAD
    elif TEAM_ID:
        current_ids, live_bank_t = fetch_squad_and_bank(TEAM_ID, gw)
        # Real bank FPL already tracks for this team — reflects every past
        # transfer and every real price move automatically, so it's used
        # in place of the hand-set BANK config for a live team.
        if live_bank_t is not None:
            bank_t = live_bank_t
    else:
        raise SystemExit("Set TEAM_ID or MANUAL_SQUAD at the top of the file.")

    if len(current_ids) != 15:
        raise SystemExit(f"Expected 15 players, got {len(current_ids)}.")

    # Stage A: a full multi-week plan — which 15 to own EACH week and how
    # many transfers to spend when, weighing the full horizon at once
    # (df's own horizon_of(df) weeks — see OPTIMISE_HORIZON above).
    prob, squad, start, cap, hits, ft, tin, tout, cost_t, budget_t = build_problem(
        df, current_ids, bank_t
    )

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        raise SystemExit(f"Solver returned {status} — usually means your budget "
                         "or transfer cap makes a legal squad impossible.")

    chosen = [p for p in df["id"] if squad[0][p].value() > 0.5]

    # Stage B: which 11 to start and who captains, weighing ONLY next GW.
    xi, captain = choose_lineup(df, chosen)

    print(f"\nOptimising for GW{gw} onward. Squad objective: "
          f"{pulp.value(prob.objective):.2f} expected points over the horizon.")
    report(df, current_ids, squad[0], hits[0], cost_t, budget_t, xi, captain)
    print_lookahead(df, gw, squad, start, cap)


if __name__ == "__main__":
    main()
