"""
FPL Stage 0 — naive expected-points ranker.

No ML, no optimiser yet. Just:
  1. pull the public FPL API
  2. build one row per player
  3. score each player with a crude xPts heuristic over the next N gameweeks
  4. print the best value picks per position

This is the baseline you must beat later. Don't skip it.

Run:  pip install requests pandas
      python fpl_stage0.py
"""

import requests
import numpy as np
import pandas as pd

BASE = "https://fantasy.premierleague.com/api"
HORIZON = 5          # how many gameweeks ahead to look
HEADERS = {"User-Agent": "Mozilla/5.0"}   # FPL rejects some default agents

# Shrinkage strength for points_per_game. Early season, a player's own ppg is
# built from very few games and one big haul dominates it (see fpl_snapshot
# from GW6: Groß and Tarkowski, both on 450/450 minutes, sat above Haaland on
# raw ppg purely off one huge return). SHRINKAGE_K is how many "pseudo-games"
# of the price-based prior we weigh a real game of evidence against. Bigger
# K = slower to trust a hot start. K=4 means a player who's played 4 full
# matches is already trusted as much as the prior; by ~10-12 games the prior
# barely matters and ppg is basically the player's own number.
SHRINKAGE_K = 4.0

# FDR 1 (easiest) .. 5 (hardest) -> multiplier on expected points.
# These numbers are guesses. Tuning them against real data is Stage 2's job.
FDR_MULTIPLIER = {1: 1.25, 2: 1.12, 3: 1.00, 4: 0.88, 5: 0.75}

# status codes in the API: a=available, d=doubtful, i=injured,
# s=suspended, u=unavailable, n=not in squad
HARD_OUT = {"i", "s", "u", "n"}

# For players with very little data this season (new signings, promoted-
# team players, the first couple of gameweeks), use the trained ML model
# (ml_predict.py) instead of the shrinkage heuristic above — see
# ml_predict.py's docstring for why only THERE: across a full season the
# ML model roughly ties a dead-simple "last 3 games" average, but at the
# cold start it has a real, repeatable edge because it can lean on price/
# position/opponent strength when there's no rolling history to shrink
# toward yet. Sits behind a flag, and any failure (no trained model yet,
# a network hiccup fetching this-season histories) falls back to the pure
# heuristic silently degrading, not crashing the whole pipeline — nothing
# about the tested heuristic path below changes when this is off or fails.
USE_ML_COLD_START = True


def get(endpoint):
    r = requests.get(f"{BASE}/{endpoint}", headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.json()


def next_gameweek(events):
    """The gameweek whose deadline hasn't passed yet."""
    for e in events:
        if e.get("is_next"):
            return e["id"]
    # preseason or end of season: fall back to the first unfinished GW
    unfinished = [e["id"] for e in events if not e["finished"]]
    return unfinished[0] if unfinished else events[-1]["id"]


def fixture_difficulties(fixtures, start_gw, horizon):
    """
    team_id -> list of FDR values for its fixtures in [start_gw, start_gw+horizon).

    A team with two fixtures in one gameweek (a double) gets two entries, which
    correctly inflates its players' xPts. A team with none (a blank) gets an
    empty list and scores zero for that week. This falls out for free.
    """
    window = range(start_gw, start_gw + horizon)
    out = {}
    for f in fixtures:
        if f["event"] is None or f["event"] not in window or f["finished"]:
            continue
        out.setdefault(f["team_h"], []).append(f["team_h_difficulty"])
        out.setdefault(f["team_a"], []).append(f["team_a_difficulty"])
    return out


def availability(row):
    """0.0 to 1.0. How likely is this player to be on the pitch at all."""
    if row["status"] in HARD_OUT:
        return 0.0
    chance = row["chance_of_playing_next_round"]
    # The API sends null for "no injury news" (i.e. fully fit), but once this
    # column sits in a pandas DataFrame alongside real percentages, pandas
    # silently upgrades that null to a float NaN. `chance is None` never
    # matches a NaN, so use pd.isna() or every fit player's avail becomes NaN
    # and poisons xpts for the whole table.
    if pd.isna(chance):
        return 1.0
    return chance / 100.0


def shrink_ppg(df, k=SHRINKAGE_K):
    """
    Blend each player's own points_per_game with a prior — the ppg a typical
    player at that price and position tends to return — weighted by how much
    evidence (minutes played) we actually have on them.

    The prior itself is fit from the data you already have: a weighted
    least-squares line of ppg against price, one line per position, weighting
    each player by minutes played so a fluke haul from a 1-game player can't
    tilt the line the prior is drawn from either. Price is a decent proxy for
    the market's long-run view of a player's quality (it's set by past
    seasons, not this one), which is exactly the outside information a small
    early-season sample needs.

    shrunk = weight * own_ppg + (1 - weight) * prior
    weight = evidence / (evidence + k),  evidence = minutes / 90
    """
    evidence = (df["minutes"] / 90.0).clip(lower=0)
    shrunk = df["ppg"].copy()
    prior_col = df["ppg"].copy()

    for pos in df["pos"].unique():
        mask = df["pos"] == pos
        sub = df.loc[mask]
        w = (evidence.loc[mask] + 0.01).to_numpy()   # tiny floor: never a zero-weight point

        if mask.sum() < 3 or w.sum() < 1e-6:
            # too few players at this position to fit a line — fall back to
            # a flat prior (the group's own weighted-average ppg)
            prior = np.average(sub["ppg"], weights=w) if w.sum() > 1e-6 else sub["ppg"].mean()
            prior_arr = np.full(mask.sum(), prior)
        else:
            slope, intercept = np.polyfit(sub["price"].to_numpy(), sub["ppg"].to_numpy(), deg=1, w=w)
            prior_arr = np.clip(intercept + slope * sub["price"].to_numpy(), a_min=0, a_max=None)

        prior_col.loc[mask] = prior_arr

    weight = evidence / (evidence + k)
    shrunk = weight * df["ppg"] + (1 - weight) * prior_col
    return shrunk, prior_col


def build_table(horizon=HORIZON):
    """
    horizon: how many gameweeks ahead to plan over. Defaults to the module
    constant, but callers (the app's sidebar, in particular) can pass a
    different value — this is the actual knob for "how many weeks should
    squad selection weigh rotation options over", not BENCH_WEIGHT. A
    longer horizon doesn't just add more fixture data: build_problem's
    per-week start[w][p] decisions mean a squad gets rewarded for having
    players who can rotate in across DIFFERENT weeks' best XIs, and that
    reward only exists for weeks actually in the horizon. horizon=1 has
    zero rotation value by construction (there's only one week to be
    optimal for); horizon=8 rewards genuine squad depth much more than
    horizon=3 does. BENCH_WEIGHT is a separate, smaller effect — a flat
    proxy for real-life bench/autosub insurance — not the same knob.
    """
    boot = get("bootstrap-static/")
    fixtures = get("fixtures/")

    gw = next_gameweek(boot["events"])
    gws_played = max(gw - 1, 1)          # avoid divide-by-zero in GW1

    # A separate fixture lookup for EACH week in the horizon, not one lump
    # window. This is what lets the optimiser plan around a specific blank or
    # double instead of averaging it into a single number and losing exactly
    # the information that made it worth planning around.
    weekly_fdr = [fixture_difficulties(fixtures, gw + w, 1) for w in range(horizon)]

    teams = {t["id"]: t["short_name"] for t in boot["teams"]}
    positions = {p["id"]: p["singular_name_short"] for p in boot["element_types"]}

    df = pd.DataFrame(boot["elements"])

    df["name"] = df["web_name"]
    df["team_name"] = df["team"].map(teams)
    df["pos"] = df["element_type"].map(positions)
    df["price"] = df["now_cost"] / 10.0                    # API stores tenths
    df["ppg"] = pd.to_numeric(df["points_per_game"], errors="coerce").fillna(0)
    df["form"] = pd.to_numeric(df["form"], errors="coerce").fillna(0)

    df["ppg_shrunk"], df["ppg_prior"] = shrink_ppg(df)

    # Share of available minutes the player has actually been on the pitch for.
    # Crude proxy for "will he start". Capped at 1.
    df["mins_share"] = (df["minutes"] / (90 * gws_played)).clip(upper=1.0)

    df["avail"] = df.apply(availability, axis=1)

    # One xw{w} column per week in the horizon — xw0 is next gameweek, xw1
    # the one after, etc. A blank week gives that player xw{w}=0 for that
    # week specifically (not the whole horizon); a double gives it roughly
    # double. ppg_shrunk/mins_share/avail are treated as constant across the
    # horizon (we don't have a way to predict THEIR future changes) — only
    # the fixture term varies week to week, which is exactly the part that's
    # actually known in advance.
    fdr_scores = []
    for w in range(horizon):
        s = df["team"].map(
            lambda t, w=w: sum(FDR_MULTIPLIER.get(d, 1.0) for d in weekly_fdr[w].get(t, []))
        ).fillna(0)
        fdr_scores.append(s)
        df[f"xw{w}"] = df["ppg_shrunk"] * df["mins_share"] * df["avail"] * s

    # Everyone starts out attributed to the heuristic; cold-start players
    # get their xw{w} columns OVERWRITTEN below if ML blending succeeds.
    # Kept as a column (not just a log message) so report()/app.py can show
    # which number a player's score actually came from.
    df["xpts_source"] = "heuristic"

    if USE_ML_COLD_START:
        try:
            # Lazy import: ml_predict.py imports fpl_stage0 (for `get` and
            # HORIZON), so importing it at module level up here would be a
            # circular import. Deferring it to inside the function, after
            # fpl_stage0 itself has finished loading, breaks the cycle.
            import ml_predict

            element_ids = df["id"].tolist()
            histories = ml_predict.fetch_current_histories(element_ids)
            ml_preds = ml_predict.predict_cold_start(boot, fixtures, histories, gw, horizon)

            if not ml_preds.empty:
                df = df.set_index("id", drop=False)
                for w in range(horizon):
                    col = f"xw{w}"
                    df.loc[ml_preds.index, col] = ml_preds[col]
                df.loc[ml_preds.index, "xpts_source"] = "ml"
                df = df.reset_index(drop=True)
        except Exception as e:
            # Anything goes wrong (no trained model yet, a network issue,
            # a schema surprise) -> stay on the pure heuristic. This path
            # must never take down the whole optimiser over an optional
            # enhancement.
            print(f"[fpl_stage0] ML cold-start blending skipped: {e}")

    # 'xpts' (horizon total) and 'xpts_gw1' (next week only) are now derived
    # sums/aliases of the per-week columns above, not separately computed —
    # one source of truth, so they can't drift out of sync with each other.
    df["fixture_score"] = sum(fdr_scores)          # kept for the report() table
    df["fixture_score_gw1"] = fdr_scores[0]
    df["xpts"] = sum(df[f"xw{w}"] for w in range(horizon))
    df["xpts_gw1"] = df["xw0"]
    df["xpts_per_m"] = df["xpts"] / df["price"]            # value, not just points

    # Safety net: NaN/inf here means PuLP will crash three files downstream
    # with an unhelpful error. Catch it here, at the source, with a message
    # that says which players and which column actually broke.
    check_cols = ["xpts", "xpts_gw1"] + [f"xw{w}" for w in range(horizon)]
    bad_mask = pd.Series(False, index=df.index)
    for c in check_cols:
        bad_mask |= df[c].isna() | ~df[c].apply(lambda v: v == v and abs(v) != float("inf"))
    bad = df[bad_mask]
    if len(bad):
        cols = ["id", "name", "ppg", "mins_share", "avail", "fixture_score", "fixture_score_gw1"]
        raise ValueError(
            f"{len(bad)} players have NaN/inf in {check_cols} — fix the source "
            f"column before running the optimiser:\n{bad[cols].to_string(index=False)}"
        )

    return df, gw


def horizon_of(df):
    """
    How many per-week xw{w} columns this table actually has — the single
    source of truth downstream code (build_problem, chips.py) uses to know
    how many weeks to plan over, instead of assuming the module's HORIZON
    constant. Inferred from the DataFrame itself so a table built with a
    custom horizon (e.g. the app's slider) can never silently mismatch
    code that assumed the default.
    """
    weeks = [int(c[2:]) for c in df.columns if c.startswith("xw") and c[2:].isdigit()]
    if not weeks:
        raise ValueError("df has no xw{w} columns — was it built by build_table()?")
    return max(weeks) + 1


def report(df, gw):
    print(f"\nNext gameweek: GW{gw}   Horizon: {horizon_of(df)} GWs\n")

    cols = ["name", "team_name", "price", "ppg", "ppg_shrunk", "mins_share",
            "avail", "fixture_score", "xpts", "xpts_gw1", "xpts_per_m", "xpts_source"]

    for pos in ["GKP", "DEF", "MID", "FWD"]:
        sub = df[(df["pos"] == pos) & (df["xpts"] > 0)]
        top = sub.nlargest(10, "xpts")[cols]
        print(f"=== {pos} — highest expected points ===")
        print(top.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        print()

    print("=== Best value overall (min 2 expected points) ===")
    value = df[df["xpts"] > 2].nlargest(15, "xpts_per_m")[cols]
    print(value.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    table, gw = build_table()
    report(table, gw)
    table.to_csv("fpl_snapshot.csv", index=False)
    print("\nSaved fpl_snapshot.csv — start keeping these weekly, you'll need them.")