"""
FPL Backtest — measures PREDICTION quality, not full squad simulation.

The question this answers: does xpts_gw1 actually correlate with what
players go on to score? Everything downstream (transfers, captaincy) is
only as good as this number. Before trusting the optimiser's picks, check
the picks it's fed are worth trusting.

Two things this deliberately does NOT do — both bigger, separate builds:
  - It does not simulate a SQUAD over time (no transfers, no bank, no price
    changes week to week). That's the "full optimiser loop" backtest — a
    much heavier project, only worth building once THIS one shows the
    underlying prediction is sound. No point stress-testing squad logic on
    top of numbers you haven't verified yet.
  - It can't recreate injury news as it stood on each historical deadline.
    `chance_of_playing_next_round` only ever reflects "now" — there's no API
    field for "what did we know on gameweek 7's deadline". Every backtested
    gameweek therefore silently assumes everyone was known-fit, which real
    gameweeks weren't. This makes every result here a slight OVERestimate of
    real-world performance — a ceiling, not a guarantee. Keep that in mind
    reading the numbers, don't try to "fix" it (you can't, the data doesn't
    exist), and don't be surprised if real-time performance runs behind this.

Run:  python backtest.py
(First run is slow — ~700 API calls, one per player, cached afterwards.)
"""

import json
import os
import time

import numpy as np
import pandas as pd

from fpl_stage0 import FDR_MULTIPLIER, SHRINKAGE_K, get, shrink_ppg

CACHE_FILE = "player_history_cache.json"
MIN_GW = 2     # GW1 is impossible (no history exists yet). GW2 is thin (one
               # game of evidence) but included anyway — early season, every
               # extra data point matters more than the noise it adds.


def fetch_all_histories(element_ids, cache_path=CACHE_FILE, pause=0.05):
    """
    Pull element-summary/{id}/ for every player once and cache to disk.
    ~700 API calls — run this alongside your weekly snapshot, not every
    time you tweak the model. Re-running only fetches players missing from
    the cache, so it's cheap after the first pull.
    """
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)

    missing = [str(i) for i in element_ids if str(i) not in cache]
    if missing:
        print(f"{len(cache)} cached, fetching {len(missing)} more...")
        for i, pid in enumerate(missing):
            data = get(f"element-summary/{pid}/")
            cache[pid] = data.get("history", [])
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(missing)}")
                with open(cache_path, "w") as f:
                    json.dump(cache, f)
            time.sleep(pause)
        with open(cache_path, "w") as f:
            json.dump(cache, f)
    else:
        print(f"{len(cache)} players cached, nothing new to fetch.")

    return cache


def snapshot_as_of(boot, fixtures, histories, target_gw, min_players=20):
    """
    Rebuild a Stage-0-style feature table using ONLY information that would
    have existed before target_gw's deadline: history rows with
    round < target_gw. Deliberately mirrors fpl_stage0.build_table's column
    names closely enough to reuse shrink_ppg() completely unchanged — the
    exact same shrinkage logic you already tested gets re-used here, not
    reimplemented.
    """
    positions = {p["id"]: p["singular_name_short"] for p in boot["element_types"]}

    rows = []
    for el in boot["elements"]:
        pid = el["id"]
        hist = histories.get(str(pid), [])
        past = [h for h in hist if h["round"] < target_gw]
        this_gw = next((h for h in hist if h["round"] == target_gw), None)

        if not past or this_gw is None:
            continue   # no prior history, or didn't feature that gameweek

        total_pts = sum(h["total_points"] for h in past)
        minutes = sum(h["minutes"] for h in past)
        games = len(past)
        price = past[-1]["value"] / 10.0   # price AS OF just before this gameweek

        rows.append({
            "id": pid,
            "pos": positions[el["element_type"]],
            "team": el["team"],
            "price": price,
            "ppg": total_pts / games,
            "minutes": minutes,
            "games_asof": games,
            "actual_points": this_gw["total_points"],
        })

    df = pd.DataFrame(rows)
    if df.empty or len(df) < min_players:
        return pd.DataFrame()

    df["ppg_shrunk"], _ = shrink_ppg(df, k=SHRINKAGE_K)

    gws_played = max(target_gw - 1, 1)
    df["mins_share"] = (df["minutes"] / (90 * gws_played)).clip(upper=1.0)

    fdr = {}
    for f in fixtures:
        if f["event"] != target_gw:
            continue
        fdr.setdefault(f["team_h"], []).append(f["team_h_difficulty"])
        fdr.setdefault(f["team_a"], []).append(f["team_a_difficulty"])
    df["fixture_score_gw1"] = df["team"].map(
        lambda t: sum(FDR_MULTIPLIER.get(d, 1.0) for d in fdr.get(t, []))
    ).fillna(0)

    # avail can't be reconstructed historically (see module docstring) —
    # every player is assumed fit. This is the "ceiling, not guarantee" caveat.
    df["xpts_gw1"] = df["ppg_shrunk"] * df["mins_share"] * df["fixture_score_gw1"]
    df["xpts_gw1_raw"] = df["ppg"] * df["mins_share"] * df["fixture_score_gw1"]   # no shrinkage, for comparison

    return df


def evaluate(df):
    """
    Two complementary views of prediction quality:
      - Spearman rank correlation: does the ORDERING match reality, across
        everyone? (robust to a few wild outlier scorelines)
      - top-11 average: if you'd picked your best XI purely by this number
        with no budget constraint, how many points would they actually
        have scored? Compared against a naive ppg-only baseline, a random
        XI, and the best XI that hindsight could have picked.

    Plus a THIRD view, calibration rather than ranking: predicted_team_total
    vs actual_team_total — take the model's own top-11-by-xpts_gw1 pick and
    its own top scorer as captain (double), and compare what it PREDICTED
    that combination would score against what it ACTUALLY scored. Same
    "starting XI + captain double" unit the dashboard's "Average per
    gameweek" metric uses, so this directly answers "does that number run
    high or low against what's actually happened" — ranking quality
    (correlation, top-11) can be good even while the absolute scale is
    off (e.g. an early-season ppg sample optimistic because it hasn't yet
    seen a player's bad patches), and this is the check that would catch
    that. Same no-budget/no-position-constraint simplification top11_shrunk
    already carries — it's an XI-shaped estimate, not a legal squad's.
    """
    corr_shrunk = df["xpts_gw1"].corr(df["actual_points"], method="spearman")
    corr_raw = df["xpts_gw1_raw"].corr(df["actual_points"], method="spearman")
    corr_naive_ppg = df["ppg"].corr(df["actual_points"], method="spearman")

    n = min(11, len(df))
    top11_df = df.nlargest(n, "xpts_gw1")
    top11_shrunk = top11_df["actual_points"].mean()
    top11_raw = df.nlargest(n, "xpts_gw1_raw")["actual_points"].mean()
    top11_ppg = df.nlargest(n, "ppg")["actual_points"].mean()
    random_11 = df["actual_points"].sample(n, random_state=0).mean()
    best_possible = df.nlargest(n, "actual_points")["actual_points"].mean()

    captain_row = top11_df.nlargest(1, "xpts_gw1").iloc[0]
    predicted_team_total = top11_df["xpts_gw1"].sum() + captain_row["xpts_gw1"]
    actual_team_total = top11_df["actual_points"].sum() + captain_row["actual_points"]

    return dict(
        n=len(df),
        corr_shrunk=corr_shrunk, corr_raw=corr_raw, corr_naive_ppg=corr_naive_ppg,
        top11_shrunk=top11_shrunk, top11_raw=top11_raw, top11_ppg=top11_ppg,
        random_11=random_11, best_possible=best_possible,
        predicted_team_total=predicted_team_total, actual_team_total=actual_team_total,
    )


def main():
    boot = get("bootstrap-static/")
    fixtures = get("fixtures/")
    element_ids = [e["id"] for e in boot["elements"]]

    histories = fetch_all_histories(element_ids)

    finished_gws = [e["id"] for e in boot["events"] if e["finished"]]
    if not finished_gws:
        raise SystemExit("No finished gameweeks yet this season — nothing to backtest.")
    test_gws = list(range(MIN_GW, max(finished_gws) + 1))

    results = []
    for gw in test_gws:
        df = snapshot_as_of(boot, fixtures, histories, gw)
        if df.empty:
            continue
        r = evaluate(df)
        r["gw"] = gw
        results.append(r)
        print(f"GW{gw:>2}  n={r['n']:<4} "
              f"corr shrunk={r['corr_shrunk']:.3f} raw={r['corr_raw']:.3f} ppg={r['corr_naive_ppg']:.3f}  "
              f"top11 shrunk={r['top11_shrunk']:.2f} raw={r['top11_raw']:.2f} "
              f"random={r['random_11']:.2f} best={r['best_possible']:.2f}  "
              f"team predicted={r['predicted_team_total']:.1f} actual={r['actual_team_total']:.1f}")

    if not results:
        raise SystemExit("No gameweeks had enough data to backtest.")

    res = pd.DataFrame(results)
    print("\n=== averages across all tested gameweeks ===")
    cols = ["corr_shrunk", "corr_raw", "corr_naive_ppg",
            "top11_shrunk", "top11_raw", "top11_ppg", "random_11", "best_possible",
            "predicted_team_total", "actual_team_total"]
    print(res[cols].mean().round(3).to_string())

    avg_pred = res["predicted_team_total"].mean()
    avg_actual = res["actual_team_total"].mean()
    bias_pct = 100 * (avg_pred - avg_actual) / avg_actual if avg_actual else float("nan")
    direction = "OVER" if bias_pct > 0 else "UNDER"
    print(f"\nCalibration: predicted team totals run {abs(bias_pct):.1f}% "
          f"{direction} actual, averaged across {len(res)} gameweek(s). "
          f"{'Too few gameweeks to call this a real bias vs noise — treat as provisional.' if len(res) < 5 else ''}")

    res.to_csv("backtest_results.csv", index=False)
    print("\nSaved backtest_results.csv")


if __name__ == "__main__":
    main()
