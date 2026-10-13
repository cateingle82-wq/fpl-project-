"""
Tracks the OPTIMISER'S OWN real recommendations over time, and checks how
they actually performed once a gameweek finishes.

Different from backtest.py: that reconstructs a GENERIC top-11-by-
prediction check using full historical data for any past gameweek,
regardless of whether this tool existed then or what it specifically told
you to do. This is the real "did it do badly" answer — what squad/XI/
captain did YOU actually get told to run with, and what did THAT exact
team really score once the games were played.

Two halves:
  log_recommendation() — called once per "Run optimiser" click (see
  app.py), appends that week's real recommendation (XI, captain, bench,
  predicted score) to recommendation_log.csv. Never overwrite — one row
  per run, so running the optimiser again mid-week before the deadline
  just adds another reading for the same gameweek, same philosophy as
  chip_log.csv.

  track_record() — reads that log, and for every row whose gameweek has
  since finished, fetches each player's REAL points for that gameweek and
  resolves real autosubs (same rule fpl_optimise.simulate_autosub_
  expected_points uses, but deterministic — real minutes, not
  probabilities, since we now know exactly what happened) to compute
  what the recommended team ACTUALLY scored. A row for a gameweek still
  in progress or not yet played is left with actual_score=None, not a
  guess.

A real limit, honestly reflected rather than worked around: if this
gameweek's recommendation was never logged (e.g. this feature didn't
exist yet, or the optimiser was never run that week), there is no way to
retroactively know what WOULD have been recommended — backtest.py's
generic check is the closest available substitute for a week like that,
not this.

Run:  python track_record.py   (prints the current track record)
"""

import csv
import os
from datetime import date

import pandas as pd

import fpl_stage0 as fs0

LOG_PATH = "recommendation_log.csv"


def log_recommendation(gw, xi_ids, captain_id, bench_ids, predicted_score, path=None):
    """Append one row for this run. `predicted_score` should be the
    autosub-adjusted prediction (fpl_optimise.simulate_autosub_expected_
    points), the same number the dashboard's "Expected points per
    gameweek" chart shows for week 0 — comparing against anything else
    would be comparing different units."""
    path = path or LOG_PATH
    is_new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["date", "gw", "xi_ids", "captain_id", "bench_ids", "predicted_score"])
        w.writerow([
            date.today().isoformat(), gw,
            ";".join(str(p) for p in xi_ids), captain_id,
            ";".join(str(p) for p in bench_ids), f"{predicted_score:.2f}",
        ])


def _resolve_real_autosubs(xi_ids, bench_ids, captain_id, pos_of, real_points, real_minutes):
    """Deterministic twin of fpl_optimise.simulate_autosub_expected_
    points's autosub rule (same GK-for-GK 1-for-1, same greedy admit-by-
    bench-order for outfield) — but resolved against REAL per-player
    minutes (who actually blanked that gameweek) instead of a probability
    distribution over who MIGHT blank, since the result is already known."""
    gk_starter = next((p for p in xi_ids if pos_of.get(p) == "GKP"), None)
    outfield_starters = [p for p in xi_ids if p != gk_starter]
    gk_bench = next((p for p in bench_ids if pos_of.get(p) == "GKP"), None)
    outfield_bench = [p for p in bench_ids if p != gk_bench]

    def played(p):
        return real_minutes.get(p, 0) > 0

    score = 0.0
    if gk_starter is not None and played(gk_starter):
        score += real_points.get(gk_starter, 0)
    elif gk_bench is not None and played(gk_bench):
        score += real_points.get(gk_bench, 0)

    final_outfield = [p for p in outfield_starters if played(p)]
    n_open = sum(1 for p in outfield_starters if not played(p))
    admitted = 0
    for b in outfield_bench:
        if admitted >= n_open:
            break
        if played(b):
            final_outfield.append(b)
            admitted += 1
    score += sum(real_points.get(p, 0) for p in final_outfield)

    if played(captain_id):
        score += real_points.get(captain_id, 0)   # captain's real double
    return score


def track_record(path=None):
    """Returns a DataFrame of every logged run, with an actual_score
    column filled in for any gameweek that's since finished (NaN while
    still in progress / not yet played — never a guess)."""
    path = path or LOG_PATH
    if not os.path.exists(path):
        return pd.DataFrame()
    log = pd.read_csv(path)
    if log.empty:
        log["actual_score"] = []
        return log

    boot = fs0.get("bootstrap-static/")
    finished_gws = {e["id"] for e in boot["events"] if e["finished"]}
    positions = {p["id"]: p["singular_name_short"] for p in boot["element_types"]}
    pos_of = {e["id"]: positions[e["element_type"]] for e in boot["elements"]}

    actual_scores = []
    history_cache = {}
    for _, row in log.iterrows():
        gw = int(row["gw"])
        if gw not in finished_gws:
            actual_scores.append(float("nan"))
            continue

        xi_ids = [int(x) for x in str(row["xi_ids"]).split(";") if x]
        bench_ids = [int(x) for x in str(row["bench_ids"]).split(";") if x]
        captain_id = int(row["captain_id"])

        real_points, real_minutes = {}, {}
        for pid in xi_ids + bench_ids:
            if pid not in history_cache:
                hist = fs0.get(f"element-summary/{pid}/").get("history", [])
                history_cache[pid] = {h["round"]: h for h in hist}
            h = history_cache[pid].get(gw)
            real_points[pid] = h["total_points"] if h else 0
            real_minutes[pid] = h["minutes"] if h else 0

        actual_scores.append(
            _resolve_real_autosubs(xi_ids, bench_ids, captain_id, pos_of, real_points, real_minutes)
        )

    log["actual_score"] = actual_scores
    return log


def main():
    rec = track_record()
    if rec.empty:
        print("No logged recommendations yet — run the optimiser at least once "
              "(each 'Run optimiser' click in the dashboard logs one).")
        return
    print(rec[["date", "gw", "predicted_score", "actual_score"]].to_string(index=False))

    done = rec.dropna(subset=["actual_score"])
    if len(done):
        avg_pred = done["predicted_score"].mean()
        avg_actual = done["actual_score"].mean()
        sign = "+" if avg_actual >= avg_pred else ""
        print(f"\nAcross {len(done)} finished gameweek(s) actually tracked: "
              f"predicted avg {avg_pred:.1f}, actual avg {avg_actual:.1f} "
              f"({sign}{avg_actual - avg_pred:.1f})")
    still_open = len(rec) - len(done)
    if still_open:
        print(f"{still_open} logged run(s) still awaiting a finished gameweek.")


if __name__ == "__main__":
    main()
