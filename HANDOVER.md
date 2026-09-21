# Session handover — 2026-09-21

Two Claude Code sessions worked on this repo in parallel today: this one
(dashboard / optimiser / chip strategy) and a second session, `cateingle-f9`
(ML feature pipeline). This doc is the join point — read it before picking
the project back up, in a new session or otherwise.

## Where things stand

All work from both sessions is committed and pushed to
`github.com/cateingle82-wq/fpl-project-`, `main`, up to `b0ba724`. Local
working tree is clean. `pytest -q` passes (see **Running things** below).

## What this session did (chronological)

1. **Estimated and explained the optimiser's runtime** on the Streamlit
   dashboard (~2-2.5s per solve at the default horizon, one-time ~3-4s data
   fetch per session).
2. **Heuristic scoring upgrades** in `fpl_stage0.py`:
   - Opponent attack/defence strength blended into every player's fixture
     score (previously only reached ML cold-start players).
   - Set-piece duty bonus (penalties/free-kicks/corners order fields).
   - Recency-weighted minutes (blends season-long `mins_share` with the
     last 4 games) for better rotation-risk detection.
3. **Live bank auto-fetch** (`fpl_optimise.fetch_squad_and_bank`) — pulls
   the real FPL account bank balance instead of a hand-typed sidebar value,
   so it self-updates every week.
4. **Autosub-aware expected points** (`fpl_optimise.simulate_autosub_expected_points`)
   — exact expectation over all 2^15 played/blanked realizations of a
   squad, so a blanked starter is correctly credited to the right bench
   cover, same as FPL's own scoring. Wired into the "Expected points per
   gameweek" chart.
5. **Dashboard restructure** (`app.py`) — merged the three tabs
   (Transfers/Chips/Backtest) into one page; one "Run optimiser" click now
   drives transfers, the weekly plan, AND chip values, since they answer
   the same "what do I do this week" question. Backtest moved to a
   collapsed sidebar expander (model-health diagnostic, not a weekly
   decision).
6. **Fixed a real chart bug**: `st.bar_chart` sorts gameweek labels
   alphabetically (GW10 before GW6) — replaced with explicit Altair charts
   pinning chronological order.
7. **Per-week chip logging**: `chips.log_row` now writes one row per
   gameweek in the horizon to `chip_log.csv` (previously only week 0),
   enabling a real week-by-week trend instead of a single collapsed
   number.
8. **Chip timing score (0-10)** (`chips.chip_scores`):
   - Bench Boost / Triple Captain scored against the *visible horizon*
     (free — the data's already computed).
   - Wildcard / Free Hit scored against *your own logged history* in
     `chip_log.csv` (needs 3+ readings; returns "not enough history yet"
     below that rather than a fake number).
   - Deliberately does **not** try to infer double/blank gameweeks from
     other seasons' fixture patterns — checked and confirmed unreliable
     (cup replay rules, European competition scheduling, and international
     breaks have all changed structurally season to season).
9. **Full-season fixture scan** (`fpl_stage0.season_fixture_counts`,
   `chips.season_outlook`) — looks at the real, currently-confirmed
   fixture list across all 38 gameweeks (not just the horizon slider) to
   flag a gameweek where the squad's teams have notably more fixtures than
   anything currently visible. Verified live: right now (GW6) the full
   season is still a clean round-robin with zero confirmed doubles/blanks
   anywhere — correctly finds nothing to flag; will start surfacing real
   signals the moment FPL confirms a rearrangement, no code changes
   needed then.
10. **Surfaced a real reporting gap**: future weeks in the multi-week plan
    showed transfers with no indication some cost points. Fixed — every
    week's tab now shows `⚠️ N transfer(s) this week, only M free — H
    hit(s) = -X pts` when relevant. (Root cause was a missing UI signal,
    not a solver bug — verified the hit accounting was already correct.)
11. **Added "Average per gameweek"** next to the horizon objective — the
    raw total always grows with a longer horizon, so it isn't a fair way
    to compare two runs at different horizon lengths; this is.
12. **Wildcard-vs-plan comparison** (`chips.wildcard_detail`) — the
    transfer optimiser and the Wildcard chip evaluation were completely
    disconnected; the optimiser could recommend an expensive multi-hit
    rebuild without ever checking whether the same or better outcome was
    free via Wildcard. Now compares the plan's real objective against a
    full free rebuild's objective directly, and surfaces it — but only
    past a real threshold (2x hit cost), since Wildcard is structurally
    unconstrained (no per-week cap, no hit cost) and will beat a capped
    plan by *some* margin almost every week regardless of whether using
    it now is smart. Points at the existing Wildcard 0-10 score (history-
    based) as the more reliable timing signal, not a standalone alarm.

## What the other session (`cateingle-f9`) did

ML feature pipeline work, commits `436eaaf`, `505628a`, `4908fce`:

- Added `rest_days` (days since a player's previous match, from real
  kickoff timestamps) as an ML feature — a genuine fixture-congestion /
  rotation-risk signal a fixed weekly gap can't see. Wired through
  `fpl_history.py` (training), `ml_predict.py` (live cold-start
  prediction), `ml_features.py` (`FEATURE_COLS`).
- Retrained and refreshed `ml_holdout_results.csv` with the new feature.
- A cosmetic fix to test print numbering in `test_ml_features.py`.

## The cross-session collision (so it doesn't repeat)

Partway through this session I found substantial, coherent uncommitted
work already on disk (`app.py` restructure, `rest_days` feature) and,
not knowing another session was active, assumed it was my own earlier
work resurfacing after context compaction — and committed + pushed it.
Turned out to be the other session's in-progress work. No data was lost
(the other session diffed it against what I'd pushed and confirmed it
matched exactly), but it was a real near-miss.

**Lesson for next time**: before assuming unexplained uncommitted changes
are your own prior work, check `ListAgents` for active peer sessions —
especially in a repo more than one session might touch. The two sessions
agreed a file split once this was caught: this session stayed on
`app.py`/`chips.py`/`fpl_stage0.py`-adjacent dashboard code; the other
stayed on `ml_*.py`/`fpl_history.py`.

## File map (what owns what)

| File | Owns |
|---|---|
| `fpl_stage0.py` | Heuristic xPts model, fixture scanning, `build_table()` |
| `fpl_optimise.py` | Stage A/B MILP (squad+transfer planning, lineup, autosub sim) |
| `chips.py` | Chip valuation (bench boost/triple captain/wildcard/free hit), timing scores, season outlook, logging |
| `app.py` | Streamlit dashboard — single page, sidebar config + model-health expander |
| `ml_features.py` / `ml_train.py` / `ml_predict.py` / `fpl_history.py` | ML cold-start-only prediction path (see `ml_predict.py`'s docstring for why it's scoped that way) |
| `backtest.py` | Live-season prediction-quality validation |
| `chip_log.csv` | Per-gameweek chip value log (tracked in git — small, meant to accumulate) |
| `ml_holdout_results.csv` | ML holdout comparison vs `roll3` baseline (tracked in git) |

Not tracked in git (regenerate locally, see `.gitignore` for why):
`ml_model.joblib`, `feature_cache.joblib`, `history_cache/`,
`player_history_cache.json`, `fpl_snapshot.csv`.

## Known issues / things to watch

- **`test_optimise.py` run standalone (`python test_optimise.py`) hangs**
  in this environment — reproduced independently of disk space. Not
  pytest-collected (no `test_*` functions, only a `run_all_checks()`
  called from `__main__`), so it doesn't affect the CI-relevant test
  suite, but worth debugging before relying on it again. Root cause not
  found this session.
- **Solve time scales badly with horizon**: ~2s at horizon 5, ~35s at
  horizon 6 on the real squad (not gradual — a step change). Worth
  profiling if anyone wants to push the horizon slider higher; likely the
  free-transfer-banking linearization or the per-week club-limit
  constraints blowing up the branch-and-bound tree.
- **"Average per gameweek" (~75-83) looks high vs real backtested
  outcomes.** Checked against `backtest_results.csv`'s `top11_shrunk`
  (real points scored by the model's top-11 picks in the 3 gameweeks
  backtested so far, GW2-4 — actual outcomes, not predictions): averages
  ~5.3 pts/player. Back-of-envelope real total: 11 × 5.3 + a captain
  bonus ≈ 66-68/week, noticeably below what the app currently shows as
  "expected." Likely cause: `ppg`/`ppg_shrunk` this early (GW6) is built
  from only ~5 real games per player, a sample that hasn't yet included
  the bad patches, dry spells, rotation and minor injuries every player
  has over a 38-game season — systematically optimistic not because the
  model is broken, but because nothing bad has happened YET. Some
  elevation above a typical manager's average is genuinely expected too
  (this is the model-optimal XI, not an average manager's), just maybe
  not this much. Caveat: only 3 backtested gameweeks exist — too few to
  call this a confirmed bias rather than noise, by the project's own
  standard elsewhere ("don't tune on fewer than a handful of GWs").

  **Suggested next step**: add a direct "predicted vs actual team total"
  check to `backtest.py` — compare what `xpts_gw1` predicted for a whole
  XI+captain going into a gameweek against what that exact combination
  really scored, in the same units the "Average per gameweek" metric
  uses (a team total, not `backtest.py`'s current per-player top-11
  metric). That gives a real, growing calibration record instead of a
  one-off gut check, and would let the dashboard eventually show "your
  expected total has historically run N% high/low" next to the number
  itself.

- **`ml_holdout_results.csv` correlation still doesn't beat the `roll3`
  baseline** (0.707-0.708 vs 0.712) even after `rest_days`/xG-xA
  features — permutation importance shows minutes-related features
  dominate by a wide margin. The next real lever for the ML path is a
  better minutes/rotation-risk signal, not more scoring features.

## Running things

```bash
# Tests (fast, ~7s, 40 tests)
python -m pytest -q

# Dashboard
streamlit run app.py
# -> http://localhost:8501

# Standalone CLI scripts (each independently runnable)
python fpl_stage0.py     # heuristic xPts table -> fpl_snapshot.csv
python fpl_optimise.py   # Stage A/B squad+transfer recommendation
python chips.py          # chip values, scores, logs to chip_log.csv
python backtest.py       # live-season prediction-quality check
python ml_train.py       # retrain the ML model (only affects cold-start players)
```

Note: this machine's `python3` on `PATH` sometimes resolves to a bare
system interpreter without the project's dependencies installed — if you
hit `ModuleNotFoundError`, try `/opt/anaconda3/bin/python3` explicitly.
