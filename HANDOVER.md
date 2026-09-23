# Session handover — 2026-09-21

Two Claude Code sessions worked on this repo in parallel today: this one
(dashboard / optimiser / chip strategy) and a second session, `cateingle-f9`
(ML feature pipeline). This doc is the join point — read it before picking
the project back up, in a new session or otherwise.

## Where things stand

All work from both sessions is committed and pushed to
`github.com/cateingle82-wq/fpl-project-`, `main`, up to `0da9ef5`. Local
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

- **`test_optimise.py` run standalone — RESOLVED, was never actually
  hanging.** It's a genuinely slow (not infinite) MILP: watched the real
  `cbc` subprocess directly (99% CPU, not stuck/deadlocked) and let it run
  to completion — all 13 checks passed in a few minutes, most of that on
  check 2's cold-start solve (building a legal 15-man squad from scratch,
  `MAX_TRANSFERS=15`, ~260 similarly-priced synthetic candidates — no
  owned squad to anchor the search, and lots of near-tied candidates,
  which is exactly what blows up a branch-and-bound tree). It only ever
  *looked* hung because it exceeds most tool runners' ~2 minute default
  timeout. Not pytest-collected (no `test_*` functions, only
  `run_all_checks()` from `__main__`), so this never affected the
  CI-relevant suite. No code changes needed — just don't run this one
  under a short timeout.
- **Solve time scales badly with horizon**: ~2s at horizon 5, ~35s at
  horizon 6 on the real squad (not gradual — a step change). Worth
  profiling if anyone wants to push the horizon slider higher; likely the
  free-transfer-banking linearization or the per-week club-limit
  constraints blowing up the branch-and-bound tree.
- **"Average per gameweek" ran high vs real backtested outcomes — RESOLVED.**
  Originally flagged as looking ~75-83 vs a ~66-68 real-world estimate.
  Built the suggested fix: `backtest.py`'s `evaluate()` now reports
  `predicted_team_total`/`actual_team_total` directly (same "XI + captain
  double" unit the dashboard uses), and `fpl_stage0.calibration_factor()`
  reads that track record back into `build_table()` as a single global
  correction applied to every `xw{w}` column. Requires 3+ backtested
  gameweeks before applying anything (same floor as elsewhere in this
  project); degrades to no correction on any missing data. Live factor is
  currently ×0.708 (confirms the original ~41% over-prediction), and
  "Average per gameweek" now shows ~55 instead of ~77 — close to the
  ~66-68 real-world estimate given the model-optimal squad should
  outscore that current-squad estimate somewhat, not match it exactly.
  Surfaced transparently in the UI with the applied factor and a pointer
  to Model Health for the gameweek count it's based on. Still worth
  re-checking as more gameweeks accumulate — the factor will drift as
  `backtest_results.csv` grows, and could over- or under-correct once
  there's more than a handful of weeks behind it.

- **`ml_holdout_results.csv` correlation still doesn't beat the `roll3`
  baseline** (0.707-0.708 vs 0.712) even after `rest_days`/xG-xA
  features — permutation importance shows minutes-related features
  dominate by a wide margin. The next real lever for the ML path is a
  better minutes/rotation-risk signal, not more scoring features.

## `cateingle-f9` session, round 2: dashboard UX (app.py)

After the ML retrain above, moved to `app.py` (with `cateingle-7c`'s
agreement — it was idle, notified before starting, file split otherwise
unchanged: `chips.py`/`fpl_stage0.py` still untouched by this session).
User explicitly asked for UX over more model/backtest work. Eight
changes, all pure rendering/translation of existing columns — no model
logic changed:

1. **"Why" reasoning** — a plain-language one-liner per transfer (fixture
   difficulty, minutes share, form vs season average, set-piece duty,
   fitness flags), built from columns `build_table()` already computes.
   Not a second opinion on the number, a translation of it.
2. **Risk tag column** (`risk_tag()`) added to every player table —
   reuses `fpl_stage0.SUB_PATTERN_GAP`'s own threshold for "impact sub"
   so this tag and that scoring discount always agree.
3. **What-if slider** — rescales a single player's own `xw0` by an
   assumed minutes share. Explicitly NOT a re-solve of the optimiser
   (caption says so) — a sensitivity check, not a new recommendation.
4. **Deadline countdown** — top of page, independent of the "Run
   optimiser" click (a clock shouldn't need a solve).
5. **Copy-paste export** (`build_export_text`) via `st.code()`.
6. **Diff vs last saved run** — persisted to `last_run_snapshot.json`
   (gitignored, per-machine state) so it survives across sessions, not
   just Streamlit reruns.
7. **Pitch view** (`render_pitch`) — GK/DEF/MID/FWD laid out as an actual
   formation via `st.columns` + a theme-neutral rgba HTML card, not a
   dataframe.
8. **Gamified backtest framing** — "beat a random XI in N of M
   gameweeks" alongside the existing raw Spearman/top11 numbers in Model
   Health.

**Architecture change required for #3/#7 to work at all**: the whole
render path moved from living inside `if st.button("Run optimiser"):` to
storing every result in `st.session_state["results"]` on click, with
rendering happening from that state on EVERY rerun. Reason: Streamlit
reruns the whole script on any widget interaction (e.g. the what-if
slider), and code that only lived inside the button's `if` block would
vanish the instant you touched any OTHER widget on the page, since the
button only evaluates `True` on the literal click event. This does NOT
change when the solver itself re-runs — only on a fresh "Run optimiser"
click, same as before.

**Verified**: `python -m pytest -q` (46 passed) plus
`streamlit.testing.v1.AppTest` driving a real "Run optimiser" click
against live FPL data — confirmed no exceptions and real rendered output
for all 8 features (deadline banner, reasoning captions, pitch-view HTML,
what-if metrics, diff caption, gamified backtest line). Test runs wrote
throwaway rows into `chip_log.csv`/`last_run_snapshot.json` — the
`chip_log.csv` pollution was reverted (`git checkout`) before committing;
`last_run_snapshot.json` is gitignored so its test content never reached
git.

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
