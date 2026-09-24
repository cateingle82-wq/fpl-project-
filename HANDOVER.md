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
| `api.py` | FastAPI HTTP wrapper over the same optimiser/chip logic — see its own round below |
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

## `cateingle-f9` session, round 3: OptimiserConfig refactor + FastAPI scaffold (api.py)

Prompted by exploring a mobile-app path for this project. Two pieces:

**1. Fixed a real (not just future-proofing) bug**: `fpl_optimise.build_problem()`
used to read `FREE_TRANSFERS`/`MAX_TRANSFERS`/`HIT_COST`/`BAN_UNAVAILABLE`/
`PURCHASE_PRICES` straight off this module's globals, and
`chips.wildcard_detail`/`free_hit_detail` TEMPORARILY OVERWROTE
`opt.MAX_TRANSFERS`/`opt.HIT_COST` mid-function to get a different
hypothetical, restoring them in a `finally`. That's a live race the
moment more than one solve could be in flight against the same process —
e.g. this dashboard open in two browser tabs against different team IDs,
since Streamlit runs each session in its own thread of the SAME process,
sharing this SAME module object. Fixed by adding `opt.OptimiserConfig` (a
frozen dataclass) — `build_problem(df, current_ids, bank_t, config=None)`
now takes one explicitly; `config=None` defaults to `opt.current_config()`
(a snapshot of the module's globals), so **every pre-existing caller
(this file's `main()`, `app.py`'s `apply_config()`, `chips.py`'s
`main()`) needed ZERO changes**. `wildcard_detail`/`free_hit_detail` now
build a local `dataclasses.replace()` copy instead of mutating shared
state — the whole `try/finally` restore dance is gone, nothing shared
left to restore. Also added an optional `base_config` param to both so a
caller with its OWN config (not sourced from these globals at all, e.g.
an API request) doesn't get silently ignored in favour of module
defaults.

**2. `api.py`** — a FastAPI scaffold wrapping the same solver/chip logic
behind HTTP instead of Streamlit widgets, as groundwork for a possible
mobile client later (PWA recommended over native React Native — same
backend either way, PWA needs no app-store review). Endpoints:
`GET /health`, `GET /deadline`, `GET /squad?team_id=`, `POST /recommend`,
`POST /chips`, `GET /chip-log`, `GET /model-health`. Every solve builds
its own `OptimiserConfig` from the request body — no shared mutable
state, so concurrent requests are actually safe (verified: `opt.
MAX_TRANSFERS` never moves while two "concurrent" configs are built and
used). Same scope as `app.py` has always had: single-user, no auth,
`chip_log.csv`/`backtest_results.csv` are still single shared files on
disk — fine for one person, not multi-user-safe.

**Real bug found and fixed while testing this**: `resolve_squad()`
originally only caught `SystemExit` from `fetch_squad_and_bank` (what it
raises deliberately for a CLI's benefit). A genuinely nonexistent team id
doesn't raise `SystemExit` at all — `fpl_stage0.get()` calls
`response.raise_for_status()`, which raises a plain
`requests.exceptions.HTTPError` on the FPL API's 404. That leaked through
as an unhandled 500 + traceback for the single most likely user mistake
against this endpoint (typo'd team id), caught live with a real curl
test against `team_id=999999999`. Fixed with a `_fetch_or_400()` helper
catching both exception types. `fetch_squad_and_bank` raising
`SystemExit` at all is itself a CLI-era wart worth fixing properly in
`fpl_optimise.py` someday — this is the pragmatic interim catch, not that
fix.

**Dependency note**: `fastapi` was stuck at `0.103.2` (installed before
this session started) and incompatible with the `starlette 1.6.0` that
an earlier `streamlit --force-reinstall` had pulled in as a side effect —
`FastAPI()` itself raised `TypeError: Router.__init__() got an
unexpected keyword argument 'on_startup'` on import. Fixed with
`pip install "fastapi>=0.115"` (landed on `0.141.1`). This flagged (but
did not create) a pre-existing conflict: `spotdl` on this machine pins
`fastapi<0.104`/`uvicorn<0.24`/`websockets<15`, already broken by the
streamlit reinstall's `uvicorn 0.53`/`websockets 16.1.1` before this
session touched anything — if `spotdl` stops working, that's why, and
it's unrelated to this project.

**Verified**: full `pytest -q` (50 passed) both before and after,
`streamlit.testing.v1.AppTest` confirming the dashboard's objective is
IDENTICAL (273.68) before/after the config refactor, and live `curl`
tests against every `api.py` endpoint with real data — `/recommend` and
`/chips` both return real solved output for team `7362936`, `/squad`
works, error paths return clean 400s (bad team id) and 422s (missing
`team_id`/`manual_squad`, wrong-length `manual_squad`) instead of raw
500s.

**Not done, worth knowing**: no `requirements.txt`/`pyproject.toml`
anywhere in this repo — every script assumes its deps are just already
installed in whatever environment runs it (`/opt/anaconda3/bin/python`
on this machine specifically, per the "Running things" note below). If
this is ever run somewhere else, that's the first gap to close, not just
for `api.py`. No auth, no request rate-limiting, no fresh-backtest
endpoint (a `/backtest/run` would need to run ~700 API calls
asynchronously, not block a request thread — left out of this scaffold
on purpose rather than done half-right).

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

# FastAPI backend (see api.py's own docstring for endpoint list)
uvicorn api:app --reload
# -> http://localhost:8000/docs for the interactive OpenAPI UI
```

Note: this machine's `python3` on `PATH` sometimes resolves to a bare
system interpreter without the project's dependencies installed — if you
hit `ModuleNotFoundError`, try `/opt/anaconda3/bin/python3` explicitly.

## `cateingle-f9` session, round 4: React Native frontend scaffold (mobile/)

Brand new `mobile/` subdirectory, zero overlap with any Python file —
scaffolded via `npx create-expo-app mobile --template blank-typescript`
(Expo SDK 57, React 19.2.3, react-native 0.86.3), then converted to
**Expo Router** (file-based routing in `src/app/`) per the template's own
`AGENTS.md`, which mandates it over any other navigation approach.

**What's built** — one real, working screen proving the whole stack
end-to-end, not a UI shell:
- `src/lib/config.ts` — API base URL + FPL Team ID, persisted in
  AsyncStorage.
- `src/lib/api.ts` — typed client mirroring `api.py`'s actual response
  shapes exactly (no shared schema between Python/TS — if one changes,
  the other needs a matching manual edit). `ApiError` surfaces FastAPI's
  `{"detail": "..."}` error bodies and gives an actionable message on a
  network-level failure (most likely first-run mistake: wrong base URL).
- `src/app/index.tsx` — the Recommend screen: fetches `POST /recommend`
  for a real team, renders the transfer recommendation, captain, XI,
  bench.
- `src/app/settings.tsx` — Team ID + API base URL, with IN-APP guidance
  on the actual gotcha here: `localhost` only resolves correctly from
  the iOS Simulator (shares the host Mac's network namespace); a
  physical phone needs the dev machine's real LAN IP with the API server
  started via `uvicorn api:app --host 0.0.0.0` (NOT the default
  `127.0.0.1`-only bind used earlier in api.py's own testing); the
  Android emulator needs its special `10.0.2.2` host alias.
- `app.json` — added `expo-build-properties` (Android
  `usesCleartextTraffic: true`) and iOS `NSAppTransportSecurity`
  exceptions, since both platforms block plain HTTP to a dev server by
  default. Checked against the actual SDK 57 docs per `AGENTS.md`'s own
  instruction not to trust training data on Expo APIs — confirmed via
  `docs.expo.dev/versions/v57.0.0/` rather than assumed.

**Verified, and what wasn't**: `npx tsc --noEmit` (clean), `npx expo
lint` (found and fixed 2 real unescaped-apostrophe errors in
`settings.tsx`), `npx expo-doctor` (21/21 checks pass), and a live
`npx expo start --web` run — Metro bundled cleanly (858 modules, zero
errors) against a REAL `uvicorn api:app --host 0.0.0.0` instance, and the
compiled bundle was checked to contain the actual screen text
("Get Recommendation", "No team set"), confirming the router/screens/API
client all wire together correctly. **This session has no browser
automation tool connected** (checked — neither the Chrome extension nor
a built-in browser were available), so nobody has actually clicked
"Get Recommendation" and watched a real result render. That's the one
verification gap here: whoever picks this up next should run `npx expo
start --web` (or on a simulator/device) and manually confirm the fetch →
render flow works, before trusting it further.

**Two dependency hiccups along the way, both fixed**: `npx expo lint`'s
first run failed installing `eslint`/`eslint-config-expo` over a
`react-dom` peer conflict (fixed with `npm install --legacy-peer-deps`);
`npx expo start --web` needed `react-dom`/`react-native-web` installed
separately (`npx expo install react-dom react-native-web -- --legacy-peer-deps`)
since the blank-typescript template doesn't include web support by
default.

**Not done, worth knowing**: no pitch-view equivalent, no app
icon/branding beyond the Expo template defaults, no tests. This is a
first couple of vertical slices (real screens, real data, real error
handling), not a port of everything `app.py` does.

## `cateingle-f9` session, round 5: Chips screen (mobile/)

Added `src/app/chips.tsx` — fetches `POST /chips` and renders Bench
Boost/Triple Captain per-week values (★ marks the best week, same idea
as `app.py`'s "Best week to..." caption) plus Wildcard/Free Hit gains,
each with its 0-10 score and verdict. Registered in `_layout.tsx` and
linked from the home screen's status row next to Settings.

Confirmed the `ChipsResponse` TypeScript shape (written against `api.py`
by reading its code, not by running it) is actually correct by curling
`POST /chips` live and comparing the real JSON keys
(`bench_boost.by_week`/`.score`, `wildcard.gain`/`.score`, etc.) against
what `chips.tsx` expects — they matched exactly, which is also why
`tsc --noEmit` passed cleanly against this screen on the first try.

One real hiccup, expected and self-resolving: adding a new route file
doesn't retroactively update Expo Router's TYPED route strings (used by
`router.push("/chips")`) until the dev server runs once and regenerates
`.expo/types/router.d.ts` — `tsc` correctly failed with `"/chips" is not
assignable to...` until a `expo start --web` run regenerated that file.
Not a bug, just something to know before assuming a typed-route error
means the code is wrong.

**Verified**: `tsc --noEmit` clean, `expo lint` clean, `expo-doctor`
21/21, live `expo start --web` bundling cleanly (869 modules, up from
858) against a real `uvicorn api:app --host 0.0.0.0`, confirmed the
compiled bundle contains "Chip Strategy"/"Get Chip Values"/"Bench Boost".
Same open gap as round 4: no browser tool connected in this session to
click through interactively.

## `cateingle-f9` session, round 6: real device verification + visual redesign (mobile/)

**Verification gap CLOSED**: the app was actually run on a real iPhone via
Expo Go — the one thing rounds 4-5 flagged as unverified. Two real
environment problems hit and fixed along the way, both worth knowing if
this comes up again:
- The dev machine's network (university-adjacent, `10.20.x.x` range)
  enforces client isolation — phone and laptop on the "same WiFi" still
  couldn't reach each other at the TCP level. No app/firewall config
  fixes this (checked: macOS firewall was already disabled). Fixed by
  putting both devices on the iPhone's own Personal Hotspot instead
  (`172.20.10.x` subnet) — guarantees no isolation since the phone IS the
  router. Whoever tests this next, on any network, should suspect this
  FIRST if "same WiFi" still can't connect.
- Recent Expo Go requires signing into the SAME Expo account (free) on
  both the phone's Expo Go app and this machine's Expo CLI to open a
  project from a local dev server on a physical iOS device — not
  optional, not a flag. `npx expo login` on the CLI side.
- A QR code from `npx expo start` doesn't render into a redirected/
  non-TTY log file (its interactive terminal UI needs a real TTY) — a
  QR image was generated directly with the `qrcode` Python package
  instead (`qrcode.make(url).save(...)`) and opened with `open` for the
  user to scan with iOS's Camera app.

**Visual redesign**, prompted by direct feedback ("the interface isnt
that nice") once the plumbing was confirmed working:
- `src/lib/theme.ts` — shared color/spacing/radius/type tokens (FPL
  purple/green/pink-inspired palette, not exact brand assets) so every
  screen draws from the same system instead of independently-styled
  inline values.
- `src/components/ui.tsx` — reusable `Card`, `PrimaryButton`,
  `PositionBadge` (colored per GK/DEF/MID/FWD), `Metric`, `ScoreBadge`
  (green/amber/red by score), `PlayerRow` — used by every screen instead
  of each one styling its own player rows independently.
- Navigation changed from a `Stack` with a "Settings" text link to a
  proper bottom **Tab bar** (`expo-router`'s `Tabs`, `@expo/vector-icons`
  for icons) — Squad / Chips / Settings. This is the single change that
  most makes it "feel like a real app" rather than a dev harness.
- `expo-doctor` caught a real gap immediately: `@expo/vector-icons`
  needs `expo-font` as a peer dependency for anything beyond Expo Go —
  installed via `npx expo install expo-font`, doctor back to 21/21.

**Verified**: `tsc --noEmit` clean, `expo lint` clean (fixed one real
unused-import warning), `expo-doctor` 21/21, live `expo start --web`
bundling cleanly (905 modules) against a real running API server with
the new screen text confirmed present in the compiled bundle. The
redesign itself was NOT clicked-through by a human in this session
(the live device test above was done BEFORE this round's changes) —
whoever picks this up should reload the app on-device and confirm the
new tab bar/cards/colors actually render as intended before trusting
this further.

## `cateingle-f9` session, round 7: fixtures, injury flags, weekly plan (api.py + mobile/)

Direct feedback after the redesign was confirmed working on-device: "why
is it doing 3 weeks" (an arbitrary hardcoded value from the first vertical
slice, not a backend default — real default is `fpl_stage0.HORIZON = 5`),
plus three feature asks: a weekly scroll through suggested transfers,
upcoming fixtures per player, and tappable injury flags.

**`api.py` changes** (backend, none of this touches the model — pure
data plumbing already computed by `build_table`/`fixture_details`):
- `player_risk(info, p)`: structured `{level, detail}` instead of a
  pre-formatted string — `level` is one of out/doubtful/impact_sub/
  fringe/ok, `detail` is FPL's own real injury news text where available
  (e.g. João Pedro → `{"level": "doubtful", "detail": "Knee injury - 75%
  chance of playing"}`, verified live). Same `SUB_PATTERN_GAP` threshold
  as `app.py`'s own `risk_tag`, duplicated rather than imported (app.py
  runs Streamlit UI code at module level just by being imported).
- `fixture_labels_for_week(df, fixtures, gw, week_offset)`: same
  duplication reasoning, mirrors `app.py`'s function of the same name
  exactly.
- `player_summary()` now always includes `risk` and `fixture` on every
  player returned.
- `/recommend` response gains a `plan` array — one entry per week of the
  horizon (captain/XI/bench/fixtures for that week, plus
  `transferred_in`/`transferred_out`/`hits`/`free_transfers_available`
  from week 1 onward, mirroring `app.py`'s per-week tabs exactly). Week 0
  has no transfer-change fields since that's the top-level `transfers`
  block instead — same convention `app.py` already uses.

**Mobile changes**:
- `src/lib/config.ts`: new `horizon` setting (persisted, default 5,
  bounded 1-8) — replaces the hardcoded `3` in both `index.tsx` and
  `chips.tsx`'s fetch calls. Added a "Planning horizon" field to
  Settings.
- `src/components/ui.tsx`'s `PlayerRow`: now shows the fixture inline
  (`ARS · £6.1m · LEE (H) FDR2`) and, when `risk.level !== "ok"`, a
  small colored icon next to the name that's `Pressable` — tapping shows
  an `Alert` with the real injury text. No new UI library; React
  Native's built-in `Alert.alert` was enough for a tap-to-expand.
- `src/app/index.tsx`: added a horizontal scrollable week-selector
  (`GW6 GW7 GW8...` pills) below the transfer recommendation — tapping a
  week shows that week's captain/XI/bench plus its planned transfer
  change, reading from the new `plan` array. Consolidated what used to
  be three always-week-0 sections (Captain/XI/Bench) into this one
  week-aware view instead of duplicating them.

**Verified**: backend — `pytest -q` (50 passed), live `curl` against
`/recommend` with `horizon: 5` confirming `plan` has 5 entries with the
right per-week keys, and confirmed real injury text flows through for
João Pedro/van Ewijk/Hughes by scanning the actual response. Mobile —
`tsc --noEmit` clean, `expo lint` clean, `expo-doctor` 21/21, live
`expo start --web` bundling cleanly (899 modules) with "Weekly Plan"/
"Planning horizon" confirmed present in the compiled bundle. As with
every mobile round, an on-device click-through by a human is the one
step this session can't do itself — flagged, not skipped.

## `cateingle-f9` session, round 8: deployment prep — requirements.txt + API key auth

Scoped hosting the FastAPI backend (Render's free tier, discussed with
the user — cold starts + ephemeral disk accepted as tradeoffs of $0/month).
Two things needed before that's safe to actually do, both done:

1. **`requirements.txt`** (didn't exist anywhere in the repo before this
   — every script assumed its deps were just already installed on this
   machine). Pinned to versions actually tested here: `fastapi==0.141.1`,
   `uvicorn==0.53.0`, `pulp==3.3.2`, `pandas==2.3.3`, `numpy==1.26.4`,
   `requests==2.34.2`, `joblib==1.5.3`, `scikit-learn==1.8.0`.
   `scikit-learn`/`joblib` are for `ml_predict.py`'s cold-start path,
   which degrades gracefully to "no ML" if `ml_model.joblib` isn't
   present (it's gitignored — a fresh deploy won't have one). Also added
   `runtime.txt` (`python-3.11.9`) for Render's Python version pinning.
   **Verified for real, not just written and hoped**: built a completely
   clean `venv`, installed ONLY from `requirements.txt`, and ran a full
   `/recommend` solve against live data from that isolated environment —
   confirms nothing's missing before this ever touches a real host.

2. **API key auth** (`api.py`) — `require_api_key()`, a FastAPI
   dependency applied to every route except `/health`. Enforced ONLY
   when the `API_KEY` environment variable is actually set — local dev
   (no `API_KEY` exported) is completely unaffected, verified by testing
   both states directly: unset → works with no header; set → wrong/
   missing header gets a clean 401, correct `X-API-Key` header gets 200,
   `/health` stays open either way. This exists because `/recommend` runs
   a real CPU-heavy MILP solve — an unauthenticated public URL is an open
   invitation to burn hosting compute for free, not a hypothetical risk.
   Mobile side: `src/lib/config.ts` gained a persisted `apiKey` (empty by
   default), a matching "API Key" field in Settings, and `api.ts`'s
   single `request()` choke point now attaches it as `X-API-Key` on
   every call when non-empty — no need to thread it through each
   exported function/call site individually.

**Not yet done**: the actual Render deployment itself (account creation,
connecting the repo, setting the `API_KEY` env var there, updating the
mobile app's default API base URL to the deployed one) — this session
prepared everything needed for that step but didn't execute it.

**Verified**: `pytest -q` (50 passed) after the auth changes, direct curl
tests of all four auth states against a real running server, the clean-
venv end-to-end solve described above, and the usual mobile trio
(`tsc`/`lint`/`expo-doctor` clean, live bundle test confirming "API Key"/
"X-API-Key" present in compiled output).
