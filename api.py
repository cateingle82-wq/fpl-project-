"""
FastAPI backend for the FPL optimiser.

Wraps fpl_stage0 / fpl_optimise / chips behind HTTP endpoints instead of
Streamlit widgets, so a future client (mobile app, PWA, or anything else)
can call the SAME solver logic app.py already uses, without re-implementing
any of it. Nothing about the underlying model changes here — every
endpoint below just calls the same tested functions app.py calls and
returns their results as JSON instead of rendering them.

This only exists as a real option because of the OptimiserConfig refactor
in fpl_optimise.py — build_problem() used to read its settings off mutable
module globals, which silently breaks the moment two requests could be in
flight at once (exactly what an API server does by default). Every solve
below builds its OWN OptimiserConfig from the request body instead of
touching opt.FREE_TRANSFERS etc., so concurrent requests can't clobber
each other's numbers.

Scope: single-user personal use, same as app.py has always been — no
auth, no per-user data isolation. chip_log.csv / backtest_results.csv are
still single shared files on disk, fine for one person, not something
this scaffold makes safe for multiple people. Extending to real
multi-user support is a separate, bigger piece of work (auth + per-user
storage), not something bolted on here.

Requires: pip install fastapi "uvicorn[standard]"

Run:   uvicorn api:app --reload
Docs:  http://localhost:8000/docs  (FastAPI's automatic interactive OpenAPI UI)
"""

import csv
import os
import secrets
import time
from typing import Optional

import pandas as pd
import pulp
import requests
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

import chips
import fpl_optimise as opt
from fpl_stage0 import HORIZON, SUB_PATTERN_GAP, build_table, fixture_details, horizon_of, next_gameweek

app = FastAPI(
    title="FPL Optimiser API",
    description="HTTP wrapper over the same optimiser/chip/backtest logic app.py uses.",
    version="0.1.0",
)


# ----------------------------------------------------------------------------
# API key auth — enforced ONLY when the API_KEY environment variable is
# actually set. Local dev (no API_KEY exported) behaves exactly as before,
# unauthenticated. Once deployed somewhere with API_KEY configured, every
# endpoint below except /health requires a matching X-API-Key header.
# Without this, a public URL running a CPU-heavy MILP solve on every
# /recommend call is an open invitation for anyone who finds it to burn
# your hosting's compute for free — this is not optional polish for a
# deployment that's actually reachable from the internet.
# ----------------------------------------------------------------------------

API_KEY = os.environ.get("API_KEY")


def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    if API_KEY is None:
        return
    if not x_api_key or not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header.")


auth = [Depends(require_api_key)]


# ----------------------------------------------------------------------------
# Simple in-process TTL cache for the slow FPL API pulls — same idea as
# app.py's st.cache_data, just without Streamlit. One process-wide cache
# since this is a single-user personal backend, not per-user isolated.
# ----------------------------------------------------------------------------

_cache: dict = {}


def _cached(key, ttl, fn):
    now = time.time()
    hit = _cache.get(key)
    if hit and now - hit[0] < ttl:
        return hit[1]
    value = fn()
    _cache[key] = (now, value)
    return value


def get_table(horizon):
    return _cached(f"table:{horizon}", ttl=300, fn=lambda: build_table(horizon=horizon))


def get_bootstrap():
    return _cached("bootstrap", ttl=300, fn=lambda: opt.get("bootstrap-static/"))


def get_fixtures():
    return _cached("fixtures", ttl=300, fn=lambda: opt.get("fixtures/"))


# ----------------------------------------------------------------------------
# Request/response models
# ----------------------------------------------------------------------------

class ConfigOverrides(BaseModel):
    """Everything an OptimiserConfig needs beyond free_transfers (which
    comes from the live account fetch or the request itself, not a
    hand-set override, mirroring how app.py's sidebar treats it)."""
    max_transfers: int = Field(5, ge=0, le=15)
    hit_cost: float = Field(4.0, ge=0)
    ban_unavailable: bool = True
    purchase_prices: dict[int, float] = Field(default_factory=dict)

    def to_optimiser_config(self, free_transfers: int) -> opt.OptimiserConfig:
        return opt.OptimiserConfig(
            free_transfers=free_transfers,
            max_transfers=self.max_transfers,
            hit_cost=self.hit_cost,
            ban_unavailable=self.ban_unavailable,
            purchase_prices={int(k): v for k, v in self.purchase_prices.items()},
        )


class SquadRequest(BaseModel):
    """Shared by /recommend and /chips — how to find out who you own.
    Exactly one of team_id / manual_squad should be set, mirroring
    app.py's sidebar (manual squad box wins if filled in)."""
    team_id: Optional[int] = None
    manual_squad: Optional[list[int]] = None
    horizon: int = Field(HORIZON, ge=1, le=8)
    bank: float = Field(0.0, description="Only used as a fallback when there's no live "
                                          "account to fetch the real bank from.")
    free_transfers: Optional[int] = Field(
        None, description="Only used as a fallback when there's no live account to "
                           "fetch the real free-transfer count from.")
    config: ConfigOverrides = ConfigOverrides()


# ----------------------------------------------------------------------------
# Shared helper: resolve squad/bank/free-transfers the same way app.py's
# get_current_squad_and_bank does — live fetch wins when a team_id is
# given, manual_squad is the only path with no live account to check.
# ----------------------------------------------------------------------------

def _fetch_or_400(fn, *args):
    """Runs an fpl_optimise fetch function and turns EITHER of its two
    real failure modes into a clean HTTP 400 instead of an unhandled 500:
    SystemExit (what fetch_squad_and_bank raises deliberately, written for
    a CLI script's benefit — kill the process, print why) AND
    requests.exceptions.RequestException (what actually happens on a
    nonexistent team id: fpl_stage0.get() calls response.raise_for_status(),
    which raises a plain requests.HTTPError on the FPL API's 404 — NOT a
    SystemExit, so catching only SystemExit here was missing this exact
    case and leaking a raw 500 + traceback to the client for the single
    most likely user mistake against this endpoint: typing a wrong team
    id. A cleaner long-term fix is making fpl_optimise.py raise one real
    exception type instead of SystemExit/whatever requests does
    internally; this is the pragmatic fix that doesn't require touching
    that file again right now."""
    try:
        return fn(*args)
    except (SystemExit, requests.exceptions.RequestException) as e:
        raise HTTPException(status_code=400, detail=str(e))


def resolve_squad(req: SquadRequest, gw: int):
    if req.manual_squad:
        if len(req.manual_squad) != 15:
            raise HTTPException(status_code=422,
                                 detail=f"manual_squad must have 15 player ids, got {len(req.manual_squad)}.")
        ids = req.manual_squad
        bank_t = int(round(req.bank * 10))
        free_transfers = req.free_transfers if req.free_transfers is not None else 1
        return ids, bank_t, free_transfers

    if not req.team_id:
        raise HTTPException(status_code=422, detail="Provide team_id or manual_squad.")

    ids, live_bank_t = _fetch_or_400(opt.fetch_squad_and_bank, req.team_id, gw)

    if len(ids) != 15:
        raise HTTPException(status_code=400, detail=f"Expected 15 players, got {len(ids)}.")

    bank_t = live_bank_t if live_bank_t is not None else int(round(req.bank * 10))
    live_ft = opt.fetch_free_transfers(req.team_id, gw)
    free_transfers = live_ft if live_ft is not None else (req.free_transfers or 1)
    return ids, bank_t, free_transfers


def fixture_labels_for_week(df, fixtures, gw, week_offset):
    """team_id -> 'OPP (H/A) FDRx' for one specific week (a double shows
    both fixtures, a blank shows '—') — same approach app.py's own
    fixture_labels_for_week uses (duplicated here rather than imported:
    app.py is a Streamlit script that runs UI-building code at module
    level just by being imported, so it can't be a shared dependency)."""
    teams_map = dict(zip(df["team"], df["team_name"]))
    details = fixture_details(fixtures, gw + week_offset, 1)
    labels = {}
    for team_id, entries in details.items():
        labels[team_id] = " + ".join(
            f"{teams_map.get(opp_id, '?')} ({'H' if was_home else 'A'}) FDR{fdr}"
            for fdr, opp_id, was_home in entries
        )
    return labels


def player_risk(info, p):
    """Structured injury/rotation-risk info — a client-side rendering
    decision (icon, color, tap-to-expand) shouldn't be baked into a
    pre-formatted string, so this returns level + detail separately.
    `level` is one of: out, doubtful, impact_sub, fringe, ok. Same
    underlying signals and SUB_PATTERN_GAP threshold as app.py's own
    risk_tag (duplicated for the same reason as fixture_labels_for_week
    above), so this and the Streamlit dashboard never disagree about
    what counts as a risk."""
    r = info.loc[p]
    avail = r.get("avail", 1.0)
    news = (r.get("news") or "").strip() or None
    if avail <= 0.0:
        return {"level": "out", "detail": news or "Ruled out this gameweek"}
    if avail < 1.0:
        chance = r.get("chance_of_playing_next_round")
        pct = f"{int(chance)}%" if pd.notna(chance) else f"{avail:.0%}"
        return {"level": "doubtful", "detail": news or f"{pct} chance of playing"}
    recent = r.get("recent_mins_share")
    starts = r.get("recent_start_share")
    if pd.notna(recent) and pd.notna(starts) and starts < recent - SUB_PATTERN_GAP:
        return {"level": "impact_sub", "detail": "Comes off the bench more than he starts — impact-sub pattern"}
    if pd.notna(recent) and recent < 0.4:
        return {"level": "fringe", "detail": "Low recent minutes — fringe squad player"}
    return {"level": "ok", "detail": None}


def player_summary(info, p, xpts_col="xpts", extra=None, fx_labels=None):
    r = info.loc[p]
    out = {
        "id": int(p), "name": r["name"], "pos": r["pos"], "team": r["team_name"],
        "price": round(float(r["price"]), 1), "xpts": round(float(r[xpts_col]), 2),
        "risk": player_risk(info, p),
        "fixture": (fx_labels or {}).get(r["team"], "—"),
    }
    if extra:
        out.update(extra)
    return out


# ----------------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/deadline", dependencies=auth)
def deadline():
    """Next gameweek id + its raw deadline_time — the client computes its
    own countdown from this rather than the server baking in a
    string-formatted one, so it stays correct across timezones/locales."""
    boot = get_bootstrap()
    gw = next_gameweek(boot["events"])
    for e in boot["events"]:
        if e["id"] == gw:
            return {"gw": gw, "deadline_time": e.get("deadline_time")}
    raise HTTPException(status_code=404, detail="No upcoming gameweek found.")


@app.get("/squad", dependencies=auth)
def squad(team_id: int):
    """Live squad/bank/free-transfers for one team — the read-only lookup
    a client would call before letting someone edit their config, same
    data app.py's sidebar auto-fetch pulls."""
    df, gw = get_table(HORIZON)
    ids, bank_t = _fetch_or_400(opt.fetch_squad_and_bank, team_id, gw)
    if len(ids) != 15:
        raise HTTPException(status_code=400, detail=f"Expected 15 players, got {len(ids)}.")
    free_transfers = opt.fetch_free_transfers(team_id, gw)
    return {
        "gw": gw, "player_ids": ids,
        "bank": (bank_t or 0) / 10,
        "free_transfers": free_transfers,
    }


@app.post("/recommend", dependencies=auth)
def recommend(req: SquadRequest):
    """The main endpoint: solve Stage A + Stage B and return the transfer
    recommendation, starting XI, and bench for week 0 (the only REAL
    decision — see build_problem's docstring), PLUS a `plan` array
    covering every week of the horizon (Stage A's own forward-looking
    hypothesis, same as app.py's per-week tabs) so a client can build a
    week-by-week scroll through the whole plan, not just this week's
    move. Minus the chip evaluation (see /chips) so a client can fetch
    that separately/in parallel if it wants to."""
    df, gw = get_table(req.horizon)
    horizon = horizon_of(df)
    current_ids, bank_t, free_transfers = resolve_squad(req, gw)
    config = req.config.to_optimiser_config(free_transfers)
    fixtures = get_fixtures()

    prob, sq, start, cap, hits, ft, tin, tout, cost_t, budget_t = opt.build_problem(
        df, current_ids, bank_t, config
    )
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        raise HTTPException(
            status_code=422,
            detail=f"Solver returned {status} — your budget or transfer cap "
                   "probably makes a legal squad impossible.",
        )

    chosen = [p for p in df["id"] if sq[0][p].value() > 0.5]
    xi, captain = opt.choose_lineup(df, chosen)
    info = df.set_index("id")
    fx_labels_wk0 = fixture_labels_for_week(df, fixtures, gw, 0)

    current = set(current_ids)
    out_ids = sorted(current - set(chosen), key=lambda p: -info.loc[p, "xpts"])
    in_ids = sorted(set(chosen) - current, key=lambda p: -info.loc[p, "xpts"])
    n_hits = int(round(hits[0].value()))
    bench = [p for p in chosen if p not in xi]

    # Per-week plan — mirrors app.py's week_data/week_transfer_info loop.
    # Week 0 gets no transferred_in/out/hits fields (that's the
    # `transfers` block above); weeks 1+ show what changes from the
    # PREVIOUS week in the plan, same convention app.py's tabs use.
    plan = []
    prev_squad = None
    for w in range(horizon):
        if w == 0:
            squad_w, xi_w, cap_w = chosen, xi, captain
        else:
            squad_w = [p for p in df["id"] if sq[w][p].value() > 0.5]
            xi_w = [p for p in squad_w if start[w][p].value() > 0.5]
            cap_w = next(p for p in squad_w if cap[w][p].value() > 0.5)
        bench_w = [p for p in squad_w if p not in xi_w]
        fx_labels_w = fixture_labels_for_week(df, fixtures, gw, w)
        xpts_col = f"xw{w}"

        entry = {
            "week_offset": w,
            "gw": gw + w,
            "captain": player_summary(info, cap_w, xpts_col, fx_labels=fx_labels_w),
            "xi": [player_summary(info, p, xpts_col, {"captain": p == cap_w}, fx_labels_w) for p in xi_w],
            "bench": [player_summary(info, p, xpts_col, fx_labels=fx_labels_w) for p in bench_w],
        }
        if prev_squad is not None:
            transferred_in = set(squad_w) - set(prev_squad)
            transferred_out = set(prev_squad) - set(squad_w)
            entry["transferred_in"] = [player_summary(info, p, xpts_col, fx_labels=fx_labels_w) for p in transferred_in]
            entry["transferred_out"] = [player_summary(info, p, fx_labels=fx_labels_w) for p in transferred_out]
            entry["hits"] = int(round(hits[w].value()))
            entry["free_transfers_available"] = free_transfers if w == 0 else int(round(ft[w].value()))
        plan.append(entry)
        prev_squad = squad_w

    return {
        "gw": gw,
        "horizon": horizon,
        "objective": round(pulp.value(prob.objective), 2),
        "average_per_gw": round(pulp.value(prob.objective) / horizon, 2),
        "transfers": {
            "out": [player_summary(info, p, fx_labels=fx_labels_wk0) for p in out_ids],
            "in": [player_summary(info, p, fx_labels=fx_labels_wk0) for p in in_ids],
            "hits": n_hits,
            "hit_cost_paid": n_hits * config.hit_cost,
        },
        "captain": player_summary(info, captain, "xw0", fx_labels=fx_labels_wk0),
        "xi": [player_summary(info, p, "xw0", {"captain": p == captain}, fx_labels_wk0) for p in xi],
        "bench": [player_summary(info, p, "xw0", fx_labels=fx_labels_wk0) for p in bench],
        "squad_cost": round(sum(cost_t[p] for p in chosen) / 10, 1),
        "budget": round(budget_t / 10, 1),
        "plan": plan,
    }


@app.post("/chips", dependencies=auth)
def chip_values(req: SquadRequest):
    """Bench Boost / Triple Captain / Wildcard / Free Hit values for the
    CURRENT squad (before any transfer /recommend suggests), plus the
    0-10 timing scores — same numbers app.py's 'Chip strategy' section
    shows, computed from the SAME config the request specifies (not
    silently falling back to whatever this process's module-level
    defaults happen to be — see wildcard_detail/free_hit_detail's
    base_config parameter)."""
    df, gw = get_table(req.horizon)
    horizon = horizon_of(df)
    current_ids, bank_t, free_transfers = resolve_squad(req, gw)
    config = req.config.to_optimiser_config(free_transfers)

    bb = chips.bench_boost_value(df, current_ids)
    tc = chips.triple_captain_value(df, current_ids)
    wc_detail = chips.wildcard_detail(df, current_ids, bank_t, config)
    fh_detail = chips.free_hit_detail(df, current_ids, bank_t, config)
    scores = chips.chip_scores(bb, tc, wc_detail["gain"], fh_detail["gain"])

    return {
        "gw": gw,
        "horizon": horizon,
        "bench_boost": {"by_week": bb, "score": scores["bench_boost"]},
        "triple_captain": {"by_week": tc, "score": scores["triple_captain"]},
        "wildcard": {"gain": round(wc_detail["gain"], 2), "score": scores["wildcard"]},
        "free_hit": {"gain": round(fh_detail["gain"], 2), "score": scores["free_hit"]},
    }


@app.get("/chip-log", dependencies=auth)
def chip_log():
    """Raw rows from chip_log.csv, for a client to plot its own trend
    chart from — same data app.py's chip-log-based scoring reads."""
    if not os.path.exists(chips.LOG_PATH):
        return []
    with open(chips.LOG_PATH) as f:
        return list(csv.DictReader(f))


@app.get("/model-health", dependencies=auth)
def model_health():
    """backtest.py's own predicted-vs-actual track record, for a client's
    'should I trust this model' view — same data app.py's Model Health
    sidebar expander reads. Does NOT run a fresh backtest (that's ~700 API
    calls on first run) — this only reads the existing results file;
    triggering a fresh run is intentionally left out of this scaffold."""
    if not os.path.exists("backtest_results.csv"):
        raise HTTPException(status_code=404,
                             detail="No backtest_results.csv yet — run backtest.py first.")
    res = pd.read_csv("backtest_results.csv")
    return res.to_dict(orient="records")
