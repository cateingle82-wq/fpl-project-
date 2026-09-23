"""
FPL Optimiser — local dashboard.

Wraps fpl_stage0 / fpl_optimise / chips / backtest in a Streamlit UI instead
of terminal output. Nothing about the underlying MODEL logic changes here
— this file only calls the same tested functions and renders their
results as tables, charts, and a few plain-language translations of them.
If you ever suspect a number here, the *_optimise.py / chips.py /
backtest.py scripts remain the source of truth and still run standalone
exactly as before.

Layout: one "Run optimiser" click solves everything (transfer plan, weekly
plan, chip values) and stores the result in st.session_state — rendering
then happens from that stored state on EVERY rerun, not just the click
that triggered the solve. This is what lets lightweight widgets added
after the solve (the what-if slider, the pitch view) react instantly
without re-running the MILP: Streamlit reruns the whole script on any
widget interaction, and code that only lived inside `if st.button(...):`
would vanish the moment you touched anything else.

Run:  pip install streamlit
      streamlit run app.py
Opens in your browser at http://localhost:8501, on your machine only —
nothing is uploaded or hosted anywhere.
"""

import json
import os
import re
from datetime import datetime, timezone

import altair as alt
import pandas as pd
import pulp
import streamlit as st

import backtest as bt
import chips
import fpl_optimise as opt
from fpl_stage0 import (HORIZON, SUB_PATTERN_GAP, build_table, fixture_details,
                         horizon_of, next_gameweek)

st.set_page_config(page_title="FPL Optimiser", layout="wide")

LAST_RUN_PATH = "last_run_snapshot.json"


# ----------------------------------------------------------------------------
# Cached data pulls — the FPL API call is the slow part, so these are cached
# for the session and only re-fetched when the sidebar button is pressed.
# ----------------------------------------------------------------------------

@st.cache_data(show_spinner="Pulling FPL data...")
def cached_build_table(horizon):
    return build_table(horizon=horizon)


def get_data(horizon, force_refresh=False):
    if force_refresh:
        cached_build_table.clear()
    return cached_build_table(horizon)


@st.cache_data(show_spinner=False)
def cached_fixtures():
    """The full season's fixture list — separate from cached_build_table
    since chips.season_outlook needs ALL 38 gameweeks, not just the
    horizon build_table computes xw{w} columns for."""
    return opt.get("fixtures/")


@st.cache_data(show_spinner=False, ttl=300)
def cached_bootstrap():
    """Only used here for the deadline countdown and next_gameweek — a
    short 5-minute TTL (not the session-long cache the data above uses)
    since a countdown that never refreshes on its own is worse than
    useless once you've had the tab open for a while."""
    return opt.get("bootstrap-static/")


def deadline_countdown(boot, gw):
    """Human string like '2d 14h 32m' until gw's transfer deadline, or
    None if it can't be found (e.g. season finished, or FPL's schedule
    changed underneath the id). Returns 'Deadline has passed' rather than
    a negative countdown if you're viewing this after it's shut."""
    for e in boot.get("events", []):
        if e["id"] == gw:
            dt_str = e.get("deadline_time")
            if not dt_str:
                return None
            deadline = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
            delta = deadline - datetime.now(timezone.utc)
            if delta.total_seconds() <= 0:
                return "Deadline has passed"
            days, rem = divmod(int(delta.total_seconds()), 86400)
            hours, rem = divmod(rem, 3600)
            minutes = rem // 60
            parts = [f"{days}d"] if days else []
            parts += [f"{hours}h"] if (hours or days) else []
            parts.append(f"{minutes}m")
            return " ".join(parts)
    return None


# ----------------------------------------------------------------------------
# Sidebar — the same config block fpl_optimise.py has at the top of the
# file, just editable per-run instead of edited-and-saved. Applying it sets
# the actual module attributes other functions read, so nothing downstream
# needs to know it's running inside a dashboard.
# ----------------------------------------------------------------------------

st.sidebar.header("Config")

horizon_weeks = st.sidebar.slider(
    "Optimise over (weeks)", min_value=1, max_value=8, value=HORIZON, step=1,
    help="How many gameweeks ahead the optimiser plans over — squad, "
         "starting XI AND transfer timing. With a longer horizon the model "
         "can genuinely decide to spend only some of your free transfers "
         "now and bank the rest for a bigger opportunity later, instead of "
         "always using what's available immediately. Set to 1 and it can "
         "only ever think about this week (no reason to bank anything).",
)

team_id = st.sidebar.number_input(
    "Team ID", value=int(opt.TEAM_ID), step=1,
    help="The number in your FPL team's URL.",
)
max_transfers = st.sidebar.number_input(
    "Max transfers in any single week", value=int(opt.MAX_TRANSFERS), min_value=0, max_value=15, step=1,
    help="A per-week ceiling, not the lever for 'how big a rehaul' — the "
         "model itself now decides how many of your available transfers to "
         "actually spend each week, and whether to bank some for later. "
         "This just stops it proposing an unrealistic same-week rebuild; "
         "the chip values below already evaluate a true wildcard/free hit "
         "separately.",
)
# Every value= below is explicitly cast to float, even though the config
# block in fpl_optimise.py "should" already hold floats — Streamlit's
# number_input demands value/min_value/max_value/step all share ONE type,
# and a config value edited by hand as a plain int (e.g. `= 9` instead of
# `= 9.0`) silently breaks that the moment it meets a float step. Casting
# here means a config edit can never crash the dashboard over this.
hit_cost = st.sidebar.number_input(
    "Hit cost (pts per extra transfer)", value=float(opt.HIT_COST), step=0.5
)
ban_unavailable = st.sidebar.checkbox("Don't buy flagged/injured players", value=opt.BAN_UNAVAILABLE)

manual_squad_input = st.sidebar.text_input(
    "Manual squad (15 comma-separated player IDs, optional)",
    help="Leave blank to fetch your live squad from Team ID instead.",
)

# Bank and free transfers are auto-fetched from your live FPL account once
# you run the optimiser (see get_current_squad_and_bank) — showing editable
# fields for them here unconditionally was misleading, since anything typed
# in got silently overridden the moment a live squad fetch succeeded. Only
# show them when there's no live account to fetch from (manual squad mode),
# where they're the only source of truth available.
if manual_squad_input.strip():
    bank = st.sidebar.number_input("Bank (£m)", value=float(opt.BANK), step=0.1, format="%.1f")
    free_transfers = st.sidebar.number_input(
        "Free transfers", value=int(opt.FREE_TRANSFERS), min_value=0, max_value=15, step=1
    )
else:
    bank, free_transfers = opt.BANK, opt.FREE_TRANSFERS
    st.sidebar.caption("Bank and free transfers are auto-fetched live from your "
                       "Team ID once you run the optimiser.")

if st.sidebar.button("Refresh FPL data", help="Re-pulls bootstrap-static/fixtures from the API."):
    get_data(horizon_weeks, force_refresh=True)
    cached_bootstrap.clear()
    st.sidebar.success("Data refreshed.")


def apply_config():
    """Push the sidebar values into fpl_optimise's module globals — the
    same globals build_problem/choose_lineup/etc. already read, so every
    function downstream behaves exactly as the standalone script would with
    this config edited in by hand."""
    opt.TEAM_ID = int(team_id)
    opt.BANK = float(bank)
    opt.FREE_TRANSFERS = int(free_transfers)
    opt.MAX_TRANSFERS = int(max_transfers)
    opt.HIT_COST = float(hit_cost)
    opt.BAN_UNAVAILABLE = bool(ban_unavailable)


def get_current_squad_and_bank(df, gw):
    """Manual squad box wins if filled in — bank AND free transfers then
    come from their sidebar fields, since there's no live account to read
    them from. Live squad fetch also pulls the REAL bank FPL tracks for
    that team (entry_history.bank) and the REAL free-transfer count
    (derived from entry/{id}/history/'s transfer log — see
    fetch_free_transfers), using both in place of their sidebar fields so
    they self-update every week instead of relying on you to track and
    type in the right numbers each time you re-run this. Free transfers
    is set as a module global on opt (opt.FREE_TRANSFERS) rather than
    returned, matching how build_problem already reads it — same pattern
    apply_config() already uses for MAX_TRANSFERS/HIT_COST/BAN_UNAVAILABLE."""
    if manual_squad_input.strip():
        ids = [int(x) for x in manual_squad_input.split(",") if x.strip()]
        bank_t = int(round(opt.BANK * 10))
    else:
        ids, live_bank_t = opt.fetch_squad_and_bank(opt.TEAM_ID, gw)
        if live_bank_t is not None:
            bank_t = live_bank_t
            st.caption(f"Bank auto-fetched from your team: £{bank_t / 10:.1f}m "
                       "(overrides the sidebar Bank field).")
        else:
            bank_t = int(round(opt.BANK * 10))

        live_ft = opt.fetch_free_transfers(opt.TEAM_ID, gw)
        if live_ft is not None:
            opt.FREE_TRANSFERS = live_ft
            st.caption(f"Free transfers auto-fetched from your team: {live_ft} "
                       "(overrides the sidebar Free transfers field).")
    if len(ids) != 15:
        st.error(f"Expected 15 players, got {len(ids)}.")
        st.stop()
    return ids, bank_t


def fixture_labels_for_week(df, fixtures, gw, week_offset):
    """team_id -> 'OPP (H/A) FDRx' for one specific week (a double shows
    both fixtures, a blank shows '—') — built once per table-rendering
    context (not per player row) and passed into player_row, since it's
    the same lookup for every player on the same team."""
    teams_map = dict(zip(df["team"], df["team_name"]))
    details = fixture_details(fixtures, gw + week_offset, 1)
    labels = {}
    for team_id, entries in details.items():
        parts = [f"{teams_map.get(opp_id, '?')} ({'H' if was_home else 'A'}) FDR{fdr}"
                 for fdr, opp_id, was_home in entries]
        labels[team_id] = " + ".join(parts)
    return labels


def risk_tag(info, p):
    """A short, SELF-EXPLANATORY flag — shows FPL's own injury news and
    percentage rather than a vague 'flagged', since 'doubtful' on its own
    doesn't say what's actually wrong or what the model did about it.
    `avail` (0.0-1.0, from fpl_stage0.availability()) is exactly the
    multiplier already applied to this player's xPts — a player at 0.75
    has already had their prediction cut by 25%, this tag is just making
    that visible instead of buried in a column nobody looks at.
    'Impact sub' reuses the exact SUB_PATTERN_GAP threshold fpl_stage0
    itself uses to discount these players' minutes, so this tag and that
    scoring effect always agree with each other."""
    r = info.loc[p]
    avail = r.get("avail", 1.0)
    news = (r.get("news") or "").strip()
    if avail <= 0.0:
        return "🔴 ruled out" + (f" — {news}" if news else "")
    if avail < 1.0:
        if news:
            return f"🟡 {news}"
        chance = r.get("chance_of_playing_next_round")
        pct = f"{int(chance)}%" if pd.notna(chance) else f"{avail:.0%}"
        return f"🟡 {pct} chance of playing — xPts already cut to match"
    recent = r.get("recent_mins_share")
    starts = r.get("recent_start_share")
    if pd.notna(recent) and pd.notna(starts) and starts < recent - SUB_PATTERN_GAP:
        return "⚠️ impact sub (comes off the bench more than he starts)"
    if pd.notna(recent) and recent < 0.4:
        return "⚠️ fringe (low recent minutes)"
    return ""


def _profile_bits(info, p, fx_labels):
    """Splits a player's profile into (positive, negative) plain-language
    bits — kept separate because the SAME positive trait (e.g. nailed-on
    starter) means something different depending on whether you're reading
    it as a reason to bring someone IN or a reason to send them OUT, and
    conflating the two produced nonsense like 'OUT: in-form, nailed-on
    starter' which reads as an argument to KEEP them, not drop them."""
    r = info.loc[p]
    pos_bits, neg_bits = [], []
    fx = (fx_labels or {}).get(r["team"], "—")
    fdrs = [int(m) for m in re.findall(r"FDR(\d)", fx)] if fx and fx != "—" else []
    if fdrs:
        worst = max(fdrs)
        if worst <= 2:
            pos_bits.append(f"favourable fixture(s) ({fx})")
        elif worst >= 4:
            neg_bits.append(f"tough fixture(s) ({fx})")
    mins = r.get("recent_mins_share", r.get("mins_share"))
    if pd.notna(mins):
        if mins >= 0.75:
            pos_bits.append("nailed-on starter")
        elif mins < 0.4:
            neg_bits.append("rotation/bench risk")
    ppg_shrunk = r.get("ppg_shrunk", 0)
    if pd.notna(r.get("form")) and ppg_shrunk and ppg_shrunk > 0:
        if r["form"] > ppg_shrunk * 1.15:
            pos_bits.append("in-form")
        elif r["form"] < ppg_shrunk * 0.7:
            neg_bits.append("out of form")
    if r.get("set_piece_bonus", 0) > 0:
        pos_bits.append("on set-pieces")
    if r.get("avail", 1.0) < 1.0:
        neg_bits.append("fitness doubt")
    return pos_bits, neg_bits


def transfer_reason_in(info, p, fx_labels):
    """Why the model wants to BUY this player — its positive traits."""
    pos_bits, _ = _profile_bits(info, p, fx_labels)
    return ", ".join(pos_bits) if pos_bits else "steady, unremarkable profile — picked on value, not a standout trait"


def transfer_reason_out(info, p, fx_labels):
    """Why the model is happy to SELL this player. Deliberately only
    surfaces NEGATIVE traits — the same profile function used for IN would
    often list this player's genuine positives too (a fine, in-form player
    can still be the correct sale if the budget's better spent elsewhere),
    and showing those without that context just reads as self-contradictory
    ('nailed-on starter, in-form' under a SELL heading). If there's no
    actual red flag, say so plainly instead of implying one exists."""
    _, neg_bits = _profile_bits(info, p, fx_labels)
    if neg_bits:
        return ", ".join(neg_bits)
    return "no red flags on this pick — simply outscored by the incoming player for the budget, not a problem with them"


def player_row(info, p, xpts_col="xpts", tag="", fx_labels=None):
    r = info.loc[p]
    row = {
        "Player": r["name"], "Pos": r["pos"], "Team": r["team_name"],
        "Price": f"£{r['price']:.1f}", "xPts": round(r[xpts_col], 2),
        "Risk": risk_tag(info, p),
    }
    if fx_labels is not None:
        row["Fixture"] = fx_labels.get(r["team"], "—")
    row[""] = tag
    return row


def render_pitch(info, xi, bench, captain, xpts_col):
    """A literal formation layout (GK/DEF/MID/FWD rows) instead of a
    dataframe — the single biggest 'does this look like a real product'
    change available for the effort, and purely a rendering choice: same
    xi/captain the tables above already show, just laid out the way you'd
    actually see a team sheet. Colours use a theme-neutral rgba overlay
    (not fixed hex values) so it reads correctly in both light and dark
    Streamlit themes."""
    order = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
    by_pos = {}
    for p in xi:
        by_pos.setdefault(info.loc[p, "pos"], []).append(p)

    for pos in ["GKP", "DEF", "MID", "FWD"]:
        players = sorted(by_pos.get(pos, []), key=lambda p: -info.loc[p, xpts_col])
        if not players:
            continue
        cols = st.columns(len(players))
        for col, p in zip(cols, players):
            r = info.loc[p]
            badge = " (C)" if p == captain else ""
            risk = risk_tag(info, p)
            with col:
                st.markdown(
                    "<div style='text-align:center;padding:10px 4px;border-radius:10px;"
                    "background:rgba(127,127,127,0.15);margin-bottom:6px;'>"
                    f"<b>{r['name']}{badge}</b><br>"
                    f"£{r['price']:.1f}m &middot; {r[xpts_col]:.1f} xPts"
                    + (f"<br><small>{risk}</small>" if risk else "")
                    + "</div>",
                    unsafe_allow_html=True,
                )
    if bench:
        bench_sorted = sorted(bench, key=lambda p: -info.loc[p, xpts_col])
        st.caption("Bench: " + " · ".join(
            f"{info.loc[p, 'name']} ({info.loc[p, xpts_col]:.1f})" for p in bench_sorted
        ))


def load_last_run():
    if os.path.exists(LAST_RUN_PATH):
        try:
            with open(LAST_RUN_PATH) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
    return None


def save_run_snapshot(gw, in_names, out_names, plan_obj):
    with open(LAST_RUN_PATH, "w") as f:
        json.dump({
            "gw": gw, "in_names": in_names, "out_names": out_names,
            "objective": plan_obj,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }, f)


def render_run_diff(last_run, gw, in_names, out_names, plan_obj):
    """Compares this run against the last saved one — including from a
    PREVIOUS session (it's a file on disk, not just Streamlit's in-memory
    state) — so re-running today shows what actually changed since you
    last checked, not just noise from re-solving the same inputs."""
    if last_run is None:
        st.caption("No previous run saved yet — this becomes a diff next time you run it.")
        return
    st.write("**Since your last saved run**")
    when = last_run.get("saved_at", "?")
    gw_note = f" (GW{last_run['gw']})" if last_run.get("gw") != gw else ""
    obj_delta = plan_obj - last_run.get("objective", plan_obj)
    st.caption(f"Last run{gw_note} was saved {when[:16].replace('T', ' ')} UTC — "
               f"objective moved {obj_delta:+.2f} pts since then.")
    new_in = set(in_names) - set(last_run.get("in_names", []))
    dropped_in = set(last_run.get("in_names", [])) - set(in_names)
    if new_in:
        st.caption(f"Newly suggested IN since last time: {', '.join(sorted(new_in))}")
    if dropped_in:
        st.caption(f"No longer suggesting (was IN last time): {', '.join(sorted(dropped_in))}")
    if not new_in and not dropped_in and set(in_names) == set(last_run.get("in_names", [])):
        st.caption("Same transfer suggestion as last time — a stable read, not a wobble.")


# ----------------------------------------------------------------------------
# Deadline countdown — independent of the "Run optimiser" click; this is a
# clock, not a model output, so it shouldn't need a solve to show up.
# ----------------------------------------------------------------------------

try:
    boot_for_deadline = cached_bootstrap()
    deadline_gw = next_gameweek(boot_for_deadline["events"])
    countdown = deadline_countdown(boot_for_deadline, deadline_gw)
    if countdown:
        st.info(f"⏰ GW{deadline_gw} deadline in **{countdown}**")
except Exception:
    pass   # a countdown is a nice-to-have, never worth blocking the page over


# ----------------------------------------------------------------------------
# Sidebar — Model health (backtest). Separate from the weekly decision above:
# this answers "should I trust the model at all", not "what do I do this
# week", so it's a diagnostic you check occasionally, not something that
# should compete with the transfer decision for main-page space.
# ----------------------------------------------------------------------------

with st.sidebar.expander("Model health (backtest)"):
    st.caption("Does xpts_gw1 actually correlate with what players go on to "
               "score? See backtest.py's docstring for what this does and "
               "doesn't cover.")

    run_backtest = st.button("Run backtest now (re-checks every finished gameweek)")

    if run_backtest:
        with st.spinner("Fetching player histories (first run is slow, ~700 API calls)..."):
            boot = bt.get("bootstrap-static/")
            fixtures = bt.get("fixtures/")
            element_ids = [e["id"] for e in boot["elements"]]
            histories = bt.fetch_all_histories(element_ids)

        finished_gws = [e["id"] for e in boot["events"] if e["finished"]]
        if not finished_gws:
            st.warning("No finished gameweeks yet this season — nothing to backtest.")
        else:
            test_gws = list(range(bt.MIN_GW, max(finished_gws) + 1))
            results = []
            progress = st.progress(0.0)
            for i, gw_ in enumerate(test_gws):
                snap = bt.snapshot_as_of(boot, fixtures, histories, gw_)
                if not snap.empty:
                    r = bt.evaluate(snap)
                    r["gw"] = gw_
                    results.append(r)
                progress.progress((i + 1) / len(test_gws))
            if results:
                res = pd.DataFrame(results)
                res.to_csv("backtest_results.csv", index=False)
                st.success(f"Backtested {len(results)} gameweek(s), saved backtest_results.csv")
            else:
                st.warning("No gameweeks had enough data to backtest.")

    if os.path.exists("backtest_results.csv"):
        res = pd.read_csv("backtest_results.csv")
        st.write(f"**{len(res)} gameweek(s) backtested** — more is better, "
                 "don't tune the model on fewer than a handful.")

        # Gamified framing of the same underlying numbers below — "beat a
        # random XI in 6 of 8 weeks" lands as a track record; the raw
        # Spearman/top11 columns underneath still exist for anyone who
        # wants the actual stats instead of the story.
        if {"top11_shrunk", "random_11"} <= set(res.columns):
            beat = int((res["top11_shrunk"] > res["random_11"]).sum())
            st.success(f"🏆 Your model's picks would have beaten a random XI "
                       f"in {beat} of {len(res)} backtested gameweek(s).")

        st.write("**Rank correlation (Spearman)** — higher is better, max 1.0")
        st.line_chart(res.set_index("gw")[["corr_shrunk", "corr_raw", "corr_naive_ppg"]])

        st.write("**Top-11 average points**, vs baselines")
        st.line_chart(res.set_index("gw")[
            ["top11_shrunk", "top11_raw", "top11_ppg", "random_11", "best_possible"]
        ])

        if "predicted_team_total" in res.columns:
            st.write("**Predicted vs actual team total** — same 'starting XI + captain "
                     "double' unit the Average per gameweek metric uses, so this is the "
                     "direct check on whether that number runs high or low.")
            st.line_chart(res.set_index("gw")[["predicted_team_total", "actual_team_total"]])
            avg_pred = res["predicted_team_total"].mean()
            avg_actual = res["actual_team_total"].mean()
            bias_pct = 100 * (avg_pred - avg_actual) / avg_actual if avg_actual else 0.0
            direction = "over" if bias_pct > 0 else "under"
            st.caption(f"Predicted team totals run **{abs(bias_pct):.1f}% {direction}** actual, "
                       f"averaged across {len(res)} gameweek(s)."
                       + (" Too few gameweeks to call this a confirmed bias rather than "
                          "noise — treat as provisional." if len(res) < 5 else ""))
            close = int((res["predicted_team_total"] - res["actual_team_total"]).abs().le(5).sum())
            st.caption(f"Landed within 5 points of the real outcome in {close} of "
                       f"{len(res)} gameweek(s).")

        st.write("**Averages across all tested gameweeks**")
        cols = ["corr_shrunk", "corr_raw", "corr_naive_ppg",
                "top11_shrunk", "top11_raw", "top11_ppg", "random_11", "best_possible"]
        if "predicted_team_total" in res.columns:
            cols += ["predicted_team_total", "actual_team_total"]
        st.dataframe(res[cols].mean().round(3).to_frame("mean"), use_container_width=True)

        st.dataframe(res, hide_index=True, use_container_width=True)
    else:
        st.caption("No backtest_results.csv yet — click **Run backtest now**.")


# ----------------------------------------------------------------------------
# Main page — one click SOLVES everything (transfers, weekly plan, chips)
# and stores it in st.session_state; rendering then reads from that state
# on every rerun, including reruns triggered by the lightweight widgets
# below the solve (what-if slider, pitch view) that don't need to re-solve.
# ----------------------------------------------------------------------------

st.subheader("Squad, transfers and starting XI")
log_chip_reading = st.checkbox(
    "Also log this week's chip readings to chip_log.csv", value=True,
    help="Chip values below are a byproduct of this same run (same squad, "
         "same bank) — don't act on one week's numbers alone, log a few "
         "readings and look for a trend.",
)

if st.button("Run optimiser", type="primary"):
    apply_config()
    df, gw = get_data(horizon_weeks)
    horizon = horizon_of(df)
    current_ids, bank_t = get_current_squad_and_bank(df, gw)
    fixtures = cached_fixtures()
    fx_labels_wk0 = fixture_labels_for_week(df, fixtures, gw, 0)

    with st.spinner("Solving Stage A (full multi-week squad + transfer plan)..."):
        prob, squad, start, cap, hits, ft, tin, tout, cost_t, budget_t = opt.build_problem(
            df, current_ids, bank_t
        )
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        st.error(f"Solver returned {status} — your budget or transfer cap "
                  "probably makes a legal squad impossible.")
        st.stop()

    chosen = [p for p in df["id"] if squad[0][p].value() > 0.5]
    xi, captain = opt.choose_lineup(df, chosen)
    info = df.set_index("id")
    plan_obj = pulp.value(prob.objective)

    current = set(current_ids)
    out_ids = sorted(current - set(chosen), key=lambda p: -info.loc[p, "xpts"])
    in_ids = sorted(set(chosen) - current, key=lambda p: -info.loc[p, "xpts"])
    n_hits = int(round(hits[0].value()))

    with st.spinner("Evaluating chips (bench boost / triple captain / wildcard / free hit)..."):
        bb = chips.bench_boost_value(df, current_ids)
        tc = chips.triple_captain_value(df, current_ids)
        wc_detail = chips.wildcard_detail(df, current_ids, bank_t)
        fh_detail = chips.free_hit_detail(df, current_ids, bank_t)
    wc = wc_detail["gain"]
    fh = fh_detail["gain"]
    wildcard_gap = wc_detail["objective"] - plan_obj

    chip_score_data = chips.chip_scores(bb, tc, wc, fh)
    outlook = chips.season_outlook(df, current_ids, fixtures, gw, horizon)

    week_data = {}
    for w in range(horizon):
        if w == 0:
            squad_w, xi_w, cap_w = chosen, xi, captain
        else:
            squad_w = [p for p in df["id"] if squad[w][p].value() > 0.5]
            xi_w = [p for p in squad_w if start[w][p].value() > 0.5]
            cap_w = next(p for p in squad_w if cap[w][p].value() > 0.5)
        week_data[w] = (squad_w, xi_w, cap_w)

    weekly_pts = []
    for w in range(horizon):
        squad_w, xi_w, cap_w = week_data[w]
        bench_w = [p for p in squad_w if p not in xi_w]
        weekly_pts.append(
            opt.simulate_autosub_expected_points(df, xi_w, bench_w, cap_w, f"xw{w}")
        )

    # Per-week transfer/hit bookkeeping extracted into plain values now
    # (not left as bare pulp variable objects to read later) — cheap here,
    # and keeps rendering below simple regardless of how it's re-triggered.
    week_transfer_info = {}
    prev_squad, prev_xi = None, None
    for w in range(horizon):
        squad_w, xi_w, cap_w = week_data[w]
        entry = {}
        if prev_squad is not None:
            transferred_in = set(squad_w) - set(prev_squad)
            transferred_out = set(prev_squad) - set(squad_w)
            entry["transferred_in"] = transferred_in
            entry["transferred_out"] = transferred_out
            if transferred_in or transferred_out:
                entry["n_hits_w"] = int(round(hits[w].value()))
                entry["n_free_w"] = opt.FREE_TRANSFERS if w == 0 else int(round(ft[w].value()))
            entry["rotated_in"] = (set(xi_w) - set(prev_xi)) - transferred_in
            entry["rotated_out"] = (set(prev_xi) - set(xi_w)) - transferred_out
        week_transfer_info[w] = entry
        prev_squad, prev_xi = squad_w, xi_w

    in_names = [info.loc[p, "name"] for p in in_ids]
    out_names = [info.loc[p, "name"] for p in out_ids]
    last_run = load_last_run()
    save_run_snapshot(gw, in_names, out_names, plan_obj)

    if log_chip_reading:
        chips.log_row(gw, bb, tc, wc, fh, horizon)

    st.session_state["results"] = dict(
        df=df, gw=gw, horizon=horizon, horizon_weeks=horizon_weeks,
        current_ids=current_ids, bank_t=bank_t, fixtures=fixtures,
        fx_labels_wk0=fx_labels_wk0, chosen=chosen, xi=xi, captain=captain,
        info=info, plan_obj=plan_obj, out_ids=out_ids, in_ids=in_ids,
        n_hits=n_hits, bb=bb, tc=tc, wc=wc, wc_detail=wc_detail, fh=fh,
        fh_detail=fh_detail, wildcard_gap=wildcard_gap,
        chip_score_data=chip_score_data, outlook=outlook, week_data=week_data,
        weekly_pts=weekly_pts, week_transfer_info=week_transfer_info,
        cost_t=cost_t, budget_t=budget_t, log_chip_reading=log_chip_reading,
        last_run=last_run, in_names=in_names, out_names=out_names,
    )

if "results" in st.session_state:
    s = st.session_state["results"]
    info, gw, horizon = s["info"], s["gw"], s["horizon"]

    obj_m, avg_m = st.columns(2)
    obj_m.metric("Squad objective (horizon expected points)", f"{s['plan_obj']:.2f}")
    avg_m.metric("Average per gameweek",
                  f"{s['plan_obj'] / horizon:.2f}",
                  help="Objective ÷ horizon length — the total on its own always "
                       "grows with a longer horizon, so it isn't a fair way to "
                       "compare two runs with different horizons. This is: "
                       "run it at 5 weeks, note this number, run it again at 8, "
                       "and compare THIS instead.")
    if s["horizon"] != s["horizon_weeks"]:
        st.caption(f"Solved over {horizon} week(s) (slider is now set to "
                    f"{horizon_weeks}, click **Run optimiser** again to match it).")
    cal_factor = info["calibration_factor"].iloc[0] if "calibration_factor" in info.columns else 1.0
    if abs(cal_factor - 1.0) > 1e-6:
        st.caption(f"📐 Calibration: all predictions above are scaled ×{cal_factor:.3f}, "
                   f"learned from backtest.py's own predicted-vs-actual track record "
                   f"(Model Health in the sidebar has the detail and the gameweek count "
                   f"it's based on — treat it as provisional until that grows).")

    if not s["in_ids"]:
        st.info("Recommendation: no transfer. Roll it.")
    else:
        st.success(f"Recommendation: {len(s['in_ids'])} transfer(s), {s['n_hits']} hit(s) "
                    f"= -{s['n_hits'] * int(opt.HIT_COST)} pts")
        c1, c2 = st.columns(2)
        with c1:
            st.write("**OUT**")
            st.dataframe(pd.DataFrame([player_row(info, p, fx_labels=s["fx_labels_wk0"])
                                        for p in s["out_ids"]]),
                          hide_index=True, use_container_width=True)
        with c2:
            st.write("**IN**")
            st.dataframe(pd.DataFrame([player_row(info, p, fx_labels=s["fx_labels_wk0"])
                                        for p in s["in_ids"]]),
                          hide_index=True, use_container_width=True)

        with st.expander("Why these transfers? (plain-language reasons)"):
            for p in s["out_ids"]:
                st.caption(f"OUT **{info.loc[p, 'name']}** — {transfer_reason_out(info, p, s['fx_labels_wk0'])}")
            for p in s["in_ids"]:
                st.caption(f"IN **{info.loc[p, 'name']}** — {transfer_reason_in(info, p, s['fx_labels_wk0'])}")
            st.caption("Built entirely from the same columns the model already computes "
                       "(fixture difficulty, minutes share, form vs season average, "
                       "set-piece duty, fitness flags) — a translation of the number, "
                       "not a second opinion on it. OUT only lists actual red flags — a "
                       "sale with none isn't a bad player, just outscored for the budget.")

    n_used_now = len(s["in_ids"])
    n_banked = opt.FREE_TRANSFERS - n_used_now
    if horizon > 1 and n_banked > 0:
        st.caption(f"Using {n_used_now} of your {opt.FREE_TRANSFERS} free transfer(s) "
                    f"this week — banking {n_banked} for later (see the week-by-week "
                    f"plan below for when it thinks that pays off). This is the model's "
                    f"own decision now, not a hand-tuned setting.")

    with st.expander("🕓 Diff vs your last saved run"):
        render_run_diff(s["last_run"], gw, s["in_names"], s["out_names"], s["plan_obj"])

    st.write("**This week's XI — pitch view**")
    xi0, cap0 = s["week_data"][0][1], s["week_data"][0][2]
    bench0 = [p for p in s["week_data"][0][0] if p not in xi0]
    render_pitch(info, xi0, bench0, cap0, "xw0")

    with st.expander("🔧 What if? test a fitness/rotation assumption"):
        st.caption("Rescales a single player's own xPts_gw1 by their assumed minutes "
                   "share, using the same heuristic build_table() already applies — "
                   "NOT a re-solve of the optimiser (the squad/XI above won't change), "
                   "just a quick 'how sensitive is this number' check.")
        squad_options = sorted(s["chosen"], key=lambda p: info.loc[p, "name"])
        pick = st.selectbox(
            "Player", squad_options,
            format_func=lambda p: f"{info.loc[p, 'name']} ({info.loc[p, 'pos']})",
        )
        if pick is not None:
            row = info.loc[pick]
            orig_share = row.get("recent_mins_share", row.get("mins_share"))
            orig_share = float(orig_share) if pd.notna(orig_share) else float(row["mins_share"])
            new_share = st.slider("Assumed minutes share", 0, 100,
                                    int(round(orig_share * 100)), step=5) / 100.0
            orig_xpts = float(row["xw0"])
            scale = (new_share / orig_share) if orig_share > 0 else 0.0
            new_xpts = orig_xpts * scale
            d1, d2, d3 = st.columns(3)
            d1.metric("Current assumption", f"{orig_share:.0%} mins")
            d2.metric("xPts_gw1 at current", f"{orig_xpts:.2f}")
            d3.metric(f"xPts_gw1 at {new_share:.0%}", f"{new_xpts:.2f}",
                       delta=f"{new_xpts - orig_xpts:+.2f}")

    st.write("**Chip strategy this week**")
    bb, tc, wc, fh = s["bb"], s["tc"], s["wc"], s["fh"]
    wc_detail, fh_detail = s["wc_detail"], s["fh_detail"]
    best_bb_w, best_tc_w = max(bb, key=bb.get), max(tc, key=tc.get)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bench Boost (this week)", f"+{bb[0]:.2f} pts")
    c2.metric("Triple Captain (this week)", f"+{tc[0]:.2f} pts")
    c3.metric(f"Wildcard (over {horizon} GWs)", f"+{wc:.2f} pts")
    c4.metric("Free Hit (this week)", f"+{fh:.2f} pts")
    st.caption(f"Best week to Bench Boost: GW{gw + best_bb_w} (+{bb[best_bb_w]:.2f} pts). "
                f"Best week to Triple Captain: GW{gw + best_tc_w} (+{tc[best_tc_w]:.2f} pts). "
                "Values are for your CURRENT squad, before the transfer above is applied.")

    if s["wildcard_gap"] > opt.HIT_COST * 2:
        st.warning(
            f"⚠️ A full Wildcard rebuild right now would be worth "
            f"{wc_detail['objective']:.2f} pts over the horizon, vs {s['plan_obj']:.2f} pts for "
            f"the transfer plan above — **+{s['wildcard_gap']:.2f} more**, since a wildcard has no "
            f"per-week transfer cap and no hit cost. Wildcard is structurally unconstrained "
            f"so it usually beats a capped plan by some margin — that alone isn't a reason "
            f"to use it, but this gap is unusually large. Worth weighing against the "
            f"Wildcard score below (scored against your own logged history) before deciding."
        )

    chip_score_data = s["chip_score_data"]
    st.write("**How good is it to use each chip THIS WEEK? (0-10)**")
    s1, s2, s3, s4 = st.columns(4)
    for col, key in zip([s1, s2, s3, s4],
                         ["bench_boost", "triple_captain", "wildcard", "free_hit"]):
        entry = chip_score_data[key]
        col.metric(key.replace("_", " ").title(),
                    f"{entry['score']}/10" if entry["score"] is not None else "n/a")
        col.caption(entry["verdict"])

    outlook = s["outlook"]
    if outlook and not outlook["within_horizon"] and outlook["best_gw_fixtures"] > outlook["this_week_fixtures"]:
        st.info(f"📅 Beyond your {horizon}-week horizon: GW{outlook['best_gw']} currently has "
                f"{outlook['best_gw_fixtures']} fixtures across your squad's teams, vs "
                f"{outlook['this_week_fixtures']} this week — worth extending the horizon "
                f"slider to see it properly before committing a chip. Based on your CURRENT "
                f"squad's teams and FPL's currently confirmed fixture list, which will change "
                f"as you transfer and as later fixtures get scheduled — a heads-up, not a plan.")

    if s["log_chip_reading"]:
        st.caption(f"Logged GW{gw}-GW{gw + horizon - 1} chip readings to {chips.LOG_PATH}.")

    with st.expander("Chip detail — week-by-week values and the Free Hit squad"):
        weeks = [f"GW{gw + w}" for w in range(horizon)]
        chart_df = pd.DataFrame({
            "GW": weeks,
            "Bench Boost": [bb[w] for w in range(horizon)],
            "Triple Captain": [tc[w] for w in range(horizon)],
        })
        long_df = chart_df.melt("GW", var_name="Chip", value_name="Points")
        chip_chart = alt.Chart(long_df).mark_bar().encode(
            x=alt.X("GW:N", sort=weeks, title=None),
            y=alt.Y("Points:Q"),
            color="Chip:N",
        )
        st.altair_chart(chip_chart, use_container_width=True)

        st.write("**Free Hit squad — what Stage A would actually build this week**")
        fh_xi, fh_cap = fh_detail["xi"], fh_detail["captain"]
        fh_bench = [p for p in fh_detail["squad"] if p not in fh_xi]
        order = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
        fh_xi_sorted = sorted(fh_xi, key=lambda p: (order[info.loc[p, "pos"]], -info.loc[p, "xw0"]))
        st.write("XI")
        st.dataframe(
            pd.DataFrame([player_row(info, p, "xw0", "(C)" if p == fh_cap else "", s["fx_labels_wk0"])
                          for p in fh_xi_sorted]),
            hide_index=True, use_container_width=True,
        )
        st.write("Bench")
        fh_bench_sorted = sorted(fh_bench, key=lambda p: -info.loc[p, "xw0"])
        st.dataframe(
            pd.DataFrame([player_row(info, p, "xw0", fx_labels=s["fx_labels_wk0"]) for p in fh_bench_sorted]),
            hide_index=True, use_container_width=True,
        )

    st.write("**Expected points per gameweek**")
    st.caption("Autosub-adjusted expected points each week (a blanked starter is "
                "covered by the right bench player, same as FPL's own scoring), "
                "captain counted twice. Weeks 1+ assume the model's own planned "
                "transfers/rotation happen.")
    gw_labels = [f"GW{gw + w}" for w in range(horizon)]
    pts_df = pd.DataFrame({"GW": gw_labels, "Expected points": s["weekly_pts"]})
    chart = alt.Chart(pts_df).mark_bar().encode(
        x=alt.X("GW:N", sort=gw_labels, title=None),
        y=alt.Y("Expected points:Q"),
    )
    st.altair_chart(chart, use_container_width=True)

    st.write(f"**Plan — all {horizon} week(s) of the horizon**")
    st.caption("Each tab reflects the solver's own plan for that week, including any "
                "further transfers it wants to make. Only THIS WEEK's transfer (above) "
                "is real — re-run fresh next week once actual news comes in.")

    order = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
    week_tabs = st.tabs([f"GW{gw + w}" for w in range(horizon)])
    for w, wtab in enumerate(week_tabs):
        with wtab:
            xpts_col = f"xw{w}"
            squad_w, xi_w, cap_w = s["week_data"][w]
            bench_w = [p for p in squad_w if p not in xi_w]
            entry = s["week_transfer_info"][w]

            if entry:
                transferred_in = entry.get("transferred_in", set())
                transferred_out = entry.get("transferred_out", set())
                if transferred_in or transferred_out:
                    in_names_w = ", ".join(info.loc[p, "name"] for p in transferred_in) or "—"
                    out_names_w = ", ".join(info.loc[p, "name"] for p in transferred_out) or "—"
                    st.caption(f"📋 Planned transfer from GW{gw + w - 1}: "
                                f"IN {in_names_w}  |  OUT {out_names_w}")
                    n_hits_w = entry.get("n_hits_w", 0)
                    n_free_w = entry.get("n_free_w", 0)
                    if n_hits_w > 0:
                        st.caption(f"⚠️ {len(transferred_out)} transfer(s) this week, only "
                                    f"{n_free_w} free — {n_hits_w} hit(s) = "
                                    f"-{n_hits_w * int(opt.HIT_COST)} pts")
                    else:
                        st.caption(f"{len(transferred_out)} transfer(s), all free "
                                    f"({n_free_w} available this week) — no hit.")
                rotated_in = entry.get("rotated_in", set())
                rotated_out = entry.get("rotated_out", set())
                if rotated_in or rotated_out:
                    in_names_w = ", ".join(info.loc[p, "name"] for p in rotated_in) or "—"
                    out_names_w = ", ".join(info.loc[p, "name"] for p in rotated_out) or "—"
                    st.caption(f"Lineup change from GW{gw + w - 1}: IN {in_names_w}  |  OUT {out_names_w}")
                elif not (transferred_in or transferred_out):
                    st.caption("No change from the previous week.")

            fx_labels_w = fixture_labels_for_week(s["df"], s["fixtures"], gw, w)
            xi_sorted = sorted(xi_w, key=lambda p: (order[info.loc[p, "pos"]], -info.loc[p, xpts_col]))
            st.dataframe(
                pd.DataFrame([player_row(info, p, xpts_col, "(C)" if p == cap_w else "", fx_labels_w)
                              for p in xi_sorted]),
                hide_index=True, use_container_width=True,
            )
            st.write("Bench")
            bench_sorted = sorted(bench_w, key=lambda p: -info.loc[p, xpts_col])
            st.dataframe(
                pd.DataFrame([player_row(info, p, xpts_col, fx_labels=fx_labels_w) for p in bench_sorted]),
                hide_index=True, use_container_width=True,
            )

    spend = sum(s["cost_t"][p] for p in s["chosen"])
    st.caption(f"Squad cost £{spend / 10:.1f}m of £{s['budget_t'] / 10:.1f}m available "
                f"(£{(s['budget_t'] - spend) / 10:.1f}m left in the bank)")
else:
    st.caption("Set your config in the sidebar, then click **Run optimiser**.")
