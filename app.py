"""
FPL Optimiser — local dashboard.

Wraps fpl_stage0 / fpl_optimise / chips / backtest in a Streamlit UI instead
of terminal output. Nothing about the underlying logic changes — this file
only calls the same tested functions and renders their results as tables
and charts. If you ever suspect a number here, the *_optimise.py /
chips.py / backtest.py scripts remain the source of truth and still run
standalone exactly as before.

Run:  pip install streamlit
      streamlit run app.py
Opens in your browser at http://localhost:8501, on your machine only —
nothing is uploaded or hosted anywhere.
"""

import pandas as pd
import pulp
import streamlit as st

import chips
import fpl_optimise as opt
from fpl_stage0 import HORIZON, build_table, horizon_of

st.set_page_config(page_title="FPL Optimiser", layout="wide")


# ----------------------------------------------------------------------------
# Cached data pull — the FPL API call is the slow part, so we cache it for
# the session and only re-fetch when the sidebar button is pressed.
# ----------------------------------------------------------------------------

@st.cache_data(show_spinner="Pulling FPL data...")
def cached_build_table(horizon):
    return build_table(horizon=horizon)


def get_data(horizon, force_refresh=False):
    if force_refresh:
        cached_build_table.clear()
    return cached_build_table(horizon)


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
bank = st.sidebar.number_input("Bank (£m)", value=float(opt.BANK), step=0.1, format="%.1f")
free_transfers = st.sidebar.number_input(
    "Free transfers", value=int(opt.FREE_TRANSFERS), min_value=0, max_value=15, step=1
)
max_transfers = st.sidebar.number_input(
    "Max transfers in any single week", value=int(opt.MAX_TRANSFERS), min_value=0, max_value=15, step=1,
    help="A per-week ceiling, not the lever for 'how big a rehaul' — the "
         "model itself now decides how many of your available transfers to "
         "actually spend each week, and whether to bank some for later. "
         "This just stops it proposing an unrealistic same-week rebuild; "
         "the Chip Advisor tab already evaluates a true wildcard/free hit "
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

if st.sidebar.button("Refresh FPL data", help="Re-pulls bootstrap-static/fixtures from the API."):
    get_data(horizon_weeks, force_refresh=True)
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
    """Manual squad box wins if filled in — bank then comes from the
    sidebar field, since there's no live account to read it from. Live
    squad fetch also pulls the REAL bank FPL tracks for that team
    (entry_history.bank) and uses that in place of the sidebar's Bank
    field, so it self-updates every week (past transfers, real price
    moves) instead of relying on you to type in the right number each
    time you re-run this."""
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
    if len(ids) != 15:
        st.error(f"Expected 15 players, got {len(ids)}.")
        st.stop()
    return ids, bank_t


def player_row(info, p, xpts_col="xpts", tag=""):
    r = info.loc[p]
    return {
        "Player": r["name"], "Pos": r["pos"], "Team": r["team_name"],
        "Price": f"£{r['price']:.1f}", "xPts": round(r[xpts_col], 2), "": tag,
    }


# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------

tab_transfers, tab_chips, tab_backtest = st.tabs(
    ["Transfers & Lineup", "Chip Advisor", "Backtest trends"]
)

# ---- Tab 1: Transfers & Lineup ---------------------------------------------
with tab_transfers:
    st.subheader("Squad, transfers and starting XI")
    if st.button("Run optimiser", type="primary"):
        apply_config()
        df, gw = get_data(horizon_weeks)
        horizon = horizon_of(df)
        current_ids, bank_t = get_current_squad_and_bank(df, gw)

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

        # Only week 0 is real — everything from week 1 on is the model's own
        # best guess at what it WOULD do next, shown below so you can see
        # WHY it's making week 0's decision the way it is, not a commitment.
        chosen = [p for p in df["id"] if squad[0][p].value() > 0.5]
        xi, captain = opt.choose_lineup(df, chosen)
        info = df.set_index("id")

        st.metric("Squad objective (horizon expected points)",
                   f"{pulp.value(prob.objective):.2f}")
        st.caption(f"Solved over {horizon} week(s) (slider was set to {horizon_weeks}). "
                    "If you change the slider, you must click **Run optimiser** again — "
                    "moving the slider alone doesn't re-solve anything.")

        current = set(current_ids)
        out_ids = sorted(current - set(chosen), key=lambda p: -info.loc[p, "xpts"])
        in_ids = sorted(set(chosen) - current, key=lambda p: -info.loc[p, "xpts"])

        if not in_ids:
            st.info("Recommendation: no transfer. Roll it.")
        else:
            n_hits = int(round(hits[0].value()))
            st.success(f"Recommendation: {len(in_ids)} transfer(s), {n_hits} hit(s) "
                        f"= -{n_hits * int(opt.HIT_COST)} pts")
            c1, c2 = st.columns(2)
            with c1:
                st.write("**OUT**")
                st.dataframe(pd.DataFrame([player_row(info, p) for p in out_ids]),
                              hide_index=True, use_container_width=True)
            with c2:
                st.write("**IN**")
                st.dataframe(pd.DataFrame([player_row(info, p) for p in in_ids]),
                              hide_index=True, use_container_width=True)

        n_used_now = len(in_ids)
        n_banked = opt.FREE_TRANSFERS - n_used_now
        if horizon > 1 and n_banked > 0:
            st.caption(f"Using {n_used_now} of your {opt.FREE_TRANSFERS} free transfer(s) "
                        f"this week — banking {n_banked} for later (see the week-by-week "
                        f"plan below for when it thinks that pays off). This is the model's "
                        f"own decision now, not a hand-tuned setting.")

        # --- Per-week squad/XI/captain, solved once and reused below for
        # both the expected-points chart and the per-week tabs — avoids
        # reading the same pulp variables twice and risking the two views
        # drifting apart.
        week_data = {}
        for w in range(horizon):
            if w == 0:
                squad_w, xi_w, cap_w = chosen, xi, captain   # already solved above
            else:
                squad_w = [p for p in df["id"] if squad[w][p].value() > 0.5]
                xi_w = [p for p in squad_w if start[w][p].value() > 0.5]
                cap_w = next(p for p in squad_w if cap[w][p].value() > 0.5)
            week_data[w] = (squad_w, xi_w, cap_w)

        # --- Expected points per gameweek -----------------------------------
        # Starting XI's xPts for that week, with the captain's counted twice
        # (the actual scoring rule) — i.e. what the plan expects to SCORE
        # each week, not the horizon-wide objective value shown above.
        weekly_pts = []
        for w in range(horizon):
            squad_w, xi_w, cap_w = week_data[w]
            xpts_col = f"xw{w}"
            weekly_pts.append(
                sum(info.loc[p, xpts_col] for p in xi_w) + info.loc[cap_w, xpts_col]
            )
        st.write("**Expected points per gameweek**")
        st.caption("Starting XI's expected points each week, captain counted twice. "
                    "Weeks 1+ assume the model's own planned transfers/rotation happen.")
        gw_labels = [f"GW{gw + w}" for w in range(horizon)]
        pts_df = pd.DataFrame({"GW": gw_labels, "Expected points": weekly_pts})
        # st.bar_chart sorts string categories alphabetically (GW10 would
        # sort before GW6) — an explicit Altair chart with sort=gw_labels
        # keeps the weeks in the actual chronological order instead.
        import altair as alt
        chart = alt.Chart(pts_df).mark_bar().encode(
            x=alt.X("GW:N", sort=gw_labels, title=None),
            y=alt.Y("Expected points:Q"),
        )
        st.altair_chart(chart, use_container_width=True)

        # --- Multi-week plan ------------------------------------------------
        # Weeks 1+ now come straight from Stage A's own multi-period solve
        # (squad[w]/start[w]/cap[w]) instead of a separate re-solve against
        # a FIXED squad — this is what actually shows the model's planned
        # transfers, not just bench/XI rotation within one unchanging 15.
        # Only week 0 is real (see build_problem's docstring); everything
        # from week 1 on is "what it would do given no new information",
        # shown so you can see WHY it made week 0's call, and WHEN it's
        # planning to use the rest of your transfers.
        st.write(f"**Plan — all {horizon} week(s) of the horizon**")
        st.caption("Each tab reflects the solver's own plan for that week, including any "
                    "further transfers it wants to make. Only THIS WEEK's transfer (above) "
                    "is real — re-run fresh next week once actual news comes in.")

        order = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
        week_tabs = st.tabs([f"GW{gw + w}" for w in range(horizon)])
        prev_squad, prev_xi = None, None
        for w, wtab in enumerate(week_tabs):
            with wtab:
                xpts_col = f"xw{w}"
                squad_w, xi_w, cap_w = week_data[w]
                bench_w = [p for p in squad_w if p not in xi_w]

                if prev_squad is not None:
                    transferred_in = set(squad_w) - set(prev_squad)
                    transferred_out = set(prev_squad) - set(squad_w)
                    if transferred_in or transferred_out:
                        in_names = ", ".join(info.loc[p, "name"] for p in transferred_in) or "—"
                        out_names = ", ".join(info.loc[p, "name"] for p in transferred_out) or "—"
                        st.caption(f"📋 Planned transfer from GW{gw + w - 1}: "
                                    f"IN {in_names}  |  OUT {out_names}")
                    rotated_in = (set(xi_w) - set(prev_xi)) - transferred_in
                    rotated_out = (set(prev_xi) - set(xi_w)) - transferred_out
                    if rotated_in or rotated_out:
                        in_names = ", ".join(info.loc[p, "name"] for p in rotated_in) or "—"
                        out_names = ", ".join(info.loc[p, "name"] for p in rotated_out) or "—"
                        st.caption(f"Lineup change from GW{gw + w - 1}: IN {in_names}  |  OUT {out_names}")
                    elif not (transferred_in or transferred_out):
                        st.caption("No change from the previous week.")
                prev_squad, prev_xi = squad_w, xi_w

                xi_sorted = sorted(xi_w, key=lambda p: (order[info.loc[p, "pos"]], -info.loc[p, xpts_col]))
                st.dataframe(
                    pd.DataFrame([player_row(info, p, xpts_col, "(C)" if p == cap_w else "")
                                  for p in xi_sorted]),
                    hide_index=True, use_container_width=True,
                )
                st.write("Bench")
                bench_sorted = sorted(bench_w, key=lambda p: -info.loc[p, xpts_col])
                st.dataframe(
                    pd.DataFrame([player_row(info, p, xpts_col) for p in bench_sorted]),
                    hide_index=True, use_container_width=True,
                )

        spend = sum(cost_t[p] for p in chosen)
        st.caption(f"Squad cost £{spend / 10:.1f}m of £{budget_t / 10:.1f}m available "
                    f"(£{(budget_t - spend) / 10:.1f}m left in the bank)")
    else:
        st.caption("Set your config in the sidebar, then click **Run optimiser**.")

# ---- Tab 2: Chip Advisor ----------------------------------------------------
with tab_chips:
    st.subheader("Chip strategy values")
    st.caption("Values for your CURRENT squad, before any transfer above is applied. "
                "Don't act on one week's numbers alone — log a few readings and look for a trend.")
    log_it = st.checkbox("Log this reading to chip_log.csv", value=True)

    if st.button("Run chip advisor", type="primary"):
        apply_config()
        df, gw = get_data(horizon_weeks)
        horizon = horizon_of(df)
        current_ids, bank_t = get_current_squad_and_bank(df, gw)
        info = df.set_index("id")

        with st.spinner("Evaluating bench boost / triple captain..."):
            bb = chips.bench_boost_value(df, current_ids)
            tc = chips.triple_captain_value(df, current_ids)
        with st.spinner("Evaluating wildcard / free hit (bigger solves)..."):
            wc = chips.wildcard_value(df, current_ids, bank_t)
            fh_detail = chips.free_hit_detail(df, current_ids, bank_t)
        fh = fh_detail["gain"]

        st.caption(f"Solved over {horizon} week(s) (slider was set to {horizon_weeks}).")
        weeks = [f"GW{gw + w}" for w in range(horizon)]
        chart_df = pd.DataFrame({
            "GW": weeks,
            "Bench Boost": [bb[w] for w in range(horizon)],
            "Triple Captain": [tc[w] for w in range(horizon)],
        })
        st.write("**Bench Boost & Triple Captain — value by week**")
        # Same alphabetical-sort issue as the Expected Points chart (GW10
        # would sort before GW6) — melt to long form and pin x-axis order
        # explicitly via sort=weeks.
        import altair as alt
        long_df = chart_df.melt("GW", var_name="Chip", value_name="Points")
        chip_chart = alt.Chart(long_df).mark_bar().encode(
            x=alt.X("GW:N", sort=weeks, title=None),
            y=alt.Y("Points:Q"),
            color="Chip:N",
        )
        st.altair_chart(chip_chart, use_container_width=True)

        c1, c2 = st.columns(2)
        c1.metric(f"Wildcard (over {horizon} GWs)", f"+{wc:.2f} pts")
        c2.metric(f"Free Hit (GW{gw} only)", f"+{fh:.2f} pts")

        best_bb_w = max(bb, key=bb.get)
        best_tc_w = max(tc, key=tc.get)
        st.caption(f"Best week to Bench Boost: GW{gw + best_bb_w} (+{bb[best_bb_w]:.2f} pts). "
                    f"Best week to Triple Captain: GW{gw + best_tc_w} (+{tc[best_tc_w]:.2f} pts).")

        with st.expander("Free Hit squad — what Stage A would actually build this week"):
            xi, cap_p = fh_detail["xi"], fh_detail["captain"]
            bench = [p for p in fh_detail["squad"] if p not in xi]
            order = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
            xi_sorted = sorted(xi, key=lambda p: (order[info.loc[p, "pos"]], -info.loc[p, "xw0"]))
            st.write("**XI**")
            st.dataframe(
                pd.DataFrame([player_row(info, p, "xw0", "(C)" if p == cap_p else "") for p in xi_sorted]),
                hide_index=True, use_container_width=True,
            )
            st.write("**Bench**")
            bench_sorted = sorted(bench, key=lambda p: -info.loc[p, "xw0"])
            st.dataframe(
                pd.DataFrame([player_row(info, p, "xw0") for p in bench_sorted]),
                hide_index=True, use_container_width=True,
            )

        if log_it:
            chips.log_row(gw, bb[0], tc[0], wc, fh)
            st.success(f"Logged GW{gw} readings to {chips.LOG_PATH}.")
    else:
        st.caption("Click **Run chip advisor** to evaluate all four chips for your current squad.")

# ---- Tab 3: Backtest trends --------------------------------------------------
with tab_backtest:
    st.subheader("Prediction-quality backtest")
    st.caption("Does xpts_gw1 actually correlate with what players go on to score? "
                "See backtest.py's docstring for what this does and doesn't cover.")

    import os
    import backtest as bt

    run_now = st.button("Run backtest now (re-checks every finished gameweek)")

    if run_now:
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
            for i, gw in enumerate(test_gws):
                snap = bt.snapshot_as_of(boot, fixtures, histories, gw)
                if not snap.empty:
                    r = bt.evaluate(snap)
                    r["gw"] = gw
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

        st.write("**Rank correlation (Spearman) — higher is better, max 1.0**")
        st.line_chart(res.set_index("gw")[["corr_shrunk", "corr_raw", "corr_naive_ppg"]])

        st.write("**Top-11 average points achieved, vs baselines**")
        st.line_chart(res.set_index("gw")[
            ["top11_shrunk", "top11_raw", "top11_ppg", "random_11", "best_possible"]
        ])

        st.write("**Averages across all tested gameweeks**")
        cols = ["corr_shrunk", "corr_raw", "corr_naive_ppg",
                "top11_shrunk", "top11_raw", "top11_ppg", "random_11", "best_possible"]
        st.dataframe(res[cols].mean().round(3).to_frame("mean"), use_container_width=True)

        st.dataframe(res, hide_index=True, use_container_width=True)
    else:
        st.caption("No backtest_results.csv yet — click **Run backtest now**.")
