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
from fpl_stage0 import HORIZON, build_table

st.set_page_config(page_title="FPL Optimiser", layout="wide")


# ----------------------------------------------------------------------------
# Cached data pull — the FPL API call is the slow part, so we cache it for
# the session and only re-fetch when the sidebar button is pressed.
# ----------------------------------------------------------------------------

@st.cache_data(show_spinner="Pulling FPL data...")
def cached_build_table():
    return build_table()


def get_data(force_refresh=False):
    if force_refresh:
        cached_build_table.clear()
    return cached_build_table()


# ----------------------------------------------------------------------------
# Sidebar — the same config block fpl_optimise.py has at the top of the
# file, just editable per-run instead of edited-and-saved. Applying it sets
# the actual module attributes other functions read, so nothing downstream
# needs to know it's running inside a dashboard.
# ----------------------------------------------------------------------------

st.sidebar.header("Config")

team_id = st.sidebar.number_input(
    "Team ID", value=opt.TEAM_ID, step=1,
    help="The number in your FPL team's URL.",
)
bank = st.sidebar.number_input("Bank (£m)", value=opt.BANK, step=0.1, format="%.1f")
free_transfers = st.sidebar.number_input(
    "Free transfers", value=opt.FREE_TRANSFERS, min_value=0, max_value=15, step=1
)
max_transfers = st.sidebar.number_input(
    "Max transfers this run", value=opt.MAX_TRANSFERS, min_value=0, max_value=15, step=1,
    help="Hard cap on the optimiser. Set to 15 to see a wildcard-style rebuild "
         "(the Chip Advisor tab does this for you already, though).",
)
hit_cost = st.sidebar.number_input("Hit cost (pts per extra transfer)", value=opt.HIT_COST, step=0.5)
opportunity_cost = st.sidebar.number_input(
    "Transfer opportunity cost", value=opt.TRANSFER_OPPORTUNITY_COST, step=0.25,
    help="Higher = more conservative; only clearly-worthwhile transfers get made.",
)
bench_weight = st.sidebar.slider("Bench weight", 0.0, 1.0, opt.BENCH_WEIGHT, step=0.05)
ban_unavailable = st.sidebar.checkbox("Don't buy flagged/injured players", value=opt.BAN_UNAVAILABLE)

manual_squad_input = st.sidebar.text_input(
    "Manual squad (15 comma-separated player IDs, optional)",
    help="Leave blank to fetch your live squad from Team ID instead.",
)

if st.sidebar.button("Refresh FPL data", help="Re-pulls bootstrap-static/fixtures from the API."):
    get_data(force_refresh=True)
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
    opt.TRANSFER_OPPORTUNITY_COST = float(opportunity_cost)
    opt.BENCH_WEIGHT = float(bench_weight)
    opt.BAN_UNAVAILABLE = bool(ban_unavailable)


def get_current_squad(df, gw):
    """Manual squad box wins if filled in; otherwise fetch live picks."""
    if manual_squad_input.strip():
        ids = [int(x) for x in manual_squad_input.split(",") if x.strip()]
    else:
        ids = opt.fetch_squad(opt.TEAM_ID, gw)
    if len(ids) != 15:
        st.error(f"Expected 15 players, got {len(ids)}.")
        st.stop()
    return ids


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
        df, gw = get_data()
        current_ids = get_current_squad(df, gw)
        bank_t = int(round(opt.BANK * 10))

        with st.spinner("Solving Stage A (squad/transfers)..."):
            prob, squad, start, cap, hits, cost_t, budget_t = opt.build_problem(
                df, current_ids, bank_t
            )
            prob.solve(pulp.PULP_CBC_CMD(msg=False))
            status = pulp.LpStatus[prob.status]
        if status != "Optimal":
            st.error(f"Solver returned {status} — your budget or transfer cap "
                      "probably makes a legal squad impossible.")
            st.stop()

        chosen = [p for p in df["id"] if squad[p].value() > 0.5]
        xi, captain = opt.choose_lineup(df, chosen)
        info = df.set_index("id")

        st.metric("Squad objective (horizon expected points)",
                   f"{pulp.value(prob.objective):.2f}")

        current = set(current_ids)
        out_ids = sorted(current - set(chosen), key=lambda p: -info.loc[p, "xpts"])
        in_ids = sorted(set(chosen) - current, key=lambda p: -info.loc[p, "xpts"])

        if not in_ids:
            st.info("Recommendation: no transfer. Roll it.")
        else:
            n_hits = int(round(hits.value()))
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

        st.write("**Starting XI** — chosen for next gameweek's fixtures only")
        order = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
        xi_sorted = sorted(xi, key=lambda p: (order[info.loc[p, "pos"]], -info.loc[p, "xpts_gw1"]))
        st.dataframe(
            pd.DataFrame([player_row(info, p, "xpts_gw1", "(C)" if p == captain else "")
                          for p in xi_sorted]),
            hide_index=True, use_container_width=True,
        )

        bench = [p for p in chosen if p not in xi]
        st.write("**Bench**")
        bench_sorted = sorted(bench, key=lambda p: -info.loc[p, "xpts_gw1"])
        st.dataframe(
            pd.DataFrame([player_row(info, p, "xpts_gw1") for p in bench_sorted]),
            hide_index=True, use_container_width=True,
        )

        spend = sum(cost_t[p] for p in chosen)
        st.caption(f"Squad cost £{spend / 10:.1f}m of £{budget_t / 10:.1f}m available "
                    f"(£{(budget_t - spend) / 10:.1f}m left in the bank)")

        with st.expander("Look-ahead — Stage A's internal per-week captain plan"):
            rows = []
            for w in range(1, HORIZON):
                captain_w = next((p for p in chosen if cap[w][p].value() > 0.5), None)
                starters_w = [p for p in chosen if start[w][p].value() > 0.5]
                blanking = [
                    p for p in starters_w
                    if info.loc[p, f"xw{w}"] == 0 and info.loc[p, "mins_share"] > 0.3
                ]
                rows.append({
                    "GW": gw + w,
                    "Captain": info.loc[captain_w, "name"] if captain_w else "?",
                    "Started despite no fixture": ", ".join(info.loc[p, "name"] for p in blanking) or "—",
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
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
        df, gw = get_data()
        current_ids = get_current_squad(df, gw)
        bank_t = int(round(opt.BANK * 10))
        info = df.set_index("id")

        with st.spinner("Evaluating bench boost / triple captain..."):
            bb = chips.bench_boost_value(df, current_ids)
            tc = chips.triple_captain_value(df, current_ids)
        with st.spinner("Evaluating wildcard / free hit (bigger solves)..."):
            wc = chips.wildcard_value(df, current_ids, bank_t)
            fh_detail = chips.free_hit_detail(df, current_ids, bank_t)
        fh = fh_detail["gain"]

        weeks = [f"GW{gw + w}" for w in range(HORIZON)]
        chart_df = pd.DataFrame({
            "GW": weeks,
            "Bench Boost": [bb[w] for w in range(HORIZON)],
            "Triple Captain": [tc[w] for w in range(HORIZON)],
        }).set_index("GW")
        st.write("**Bench Boost & Triple Captain — value by week**")
        st.bar_chart(chart_df)

        c1, c2 = st.columns(2)
        c1.metric(f"Wildcard (over {HORIZON} GWs)", f"+{wc:.2f} pts")
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
