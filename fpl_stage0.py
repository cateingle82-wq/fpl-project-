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

import os
import re
from datetime import date, datetime

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

# Opponent attack/defence strength, home/away-aware, blended in alongside
# the FDR multiplier above (see opponent_strength_multiplier). Previously
# this real per-fixture signal (bootstrap-static's strength_attack/
# defence_home/away, the same fields the ML cold-start path already used)
# only ever reached players with almost no history this season — everyone
# else was stuck on FDR's single 1-5 bucket, which doesn't know a fixture
# against a weak defence is a better attacking chance than a fixture
# against a merely "medium FDR" strong one that happens to be poor at
# defending set pieces, or vice versa. Centred so a league-average
# opponent gives a multiplier of 1.0.
STRENGTH_BASELINE = 1200
# Below this, treat a team's strength fields as "not populated yet" (early
# preseason before FPL has computed a rating for the current season) and
# fall back to FDR alone for that fixture, rather than dividing by a
# near-zero number and producing a nonsense multiplier.
STRENGTH_MIN_VALID = 100

# status codes in the API: a=available, d=doubtful, i=injured,
# s=suspended, u=unavailable, n=not in squad
HARD_OUT = {"i", "s", "u", "n"}

# Extra expected points/game for being the team's set-piece taker — goal
# (penalties) or assist (corners/free-kicks) threat that two otherwise
# similar-priced teammates can differ on hugely, and which points_per_game
# already reflects only AFTER the fact (so it's invisible to a player who
# just inherited the duty, e.g. after a summer signing or an injury to the
# old taker). *_order fields come straight off the bootstrap API: 1 =
# primary taker, 2/3 = backup. Order 4+ is treated as no bonus (residual
# duty, rarely taken).  These figures are hand-picked estimates of the
# points/game a duty is worth, not fitted — reasonable starting priors,
# not a claim of precision.
SET_PIECE_BONUS = {
    "penalties_order": {1: 1.0, 2: 0.25, 3: 0.05},
    "direct_freekicks_order": {1: 0.15, 2: 0.05},
    "corners_and_indirect_freekicks_order": {1: 0.15, 2: 0.05},
}

# Rotation-risk proxy: blend season-long mins_share with a MORE RECENT
# window so a player who's fallen out of the XI in the last few
# gameweeks (rotation, a new signing, returning from injury and being
# eased back in) gets discounted even while his season total still looks
# fine — mins_share alone is backward-looking over the WHOLE season and
# can't see a trend like that.
USE_RECENT_MINUTES = True
RECENT_GAMES_WINDOW = 4     # how many of the player's most recent games to look at
RECENT_MINUTES_WEIGHT = 0.6  # weight on the recent window vs the season-long mins_share

# recent_mins_share alone can't tell apart two players with identical
# recent minutes but very different reliability: a nailed starter rested
# for one game, vs an impact substitute who gets the same total minutes
# in 15-30 minute cameos every week. The latter has real week-to-week
# blank risk mins_share can't see (an unused-sub week scores 0, and a
# start isn't guaranteed even when they do get minutes) — recent_start_
# share (fraction of recent games actually STARTED, not just featured)
# is the signal that distinguishes them. Only applied when there's a
# real gap between the two (minutes coming disproportionately from sub
# appearances, not starts) — a small gap is normal squad-rotation noise,
# not a real substitute pattern worth discounting for.
SUB_PATTERN_GAP = 0.15       # recent_start_share below recent_mins_share by more than this -> flagged
SUB_PATTERN_DISCOUNT = 0.85  # mins_share multiplier applied when flagged — a modest, not punitive, haircut

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

# A single global correction applied to every xw{w}/xpts value at the end
# of build_table(), derived from backtest.py's own track record
# (predicted_team_total vs actual_team_total in backtest_results.csv) —
# the model's own measured bias fed back into itself, not a hand-picked
# fudge factor. Deliberately ONE number, not per-position/per-price:
# MIN_CALIBRATION_GAMEWEEKS worth of data is nowhere near enough to fit
# anything finer without overfitting to noise, the same "don't tune on a
# handful of gameweeks" discipline this project already applies in
# backtest.py, ml_train.py and chips.chip_scores. Set USE_CALIBRATION =
# False to see the raw, uncorrected model (e.g. while investigating
# whether a discrepancy is upstream of calibration).
USE_CALIBRATION = True
CALIBRATION_RESULTS_PATH = "backtest_results.csv"
MIN_CALIBRATION_GAMEWEEKS = 3


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


def fixture_details(fixtures, start_gw, horizon):
    """
    team_id -> list of (fdr, opponent_team_id, was_home) for its fixtures in
    [start_gw, start_gw+horizon) — the opponent id and venue are what let
    opponent_strength_multiplier look up the RIGHT strength field (their
    defence if we're facing an attacker, their attack if we're facing a
    defender/keeper, home or away).

    A team with two fixtures in one gameweek (a double) gets two entries, which
    correctly inflates its players' xPts. A team with none (a blank) gets an
    empty list and scores zero for that week. This falls out for free.
    """
    window = range(start_gw, start_gw + horizon)
    out = {}
    for f in fixtures:
        if f["event"] is None or f["event"] not in window or f["finished"]:
            continue
        out.setdefault(f["team_h"], []).append((f["team_h_difficulty"], f["team_a"], True))
        out.setdefault(f["team_a"], []).append((f["team_a_difficulty"], f["team_h"], False))
    return out


def season_fixture_counts(fixtures, start_gw, end_gw=38):
    """
    team_id -> {gw: n_fixtures} across the WHOLE rest of the season
    (start_gw..end_gw inclusive), not just the optimiser's own horizon —
    cheap (just counting fixtures already fetched, no solving) and, unlike
    trying to guess a double/blank gameweek from OTHER seasons' patterns
    (unreliable — cup replay rules, European competition scheduling and
    international breaks have all changed structurally season to season,
    see chips.chip_scores' docstring), this reads the REAL confirmed
    fixture list for THIS season. Its one real limit: a gameweek the FPL
    API hasn't scheduled yet (common for the back half of the season,
    pending cup outcomes) simply won't show up here until it's announced
    — same blind spot every other fixture-reading function in this
    project already has, not a new one.

    n_fixtures is 0 for a blank, 1 normally, 2+ for a double.
    """
    counts = {}
    for f in fixtures:
        gw = f["event"]
        if gw is None or gw < start_gw or gw > end_gw:
            continue
        counts.setdefault(f["team_h"], {}).setdefault(gw, 0)
        counts.setdefault(f["team_a"], {}).setdefault(gw, 0)
        counts[f["team_h"]][gw] += 1
        counts[f["team_a"]][gw] += 1
    return counts


def team_strength_dict(teams):
    """team_id -> {attack_home, attack_away, defence_home, defence_away},
    straight off bootstrap-static's 'teams' list (same fields ml_predict.py
    already uses for cold-start players via team_strength_from_boot)."""
    out = {}
    for t in teams:
        out[t["id"]] = {
            "attack_home": t.get("strength_attack_home"),
            "attack_away": t.get("strength_attack_away"),
            "defence_home": t.get("strength_defence_home"),
            "defence_away": t.get("strength_defence_away"),
        }
    return out


def opponent_strength_multiplier(pos, opp_strength, was_home):
    """
    >1.0 = easier-than-average fixture for a player in this position,
    <1.0 = harder. Attackers (MID/FWD) care about the opponent's DEFENCE
    rating (a weak defence is a better scoring chance); defenders/keepers
    care about the opponent's ATTACK rating (a weak attack is a better
    clean-sheet chance).

    Returns None — not a degenerate multiplier — when the relevant field
    is missing or not yet populated for this season (see
    STRENGTH_MIN_VALID), so the caller can fall back to FDR alone instead
    of dividing by a near-zero number.
    """
    if opp_strength is None:
        return None
    field = ("defence_away" if was_home else "defence_home") if pos in ("MID", "FWD") \
        else ("attack_away" if was_home else "attack_home")
    relevant = opp_strength.get(field)
    if not relevant or relevant < STRENGTH_MIN_VALID:
        return None
    return STRENGTH_BASELINE / relevant


def week_fixture_scores(weekly_details, team_strength):
    """
    team_id -> (attack_role_score, defence_role_score) summed over one
    week's fixtures for that team (0 fixtures = blank = 0, 2 fixtures = a
    double = roughly double). attack_role_score is what MID/FWD players
    use; defence_role_score is what GKP/DEF players use — see
    opponent_strength_multiplier for why they differ.

    Each fixture blends the official FDR multiplier with the real
    opponent-strength multiplier 50/50 when strength data is populated
    this season, and falls back to FDR alone (weight 1.0) when it isn't —
    see opponent_strength_multiplier's None case. Blending rather than
    replacing FDR outright means a bad/stale strength number can only
    ever pull the score halfway off FDR's own estimate, not override it.
    """
    attack_role, defence_role = {}, {}
    for team_id, entries in weekly_details.items():
        a_total = d_total = 0.0
        for fdr, opp_id, was_home in entries:
            fdr_mult = FDR_MULTIPLIER.get(fdr, 1.0)
            opp = team_strength.get(opp_id)
            a_mult = opponent_strength_multiplier("MID", opp, was_home)
            d_mult = opponent_strength_multiplier("DEF", opp, was_home)
            a_total += fdr_mult if a_mult is None else 0.5 * (fdr_mult + a_mult)
            d_total += fdr_mult if d_mult is None else 0.5 * (fdr_mult + d_mult)
        attack_role[team_id] = a_total
        defence_role[team_id] = d_total
    return attack_role, defence_role


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


# "Hamstring injury - Expected back 10 Oct" / "Back injury - Unknown
# return date" / "Groin injury - 75% chance of playing" / a transfer
# announcement ("Has joined X permanently") with no injury at all — see
# parse_expected_return's docstring for what this is used for and why
# chance_of_playing_next_round alone can't do the same job.
_RETURN_DATE_RE = re.compile(r"[Ee]xpected back (\d{1,2}) (\w{3})")
_MONTH_ABBR = {m: i + 1 for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
)}


def parse_expected_return(news, news_added):
    """
    Extracts a real expected-RETURN date from the API's free-text `news`
    field, when it contains one ("Hamstring injury - Expected back 10
    Oct") — the only source of this information anywhere in the API.
    chance_of_playing_next_round only ever answers "will they play THE
    VERY NEXT gameweek", nothing about a specific week further into the
    horizon; a player ruled out for a month currently looks IDENTICAL,
    every single week, to one week-to-week doubtful, even though the
    honest answer for week 4 of a 5-week horizon is "yes, they're back."

    Returns None (not a guess) when the text doesn't contain a
    parseable return date — "Unknown return date", a loan/transfer
    announcement, an unusual date format, or no news at all. A None
    here means "stay on the existing week-to-week avail estimate",
    never "assume fit" or "assume out forever".

    The API gives no year for the return date, so it's inferred
    relative to news_added's own year — rolling over to the NEXT year
    if the return month is earlier than the announcement month (news
    posted in November about a return "10 Jan" means January of next
    year, not a date already in the past).
    """
    if not news or not isinstance(news, str):
        return None
    m = _RETURN_DATE_RE.search(news)
    if not m:
        return None
    day, month = int(m.group(1)), _MONTH_ABBR.get(m.group(2))
    if month is None:
        return None
    if not news_added:
        return None
    try:
        anchor = datetime.fromisoformat(str(news_added).replace("Z", "+00:00"))
    except ValueError:
        return None
    year = anchor.year if month >= anchor.month else anchor.year + 1
    try:
        return date(year, month, day)
    except ValueError:
        return None


def set_piece_bonus(df):
    """Per-player flat points/game bonus for known penalty/free-kick/corner
    duty (see SET_PIECE_BONUS above). Missing columns (e.g. a synthetic
    test DataFrame that doesn't include them) contribute nothing rather
    than raising — this is a bonus on top of the heuristic, never a
    requirement of it."""
    bonus = pd.Series(0.0, index=df.index)
    for col, table in SET_PIECE_BONUS.items():
        if col not in df.columns:
            continue
        orders = pd.to_numeric(df[col], errors="coerce")
        for order, pts in table.items():
            bonus = bonus + np.where(orders == order, pts, 0.0)
    return bonus


def recent_mins_share(element_ids, histories, n=RECENT_GAMES_WINDOW):
    """id -> average minutes/90 over that player's last `n` played games,
    from the same real gameweek history ml_predict.py already fetches and
    caches (element-summary/{id}/) — no extra API calls beyond what
    USE_ML_COLD_START was already making. A player missing from
    `histories`, or with no games in it yet, is simply left out of the
    result — callers must fall back to season-long mins_share for them."""
    out = {}
    for pid in element_ids:
        hist = histories.get(str(pid)) or histories.get(pid)
        if not hist:
            continue
        recent = hist[-n:]
        out[pid] = sum(h.get("minutes", 0) for h in recent) / (90.0 * len(recent))
    return out


def recent_start_share(element_ids, histories, n=RECENT_GAMES_WINDOW):
    """id -> fraction of the player's last `n` played games they actually
    STARTED (not just featured) — see SUB_PATTERN_GAP's comment for why
    this is a genuinely different signal from recent_mins_share, not a
    duplicate of it. Same source data as recent_mins_share (no extra API
    calls); a player missing history, or whose rows don't carry a
    `starts` field at all (should always be present live, but kept
    defensive to match recent_mins_share's own style), is left out of
    the result the same way."""
    out = {}
    for pid in element_ids:
        hist = histories.get(str(pid)) or histories.get(pid)
        if not hist:
            continue
        recent = hist[-n:]
        starts = [h.get("starts") for h in recent if h.get("starts") is not None]
        if not starts:
            continue
        out[pid] = sum(starts) / len(starts)
    return out


def calibration_factor(path=CALIBRATION_RESULTS_PATH, min_gameweeks=MIN_CALIBRATION_GAMEWEEKS):
    """
    A single multiplicative correction for every xw{w}/xpts value, i.e.
    total real points scored / total points the model predicted, summed
    across every gameweek backtest.py has checked so far (see its
    predicted_team_total/actual_team_total columns) — the model's own
    measured track record fed back into itself.

    Summing both totals first, THEN dividing (rather than averaging each
    gameweek's own ratio) weights every gameweek's real points equally,
    not every gameweek's RATIO equally — a gameweek with more players
    fielding shouldn't count the same as a thin one.

    Returns 1.0 (no correction — trust the raw model) if:
    backtest_results.csv doesn't exist yet, can't be read, predates this
    feature (missing the calibration columns), or has fewer than
    min_gameweeks rows — the same "don't tune on a handful of gameweeks"
    floor used elsewhere in this project. Never raises: this is an
    optional refinement layered on top of an already-working model, not
    something that should be able to take the whole pipeline down.
    """
    try:
        if not os.path.exists(path):
            return 1.0
        res = pd.read_csv(path)
        if ("predicted_team_total" not in res.columns
                or "actual_team_total" not in res.columns
                or len(res) < min_gameweeks):
            return 1.0
        total_pred = res["predicted_team_total"].sum()
        total_actual = res["actual_team_total"].sum()
        if total_pred <= 0:
            return 1.0
        return total_actual / total_pred
    except Exception:
        return 1.0


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
    weekly_details = [fixture_details(fixtures, gw + w, 1) for w in range(horizon)]
    team_strength = team_strength_dict(boot["teams"])

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
    df["mins_share_season"] = df["mins_share"]   # kept for comparison/debugging

    df["avail"] = df.apply(availability, axis=1)

    # A real expected-return date parsed from injury news, when one
    # exists (see parse_expected_return's docstring) — used just below to
    # zero out specifically the weeks a player is CONFIRMED still out for,
    # instead of every week in the horizon getting the same flat avail
    # discount regardless of how far away it is.
    news_col = df["news"] if "news" in df.columns else pd.Series([None] * len(df), index=df.index)
    news_added_col = (df["news_added"] if "news_added" in df.columns
                       else pd.Series([None] * len(df), index=df.index))
    df["expected_return"] = [
        parse_expected_return(n, na) for n, na in zip(news_col, news_added_col)
    ]

    # Fetched once, reused below both for the recent-minutes blend and (if
    # enabled) the ML cold-start section further down — same cache file, so
    # this costs nothing extra beyond what USE_ML_COLD_START was already
    # fetching, and nothing here breaks build_table() if it fails (no
    # trained model / no network yet): stays on the pure season-long
    # mins_share instead.
    histories = None
    if USE_RECENT_MINUTES or USE_ML_COLD_START:
        try:
            import ml_predict
            histories = ml_predict.fetch_current_histories(df["id"].tolist())
        except Exception as e:
            print(f"[fpl_stage0] player history fetch skipped: {e}")

    if USE_RECENT_MINUTES and histories:
        recent = recent_mins_share(df["id"].tolist(), histories)
        df["recent_mins_share"] = df["id"].map(recent)
        has_recent = df["recent_mins_share"].notna()
        df.loc[has_recent, "mins_share"] = (
            RECENT_MINUTES_WEIGHT * df.loc[has_recent, "recent_mins_share"].clip(upper=1.0)
            + (1 - RECENT_MINUTES_WEIGHT) * df.loc[has_recent, "mins_share_season"]
        )

        # Impact-substitute discount: same recent minutes, but coming
        # disproportionately from sub appearances rather than starts —
        # see SUB_PATTERN_GAP's comment for why this is worth catching
        # separately from the mins_share blend above.
        start_share = recent_start_share(df["id"].tolist(), histories)
        df["recent_start_share"] = df["id"].map(start_share)
        sub_pattern = (
            df["recent_mins_share"].notna() & df["recent_start_share"].notna()
            & (df["recent_start_share"] < df["recent_mins_share"] - SUB_PATTERN_GAP)
        )
        df.loc[sub_pattern, "mins_share"] = df.loc[sub_pattern, "mins_share"] * SUB_PATTERN_DISCOUNT

    df["set_piece_bonus"] = set_piece_bonus(df)

    # One xw{w} column per week in the horizon — xw0 is next gameweek, xw1
    # the one after, etc. A blank week gives that player xw{w}=0 for that
    # week specifically (not the whole horizon); a double gives it roughly
    # double. ppg_shrunk/mins_share/avail/set_piece_bonus are treated as
    # constant across the horizon (we don't have a way to predict THEIR
    # future changes) — only the fixture term varies week to week, which is
    # exactly the part that's actually known in advance. The fixture term
    # itself now blends the official FDR bucket with each fixture's real
    # opponent attack/defence rating (position- and venue-aware) instead of
    # FDR alone — see week_fixture_scores.
    fdr_scores = []
    is_attacker = df["pos"].isin(["MID", "FWD"])
    for w in range(horizon):
        attack_role, defence_role = week_fixture_scores(weekly_details[w], team_strength)
        s_attack = df["team"].map(attack_role).fillna(0)
        s_defence = df["team"].map(defence_role).fillna(0)
        s = s_attack.where(is_attacker, s_defence)
        fdr_scores.append(s)
        df[f"xw{w}"] = (df["ppg_shrunk"] + df["set_piece_bonus"]) * df["mins_share"] * df["avail"] * s

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

            # Reuse the histories fetched above (for the recent-minutes
            # blend) if that already happened; only fetch here if it
            # didn't (e.g. USE_RECENT_MINUTES was off), same cache either way.
            if histories is None:
                histories = ml_predict.fetch_current_histories(df["id"].tolist())
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

    # Calibration: a single global correction learned from backtest.py's
    # own predicted-vs-actual track record (see calibration_factor's
    # docstring), applied uniformly to every xw{w} column — heuristic AND
    # ML-sourced alike, since the bias was measured against the blended
    # xpts_gw1 the app actually uses, not either source in isolation.
    # Applied here, before xpts/xpts_gw1 are derived below, so everything
    # downstream (the optimiser, chip values, the dashboard's own
    # "Average per gameweek") sees the corrected numbers automatically.
    # Pass the current module-level config explicitly rather than relying
    # on calibration_factor's own default parameters — those are bound at
    # function-definition time, so a monkeypatched CALIBRATION_RESULTS_PATH
    # (tests) or a config edit (callers) would silently be ignored otherwise.
    cal_factor = (
        calibration_factor(CALIBRATION_RESULTS_PATH, MIN_CALIBRATION_GAMEWEEKS)
        if USE_CALIBRATION else 1.0
    )
    df["calibration_factor"] = cal_factor
    if cal_factor != 1.0:
        for w in range(horizon):
            df[f"xw{w}"] = df[f"xw{w}"] * cal_factor

    # Confirmed-comeback restore: a player with HARD_OUT status (injured/
    # suspended) already gets avail=0.0 for EVERY week via availability()
    # above — correct for as long as they're actually out, but that flat
    # discount has no way to know when they're coming BACK, so a player
    # confirmed to return in 3 weeks currently looks exactly as
    # unavailable in week 4 of the horizon as week 0. Where a parseable
    # return date exists (see parse_expected_return) and it falls ON OR
    # BEFORE a given week's deadline, that week's score is recomputed as
    # if avail were 1.0 (confirmed fit again) instead of the stale 0.0.
    #
    # mins_share_season (their typical involvement over HEALTHY stretches
    # this season), not the current recency-blended mins_share, is used
    # for the recovered estimate — mins_share right now is collapsed by
    # the injury itself (weeks of 0 minutes), which would understate a
    # returning regular starter. Known simplification: a first game back
    # is often more limited than this suggests (a late cameo, an
    # abundance-of-caution substitution) — there's no data field for
    # "how eased-in will the comeback be", so this is a reasonable prior,
    # not a promise.
    gw_deadlines = {}
    for e in boot.get("events", []):
        dt_str = e.get("deadline_time")
        if dt_str:
            gw_deadlines[e["id"]] = datetime.fromisoformat(dt_str.replace("Z", "+00:00")).date()
    has_return_date = df["expected_return"].notna()
    if has_return_date.any():
        for w in range(horizon):
            deadline = gw_deadlines.get(gw + w)
            if deadline is None:
                continue
            recovered = has_return_date & (df["expected_return"] <= deadline)
            if recovered.any():
                df.loc[recovered, f"xw{w}"] = (
                    (df.loc[recovered, "ppg_shrunk"] + df.loc[recovered, "set_piece_bonus"])
                    * df.loc[recovered, "mins_share_season"]
                    * fdr_scores[w].loc[recovered]
                    * cal_factor
                )

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
            "set_piece_bonus", "avail", "fixture_score", "xpts", "xpts_gw1",
            "xpts_per_m", "xpts_source"]

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