"""
Trains the ML points-predictor on prior COMPLETE seasons and validates it
on a held-out complete season it never trained on — the standard way to
check a model generalizes rather than just memorizing the seasons it saw.

This mirrors backtest.py's philosophy exactly (leakage-safe, evaluated
with Spearman correlation + top-11 average, compared against baselines),
just applied to historical seasons instead of the live current season —
because the live season doesn't have enough gameweeks yet to train OR
validate on by itself.

TRAIN_SEASONS / HOLDOUT_SEASON: chronological split. 2025-26 is the most
recent COMPLETE season (this season, 2026-27, is still in progress and
stays out of both — it's what the model will eventually predict for, and
you don't validate on data the model will later be asked to predict).

Baselines it's compared against, all computed from columns already in the
feature table (no extra work, same idea as backtest.py's corr_raw/
corr_naive_ppg):
  - "roll3": just total_points_roll3 — a player's own past-3-game average,
    no ML at all. If the trained model can't beat this, the ML isn't
    earning its complexity.
  - "random_11" / "best_possible": same sanity-check floor/ceiling
    backtest.py uses.

Run:  python ml_train.py
(First run is slow: ~45s feature engineering + ~1-2 min training. Cached
CSVs mean re-running is fast except for that.)
"""

import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import fpl_history as fh
import ml_features as mf

TRAIN_SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
HOLDOUT_SEASON = "2025-26"

MODEL_PATH = "ml_model.joblib"

# Featurizing 6 seasons takes ~1.5 min (the rolling-window groupby is the
# slow part). Cached here so trying a different model/hyperparameters
# doesn't re-pay that cost every time — only fpl_history.py's own raw CSV
# cache or a change to ml_features.py should invalidate this. Delete
# feature_cache.joblib by hand if you change ml_features.py and want the
# new feature set picked up (there's no automatic invalidation — this is
# a dev-speed shortcut, not a correctness-critical cache).
FEATURE_CACHE = "feature_cache.joblib"


def featurize_cached(seasons, cache_key):
    if os.path.exists(FEATURE_CACHE):
        cache = joblib.load(FEATURE_CACHE)
    else:
        cache = {}
    if cache_key in cache:
        return cache[cache_key]

    df, strength = fh.load_seasons(seasons)
    feat = mf.build_features(df, strength)
    cache[cache_key] = feat
    joblib.dump(cache, FEATURE_CACHE)
    return feat


def evaluate_gw(df_gw, pred_col, actual_col="total_points"):
    """Same two complementary metrics as backtest.py's evaluate(): rank
    correlation (robust to a couple of huge outlier scorelines) and a
    top-11 average (what picking your best XI by this number would
    actually have scored), against a random-11 floor and best-possible
    ceiling for scale."""
    corr = df_gw[pred_col].corr(df_gw[actual_col], method="spearman")
    n = min(11, len(df_gw))
    top11 = df_gw.nlargest(n, pred_col)[actual_col].mean()
    return corr, top11


def main():
    print("Loading + featurizing training seasons...")
    t0 = time.time()
    train_feat = featurize_cached(TRAIN_SEASONS, "train:" + ",".join(TRAIN_SEASONS))
    print(f"  {len(train_feat)} rows, {time.time()-t0:.0f}s")

    print("Loading + featurizing holdout season...")
    hold_feat = featurize_cached([HOLDOUT_SEASON], "holdout:" + HOLDOUT_SEASON)
    print(f"  {len(hold_feat)} rows")

    X_train = train_feat[mf.FEATURE_COLS]
    y_train = train_feat[mf.TARGET_COL]

    # Two candidate models, same features, different loss function. FPL
    # points are non-negative, integer, zero-inflated counts (a huge chunk
    # of rows are "played 0 minutes, scored 0 points") — squared-error loss
    # (the default) treats that the same as any other regression target,
    # while Poisson loss is built for exactly this shape of data. Whether
    # that actually helps HERE is an empirical question, not something to
    # assume — hence training both and comparing on the same holdout below,
    # rather than swapping the loss and taking it on faith.
    candidates = {
        "squared_error": HistGradientBoostingRegressor(
            loss="squared_error", max_depth=6, learning_rate=0.05,
            max_iter=300, l2_regularization=1.0, random_state=42,
        ),
        "poisson": HistGradientBoostingRegressor(
            loss="poisson", max_depth=6, learning_rate=0.05,
            max_iter=300, l2_regularization=1.0, random_state=42,
        ),
    }

    # Poisson loss requires a non-negative target, but FPL points CAN go
    # negative (a red card is -3, an own goal -2) — rare, but real. Clip
    # only the copy fed to the Poisson model's training; squared_error
    # trains on, and everything is EVALUATED against, the true points
    # (clipping the evaluation target would be grading on an easier test).
    y_train_nonneg = y_train.clip(lower=0)

    print("Training candidate models...")
    for name, m in candidates.items():
        t0 = time.time()
        m.fit(X_train, y_train_nonneg if name == "poisson" else y_train)
        print(f"  {name}: trained in {time.time()-t0:.0f}s")

    hold_feat = hold_feat.copy()
    for name, m in candidates.items():
        hold_feat[f"pred_{name}"] = m.predict(hold_feat[mf.FEATURE_COLS])
    # roll3 baseline: a player's own past-3-game average, no ML. NaN (no
    # prior games yet) becomes 0 — "no evidence" is a fair naive guess.
    hold_feat["pred_roll3"] = hold_feat["total_points_roll3"].fillna(0)

    pred_cols = [f"pred_{name}" for name in candidates] + ["pred_roll3"]

    results = []
    for gw, sub in hold_feat.groupby("gw"):
        # a gameweek's first-appearance rows aren't excluded — a
        # prediction of "will barely play" scoring near 0 is itself a
        # meaningful, checkable prediction, not noise to filter out.
        n = min(11, len(sub))
        row = {"gw": gw, "n": len(sub)}
        for col in pred_cols:
            corr, top11 = evaluate_gw(sub, col)
            row[f"corr_{col}"] = corr
            row[f"top11_{col}"] = top11
        row["random_11"] = sub["total_points"].sample(n, random_state=0).mean()
        row["best_possible"] = sub.nlargest(n, "total_points")["total_points"].mean()
        results.append(row)

    res = pd.DataFrame(results).sort_values("gw")
    print(f"\n=== {HOLDOUT_SEASON} holdout, {len(res)} gameweeks ===")
    print(res.round(2).to_string(index=False))

    print("\n=== averages across the holdout season ===")
    avg_cols = [c for c in res.columns if c not in ("gw", "n")]
    avg = res[avg_cols].mean().round(3)
    print(avg.to_string())

    print()
    for name in candidates:
        c_this, c_base = avg[f"corr_pred_{name}"], avg["corr_pred_roll3"]
        verdict = "beats" if c_this > c_base else "does NOT beat"
        print(f"{name}: {verdict} the roll3 baseline on rank correlation "
              f"({c_this:.3f} vs {c_base:.3f})")

    best_name = max(candidates, key=lambda n: avg[f"corr_pred_{n}"])
    best_model = candidates[best_name]
    print(f"\nBest of the two by rank correlation: {best_name} — saving this one.")

    joblib.dump({"model": best_model, "feature_cols": mf.FEATURE_COLS,
                 "loss": best_name}, MODEL_PATH)
    res.to_csv("ml_holdout_results.csv", index=False)
    print(f"Saved {MODEL_PATH} and ml_holdout_results.csv")
    model = best_model

    # Feature importances — sklearn's HistGradientBoostingRegressor doesn't
    # expose these directly (it's a gradient-boosted histogram model, not
    # a single tree), so this uses permutation importance: how much worse
    # the model gets when one column's values are shuffled, holding
    # everything else fixed. Cheap sanity check that it's learning
    # something sensible (form/minutes should matter a lot; team_id
    # shouldn't matter much since opponent strength already captures that).
    from sklearn.inspection import permutation_importance
    print("\nComputing permutation importance on a holdout sample "
          "(sanity check on what the model actually learned)...")
    sample = hold_feat.sample(min(5000, len(hold_feat)), random_state=0)
    imp = permutation_importance(
        model, sample[mf.FEATURE_COLS], sample[mf.TARGET_COL],
        n_repeats=3, random_state=0, n_jobs=-1,
    )
    order = np.argsort(imp.importances_mean)[::-1]
    for i in order[:10]:
        print(f"  {mf.FEATURE_COLS[i]:<32} {imp.importances_mean[i]:.4f}")


if __name__ == "__main__":
    main()
