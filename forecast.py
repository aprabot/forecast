#!/usr/bin/env python3
"""
Forecast shipped units at the ASIN x postal_code x day grain.

Pipeline
--------
1. Load shipment-level data and aggregate to ASIN x postal_code x ship_day.
2. Densify each series onto a full daily calendar (zero-fill no-ship days)
   so lag/rolling features and intermittency are represented correctly.
3. Engineer lag, rolling-window, calendar, trend, and price features.
4. Train a single GLOBAL LightGBM regressor across all series.
5. Backtest with a time-based holdout and compare against
   Seasonal-Naive (lag-7) and Croston/TSB baselines.
6. Refit on all data and produce a forward forecast for HORIZON days.

Why this model
--------------
EDA showed 217 heterogeneous series (smooth -> lumpy), strong lag-1/lag-7
autocorrelation, weekly + annual seasonality, and ~25%/yr upward trend.
A single global gradient-boosted model shares strength across series,
handles intermittency via zero-valued targets, and captures trend +
seasonality through engineered features. See README docstring for rationale.

Usage
-----
    python forecast.py --input /Users/apurv/Downloads/With_Price.tsv
    python forecast.py --input ... --horizon 28 --no-price
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

import lightgbm as lgb

try:
    import jpholiday
    _HAS_JPHOLIDAY = True
except Exception:  # noqa: BLE001
    _HAS_JPHOLIDAY = False

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
KEY = ["ASIN", "postal_code"]
TARGET = "shipment_units"
DATE = "ship_day"

LAGS = [1, 2, 3, 7, 14, 21, 28]
# Yearly-seasonality lags: same day / same weekday window one year back.
# These point at prior-year ACTUALS, so they anchor annual seasonality and
# the YoY growth trend that short lags + dayofyear cannot reconstruct well.
YEAR_LAGS = [364, 365, 371]
ROLL_WINDOWS = [7, 14, 28]
DEFAULT_HORIZON = 28          # days to forecast forward
DEFAULT_BACKTEST_DAYS = 28    # length of the holdout window

# Holiday / calendar feature names (added when jpholiday is available).
HOLIDAY_COLS = ["is_holiday", "is_holiday_eve", "is_holiday_next",
                "days_to_holiday", "days_from_holiday",
                "is_golden_week", "is_obon", "is_year_end_new_year"]

# Peak-anticipation feature names. Created in add_features when their inputs
# are present; feature_columns includes only those actually built. These target
# the high-demand contexts the model was found to under-forecast.
PEAK_COLS = ["burstiness", "discount_frac", "discount_vs_tr28", "deep_promo",
             "promo_x_weekend", "promo_x_holiday_eve", "promo_x_hot",
             "hot_x_weekend"]


# --------------------------------------------------------------------------- #
# 1. Load & aggregate
# --------------------------------------------------------------------------- #
def load_and_aggregate(input_path: str, use_price: bool) -> pd.DataFrame:
    """Read shipment-level TSV and aggregate to ASIN x postal_code x day.

    If the file is already aggregated (has shipment_units), it is used as-is.
    """
    df = pd.read_csv(input_path, sep="\t", dtype={"asin": "string",
                                                  "postal_code": "string",
                                                  "ASIN": "string"})
    df.columns = [c.strip() for c in df.columns]

    # Normalize column naming between raw and pre-aggregated inputs.
    if "asin" in df.columns and "ASIN" not in df.columns:
        df = df.rename(columns={"asin": "ASIN"})
    unit_col = "shipped_units" if "shipped_units" in df.columns else TARGET
    df = df.rename(columns={unit_col: TARGET})

    df[DATE] = pd.to_datetime(df[DATE], errors="coerce").dt.normalize()
    for c in KEY:
        df[c] = df[c].astype("string").str.strip()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

    df = df.dropna(subset=[DATE] + KEY + [TARGET])
    df = df[(df["ASIN"] != "") & (df["postal_code"] != "") & (df[TARGET] > 0)]

    agg_map = {TARGET: "sum"}
    # Carry price/discount as demand drivers if present and requested.
    price_cols = []
    if use_price:
        for c in ["avg_our_price", "avg_discount_amt"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
                agg_map[c] = "mean"
                price_cols.append(c)

    agg = df.groupby(KEY + [DATE], as_index=False).agg(agg_map)
    agg.attrs["price_cols"] = price_cols
    print(f"[load] aggregated to {len(agg):,} rows | price features: {price_cols or 'none'}")
    return agg


# --------------------------------------------------------------------------- #
# 2. Densify onto full daily calendar
# --------------------------------------------------------------------------- #
def densify(agg: pd.DataFrame) -> pd.DataFrame:
    """Reindex every series to the full daily calendar, zero-filling units."""
    price_cols = agg.attrs.get("price_cols", [])
    start, end = agg[DATE].min(), agg[DATE].max()
    full_idx = pd.date_range(start, end, freq="D", name=DATE)

    frames = []
    for (asin, pc), g in agg.groupby(KEY, sort=False):
        g = g.set_index(DATE).reindex(full_idx)
        g[TARGET] = g[TARGET].fillna(0.0)
        g["ASIN"] = asin
        g["postal_code"] = pc
        if price_cols:
            # Price persists on no-ship days; forward/back fill then default.
            g[price_cols] = g[price_cols].ffill().bfill()
        frames.append(g.reset_index())

    dense = pd.concat(frames, ignore_index=True)
    dense.attrs["price_cols"] = price_cols
    dense.attrs["origin"] = start
    print(f"[densify] {dense[KEY].drop_duplicates().shape[0]} series x "
          f"{full_idx.size} days = {len(dense):,} rows")
    return dense


# --------------------------------------------------------------------------- #
# 3. Feature engineering
# --------------------------------------------------------------------------- #
_HOLIDAY_CACHE: dict | None = None


def _holiday_ordinals() -> set:
    """Cache the set of Japanese-holiday date ordinals once (2021-2027)."""
    global _HOLIDAY_CACHE
    if _HOLIDAY_CACHE is None:
        if _HAS_JPHOLIDAY:
            rng = pd.date_range("2021-01-01", "2027-12-31", freq="D")
            ords = {d.toordinal() for d in rng if jpholiday.is_holiday(d.date())}
        else:
            ords = set()
        _HOLIDAY_CACHE = {"ords": ords,
                          "sorted": np.array(sorted(ords)) if ords else np.array([])}
    return _HOLIDAY_CACHE


def _build_holiday_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Compute Japanese-holiday calendar features for a set of unique dates.

    Cheap, deterministic, and available for any future date — so these are
    valid features for both backtest and forward forecasting. Backed by a
    one-time cached holiday set (the recursive roller calls this per day).
    """
    uniq = pd.DatetimeIndex(sorted(pd.unique(dates)))
    out = pd.DataFrame({DATE: uniq})
    cache = _holiday_ordinals()
    hol_set, hol_ord = cache["ords"], cache["sorted"]
    if not _HAS_JPHOLIDAY or len(hol_ord) == 0:
        for c in HOLIDAY_COLS:
            out[c] = 0
        return out

    def _days_to(o):  # next holiday on/after o
        i = np.searchsorted(hol_ord, o, side="left")
        return int(hol_ord[i] - o) if i < len(hol_ord) else 99

    def _days_from(o):  # last holiday on/before o
        i = np.searchsorted(hol_ord, o, side="right") - 1
        return int(o - hol_ord[i]) if i >= 0 else 99

    recs = []
    for d in uniq:
        o = d.toordinal()
        recs.append({
            "is_holiday": int(o in hol_set),
            "is_holiday_eve": int((o + 1) in hol_set),
            "is_holiday_next": int((o - 1) in hol_set),
            "days_to_holiday": min(_days_to(o), 99),
            "days_from_holiday": min(_days_from(o), 99),
            # Golden Week (late Apr - early May), Obon (mid-Aug), New Year window.
            "is_golden_week": int((d.month == 4 and d.day >= 29) or
                                  (d.month == 5 and d.day <= 6)),
            "is_obon": int(d.month == 8 and 11 <= d.day <= 16),
            "is_year_end_new_year": int((d.month == 12 and d.day >= 28) or
                                        (d.month == 1 and d.day <= 5)),
        })
    feat = pd.DataFrame(recs)
    for c in HOLIDAY_COLS:
        out[c] = feat[c].to_numpy()
    return out


def add_features(dense: pd.DataFrame) -> pd.DataFrame:
    price_cols = dense.attrs.get("price_cols", [])
    # Fixed trend origin so features computed on a small recursive buffer
    # match those computed on the full training frame.
    origin = dense.attrs.get("origin", dense[DATE].min())
    df = dense.sort_values(KEY + [DATE]).reset_index(drop=True)
    grp = df.groupby(KEY, sort=False)[TARGET]

    # Lag features (short-horizon autocorrelation + weekly cycle).
    for lag in LAGS:
        df[f"lag_{lag}"] = grp.shift(lag)

    # Yearly-seasonality lags: same day/week one year back (prior-year actuals).
    for lag in YEAR_LAGS:
        df[f"lag_{lag}"] = grp.shift(lag)
    # Same-week-last-year level: mean of the 7 days ending ~1 year ago.
    df["ly_week_mean"] = grp.shift(365).rolling(7, min_periods=1).mean().reset_index(drop=True)

    # Rolling stats computed on lagged values (shift(1) avoids leakage).
    for w in ROLL_WINDOWS:
        shifted = grp.shift(1)
        df[f"roll_mean_{w}"] = shifted.rolling(w, min_periods=1).mean().reset_index(drop=True)
        df[f"roll_std_{w}"] = shifted.rolling(w, min_periods=1).std().reset_index(drop=True)
        df[f"roll_max_{w}"] = shifted.rolling(w, min_periods=1).max().reset_index(drop=True)
        # Demand frequency: share of active days in the window (intermittency).
        df[f"active_rate_{w}"] = (shifted > 0).rolling(w, min_periods=1).mean().reset_index(drop=True)

    # YoY growth ratio: trailing-28d level now vs ~1 year ago (carries trend).
    ly_level = grp.shift(365).rolling(28, min_periods=7).mean().reset_index(drop=True)
    df["yoy_ratio"] = (df["roll_mean_28"] + 1.0) / (ly_level + 1.0)

    # Calendar / seasonality features
    d = df[DATE].dt
    df["dow"] = d.dayofweek
    df["is_weekend"] = (d.dayofweek >= 5).astype("int8")
    df["day"] = d.day
    df["month"] = d.month
    df["weekofyear"] = d.isocalendar().week.astype("int32")
    df["dayofyear"] = d.dayofyear
    # Trend index (days since global start) captures YoY growth.
    df["time_idx"] = (df[DATE] - origin).dt.days

    # Japanese-holiday calendar features (merged on unique dates for speed).
    # Drop any pre-existing holiday columns first so re-running add_features on
    # a feature frame (recursive roller) doesn't create _x/_y suffix collisions.
    df = df.drop(columns=[c for c in HOLIDAY_COLS if c in df.columns])
    hol = _build_holiday_features(df[DATE])
    df = df.merge(hol, on=DATE, how="left")

    # --- Peak-anticipation features --------------------------------------- #
    # Target the high-demand days the model under-forecasts. Drop any stale
    # copies first so re-running on a feature frame stays clean.
    df = df.drop(columns=[c for c in PEAK_COLS if c in df.columns])
    # Demand burstiness: recent 7-day peak relative to 28-day level.
    df["burstiness"] = (df["roll_max_7"] + 1.0) / (df["roll_mean_28"] + 1.0)
    if {"avg_discount_amt", "avg_our_price"}.issubset(df.columns):
        disc = df["avg_discount_amt"].fillna(0.0)
        price = df["avg_our_price"].astype(float)
        # Promo depth: discount as a fraction of list (price + discount).
        df["discount_frac"] = disc / (price.abs() + disc.abs() + 1e-6)
        # Promo intensity: today's discount vs the series' trailing-28d discount.
        tr_disc = (df.groupby(KEY, sort=False)["avg_discount_amt"]
                   .shift(1).rolling(28, min_periods=1).mean().reset_index(drop=True))
        df["discount_vs_tr28"] = (disc + 1.0) / (tr_disc.fillna(0.0) + 1.0)
        df["deep_promo"] = (df["discount_frac"] >= 0.15).astype("int8")
        # Interactions: promos amplify demand on weekends / holiday-eves / heat.
        df["promo_x_weekend"] = df["discount_frac"] * df["is_weekend"]
        df["promo_x_holiday_eve"] = df["discount_frac"] * df["is_holiday_eve"]
        if "is_hot" in df.columns:
            df["promo_x_hot"] = df["discount_frac"] * df["is_hot"].fillna(0.0)
    if "is_hot" in df.columns:
        df["hot_x_weekend"] = df["is_hot"].fillna(0.0) * df["is_weekend"]

    # Categorical keys for the global model.
    df["ASIN"] = df["ASIN"].astype("category")
    df["postal_code"] = df["postal_code"].astype("category")

    df.attrs["price_cols"] = price_cols
    df.attrs["origin"] = origin
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    price_cols = df.attrs.get("price_cols", [])
    weather_cols = df.attrs.get("weather_cols", [])
    feats = (
        [f"lag_{l}" for l in LAGS]
        + [f"lag_{l}" for l in YEAR_LAGS]
        + ["ly_week_mean", "yoy_ratio"]
        + [f"roll_mean_{w}" for w in ROLL_WINDOWS]
        + [f"roll_std_{w}" for w in ROLL_WINDOWS]
        + [f"roll_max_{w}" for w in ROLL_WINDOWS]
        + [f"active_rate_{w}" for w in ROLL_WINDOWS]
        + ["dow", "is_weekend", "day", "month", "weekofyear", "dayofyear", "time_idx"]
        + HOLIDAY_COLS
        + [c for c in PEAK_COLS if c in df.columns]
        + ["ASIN", "postal_code"]
        + price_cols
        + weather_cols
    )
    return feats


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def wape(y_true, y_pred):
    denom = np.sum(np.abs(y_true))
    return np.sum(np.abs(y_true - y_pred)) / denom if denom else np.nan


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def bias(y_true, y_pred):
    return np.sum(y_pred - y_true) / np.sum(y_true) if np.sum(y_true) else np.nan


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def seasonal_naive(df_hist: pd.DataFrame, eval_df: pd.DataFrame, season=7) -> np.ndarray:
    """Predict each holdout day with the value 'season' days earlier."""
    hist = df_hist.set_index(KEY + [DATE])[TARGET]
    preds = []
    for _, r in eval_df.iterrows():
        key = (r["ASIN"], r["postal_code"], r[DATE] - pd.Timedelta(days=season))
        preds.append(hist.get(key, 0.0))
    return np.array(preds, dtype=float)


def croston_tsb(series: np.ndarray, alpha=0.1, beta=0.1) -> float:
    """TSB (Teunter-Syntetos-Babai) one-step intermittent-demand forecast.
    Returns the steady-state demand-rate estimate (units/day)."""
    p = np.mean(series > 0) if len(series) else 0.0  # prob of demand
    nz = series[series > 0]
    z = nz.mean() if len(nz) else 0.0                 # demand size
    for x in series:
        if x > 0:
            z = z + alpha * (x - z)
            p = p + beta * (1 - p)
        else:
            p = p + beta * (0 - p)
    return p * z


# --------------------------------------------------------------------------- #
# 4-5. Train + backtest
# --------------------------------------------------------------------------- #
# Runtime toggles (set from CLI in main()). Both default OFF: on this data
# volume sample-weighting and monotone price/discount constraints each
# regressed WAPE (they push the model to over-forecast), so they are opt-in.
USE_WEIGHTS = False      # volume sample-weighting toward high-volume series
USE_MONOTONE = False     # monotone: discount up -> demand up, price up -> down


def _monotone_constraints(feats: list[str]) -> list[int]:
    """+1 = non-decreasing in feature, -1 = non-increasing, 0 = unconstrained."""
    pos = {"avg_discount_amt"}
    neg = {"avg_our_price"}
    return [1 if f in pos else (-1 if f in neg else 0) for f in feats]


def _sample_weights(train_df: pd.DataFrame) -> np.ndarray:
    """Per-row weight = series mean daily volume (sqrt-damped).

    WAPE is volume-weighted, so steering training toward high-volume series
    aligns the loss with the evaluation metric without letting a few huge
    series dominate entirely (sqrt damping)."""
    series_mean = train_df.groupby(KEY)[TARGET].transform("mean")
    w = np.sqrt(series_mean.to_numpy() + 1e-6)
    # Normalize to mean 1 for stable learning rates.
    return w / w.mean()


def lgb_params(monotone=None) -> dict:
    p = dict(
        objective="tweedie",          # handles non-negative, zero-inflated counts
        tweedie_variance_power=1.2,
        metric="mae",
        learning_rate=0.05,
        num_leaves=63,
        min_data_in_leaf=50,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=1.0,
        num_threads=0,
        verbose=-1,
    )
    if monotone is not None and any(monotone):
        p["monotone_constraints"] = monotone
        p["monotone_constraints_method"] = "advanced"
    return p


def train_lgb(train_df, feats, valid_df=None, num_boost_round=1500):
    monotone = _monotone_constraints(feats) if USE_MONOTONE else None
    weights = _sample_weights(train_df) if USE_WEIGHTS else None
    dtrain = lgb.Dataset(train_df[feats], label=train_df[TARGET],
                         weight=weights,
                         categorical_feature=["ASIN", "postal_code"],
                         free_raw_data=False)
    callbacks = [lgb.log_evaluation(period=0)]
    valid_sets = [dtrain]
    if valid_df is not None:
        vweights = _sample_weights(valid_df) if USE_WEIGHTS else None
        dvalid = lgb.Dataset(valid_df[feats], label=valid_df[TARGET],
                             weight=vweights,
                             reference=dtrain,
                             categorical_feature=["ASIN", "postal_code"],
                             free_raw_data=False)
        valid_sets.append(dvalid)
        callbacks.append(lgb.early_stopping(stopping_rounds=100, verbose=False))
    model = lgb.train(lgb_params(monotone), dtrain, num_boost_round=num_boost_round,
                      valid_sets=valid_sets, callbacks=callbacks)
    return model


BUFFER_DAYS = max(LAGS + YEAR_LAGS) + max(ROLL_WINDOWS) + 5  # history for features

# forecast_future()'s calibration curve: a real, uninterrupted 365-day
# forward forecast has no future actuals to periodically reseed lags with
# (unlike backtest_period()'s --refresh-days mode), so the model's short-
# term lag_*/roll_* features become entirely self-referential after ~28-56
# days and settle into a persistently-biased-low equilibrium — the model's
# feature importance is dominated by those short-term features, not the
# YEAR_LAGS features meant to carry YoY growth forward, so growth visible
# in the actuals largely doesn't make it into the forward forecast.
#
# Measured directly (2026-09-04) by truncating With_Price.tsv to
# 2024-12-31 and calling forecast_future() for the next 365 days — i.e.
# reproducing forecast_future()'s exact handicap (trained on everything
# available, no future price/weather calendar, single uninterrupted
# recursive walk) — then scoring that forecast against the real, known
# 2025 actuals it never saw. Result (month of horizon -> actual/forecast):
# month 1 needed the least correction (+24%), growing to a plateau around
# +55-65% by months 6-12. These are MONTHLY buckets, not a smoothed daily
# curve, because the raw day-level actual/forecast ratio is far too noisy
# (217 heterogeneous, sometimes-lumpy series x lands-on-a-given-day) for a
# per-day lookup to be anything but overfit to that noise.
#
# Only valid for the "no real future exogenous data" case this was measured
# under — see forecast_future()'s own calibrate handling for why it's
# skipped whenever a real --future-prices calendar is supplied instead.
# Re-measure periodically as the underlying data/features drift (rerun the
# same truncate-and-score methodology) rather than assuming this stays
# accurate indefinitely.
MONTHLY_BIAS_CURVE = [1.239, 1.306, 1.316, 1.346, 1.479,
                      1.553, 1.567, 1.486, 1.499, 1.550, 1.629, 1.623]


def recursive_forecast(model, dense_history, feats, start_day, end_day,
                       best_iter, future_price=None, price_calendar=None,
                       explain=False, calibrate_factor=1.0):
    """Roll the model forward day-by-day from start_day..end_day inclusive.

    Each day's prediction is written back into the history so it feeds the
    lag/rolling features of subsequent days (true multi-step forecast — no
    future actuals are used). Only a trailing BUFFER_DAYS window is retained
    per step for speed, which is sufficient because every feature depends on
    at most ~max(LAGS)+max(ROLL_WINDOWS) prior days.

    calibrate_factor: a multiplicative bias correction applied ONLY to the
    returned/reported forecast_units — never to what's written back into
    the buffer, which stays the model's raw, uncorrected prediction. Either
    a single float (flat correction for the whole call) or a sequence
    indexed by day-offset-from-start_day (0-based; an index past the end of
    the sequence reuses its last value), for a horizon-dependent correction
    like forecast_future()'s own calibration curve.

    IMPORTANT: this must NOT be applied before the buf write-back. Tried
    that first (2026-09-04) reasoning it would also fix the compounding
    decay at its source, not just cosmetically rescale the output — instead
    it created a WORSE, opposite compounding problem: day N's inflated
    prediction becomes day N+1's dominant lag_1 feature, so day N+1 already
    predicts higher before its own factor is even applied, then gets
    multiplied again on top of that — a sustained >1 factor compounds like
    daily interest across the whole walk (a ~1.2-1.6x monthly factor this
    way produced +18% by month 1 growing to +103% by month 12). Output-only
    calibration matches how backtest_period()'s own calibration already
    works safely (it scales the completed block's output column, never
    feeds back into that block's own buffer) and matches how
    MONTHLY_BIAS_CURVE was actually measured (comparing raw model OUTPUT
    against real actuals, not a self-correcting walk).

    explain: if True, also computes LightGBM's exact per-feature contribution
    to each row's prediction (model.predict(..., pred_contrib=True) — exact
    for tree ensembles, not an approximation) and keeps the top 5 by absolute
    contribution per row. Real added per-step cost (roughly doubles each
    day's predict call), so it's off by default and meant for the forward
    horizon, not a 365-day backtest.

    IMPORTANT: this model uses objective="tweedie" (log link), so pred_contrib
    values are additive in LOG-space, not in forecast units — sum(contrib) ==
    log(raw_prediction), NOT raw_prediction itself (verified: exp(sum(contrib))
    matches the unclipped raw prediction to float precision). Reporting the raw
    log-space numbers as if they were unit contributions would be quantitatively
    wrong (e.g. -1.6 doesn't mean "-1.6 units"). Instead each feature's effect is
    converted to a multiplicative pct_effect = exp(contribution) - 1, which is
    exact and composable: raw_prediction == exp(base_value) * prod(1 + pct_effect_j)
    over ALL features j (not just the persisted top 5) — i.e. "this feature alone
    scaled the forecast by roughly (1 + pct_effect)x, holding every other factor's
    effect fixed."

    Returns (forecast_df, explain_dict) — explain_dict is None when
    explain=False, otherwise {ASIN: {ship_day_str: [{postal_code, units,
    features: [{feature, pct_effect}, ...top 5 by |log-space contribution|]},
    ...]}}. `units` is that row's own predicted volume (see note above on
    why it's needed to aggregate pct_effect across postal codes correctly).
    """
    price_cols = dense_history.attrs.get("price_cols", [])
    weather_cols = dense_history.attrs.get("weather_cols", [])
    exog_cols = price_cols + weather_cols
    origin = dense_history.attrs.get("origin", dense_history[DATE].min())
    keys = dense_history[KEY].drop_duplicates().reset_index(drop=True)
    explain_out = {} if explain else None

    # Seed the working buffer with the tail of real history before start_day.
    buf = dense_history[dense_history[DATE] >= start_day - pd.Timedelta(days=BUFFER_DAYS)].copy()

    # Fallback for future rows: last-known exog value per series (carry forward).
    if exog_cols and future_price is None:
        future_price = (dense_history.sort_values(DATE).groupby(KEY)[exog_cols]
                        .last().to_dict("index"))
    # Known covariate calendar (actual/planned price + actual weather) by day:
    # {Timestamp -> DataFrame[KEY + cal_cols]}.
    price_by_day, cal_cols = {}, []
    if exog_cols and price_calendar is not None:
        cal_cols = [c for c in exog_cols if c in price_calendar.columns]
        price_by_day = {ts: sub[KEY + cal_cols]
                        for ts, sub in price_calendar.groupby(DATE)}

    # calibrate_factor may be a flat float or a per-horizon-day sequence
    # (see docstring) — normalize lookup into one closure either way.
    if isinstance(calibrate_factor, (int, float)):
        _cal_factor_for = lambda day_idx: calibrate_factor
    else:
        _cal_curve = calibrate_factor
        _cal_factor_for = lambda day_idx: _cal_curve[min(day_idx, len(_cal_curve) - 1)]

    out_rows = []
    day = start_day
    day_idx = 0
    while day <= end_day:
        block = keys.copy()
        block[DATE] = day
        block[TARGET] = np.nan
        if exog_cols:
            # Start from last-known value (carry-forward fallback).
            for c in exog_cols:
                block[c] = [future_price.get((a, p), {}).get(c, np.nan)
                            for a, p in zip(block["ASIN"], block["postal_code"])]
            # Override with the day's known covariate values where available.
            pday = price_by_day.get(day)
            if pday is not None:
                block = block.merge(pday, on=KEY, how="left", suffixes=("", "_plan"))
                for c in cal_cols:
                    pc = f"{c}_plan"
                    if pc in block.columns:
                        block[c] = block[pc].combine_first(block[c])
                        block = block.drop(columns=[pc])
        buf = pd.concat([buf, block], ignore_index=True)
        buf.attrs["price_cols"] = price_cols
        buf.attrs["origin"] = origin

        fe = add_features(buf)
        mask = fe[DATE] == day
        # pred stays RAW (uncalibrated) — it's what gets written back into
        # buf below to feed subsequent days' lag_*/roll_* features, and it
        # must stay genuine model output for that. A day_factor != 1.0
        # applied here compounds: day N's inflated pred becomes day N+1's
        # dominant lag_1 feature, day N+1 predicts higher BEFORE its own
        # factor is even applied, then gets multiplied again — a sustained
        # >1 (or <1) factor compounds like daily interest across the whole
        # walk (confirmed 2026-09-04: a ~1.2-1.6x monthly factor applied
        # this way produced +18% by month 1 growing to +103% by month 12,
        # nowhere near the intended correction). The reported value is
        # calibrated separately, below, from this same raw pred.
        pred = np.clip(model.predict(fe.loc[mask, feats], num_iteration=best_iter), 0, None)
        day_factor = _cal_factor_for(day_idx)
        reported_pred = pred * day_factor if day_factor != 1.0 else pred

        if explain:
            # pred_contrib columns are in `feats` order, with one extra final
            # column (the base/expected value) — exclude it, since we only
            # want per-feature attribution, not the base itself. These are
            # additive in LOG-space (tweedie's link function) — see the
            # docstring for why they're converted to a multiplicative
            # pct_effect before being persisted, rather than reported raw.
            contrib = model.predict(fe.loc[mask, feats], num_iteration=best_iter, pred_contrib=True)
            rows_meta = fe.loc[mask, KEY + [DATE]].reset_index(drop=True)
            for i in range(contrib.shape[0]):
                row_contrib = contrib[i, :-1]
                top_idx = np.argsort(-np.abs(row_contrib))[:5]
                top = [{"feature": feats[j],
                        "pct_effect": round(float(np.expm1(row_contrib[j])) * 100, 1)}
                       for j in top_idx]
                asin = rows_meta.loc[i, "ASIN"]
                day_str = rows_meta.loc[i, DATE].strftime("%Y-%m-%d")
                # units = this row's own predicted volume (postal_code x day),
                # carried alongside pct_effect so a consumer aggregating
                # multiple postal codes for the same SKU+day can convert each
                # ZIP's log-space percentage back to a real unit delta
                # (units - units / (1 + pct_effect/100)) BEFORE summing —
                # percentages themselves aren't additive across postal codes
                # since each ZIP has a different base prediction.
                explain_out.setdefault(asin, {}).setdefault(day_str, []).append(
                    {"postal_code": rows_meta.loc[i, "postal_code"],
                     "units": round(float(pred[i]), 3), "features": top})

        # Write the RAW prediction back so it feeds future lags — see the
        # comment above pred's computation for why this must not be the
        # calibrated value.
        buf.loc[buf[DATE] == day, TARGET] = pred
        out = fe.loc[mask, KEY + [DATE]].copy()
        out["forecast_units"] = reported_pred
        out_rows.append(out)

        # Trim buffer to a trailing window to keep each step cheap.
        buf = buf[buf[DATE] > day - pd.Timedelta(days=BUFFER_DAYS)].copy()
        day += pd.Timedelta(days=1)
        day_idx += 1

    return pd.concat(out_rows, ignore_index=True), (explain_out if explain else None)


def _metric_table(y_true, preds: dict):
    print(f"{'model':<20}{'WAPE':>9}{'MAE':>9}{'RMSE':>9}{'BIAS':>9}")
    for name, pred in preds.items():
        print(f"{name:<20}{wape(y_true, pred):>9.3f}{mae(y_true, pred):>9.3f}"
              f"{rmse(y_true, pred):>9.3f}{bias(y_true, pred):>9.3f}")


def backtest_period(df, feats, train_end, out_path, refresh_days=0, known_prices=False,
                    calibrate=False):
    """Train on data <= train_end, forecast every day after it, and score
    against held-out actuals (e.g. train 2023-24, test all 2025).

    refresh_days = 0  -> single-shot recursive forecast for the whole horizon
                         (predictions feed their own lags; error compounds).
    refresh_days = N  -> rolling-origin: model stays trained on <=train_end,
                         but every N days the lag state is re-seeded with the
                         ACTUAL values up to that origin, then the next N days
                         are forecast recursively. Realistic operational backtest.
    """
    train_end = pd.Timestamp(train_end).normalize()
    test = df[df[DATE] > train_end].copy()
    if test.empty:
        raise SystemExit(f"No data after train_end={train_end.date()} to backtest.")
    test_start, test_end = test[DATE].min(), test[DATE].max()

    train = df[df[DATE] <= train_end].dropna(subset=[f"lag_{max(LAGS)}"])
    val_cut = train_end - pd.Timedelta(days=28)
    tr, val = train[train[DATE] <= val_cut], train[train[DATE] > val_cut]

    mode = "single-shot recursive" if not refresh_days else f"rolling-origin (refresh {refresh_days}d)"
    print(f"\n[backtest] train <= {train_end.date()} ({len(tr):,} rows) | "
          f"val ({len(val):,}) | TEST {test_start.date()}..{test_end.date()} "
          f"({len(test):,} rows, {mode})")

    model = train_lgb(tr, feats, val)
    best_iter = model.best_iteration or model.current_iteration()

    base_attrs = {"price_cols": df.attrs.get("price_cols", []),
                  "weather_cols": df.attrs.get("weather_cols", []),
                  "origin": df.attrs.get("origin", df[DATE].min())}
    price_cols = df.attrs.get("price_cols", [])
    weather_cols = df.attrs.get("weather_cols", [])

    # Covariate calendar: weather is always supplied as a known signal (we have
    # the historical actuals); price is supplied only with --known-prices,
    # otherwise it is carried forward flat.
    cal_cols = list(weather_cols)
    if known_prices and price_cols:
        cal_cols += price_cols
    price_cal = test[KEY + [DATE] + cal_cols].copy() if cal_cols else None
    if cal_cols:
        print(f"[backtest] known covariate calendar: {cal_cols}")

    if not refresh_days:
        hist = df[df[DATE] <= train_end].copy()
        hist.attrs.update(base_attrs)
        fc, _ = recursive_forecast(model, hist, feats, test_start, test_end, best_iter,
                                   price_calendar=price_cal)
    else:
        # Walk origins through the test window, re-seeding lags with actuals.
        chunks = []
        origin = test_start
        cum_a = cum_f = 0.0   # cumulative realized actual vs RAW forecast
        while origin <= test_end:
            block_end = min(origin + pd.Timedelta(days=refresh_days - 1), test_end)
            seed = df[df[DATE] < origin].copy()      # ACTUALS up to this origin
            seed.attrs.update(base_attrs)
            bf, _ = recursive_forecast(model, seed, feats, origin, block_end,
                                       best_iter, price_calendar=price_cal)
            raw_sum = float(bf["forecast_units"].sum())
            # Rolling calibration: scale by the bias seen on already-realized
            # blocks only (leakage-free — uses past actuals, like real ops).
            if calibrate and cum_f > 0:
                factor = float(np.clip(cum_a / cum_f, 0.7, 1.3))
                bf["forecast_units"] = bf["forecast_units"] * factor
            chunks.append(bf)
            # Accumulate this block's realized actual vs its RAW forecast.
            blk_actual = float(df[(df[DATE] >= origin) & (df[DATE] <= block_end)][TARGET].sum())
            cum_a += blk_actual
            cum_f += raw_sum
            origin = block_end + pd.Timedelta(days=1)
        fc = pd.concat(chunks, ignore_index=True)

    # Align predictions to actuals.
    merged = test[KEY + [DATE, TARGET]].merge(fc, on=KEY + [DATE], how="left")
    merged["forecast_units"] = merged["forecast_units"].fillna(0.0)
    y_true = merged[TARGET].to_numpy()
    p_lgb = merged["forecast_units"].to_numpy()

    # Baseline 1: same-day-last-year (lag-365) — natural for a 1-year horizon.
    hist_idx = df.set_index(KEY + [DATE])[TARGET]
    p_ly = np.array([hist_idx.get((a, p, d - pd.Timedelta(days=365)), 0.0)
                     for a, p, d in zip(merged["ASIN"], merged["postal_code"], merged[DATE])])

    # Baseline 2: Croston/TSB flat rate from training history.
    hist_by_key = {k: g.sort_values(DATE)[TARGET].to_numpy()
                   for k, g in df[df[DATE] <= train_end].groupby(KEY, sort=False)}
    tsb_rate = {k: croston_tsb(v) for k, v in hist_by_key.items()}
    p_tsb = np.array([tsb_rate.get((a, p), 0.0)
                      for a, p in zip(merged["ASIN"], merged["postal_code"])])

    print(f"\n========== DAILY-LEVEL BACKTEST ({test_start.date()}..{test_end.date()}) ==========")
    _metric_table(y_true, {"LightGBM (global)": p_lgb,
                           "LastYear(lag365)": p_ly,
                           "Croston/TSB": p_tsb})
    tot_a, tot_f = y_true.sum(), p_lgb.sum()
    print(f"\nTotal actual units : {tot_a:,.0f}")
    print(f"Total forecast units (LGBM): {tot_f:,.0f}  "
          f"(total-volume error {100*(tot_f-tot_a)/tot_a:+.1f}%)")

    # Monthly-aggregated accuracy (per series -> month), more stable than daily.
    merged["month"] = merged[DATE].dt.to_period("M").astype(str)
    m = merged.groupby(KEY + ["month"]).agg(
        actual=(TARGET, "sum"), fcst=("forecast_units", "sum")).reset_index()
    print(f"\nMonthly-aggregated WAPE (per series x month): "
          f"{wape(m['actual'].to_numpy(), m['fcst'].to_numpy()):.3f}")

    # Month-by-month totals: actual vs forecast.
    mt = merged.groupby("month").agg(actual=(TARGET, "sum"),
                                     forecast=("forecast_units", "sum"))
    mt["err_%"] = (100 * (mt["forecast"] - mt["actual"]) / mt["actual"]).round(1)
    mt["wape"] = [round(wape(merged.loc[merged["month"] == mo, TARGET].to_numpy(),
                             merged.loc[merged["month"] == mo, "forecast_units"].to_numpy()), 3)
                  for mo in mt.index]
    mt["actual"] = mt["actual"].round(0)
    mt["forecast"] = mt["forecast"].round(0)
    print("\nMonth-by-month (totals):")
    print(mt.to_string())

    # Feature importance.
    imp = pd.DataFrame({"feature": model.feature_name(),
                        "gain": model.feature_importance("gain")}
                       ).sort_values("gain", ascending=False).head(15)
    print("\nTop 15 features by gain:")
    print(imp.to_string(index=False))

    # Save daily forecast vs actual for inspection.
    merged = merged.sort_values(KEY + [DATE])
    merged["forecast_units"] = merged["forecast_units"].round(2)
    merged[KEY + [DATE, TARGET, "forecast_units"]].rename(
        columns={TARGET: "actual_units"}).to_csv(out_path, sep="\t", index=False)
    print(f"\n[backtest] wrote daily forecast vs actual -> {out_path}")

    # Sidecar with the actual run config, so downstream report generators can
    # describe what was really used instead of assuming a fixed methodology.
    meta_path = out_path[:-4] + ".meta.json" if out_path.endswith(".tsv") else out_path + ".meta.json"
    meta = {
        "model": "LightGBM (global, tweedie)",
        "train_end": str(train_end.date()),
        "test_start": str(test_start.date()),
        "test_end": str(test_end.date()),
        "mode": mode,
        "refresh_days": refresh_days,
        "known_prices": bool(known_prices and price_cols),
        "calibrate": bool(calibrate),
        "price_cols": price_cols,
        "weather_cols": weather_cols,
        # Lets a later --forecast-future-only invocation (e.g. a dedicated
        # --explain pass) pass --best-iter <this value> to reuse the same
        # refit-on-all-data model this run already produced, instead of
        # deriving its own (possibly slightly different) best_iter.
        "best_iter": int(best_iter),
    }
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[backtest] wrote run metadata -> {meta_path}")
    return best_iter


def backtest_trailing(df, feats, backtest_days):
    """One-step-ahead holdout over the last `backtest_days` (uses actual lags)."""
    cutoff = df[DATE].max() - pd.Timedelta(days=backtest_days)
    train = df[df[DATE] <= cutoff].dropna(subset=[f"lag_{max(LAGS)}"])
    test = df[df[DATE] > cutoff]
    val_cut = cutoff - pd.Timedelta(days=backtest_days)
    tr, val = train[train[DATE] <= val_cut], train[train[DATE] > val_cut]
    print(f"\n[backtest] (trailing 1-step) train<= {val_cut.date()} | "
          f"test> {cutoff.date()} ({len(test):,})")
    model = train_lgb(tr, feats, val)
    best_iter = model.best_iteration or model.current_iteration()
    y_true = test[TARGET].to_numpy()
    p_lgb = np.clip(model.predict(test[feats], num_iteration=best_iter), 0, None)
    p_sn = seasonal_naive(df[df[DATE] <= cutoff], test, season=7)
    print("\n========== 1-STEP-AHEAD BACKTEST ==========")
    _metric_table(y_true, {"LightGBM (global)": p_lgb, "SeasonalNaive(7)": p_sn})
    return best_iter


# --------------------------------------------------------------------------- #
# 6. Recursive forward forecast (future, beyond all data)
# --------------------------------------------------------------------------- #
def forecast_future(df, feats, horizon, best_iter, out_path, price_calendar=None,
                    explain=False, explain_path=None, calibrate=False):
    """Refit on all history, then roll forward `horizon` days recursively.

    price_calendar (optional): known/planned price+discount rows for dates
    within the forward horizon, same shape backtest_period() already builds
    for its own known-prices mode (KEY + ship_day + avg_our_price/
    avg_discount_amt). Without it, recursive_forecast() carries the last
    historical value forward flat for the whole horizon.

    explain (optional): also compute and write real per-feature contribution
    data (top 5 per SKU x postal_code x day) to explain_path — see
    recursive_forecast()'s own docstring. Deliberately only offered here,
    not in backtest_period(): a year of backtest days at full catalog scale
    would multiply the already-real cost of this by ~10x for no proven need.

    calibrate (optional): apply MONTHLY_BIAS_CURVE (see its own module-level
    docstring for what it corrects and how it was measured) to every day of
    this call's recursive walk, indexed by month-of-horizon. Skipped even
    when True if price_calendar is given — a real future price/discount
    plan changes the model's information enough that the curve (measured
    with no such plan) no longer applies; use it un-recalibrated in that
    case rather than apply a correction measured for a different situation.
    """
    train_full = df.dropna(subset=[f"lag_{max(LAGS)}"])
    model = train_lgb(train_full, feats, valid_df=None,
                      num_boost_round=max(best_iter, 200))
    last_day = df[DATE].max()

    calibrate_factor = 1.0
    if calibrate and price_calendar is None:
        # One MONTHLY_BIAS_CURVE value per ~30-day bucket of the horizon,
        # reused (via recursive_forecast()'s "past the end" clamp) for any
        # day beyond the last bucket.
        calibrate_factor = [MONTHLY_BIAS_CURVE[min(d // 30, len(MONTHLY_BIAS_CURVE) - 1)]
                            for d in range(horizon)]
        print(f"[forecast] applying MONTHLY_BIAS_CURVE ({calibrate_factor[0]:.3f} -> "
              f"{calibrate_factor[-1]:.3f} over the horizon) to the forward forecast")
    elif calibrate and price_calendar is not None:
        print("[forecast] --calibrate requested but a --future-prices calendar was "
              "supplied — skipping MONTHLY_BIAS_CURVE (measured without one, doesn't apply here)")

    fc, explain_data = recursive_forecast(model, df, feats,
                            last_day + pd.Timedelta(days=1),
                            last_day + pd.Timedelta(days=horizon), best_iter,
                            price_calendar=price_calendar, explain=explain,
                            calibrate_factor=calibrate_factor)
    fc = fc.sort_values(KEY + [DATE])
    fc["forecast_units"] = fc["forecast_units"].round(2)
    fc.to_csv(out_path, sep="\t", index=False)
    print(f"\n[forecast] wrote {len(fc):,} rows ({horizon} days x "
          f"{df[KEY].drop_duplicates().shape[0]} series) -> {out_path}")
    if explain and explain_data is not None:
        ep = explain_path or (os.path.join(os.path.dirname(out_path), "forecast_explain.json")
                               if os.path.dirname(out_path) else "forecast_explain.json")
        with open(ep, "w") as fh:
            json.dump(explain_data, fh)
        print(f"[forecast] wrote per-day explainability -> {ep}")
    print("\nForecast sample:")
    print(fc.head(10).to_string(index=False))
    return fc


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="/Users/apurv/Downloads/With_Price.tsv",
                    help="Shipment-level or pre-aggregated TSV")
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)),
                    help="Directory for output files")
    ap.add_argument("--horizon", type=int, default=DEFAULT_HORIZON,
                    help="Forecast horizon in days")
    ap.add_argument("--backtest-days", type=int, default=DEFAULT_BACKTEST_DAYS,
                    help="Holdout window length for backtest")
    ap.add_argument("--no-price", action="store_true",
                    help="Ignore price/discount columns as features")
    ap.add_argument("--train-end", default=None,
                    help="Train on data <= this date, recursively backtest all "
                         "data after it (e.g. 2024-12-31 to backtest 2025). "
                         "When set, overrides the trailing --backtest-days mode.")
    ap.add_argument("--forecast-future", action="store_true",
                    help="Also refit on ALL data and forecast --horizon days beyond it")
    ap.add_argument("--refresh-days", type=int, default=0,
                    help="Rolling-origin re-seed cadence for --train-end backtest "
                         "(e.g. 28). 0 = single-shot recursive over the whole horizon.")
    ap.add_argument("--known-prices", action="store_true",
                    help="Feed actual test-period prices as a known promo calendar "
                         "during the --train-end backtest (instead of flat carry-forward).")
    ap.add_argument("--weather", default=None,
                    help="Path to weather TSV (postal_code, ship_day, temp_*, precip_mm, "
                         "is_hot, is_cold) to add as exogenous signals. See fetch_weather.py.")
    ap.add_argument("--future-prices", default=None,
                    help="Path to a TSV of known/planned price+discount rows for dates within "
                         "the --forecast-future horizon (ASIN/asin, postal_code, ship_day, and "
                         "avg_our_price and/or avg_discount_amt). Without this, the forward "
                         "forecast carries the last historical price/discount forward flat.")
    ap.add_argument("--weights", action="store_true",
                    help="Enable volume sample-weighting during training "
                         "(off by default; regressed WAPE on this data).")
    ap.add_argument("--monotone", action="store_true",
                    help="Enable monotone price/discount constraints "
                         "(off by default; regressed WAPE on this data).")
    ap.add_argument("--calibrate", action="store_true",
                    help="Rolling leakage-free bias correction: scale each "
                         "refresh block by the actual/forecast ratio of "
                         "already-realized blocks. Removes systematic volume bias.")
    ap.add_argument("--explain", action="store_true",
                    help="Also compute real per-feature contributions (LightGBM "
                         "pred_contrib, exact for tree ensembles) for each day of "
                         "the --forecast-future horizon, top 5 by |contribution| "
                         "per SKU x postal_code x day. Written to "
                         "forecast_explain.json alongside the forecast output. "
                         "Off by default — real added cost, and only applies to "
                         "the forward horizon, not the backtest.")
    ap.add_argument("--best-iter", type=int, default=None,
                    help="Skip backtesting (both --train-end and the trailing "
                         "default) and use this exact boosting-round count. For "
                         "a fast --forecast-future-only run (e.g. a dedicated "
                         "--explain pass) that reuses a prior run's best_iter "
                         "(see backtest_2025.meta.json's \"best_iter\" field) so "
                         "the refit-on-all-data model matches what that prior "
                         "run already produced, instead of drifting from an "
                         "independently-recomputed best_iter.")
    args = ap.parse_args()

    global USE_WEIGHTS, USE_MONOTONE
    USE_WEIGHTS = args.weights
    USE_MONOTONE = args.monotone
    print(f"[config] sample_weights={USE_WEIGHTS} monotone={USE_MONOTONE} "
          f"jpholiday={'on' if _HAS_JPHOLIDAY else 'OFF'}")

    agg = load_and_aggregate(args.input, use_price=not args.no_price)
    dense = densify(agg)

    # Merge weather (exogenous, known by postal_code x day) if provided.
    weather_cols = []
    if args.weather:
        wx = pd.read_csv(args.weather, sep="\t", dtype={"postal_code": "string"})
        wx["postal_code"] = wx["postal_code"].str.strip()
        wx[DATE] = pd.to_datetime(wx["ship_day"]).dt.normalize()
        wx = wx.sort_values(["postal_code", DATE])

        # Derived temperature signals (functions of known weather only, so they
        # remain valid "known" covariates during the recursive forecast).
        if "temp_max" in wx.columns:
            g = wx.groupby("postal_code", sort=False)["temp_max"]
            wx["temp_max_roll7"] = g.transform(lambda s: s.rolling(7, min_periods=1).mean())
            wx["temp_max_lag1"] = g.shift(1)
            # Sustained heat: 3+ consecutive days at/above 28C.
            hot = (wx["temp_max"] >= 28).astype("float")
            hot3 = hot.groupby(wx["postal_code"]).transform(
                lambda s: s.rolling(3, min_periods=3).sum())
            wx["heatwave"] = (hot3 >= 3).fillna(False).astype("int8")
        if {"temp_max", "temp_min"}.issubset(wx.columns):
            wx["temp_range"] = wx["temp_max"] - wx["temp_min"]

        weather_cols = [c for c in wx.columns if c not in ("postal_code", "ship_day", DATE)]
        price_cols_saved = dense.attrs.get("price_cols", [])
        origin_saved = dense.attrs.get("origin", dense[DATE].min())
        dense = dense.merge(wx[["postal_code", DATE] + weather_cols],
                            on=["postal_code", DATE], how="left")
        dense.attrs["price_cols"] = price_cols_saved
        dense.attrs["origin"] = origin_saved
        dense.attrs["weather_cols"] = weather_cols
        miss = 100 * dense[weather_cols[0]].isna().mean()
        print(f"[weather] merged {len(weather_cols)} signals "
              f"({weather_cols}); {miss:.1f}% rows missing weather")

    feat_df = add_features(dense)
    # Preserve metadata that pandas drops across operations.
    feat_df.attrs["price_cols"] = dense.attrs.get("price_cols", [])
    feat_df.attrs["weather_cols"] = weather_cols
    feat_df.attrs["origin"] = dense.attrs.get("origin", dense[DATE].min())
    feats = feature_columns(feat_df)
    print(f"[features] {len(feats)} features: {feats}")

    if args.best_iter is not None:
        best_iter = args.best_iter
        print(f"[config] skipping backtest — using given --best-iter {best_iter}")
    elif args.train_end:
        bt_path = os.path.join(args.outdir, "backtest_2025.tsv")
        best_iter = backtest_period(feat_df, feats, args.train_end, bt_path,
                                    refresh_days=args.refresh_days,
                                    known_prices=args.known_prices,
                                    calibrate=args.calibrate)
    else:
        best_iter = backtest_trailing(feat_df, feats, args.backtest_days)

    if args.forecast_future or not args.train_end:
        future_price_cal = None
        if args.future_prices:
            future_price_cal = pd.read_csv(args.future_prices, sep="\t",
                                           dtype={"asin": "string", "postal_code": "string",
                                                  "ASIN": "string"})
            future_price_cal.columns = [c.strip() for c in future_price_cal.columns]
            if "asin" in future_price_cal.columns and "ASIN" not in future_price_cal.columns:
                future_price_cal = future_price_cal.rename(columns={"asin": "ASIN"})
            future_price_cal[DATE] = pd.to_datetime(future_price_cal[DATE], errors="coerce").dt.normalize()
            for c in KEY:
                future_price_cal[c] = future_price_cal[c].astype("string").str.strip()
            for c in ("avg_our_price", "avg_discount_amt"):
                if c in future_price_cal.columns:
                    future_price_cal[c] = pd.to_numeric(future_price_cal[c], errors="coerce")
            future_price_cal = future_price_cal.dropna(subset=[DATE] + KEY, how="any")
            print(f"[future-prices] loaded {len(future_price_cal):,} planned price/discount rows "
                  f"from {args.future_prices}")

        out_path = os.path.join(args.outdir, "forecast_output.tsv")
        forecast_future(feat_df, feats, args.horizon, best_iter, out_path,
                        price_calendar=future_price_cal, explain=args.explain,
                        calibrate=args.calibrate)


if __name__ == "__main__":
    main()
