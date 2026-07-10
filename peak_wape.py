#!/usr/bin/env python3
"""Segregate forecast accuracy on PEAK vs NON-PEAK demand days.

A (series, day) row is labelled PEAK if its ACTUAL exceeds `mult` times that
series' mean daily demand (computed over the backtest window) and is non-zero.
Peaks are defined on actuals only, so the labelling is independent of the
forecast. Reports WAPE / MAE / bias / row-count / unit-share per segment, at
both the native SKU-ZIP grain and the SKU (ASIN x day) grain.

Usage:
    python peak_wape.py                     # default mult=2.0
    python peak_wape.py --mult 1.5
    python peak_wape.py --input backtest_2025.tsv
"""
import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
KEY = ["ASIN", "postal_code"]


def wape(a, f):
    a, f = np.asarray(a, float), np.asarray(f, float)
    s = np.abs(a).sum()
    return np.abs(a - f).sum() / s if s else np.nan


def bias(a, f):
    a, f = np.asarray(a, float), np.asarray(f, float)
    s = a.sum()
    return (f.sum() - a.sum()) / s if s else np.nan


def mae(a, f):
    a, f = np.asarray(a, float), np.asarray(f, float)
    return np.abs(a - f).mean()


def seg_report(df, label):
    n = len(df)
    units = df["actual_units"].sum()
    print(f"{label:<26}{wape(df['actual_units'], df['forecast_units']):>8.3f}"
          f"{mae(df['actual_units'], df['forecast_units']):>8.3f}"
          f"{bias(df['actual_units'], df['forecast_units']):>+8.1%}"
          f"{n:>10,}{units:>12,.0f}")


def run_grain(df, grain_name):
    print(f"\n{'='*72}\n{grain_name}\n{'='*72}")
    print(f"{'segment':<26}{'WAPE':>8}{'MAE':>8}{'bias':>8}{'#rows':>10}{'units':>12}")
    print("-" * 72)
    seg_report(df, "ALL")
    seg_report(df[df["is_peak"]], "PEAK (actual high)")
    seg_report(df[(~df["is_peak"]) & (df["actual_units"] > 0)], "NON-PEAK active (>0)")
    seg_report(df[df["actual_units"] == 0], "ZERO-demand days")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(HERE, "backtest_2025.tsv"))
    ap.add_argument("--mult", type=float, default=2.0,
                    help="Peak if actual >= mult * series mean daily demand")
    args = ap.parse_args()

    df = pd.read_csv(args.input, sep="\t", parse_dates=["ship_day"])

    # Per-series mean daily demand over the backtest window (actuals only).
    series_mean = df.groupby(KEY)["actual_units"].transform("mean")
    df["is_peak"] = (df["actual_units"] >= args.mult * series_mean) & (df["actual_units"] > 0)

    tot = df["actual_units"].sum()
    peak_units = df.loc[df["is_peak"], "actual_units"].sum()
    print(f"Peak definition: actual >= {args.mult}x series mean daily demand (and > 0)")
    print(f"Peak rows: {df['is_peak'].sum():,} / {len(df):,} "
          f"({100*df['is_peak'].mean():.1f}% of rows) carrying "
          f"{100*peak_units/tot:.1f}% of all units")

    # 1) Native SKU-ZIP x day grain.
    run_grain(df, "SKU-ZIP x DAY  (native grain)")

    # 2) SKU x day grain: aggregate across ZIPs, then re-label peaks per SKU.
    sku = df.groupby(["ASIN", "ship_day"]).agg(
        actual_units=("actual_units", "sum"),
        forecast_units=("forecast_units", "sum")).reset_index()
    sku_mean = sku.groupby("ASIN")["actual_units"].transform("mean")
    sku["is_peak"] = (sku["actual_units"] >= args.mult * sku_mean) & (sku["actual_units"] > 0)
    run_grain(sku, "SKU x DAY  (summed across ZIPs)")


if __name__ == "__main__":
    main()
