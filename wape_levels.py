#!/usr/bin/env python3
"""Compute WAPE at multiple aggregation levels from a backtest output file
(columns: ASIN, postal_code, ship_day, actual_units, forecast_units).

Levels:
  1. Overall daily total demand (summed across all ASINs and postal codes)
  2. SKU level  (ASIN x day, summed across postal codes)
  3. SKU-ZIP level (ASIN x postal_code x day -- native grain)
"""
import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def wape(a, f):
    a, f = np.asarray(a, float), np.asarray(f, float)
    s = np.abs(a).sum()
    return np.abs(a - f).sum() / s if s else np.nan


def bias(a, f):
    a, f = np.asarray(a, float), np.asarray(f, float)
    s = a.sum()
    return (f.sum() - a.sum()) / s if s else np.nan


def report(df, group, label):
    g = df.groupby(group).agg(actual=("actual_units", "sum"),
                              forecast=("forecast_units", "sum")).reset_index()
    w = wape(g["actual"], g["forecast"])
    b = bias(g["actual"], g["forecast"])
    n = len(g)
    print(f"{label:<42}{w:>8.3f}{b:>+9.1%}{n:>12,}")
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(HERE, "backtest_2025.tsv"))
    args = ap.parse_args()

    df = pd.read_csv(args.input, sep="\t", parse_dates=["ship_day"])
    tot_a = df["actual_units"].sum()
    tot_f = df["forecast_units"].sum()
    print(f"Backtest: {df['ship_day'].min().date()}..{df['ship_day'].max().date()} | "
          f"actual {tot_a:,.0f} units | forecast {tot_f:,.0f} units\n")

    print(f"{'Level':<42}{'WAPE':>8}{'Bias':>9}{'#rows':>12}")
    print("-" * 71)
    # 1. Overall daily total (across all ASINs and ZIPs)
    report(df, ["ship_day"], "1. Overall daily total (all ASIN+ZIP)")
    # 2. SKU level (ASIN x day, across ZIPs)
    report(df, ["ASIN", "ship_day"], "2. SKU level  (ASIN x day, no ZIP)")
    # 3. SKU-ZIP level (native grain)
    report(df, ["ASIN", "postal_code", "ship_day"], "3. SKU-ZIP level (ASIN x ZIP x day)")
    print("-" * 71)
    print("Bias = (sum forecast - sum actual) / sum actual  (+ = over-forecast)")


if __name__ == "__main__":
    main()
