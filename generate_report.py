#!/usr/bin/env python3
"""Generate a self-contained HTML WAPE report from a backtest output file
(columns: ASIN, postal_code, ship_day, actual_units, forecast_units).

Reports WAPE / MAE / Bias for ALL, PEAK, and NON-PEAK demand at three grains:
  1. Day level        - total demand per day across all SKU x ZIP
  2. Day-SKU level    - per ASIN per day, summed across ZIPs
  3. Day-SKU-ZIP level- native grain (ASIN x postal_code x day)

A row is PEAK if its actual >= `mult` x the mean actual of its own group
(group = the entity at that grain), computed on actuals only so the labelling
is independent of the forecast.
"""
import argparse
import datetime as dt
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def wape(a, f):
    a, f = np.asarray(a, float), np.asarray(f, float)
    s = np.abs(a).sum()
    return np.abs(a - f).sum() / s if s else float("nan")


def bias(a, f):
    a, f = np.asarray(a, float), np.asarray(f, float)
    s = a.sum()
    return (f.sum() - a.sum()) / s if s else float("nan")


def mae(a, f):
    a, f = np.asarray(a, float), np.asarray(f, float)
    return np.abs(a - f).mean() if len(a) else float("nan")


def metrics(df):
    a, f = df["actual_units"], df["forecast_units"]
    return {"wape": wape(a, f), "mae": mae(a, f), "bias": bias(a, f),
            "rows": len(df), "units": float(a.sum())}


def label_peaks(df, group_cols, q):
    """Label PEAK rows: actual in the top (1-q) of the group's demand.

    Quantile-based so it populates at every grain (a fixed mean-multiplier can
    yield zero peaks at the stable aggregate day level). Forecast-independent.
    """
    df = df.copy()
    if group_cols:
        thr = df.groupby(group_cols)["actual_units"].transform(lambda s: s.quantile(q))
    else:
        thr = df["actual_units"].quantile(q)
    df["is_peak"] = (df["actual_units"] >= thr) & (df["actual_units"] > 0)
    return df


def fmt_pct(x):
    return "-" if pd.isna(x) else f"{x:.1%}"


def fmt_wape(x):
    return "-" if pd.isna(x) else f"{x:.3f}"


def grain_section(title, desc, df, mult):
    """Return (html, rows) for one grain: ALL / PEAK / NON-PEAK metric rows."""
    segs = {
        "All": df,
        "Peak": df[df["is_peak"]],
        "Non-peak": df[~df["is_peak"]],
    }
    total_units = df["actual_units"].sum()
    body = ""
    for name, sub in segs.items():
        m = metrics(sub)
        share = (m["units"] / total_units) if total_units else float("nan")
        cls = {"All": "all", "Peak": "peak", "Non-peak": "nonpeak"}[name]
        bias_cls = "pos" if (not pd.isna(m["bias"]) and m["bias"] > 0) else "neg"
        body += (
            f'<tr class="{cls}">'
            f'<td class="seg">{name}</td>'
            f'<td class="num big">{fmt_wape(m["wape"])}</td>'
            f'<td class="num">{m["mae"]:.2f}</td>'
            f'<td class="num {bias_cls}">{fmt_pct(m["bias"])}</td>'
            f'<td class="num">{m["rows"]:,}</td>'
            f'<td class="num">{m["units"]:,.0f}</td>'
            f'<td class="num">{fmt_pct(share)}</td>'
            f"</tr>"
        )
    html = f"""
    <section>
      <h2>{title}</h2>
      <p class="desc">{desc}</p>
      <table>
        <thead>
          <tr><th>Segment</th><th>WAPE</th><th>MAE</th><th>Bias</th>
              <th>Rows</th><th>Units</th><th>Unit&nbsp;share</th></tr>
        </thead>
        <tbody>{body}</tbody>
      </table>
    </section>"""
    return html


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(HERE, "backtest_2025.tsv"))
    ap.add_argument("--output", default=os.path.join(HERE, "wape_report.html"))
    ap.add_argument("--peak-quantile", type=float, default=0.90,
                    help="Peak = actual in the top (1-q) of the group (default 0.90 = busiest decile)")
    ap.add_argument("--top-asins", type=int, default=None,
                    help="Limit the report to the N ASINs with the highest total actual "
                         "units (default: all ASINs)")
    args = ap.parse_args()
    q = args.peak_quantile
    pct = f"{(1-q)*100:.0f}%"

    df = pd.read_csv(args.input, sep="\t", parse_dates=["ship_day"])

    top_asins = None
    if args.top_asins:
        totals = df.groupby("ASIN")["actual_units"].sum().sort_values(ascending=False)
        top_asins = totals.head(args.top_asins).index.tolist()
        df = df[df["ASIN"].isin(top_asins)]
        print(f"[filter] top {len(top_asins)} ASINs by actual units "
              f"({len(df):,} rows, {df['ASIN'].nunique()} ASINs)")

    d0, d1 = df["ship_day"].min().date(), df["ship_day"].max().date()
    tot_a, tot_f = df["actual_units"].sum(), df["forecast_units"].sum()
    overall_wape = wape(df["actual_units"], df["forecast_units"])

    # Describe the run from forecast.py's sidecar metadata if present, instead
    # of assuming a fixed methodology that may not match how this backtest
    # was actually produced.
    meta_path = (args.input[:-4] if args.input.endswith(".tsv") else args.input) + ".meta.json"
    meta = None
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            meta = json.load(fh)

    if meta:
        bits = [meta.get("model", "LightGBM global model"), meta.get("mode", "")]
        bits.append("known prices" if meta.get("known_prices") else "flat-carried prices")
        bits.append("weather" if meta.get("weather_cols") else "no weather")
        bits.append("rolling calibration" if meta.get("calibrate") else "no calibration")
        method_desc = ", ".join(b for b in bits if b)
    else:
        method_desc = "methodology unknown (no .meta.json sidecar found next to the input)"

    scope_desc = f" &middot; top {len(top_asins)} ASINs by units" if top_asins else ""
    day_scope = (f"the top {len(top_asins)} ASINs (by units)" if top_asins
                 else "every ASIN")
    top_flag = f" --top-asins {len(top_asins)}" if top_asins else ""

    # Grain 1: Day level (total across all SKU x ZIP).
    day = df.groupby("ship_day", as_index=False).agg(
        actual_units=("actual_units", "sum"),
        forecast_units=("forecast_units", "sum"))
    day = label_peaks(day, [], q)  # busiest days of the year

    # Grain 2: Day-SKU level (sum across ZIPs), peak per SKU.
    sku = df.groupby(["ASIN", "ship_day"], as_index=False).agg(
        actual_units=("actual_units", "sum"),
        forecast_units=("forecast_units", "sum"))
    sku = label_peaks(sku, ["ASIN"], q)

    # Grain 3: Day-SKU-ZIP (native), peak per series.
    szp = label_peaks(df, ["ASIN", "postal_code"], q)

    sections = (
        grain_section(
            "1. Day level &mdash; total demand across all SKU&times;ZIP",
            f"Total shipped units per day, summed across {day_scope} and postal code "
            f"(one row per day). Peak = the busiest {pct} of days.",
            day, q)
        + grain_section(
            "2. Day-SKU level &mdash; per ASIN, across all ZIPs",
            "Units per ASIN per day, summed across postal codes. "
            f"Peak = the busiest {pct} of days for each ASIN.",
            sku, q)
        + grain_section(
            "3. Day-SKU-ZIP level &mdash; native grain",
            "Units per ASIN &times; postal code &times; day (the forecast's native grain). "
            f"Peak = the busiest {pct} of days for each series.",
            szp, q)
    )

    generated = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Forecast WAPE Report &mdash; Peak vs Non-Peak</title>
<style>
  :root {{ --bg:#0f1320; --card:#1a2032; --ink:#e8ecf5; --muted:#9aa6c2;
           --line:#2a3147; --accent:#5b8cff; --peak:#ff8a5b; --ok:#46d29a; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
          font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }}
  .wrap {{ max-width:980px; margin:0 auto; padding:40px 24px 64px; }}
  h1 {{ font-size:26px; margin:0 0 6px; }}
  .sub {{ color:var(--muted); margin:0 0 28px; }}
  .kpis {{ display:flex; gap:16px; flex-wrap:wrap; margin:0 0 36px; }}
  .kpi {{ background:var(--card); border:1px solid var(--line); border-radius:14px;
          padding:16px 20px; min-width:170px; }}
  .kpi .label {{ color:var(--muted); font-size:12px; text-transform:uppercase;
                 letter-spacing:.06em; }}
  .kpi .value {{ font-size:26px; font-weight:700; margin-top:4px; }}
  section {{ background:var(--card); border:1px solid var(--line); border-radius:14px;
             padding:22px 24px; margin:0 0 22px; }}
  h2 {{ font-size:18px; margin:0 0 4px; }}
  .desc {{ color:var(--muted); font-size:13px; margin:0 0 16px; }}
  table {{ width:100%; border-collapse:collapse; }}
  th,td {{ padding:10px 12px; text-align:right; border-bottom:1px solid var(--line); }}
  th:first-child, td:first-child {{ text-align:left; }}
  thead th {{ color:var(--muted); font-size:12px; text-transform:uppercase;
              letter-spacing:.05em; border-bottom:2px solid var(--line); }}
  td.seg {{ font-weight:600; }}
  td.big {{ font-size:17px; font-weight:700; }}
  tr.peak td.seg {{ color:var(--peak); }}
  tr.peak td.big {{ color:var(--peak); }}
  tr.nonpeak td.seg {{ color:var(--ok); }}
  tr.all {{ background:rgba(91,140,255,.07); }}
  td.pos {{ color:var(--peak); }}
  td.neg {{ color:var(--ok); }}
  .note {{ color:var(--muted); font-size:12.5px; margin-top:8px; }}
  .foot {{ color:var(--muted); font-size:12px; margin-top:30px; }}
  code {{ background:#0a0d18; padding:1px 6px; border-radius:5px; }}
</style></head>
<body><div class="wrap">
  <h1>Forecast WAPE Report &mdash; Peak vs Non-Peak</h1>
  <p class="sub">Backtest {d0} &rarr; {d1} &middot; {method_desc}{scope_desc}.
     Peak threshold = busiest {pct} of each group.</p>

  <div class="kpis">
    <div class="kpi"><div class="label">Overall WAPE (SKU-ZIP)</div>
        <div class="value">{overall_wape:.3f}</div></div>
    <div class="kpi"><div class="label">ASINs</div>
        <div class="value">{df['ASIN'].nunique():,}</div></div>
    <div class="kpi"><div class="label">Actual units</div>
        <div class="value">{tot_a:,.0f}</div></div>
    <div class="kpi"><div class="label">Forecast units</div>
        <div class="value">{tot_f:,.0f}</div></div>
    <div class="kpi"><div class="label">Volume error</div>
        <div class="value">{(tot_f-tot_a)/tot_a:+.1%}</div></div>
  </div>

  {sections}

  <p class="note">WAPE = &Sigma;|actual&minus;forecast| / &Sigma;actual.
     Bias = (&Sigma;forecast&minus;&Sigma;actual)/&Sigma;actual; positive = over-forecast.
     Peak/non-peak is a partition of all rows at each grain (non-peak includes
     zero-demand days at the SKU and SKU-ZIP grains).</p>
  <p class="foot">Generated {generated} from <code>{args.input.split("/")[-1]}</code>
     &middot; reproduce with <code>python generate_report.py --peak-quantile {q}{top_flag}</code></p>
</div></body></html>"""

    with open(args.output, "w") as fh:
        fh.write(html)
    print(f"Wrote report -> {args.output}")

    # Console echo of the headline numbers for quick verification.
    for nm, frame, gc in [("Day", day, []), ("Day-SKU", sku, ["ASIN"]),
                          ("Day-SKU-ZIP", szp, ["ASIN", "postal_code"])]:
        a = metrics(frame); p = metrics(frame[frame["is_peak"]])
        npk = metrics(frame[~frame["is_peak"]])
        print(f"{nm:<12} ALL wape={a['wape']:.3f} | "
              f"PEAK wape={p['wape']:.3f} bias={p['bias']:+.1%} | "
              f"NON-PEAK wape={npk['wape']:.3f} bias={npk['bias']:+.1%}")


if __name__ == "__main__":
    main()
