#!/usr/bin/env python3
"""
Fetch historical daily weather for the Japanese postal codes in the dataset
and cache it as a join-ready TSV for the forecast model.

Sources (both free, no API key):
  - Geocoding : HeartRails Geo API (Japanese postal code -> lat/lon)
  - Weather   : Open-Meteo Archive API (daily historical reanalysis)

Output columns: postal_code, ship_day, temp_mean, temp_max, temp_min,
                precip_mm, is_hot(>=28C), is_cold(<=5C)
"""

import argparse
import json
import os
import ssl
import time
import urllib.parse
import urllib.request

import pandas as pd

try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:  # noqa: BLE001
    _SSL_CTX = ssl.create_default_context()
    _SSL_CTX.check_hostname = False
    _SSL_CTX.verify_mode = ssl.CERT_NONE

GEO_URL = "https://geoapi.heartrails.com/api/json"
WX_URL = "https://archive-api.open-meteo.com/v1/archive"
DAILY_VARS = ["temperature_2m_mean", "temperature_2m_max",
              "temperature_2m_min", "precipitation_sum"]


def _get(url: str, params: dict, retries: int = 3) -> dict:
    qs = urllib.parse.urlencode(params)
    full = f"{url}?{qs}"
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(full, timeout=30, context=_SSL_CTX) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries} tries: {full}\n{last_err}")


def geocode(postal_codes) -> dict:
    """Map each postal_code (original format) -> (lat, lon)."""
    out = {}
    for pc in postal_codes:
        seven = pc.replace("-", "").strip()
        data = _get(GEO_URL, {"method": "searchByPostal", "postal": seven})
        locs = data.get("response", {}).get("location") or []
        if not locs:
            print(f"  [warn] no geocode for {pc}")
            continue
        loc = locs[0]
        out[pc] = (float(loc["y"]), float(loc["x"]))  # y=lat, x=lon
        print(f"  {pc} -> {loc['prefecture']}{loc['city']} "
              f"({out[pc][0]:.3f}, {out[pc][1]:.3f})")
        time.sleep(0.3)  # be polite to the free API
    return out


def fetch_weather(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    data = _get(WX_URL, {
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "daily": ",".join(DAILY_VARS),
        "timezone": "Asia/Tokyo",
    })
    d = data["daily"]
    return pd.DataFrame({
        "ship_day": pd.to_datetime(d["time"]),
        "temp_mean": d["temperature_2m_mean"],
        "temp_max": d["temperature_2m_max"],
        "temp_min": d["temperature_2m_min"],
        "precip_mm": d["precipitation_sum"],
    })


HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/Users/apurv/Downloads/With_Price.tsv")
    ap.add_argument("--output", default=os.path.join(HERE, "weather.tsv"))
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default="2025-12-31")
    args = ap.parse_args()

    df = pd.read_csv(args.input, sep="\t", usecols=["postal_code"], dtype="string")
    postal_codes = sorted(df["postal_code"].dropna().str.strip().unique())
    print(f"Geocoding {len(postal_codes)} postal codes...")
    coords = geocode(postal_codes)

    print(f"\nFetching weather {args.start}..{args.end} for {len(coords)} locations...")
    frames = []
    for pc, (lat, lon) in coords.items():
        wx = fetch_weather(lat, lon, args.start, args.end)
        wx.insert(0, "postal_code", pc)
        frames.append(wx)
        print(f"  {pc}: {len(wx)} days "
              f"(temp {wx['temp_mean'].min():.0f}..{wx['temp_mean'].max():.0f}C)")
        time.sleep(0.3)

    weather = pd.concat(frames, ignore_index=True)
    # Derived signals relevant to beverage demand.
    weather["is_hot"] = (weather["temp_max"] >= 28).astype("int8")
    weather["is_cold"] = (weather["temp_max"] <= 5).astype("int8")
    weather = weather.sort_values(["postal_code", "ship_day"])
    weather.to_csv(args.output, sep="\t", index=False)
    print(f"\nWrote {len(weather):,} rows ({weather['postal_code'].nunique()} "
          f"postal codes) -> {args.output}")


if __name__ == "__main__":
    main()
