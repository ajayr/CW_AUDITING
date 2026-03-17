# data/GenerateJoinedTable.py
#DONT NEED????
import sys
import pandas as pd
from pathlib import Path

# ── Add project root to path so analytics can be imported ─────────────────
sys.path.append(str(Path(__file__).resolve().parent.parent))
from analytics.DataLoader import DataLoaderClass

BASE        = Path(__file__).resolve().parent
OUTPUT_PATH = BASE / "JoinedRunWeather.csv"

if OUTPUT_PATH.exists():
    print(f"JoinedRunWeather.csv already exists at '{OUTPUT_PATH}' — skipping generation.")
else:
    garmin  = pd.read_csv(BASE / "GarminFullRunning.csv")
    weather = pd.read_csv(BASE / "WeatherData.csv")

    garmin["Date"]  = pd.to_datetime(garmin["Date"],  errors="coerce")
    weather["date"] = pd.to_datetime(weather["date"], utc=True)

    range_start = pd.Timestamp("2020-12-13")
    range_end   = pd.Timestamp("2026-02-06")
    gap_start   = pd.Timestamp("2021-12-23")
    gap_end     = pd.Timestamp("2024-03-28")

    garmin = garmin[garmin["Date"].between(range_start, range_end)]
    garmin = garmin[~garmin["Date"].between(gap_start, gap_end)]
    garmin = garmin.reset_index(drop=True)

    garmin["_date"]  = garmin["Date"].dt.date
    garmin["_hour"]  = garmin["Date"].dt.hour
    weather["_date"] = weather["date"].dt.date
    weather["_hour"] = weather["date"].dt.hour
    weather = weather.drop(columns=["date"])

    joined = garmin.merge(weather, on=["_date", "_hour"], how="left")
    joined = joined.drop(columns=["_date", "_hour"])

    joined.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(joined):,} rows to '{OUTPUT_PATH}'")
    print(f"Columns: {joined.columns.tolist()}")
    print(joined.head())

# ── Dtype check via DataLoader ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
loader   = DataLoaderClass(BASE_DIR / "data" / "JoinedRunWeather.csv")

