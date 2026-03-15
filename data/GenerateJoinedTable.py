# data/GenerateJoinedTable.py
#DONT NEED????
import sys
import pandas as pd
from pathlib import Path

# ── Add project root to path so analytics can be imported ─────────────────
sys.path.append(str(Path(__file__).resolve().parent.parent))
from analytics.DataLoader import DataLoaderClass

Base       = Path(__file__).resolve().parent
OutputPath = Base / "JoinedRunWeather.csv"

if OutputPath.exists():
    print(f"JoinedRunWeather.csv already exists at '{OutputPath}' — skipping generation.")
else:
    garmin  = pd.read_csv(Base / "GarminFullRunning.csv")
    weather = pd.read_csv(Base / "WeatherData.csv")

    garmin["Date"]  = pd.to_datetime(garmin["Date"],  errors="coerce")
    weather["date"] = pd.to_datetime(weather["date"], utc=True)

    rangeStart = pd.Timestamp("2020-12-13")
    rangeEnd   = pd.Timestamp("2026-02-06")
    gapStart   = pd.Timestamp("2021-12-23")
    gapEnd     = pd.Timestamp("2024-03-28")

    garmin = garmin[garmin["Date"].between(rangeStart, rangeEnd)]
    garmin = garmin[~garmin["Date"].between(gapStart, gapEnd)]
    garmin = garmin.reset_index(drop=True)

    garmin["_date"]  = garmin["Date"].dt.date
    garmin["_hour"]  = garmin["Date"].dt.hour
    weather["_date"] = weather["date"].dt.date
    weather["_hour"] = weather["date"].dt.hour
    weather = weather.drop(columns=["date"])

    joined = garmin.merge(weather, on=["_date", "_hour"], how="left")
    joined = joined.drop(columns=["_date", "_hour"])

    joined.to_csv(OutputPath, index=False)
    print(f"Saved {len(joined):,} rows to '{OutputPath}'")
    print(f"Columns: {joined.columns.tolist()}")
    print(joined.head())

# ── Dtype check via DataLoader ─────────────────────────────────────────────
BaseDir = Path(__file__).resolve().parent.parent
loader  = DataLoaderClass(BaseDir / "data" / "JoinedRunWeather.csv")