# data/GenerateClimateData.py

import openmeteo_requests
import pandas as pd
import requests_cache
from analytics.mergesort import mergesort_dataframe
from pathlib import Path
from retry_requests import retry

OutputPath = Path(__file__).resolve().parent / "WeatherData.csv"

if OutputPath.exists():
    print(f"WeatherData.csv already exists at '{OutputPath}' -- skipping generation.")
else:
    # ─────────────────────────────────────────────────────────── #API setup
    cacheSession = requests_cache.CachedSession('.cache', expire_after=-1)
    retrySession = retry(cacheSession, retries=5, backoff_factor=0.2)
    openmeteo    = openmeteo_requests.Client(session=retrySession)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":   51.601,
        "longitude":  -0.193,
        "start_date": "2020-01-01",
        "end_date":   "2026-02-06",
        "hourly": [
            "temperature_2m", "relative_humidity_2m", "dew_point_2m",
            "apparent_temperature", "cloud_cover",
            "wind_speed_10m", "wind_gusts_10m"
        ],
    }

    # ───────────────────────────────────────────────────────────────
    responses = openmeteo.weather_api(url, params=params)
    response  = responses[0]

    print(f"Coordinates : {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation   : {response.Elevation()} m asl")

    # ─────────────────────────────────────────────────────
    hourly = response.Hourly()

    df = pd.DataFrame({
        "date": pd.date_range(
            start     = pd.to_datetime(hourly.Time(),    unit="s", utc=True),
            end       = pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq      = pd.Timedelta(seconds=hourly.Interval()),
            inclusive = "left"
        ),
        "temperature_2m":       hourly.Variables(0).ValuesAsNumpy(),
        "relative_humidity_2m": hourly.Variables(1).ValuesAsNumpy(),
        "dew_point_2m":         hourly.Variables(2).ValuesAsNumpy(),
        "apparent_temperature": hourly.Variables(3).ValuesAsNumpy(),
        "cloud_cover":          hourly.Variables(4).ValuesAsNumpy(),
        "wind_speed_10m":       hourly.Variables(5).ValuesAsNumpy(),
        "wind_gusts_10m":       hourly.Variables(6).ValuesAsNumpy(),
    })

    # ── Filter 1: remove sleeping hours ───────────────────────────────────
    df = df[df["date"].dt.hour.between(4, 22)]

    # ── Filter 2: remove gap period ───────────────────────────────────────
    gapStart = pd.Timestamp("2021-12-23", tz="UTC")
    gapEnd   = pd.Timestamp("2024-03-28", tz="UTC")
    df = df[~df["date"].between(gapStart, gapEnd)]

    # ── Filter 3: keep only the exact run hour on days I ran ──────────────
    GarminPath     = Path(__file__).resolve().parent / "GarminFullRunning.csv"
    runsdf         = pd.read_csv(GarminPath, usecols=["Date"])
    runsdf["Date"] = pd.to_datetime(runsdf["Date"], errors="coerce")

    runDateHours = set(
        zip(runsdf["Date"].dt.date, runsdf["Date"].dt.hour)
    )

    df["_date_hour"] = list(zip(df["date"].dt.date, df["date"].dt.hour))
    df = df[df["_date_hour"].isin(runDateHours)]
    df = df.drop(columns=["_date_hour"])

    # ── Deduplicate: one weather row per timestamp ─────────────────────────
    df = df.drop_duplicates(subset=["date"])

    # ── Save ──────────────────────────────────────────────────────────────
    df.to_csv(OutputPath, index=False)
    print(f"\nSaved {len(df):,} rows to '{OutputPath}'")
    print(df.head())

garmin = pd.read_csv("/Users/aadithatg/Documents/CWFinal/data/GarminFullRunning.csv", usecols=["Date", "Title"])
garmin["Date"] = pd.to_datetime(garmin["Date"], errors="coerce")
garmin["date_only"] = garmin["Date"].dt.date

dupes = garmin[garmin.duplicated(subset=["date_only"], keep=False)]
print(mergesort_dataframe(dupes[["Date", "Title"]], by="Date"))