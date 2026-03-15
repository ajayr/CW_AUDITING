import pandas as pd
import numpy as np
from pathlib import Path
import re

class DataLoaderClass:

    PaceCols = ['Avg Pace', 'Best Pace', 'Avg GAP']
    TimeCols = ['Time', 'Moving Time', 'Elapsed Time', 'Best Lap Time']
    NumCols  = [
        'Distance', 'Calories', 'Avg HR', 'Max HR', 'Aerobic TE',
        'Avg Run Cadence', 'Max Run Cadence', 'Avg Stride Length',
        'Avg Vertical Ratio', 'Avg Vertical Oscillation',
        'Avg Ground Contact Time', 'Total Ascent', 'Total Descent',
        'Training Stress Score®', 'Normalized Power® (NP®)',
        'Avg Power', 'Max Power', 'Steps',
        'Body Battery Drain', 'Min Temp', 'Max Temp',
        'Min Elevation', 'Max Elevation', 'Number of Laps',
        'Avg Resp', 'Min Resp', 'Max Resp',
        'temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
        'apparent_temperature', 'cloud_cover',
        'wind_speed_10m', 'wind_gusts_10m',
    ]

    _sentinels = {'--', 'nan', '', 'none', 'null', 'n/a'}

    def __init__(self, Filepath):
        self.Filepath = Path(Filepath)
        if not self.Filepath.exists():
            raise FileNotFoundError(f"File not found: {self.Filepath}")
        if self.Filepath.stat().st_size == 0:
            raise ValueError("CSV file is empty")
        self.df = self.LoadData()

    @classmethod
    def FromDataframe(cls, df: pd.DataFrame):
        instance = cls.__new__(cls)
        instance.Filepath = Path("in-memory")
        instance.df = instance.ProcessData(df.copy())
        return instance

    def LoadData(self):
        df = pd.read_csv(self.Filepath, low_memory=False)
        return self.ProcessData(df)

    def ProcessData(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.RemoveDuplicateColumns(df)
        df = self.ConvertDates(df)
        df = self.ConvertNumericColumns(df)
        df = self.ConvertPaceColumns(df)
        df = self.ConvertTimeColumns(df)
        df = self.CreateExtraColumns(df)
        df = df.sort_values("Date")
        df = df.reset_index(drop=True)
        return df

    def RemoveDuplicateColumns(self, df):
        ColsToDrop = [col for col in df.columns if re.match(r'^.+\.\d+$', col)]
        if ColsToDrop:
            df = df.drop(columns=ColsToDrop)
        return df

    def ConvertDates(self, df):
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        return df

    def ConvertNumericColumns(self, df):
        for col in self.NumCols:
            if col not in df.columns:
                continue
            series = df[col].astype(str).str.strip()
            series = series.str.replace(",", "", regex=False)
            series = series.str.lower()
            series = series.replace(self._sentinels, np.nan)
            df[col] = pd.to_numeric(series, errors="coerce")
        return df

    def _PaceToSeconds(self, value):
        try:
            text = str(value).strip().lower()
            if text in self._sentinels or "nan" in text:
                return np.nan
            mins, secs = text.split(":")
            return int(mins) * 60 + int(secs)
        except:
            return np.nan

    def _TimeToSeconds(self, value):
        try:
            text = str(value).strip().lower()
            if text in self._sentinels or "nan" in text or "--" in text:
                return np.nan
            parts = text.split(":")
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            if len(parts) == 2:
                return int(parts[0]) * 60 + float(parts[1])
        except:
            return np.nan
        return np.nan

    def ConvertPaceColumns(self, df):
        for col in self.PaceCols:
            if col in df.columns:
                df[col + "_sec"] = df[col].apply(self._PaceToSeconds)
        return df

    def ConvertTimeColumns(self, df):
        for col in self.TimeCols:
            if col in df.columns:
                df[col + "_sec"] = df[col].apply(self._TimeToSeconds)
        return df

    def CreateExtraColumns(self, df):
        df["hr_efficiency"] = df["Avg Pace_sec"] / df["Avg HR"]
        df["speed_kmh"]     = 3600 / df["Avg Pace_sec"].replace(0, np.nan)
        df["week_start"]    = (
            df["Date"] - pd.to_timedelta(df["Date"].dt.dayofweek, unit="d")
        ).dt.normalize()
        df["month"]        = df["Date"].dt.to_period("M").dt.to_timestamp()
        df["year"]         = df["Date"].dt.year
        df["duration_min"] = df["Time_sec"] / 60
        df["YearMonth"]    = df["Date"].dt.to_period("M").astype(str)
        return df

    def GetDataframe(self):
        return self.df.copy()