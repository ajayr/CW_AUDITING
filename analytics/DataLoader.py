import pandas as pd
import numpy as np
from pathlib import Path
import re
from analytics.mergesort import mergesort_dataframe
from analytics.base_processor import BaseDataProcessor

class DataLoaderClass(BaseDataProcessor):
    """Handles loading and cleaning Garmin running data from CSV files.

    This is the workhorse of the data pipeline -- it reads a CSV, cleans up
    all the messy string values Garmin exports, converts paces and times
    into usable numbers, and adds derived columns like speed and efficiency.
    """

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

    # Garmin exports use these strings for missing values
    _sentinels = {'--', 'nan', '', 'none', 'null', 'n/a'}

    def __init__(self, Filepath):
        """Load and process a CSV file right away.

        Raises FileNotFoundError or ValueError if the file is missing or empty.
        """
        self.Filepath = Path(Filepath)
        if not self.Filepath.exists():
            raise FileNotFoundError(f"File not found: {self.Filepath}")
        if self.Filepath.stat().st_size == 0:
            raise ValueError("CSV file is empty")
        self.df = self.LoadData()

    @classmethod
    def FromDataframe(cls, df: pd.DataFrame):
        """Create an instance from an existing DataFrame instead of a CSV file.

        Handy for testing or when you've already got the data in memory.
        """
        instance = cls.__new__(cls)
        instance.Filepath = Path("in-memory")
        instance.df = instance.process(df.copy())
        return instance

    def LoadData(self):
        """Read the CSV and run it through the full processing pipeline."""
        df = pd.read_csv(self.Filepath, low_memory=False)
        return self.process(df)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the entire cleaning pipeline: dates, numerics, paces, times,
        derived columns, then sort everything chronologically.
        """
        df = self.RemoveDuplicateColumns(df)
        df = self.ConvertDates(df)
        df = self.ConvertNumericColumns(df)
        df = self.ConvertPaceColumns(df)
        df = self.ConvertTimeColumns(df)
        df = self.CreateExtraColumns(df)
        df = mergesort_dataframe(df, by="Date")
        df = df.reset_index(drop=True)
        return df

    def RemoveDuplicateColumns(self, df):
        """Drop columns that pandas added a '.1', '.2' suffix to during import.

        This happens when the CSV has duplicate column headers.
        """
        ColsToDrop = [col for col in df.columns if re.match(r'^.+\.\d+$', col)]
        if ColsToDrop:
            df = df.drop(columns=ColsToDrop)
        return df

    def ConvertDates(self, df):
        """Parse the Date column into proper datetime objects.

        Anything that can't be parsed becomes NaT so it doesn't break downstream code.
        """
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        return df

    def ConvertNumericColumns(self, df):
        """Clean up numeric columns -- strip commas, handle Garmin's placeholder
        strings like '--' and 'n/a', and coerce everything to actual numbers.
        """
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
        """Convert a pace string like '5:30' (min:sec per km) into total seconds.

        Returns NaN for anything that doesn't look like a valid pace.
        """
        try:
            text = str(value).strip().lower()
            if text in self._sentinels or "nan" in text:
                return np.nan
            mins, secs = text.split(":")
            return int(mins) * 60 + int(secs)
        except:
            return np.nan

    def _TimeToSeconds(self, value):
        """Convert a time string like '1:23:45' or '23:45' into total seconds.

        Handles both HH:MM:SS and MM:SS formats. Returns NaN for junk values.
        """
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
        """Turn pace columns (MM:SS strings) into numeric '_sec' columns."""
        for col in self.PaceCols:
            if col in df.columns:
                df[col + "_sec"] = df[col].apply(self._PaceToSeconds)
        return df

    def ConvertTimeColumns(self, df):
        """Turn time columns (HH:MM:SS strings) into numeric '_sec' columns."""
        for col in self.TimeCols:
            if col in df.columns:
                df[col + "_sec"] = df[col].apply(self._TimeToSeconds)
        return df

    def CreateExtraColumns(self, df):
        """Derive a bunch of useful columns from the raw data.

        Things like running efficiency (pace divided by heart rate), speed in km/h,
        which week/month/year each run belongs to, and duration in minutes.
        """
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
        """Return a copy of the processed data so callers can't accidentally modify ours."""
        return self.df.copy()
