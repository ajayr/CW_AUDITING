import pandas as pd
import numpy as np
import os
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from analytics.DataLoader import DataLoaderClass


class JoinedDataLoaderClass(DataLoaderClass):
    """Extends the base loader with weather data and correlation analysis.

    This handles the joined dataset that combines Garmin running data with
    hourly weather observations. It derives things like heat index, wind chill,
    and a running stress score so we can see how weather affects performance.
    """

    WeatherCols = [
        'temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
        'apparent_temperature', 'cloud_cover',
        'wind_speed_10m', 'wind_gusts_10m',
    ]

    AllSelectableCols = [
        # Performance metrics from the watch
        "Distance", "Avg Pace_sec", "Avg HR", "Max HR", "Calories",
        "duration_min", "speed_kmh", "hr_efficiency", "Aerobic TE",
        "Avg Run Cadence", "Avg Stride Length", "Avg Vertical Ratio",
        "Avg Vertical Oscillation", "Avg Ground Contact Time",
        "Total Ascent", "Total Descent", "Avg Power",
        # Raw weather readings
        "temperature_2m", "apparent_temperature", "relative_humidity_2m",
        "dew_point_2m", "cloud_cover", "wind_speed_10m", "wind_gusts_10m",
        # Weather-derived indices we calculate
        "heat_index", "humidity_temp_product", "temp_deviation",
        "dew_point_discomfort", "wind_chill", "gust_ratio",
        "running_stress_score", "cloud_fraction",
    ]

    Themes = [
        {"cmap": "RdBu_r",   "title_colour": "#2c3e50", "name": "Classic Blue-Red"},
        {"cmap": "coolwarm", "title_colour": "#2c3e50", "name": "Cool-Warm"},
        {"cmap": "PiYG",     "title_colour": "#2c3e50", "name": "Pink-Green"},
        {"cmap": "BrBG",     "title_colour": "#2c3e50", "name": "Brown-Teal"},
        {"cmap": "PRGn",     "title_colour": "#2c3e50", "name": "Purple-Green"},
        {"cmap": "RdBu_r",   "title_colour": "#2e4057", "name": "Classic Blue-Red"},
        {"cmap": "bwr",      "title_colour": "#333333", "name": "Blue-White-Red"},
    ]

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the standard data cleaning, then tack on weather-derived columns."""
        df = super().process(df)
        df = self._CreateWeatherColumns(df)
        return df

    def _CreateWeatherColumns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate weather indices that might explain running performance.

        Things like heat index, wind chill, and a composite 'running stress score'
        that combines temperature discomfort, humidity, and wind into one number.
        Each calculation only runs if the required raw columns are present.
        """
        temperature   = df.get("temperature_2m")
        humidity      = df.get("relative_humidity_2m")
        dewPoint      = df.get("dew_point_2m")
        apparentTemp  = df.get("apparent_temperature")
        cloudCover    = df.get("cloud_cover")
        windSpeed     = df.get("wind_speed_10m")
        windGusts     = df.get("wind_gusts_10m")

        if temperature is not None and humidity is not None:
            df["heat_index"]            = temperature + 0.33 * humidity * 0.1 - 4
            df["humidity_temp_product"] = humidity * temperature

        if temperature is not None:
            df["temp_deviation"] = temperature - temperature.mean()

        if dewPoint is not None:
            df["dew_point_discomfort"] = dewPoint - 10

        if temperature is not None and windSpeed is not None:
            df["wind_chill"] = (
                13.12
                + 0.6215 * temperature
                - 11.37 * windSpeed.clip(lower=0.1) ** 0.16
                + 0.3965 * temperature * windSpeed.clip(lower=0.1) ** 0.16
            )

        if windSpeed is not None and windGusts is not None:
            df["gust_ratio"] = windGusts / windSpeed.replace(0, np.nan)

        if apparentTemp is not None and humidity is not None and windSpeed is not None:
            df["running_stress_score"] = (
                (apparentTemp - 15).abs()
                + (humidity / 10)
                + (windSpeed / 5)
            )

        if cloudCover is not None:
            df["cloud_fraction"] = cloudCover / 100

        return df

    def GetWeatherSummary(self) -> pd.DataFrame:
        """Get average weather conditions for each month — useful for spotting
        seasonal patterns in the data.
        """
        return (
            self.df.groupby("YearMonth")[self.WeatherCols]
            .mean()
            .reset_index()
        )

    def GetAvailableCols(self) -> list:
        """Figure out which of our selectable columns actually exist in this dataset.

        Not every column is always present (depends on what the watch recorded),
        so we filter down to what's actually there.
        """
        return [col for col in self.AllSelectableCols if col in self.df.columns]

    def CorrelationMatrixPng(self, SelectedCols: list, ThemeIndex: int = 0) -> bytes:
        """Generate a correlation heatmap as a PNG image.

        You need at least 2 valid columns or this will raise a ValueError.
        The theme cycles through different color schemes each time you call it.
        """
        validCols = [col for col in SelectedCols if col in self.df.columns]

        if len(validCols) < 2:
            raise ValueError("Select at least 2 valid columns.")

        theme      = self.Themes[ThemeIndex % len(self.Themes)]
        corrMatrix = self.df[validCols].corr()

        size = max(10, len(validCols) * 0.8)
        fig, ax = plt.subplots(figsize=(size, size * 0.85))

        sns.heatmap(
            corrMatrix,
            ax=ax,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 8},
            cmap=theme["cmap"],
            center=0,
            vmin=-1,
            vmax=1,
            linewidths=0.4,
            linecolor="white",
            square=True,
            cbar_kws={"shrink": 0.6, "label": "Pearson r"},
        )

        ax.set_title(
            f"Correlation Matrix  \u00b7  {theme['name']}",
            fontsize=14,
            pad=16,
            color=theme["title_colour"]
        )
        ax.tick_params(axis="x", labelsize=9, rotation=45)
        ax.tick_params(axis="y", labelsize=9, rotation=0)
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
        buf.seek(0)
        plt.close(fig)
        return buf.read()
