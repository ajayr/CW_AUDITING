import pandas as pd
from analytics.DataLoader import DataLoaderClass

class RunningAnalyticsClass(DataLoaderClass):
    def MonthlySummary(self) -> pd.DataFrame:
        return (
            self.df.groupby("YearMonth")
            .agg(
                total_distance=("Distance",     "sum"),
                avg_hr=        ("Avg HR",       "mean"),
                avg_pace_s=    ("Avg Pace_sec", "mean"),
                run_count=     ("Distance",     "count"),
                total_calories=("Calories",     "sum"),
            )
            .reset_index()
        )

    def YearlySummary(self) -> pd.DataFrame:
        return (
            self.df.groupby("year")
            .agg(
                total_distance=("Distance",     "sum"),
                avg_hr=        ("Avg HR",       "mean"),
                avg_pace_s=    ("Avg Pace_sec", "mean"),
                run_count=     ("Distance",     "count"),
                total_calories=("Calories",     "sum"),
            )
            .reset_index()
        )