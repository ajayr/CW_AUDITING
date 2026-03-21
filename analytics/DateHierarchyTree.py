import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List


def _isnan(val) -> bool:
    try:
        return pd.isna(val)
    except (TypeError, ValueError):
        return False


def _mean(values: list) -> float:
    if not values:
        return np.nan
    return sum(values) / len(values)


@dataclass
class RunNode:
    """Leaf node: one individual run."""
    date: str
    distance: float
    avg_hr: float
    avg_pace_sec: float
    calories: float


@dataclass
class MonthNode:
    """Intermediate node: one Year-Month period (e.g. '2024-03')."""
    year_month: str
    runs: List[RunNode] = field(default_factory=list)

    def aggregate(self) -> dict:
        """Recursively collect leaf data and compute monthly aggregation.

        Matches pandas groupby behavior:
        - sum: skips NaN, returns 0 if all NaN
        - mean: skips NaN, returns NaN if all NaN
        - count: counts non-NaN Distance values
        """
        distances = [r.distance for r in self.runs if not _isnan(r.distance)]
        hrs = [r.avg_hr for r in self.runs if not _isnan(r.avg_hr)]
        paces = [r.avg_pace_sec for r in self.runs if not _isnan(r.avg_pace_sec)]
        cals = [r.calories for r in self.runs if not _isnan(r.calories)]

        return {
            "YearMonth": self.year_month,
            "total_distance": sum(distances) if distances else 0.0,
            "avg_hr": _mean(hrs),
            "avg_pace_s": _mean(paces),
            "run_count": len(distances),
            "total_calories": sum(cals) if cals else 0.0,
        }


@dataclass
class YearNode:
    """Intermediate node: one year (e.g. 2024)."""
    year: int
    months: Dict[str, MonthNode] = field(default_factory=dict)

    def get_or_create_month(self, year_month: str) -> MonthNode:
        if year_month not in self.months:
            self.months[year_month] = MonthNode(year_month=year_month)
        return self.months[year_month]

    def _collect_all_runs(self) -> List[RunNode]:
        """Recursively gather all runs from child month nodes."""
        all_runs = []
        for month_node in self.months.values():
            all_runs.extend(month_node.runs)
        return all_runs

    def aggregate(self) -> dict:
        """Recursively traverse month nodes to compute yearly aggregation."""
        all_runs = self._collect_all_runs()

        distances = [r.distance for r in all_runs if not _isnan(r.distance)]
        hrs = [r.avg_hr for r in all_runs if not _isnan(r.avg_hr)]
        paces = [r.avg_pace_sec for r in all_runs if not _isnan(r.avg_pace_sec)]
        cals = [r.calories for r in all_runs if not _isnan(r.calories)]

        return {
            "year": self.year,
            "total_distance": sum(distances) if distances else 0.0,
            "avg_hr": _mean(hrs),
            "avg_pace_s": _mean(paces),
            "run_count": len(distances),
            "total_calories": sum(cals) if cals else 0.0,
        }


class DateHierarchyTree:
    """Root of the date hierarchy: Year -> Month -> Run.

    Built once from a DataFrame, then queried via recursive traversal.
    """

    def __init__(self, df: pd.DataFrame):
        self.years: Dict[int, YearNode] = {}
        self._year_dtype = df["year"].dtype if "year" in df.columns else np.int64
        self._build(df)

    def _get_or_create_year(self, year: int) -> YearNode:
        if year not in self.years:
            self.years[year] = YearNode(year=year)
        return self.years[year]

    def _build(self, df: pd.DataFrame) -> None:
        for _, row in df.iterrows():
            year_val = row.get("year")
            ym_val = row.get("YearMonth")
            if _isnan(year_val) or pd.isna(ym_val):
                continue

            year_int = int(year_val)
            ym_str = str(ym_val)

            year_node = self._get_or_create_year(year_int)
            month_node = year_node.get_or_create_month(ym_str)
            month_node.runs.append(RunNode(
                date=str(row.get("Date", "")),
                distance=row.get("Distance", np.nan),
                avg_hr=row.get("Avg HR", np.nan),
                avg_pace_sec=row.get("Avg Pace_sec", np.nan),
                calories=row.get("Calories", np.nan),
            ))

    def monthly_summary(self) -> pd.DataFrame:
        """Recursive traversal: each MonthNode aggregates its RunNode children."""
        records = []
        for year_node in sorted(self.years.values(), key=lambda y: y.year):
            for ym_key in sorted(year_node.months.keys()):
                records.append(year_node.months[ym_key].aggregate())
        return pd.DataFrame(records)

    def yearly_summary(self) -> pd.DataFrame:
        """Recursive traversal: each YearNode aggregates across all its MonthNodes."""
        records = []
        for year_node in sorted(self.years.values(), key=lambda y: y.year):
            records.append(year_node.aggregate())
        result = pd.DataFrame(records)
        if "year" in result.columns:
            result["year"] = result["year"].astype(self._year_dtype)
        return result
