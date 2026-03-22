import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List
from analytics.mergesort import mergesort


def _isnan(val) -> bool:
    """Safely check if something is NaN or NaT without blowing up on weird types."""
    try:
        return pd.isna(val)
    except (TypeError, ValueError):
        return False


def _mean(values: list) -> float:
    """Simple average that returns NaN for an empty list, matching how pandas does it."""
    if not values:
        return np.nan
    return sum(values) / len(values)


@dataclass
class RunNode:
    """A single run -- the leaf node of our date tree.

    Stores just the metrics we need for aggregation: distance, heart rate,
    pace, and calories.
    """
    date: str
    distance: float
    avg_hr: float
    avg_pace_sec: float
    calories: float


@dataclass
class MonthNode:
    """Groups all runs within a single month (like '2024-03').

    The runs list grows as we build the tree, then aggregate() rolls
    everything up into totals and averages.
    """
    year_month: str
    runs: List[RunNode] = field(default_factory=list)

    def aggregate(self) -> dict:
        """Roll up all runs in this month into summary stats.

        We skip NaN values the same way pandas groupby does -- sums ignore them,
        means exclude them, and counts only tally non-NaN distances.
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
    """Groups all months within a single year.

    Each month is stored by its YearMonth key (e.g. '2024-03') so we can
    look them up or iterate in order.
    """
    year: int
    months: Dict[str, MonthNode] = field(default_factory=dict)

    def get_or_create_month(self, year_month: str) -> MonthNode:
        """Grab the MonthNode for this key, creating it if it doesn't exist yet."""
        if year_month not in self.months:
            self.months[year_month] = MonthNode(year_month=year_month)
        return self.months[year_month]

    def _collect_all_runs(self) -> List[RunNode]:
        """Gather every run across all months in this year into a flat list."""
        all_runs = []
        for month_node in self.months.values():
            all_runs.extend(month_node.runs)
        return all_runs

    def aggregate(self) -> dict:
        """Roll up the entire year by collecting runs from every month."""
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
    """A tree that organises runs by Year -> Month -> individual Run.

    Built once from a DataFrame at startup. Instead of using pandas groupby
    for monthly/yearly summaries, we walk the tree recursively -- same results,
    but it's a proper tree traversal.
    """

    def __init__(self, df: pd.DataFrame):
        """Build the tree from a processed DataFrame."""
        self.years: Dict[int, YearNode] = {}
        self._year_dtype = df["year"].dtype if "year" in df.columns else np.int64
        self._build(df)

    def _get_or_create_year(self, year: int) -> YearNode:
        """Grab the YearNode, creating it on first access."""
        if year not in self.years:
            self.years[year] = YearNode(year=year)
        return self.years[year]

    def _build(self, df: pd.DataFrame) -> None:
        """Walk through every row of the DataFrame and slot each run into
        the right Year -> Month bucket.
        """
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
        """Walk every month node and aggregate its runs into one row per month."""
        records = []
        for year_node in mergesort(list(self.years.values()), key=lambda y: y.year):
            for ym_key in mergesort(list(year_node.months.keys())):
                records.append(year_node.months[ym_key].aggregate())
        return pd.DataFrame(records)

    def yearly_summary(self) -> pd.DataFrame:
        """Walk every year node and aggregate all its months into one row per year."""
        records = []
        for year_node in mergesort(list(self.years.values()), key=lambda y: y.year):
            records.append(year_node.aggregate())
        result = pd.DataFrame(records)
        if "year" in result.columns:
            result["year"] = result["year"].astype(self._year_dtype)
        return result
