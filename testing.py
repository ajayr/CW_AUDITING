# testing.py
"""
Pytest file – Input & Inference section
Tests 1–2: time string → minutes conversion using DataLoaderClass._TimeToSeconds
"""

import pytest
import numpy as np

# Change 'dataloader' to match your actual file name
# Examples:
#   from dataloader import DataLoaderClass
#   from DataLoader import DataLoaderClass
from analytics.DataLoader import DataLoaderClass   # ← ← ← ADJUST THIS LINE


# Helper wrapper – mimics how your inference code probably uses this method
def time_str_to_minutes(time_str):
    """
    Calls the internal _TimeToSeconds method and converts result to minutes.
    Returns float minutes or np.nan if invalid/unparseable.
    """
    # We create a dummy instance just to access the method (no real file needed)
    dummy_loader = DataLoaderClass.__new__(DataLoaderClass)
    seconds = dummy_loader._TimeToSeconds(time_str)
    if np.isnan(seconds):
        return np.nan
    return seconds / 60.0


# ───────────────────────────────────────────────
# Test 1: Normal cases – valid HH:MM strings
# ───────────────────────────────────────────────
def test_time_conversion_normal():
    assert time_str_to_minutes("00:08:45") == 525.0
    assert time_str_to_minutes("00:14:30") == 870.0
    assert time_str_to_minutes("00:00:15") == 15.0
    assert time_str_to_minutes("00:23:00") == 1380.0
    assert time_str_to_minutes("00:19:22") == (19 * 60 + 22)          # 1162.0
    assert time_str_to_minutes("00:05:09") == pytest.approx(309.0)    # allows tiny float diff


# ───────────────────────────────────────────────
# Test 2: Boundary minimum – 00:00
# ───────────────────────────────────────────────
def test_time_conversion_boundary_min():
    assert time_str_to_minutes("00:00") == 0.0


# Optional: quick smoke test that invalid inputs become nan
# (you can move this to test 4/5 later)
def test_time_conversion_invalid_becomes_nan():
    assert np.isnan(time_str_to_minutes("abc"))
    assert np.isnan(time_str_to_minutes("25:00"))
    assert np.isnan(time_str_to_minutes(""))
    assert np.isnan(time_str_to_minutes("--"))


# ───────────────────────────────────────────────
# Tests: DateHierarchyTree matches pandas groupby
# ───────────────────────────────────────────────
import pandas as pd
from pathlib import Path
from analytics.DateHierarchyTree import DateHierarchyTree


def _load_analytics():
    from analytics.RunningAnalytics import RunningAnalyticsClass
    csv_path = Path(__file__).resolve().parent / "data" / "GarminFullRunning.csv"
    return RunningAnalyticsClass(csv_path)


def _pandas_monthly(df):
    return (
        df.groupby("YearMonth")
        .agg(
            total_distance=("Distance",     "sum"),
            avg_hr=        ("Avg HR",       "mean"),
            avg_pace_s=    ("Avg Pace_sec", "mean"),
            run_count=     ("Distance",     "count"),
            total_calories=("Calories",     "sum"),
        )
        .reset_index()
    )


def _pandas_yearly(df):
    return (
        df.groupby("year")
        .agg(
            total_distance=("Distance",     "sum"),
            avg_hr=        ("Avg HR",       "mean"),
            avg_pace_s=    ("Avg Pace_sec", "mean"),
            run_count=     ("Distance",     "count"),
            total_calories=("Calories",     "sum"),
        )
        .reset_index()
    )


def test_tree_monthly_matches_pandas():
    ra = _load_analytics()
    expected = _pandas_monthly(ra.df)
    actual = ra.MonthlySummary()
    pd.testing.assert_frame_equal(actual, expected)


def test_tree_yearly_matches_pandas():
    ra = _load_analytics()
    expected = _pandas_yearly(ra.df)
    actual = ra.YearlySummary()
    pd.testing.assert_frame_equal(actual, expected)


# ───────────────────────────────────────────────
# Tests: Custom mergesort
# ───────────────────────────────────────────────
from analytics.mergesort import mergesort, mergesort_dataframe


def test_mergesort_integers():
    assert mergesort([3, 1, 4, 1, 5, 9, 2, 6]) == [1, 1, 2, 3, 4, 5, 6, 9]


def test_mergesort_strings():
    assert mergesort(["banana", "apple", "cherry"]) == ["apple", "banana", "cherry"]


def test_mergesort_with_key():
    items = [{"name": "b", "val": 2}, {"name": "a", "val": 1}, {"name": "c", "val": 3}]
    result = mergesort(items, key=lambda x: x["val"])
    assert [r["name"] for r in result] == ["a", "b", "c"]


def test_mergesort_reverse():
    assert mergesort([3, 1, 2], reverse=True) == [3, 2, 1]


def test_mergesort_empty_and_single():
    assert mergesort([]) == []
    assert mergesort([42]) == [42]


def test_mergesort_dataframe_matches_sort_values():
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2024-03-01", "2024-01-15", "2024-02-10"]),
        "value": [30, 10, 20],
    })
    expected = df.sort_values("Date").reset_index(drop=True)
    actual = mergesort_dataframe(df, by="Date")
    pd.testing.assert_frame_equal(actual, expected)


def test_mergesort_dataframe_with_nan():
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2024-03-01", pd.NaT, "2024-01-15"]),
        "value": [30, 20, 10],
    })
    expected = df.sort_values("Date").reset_index(drop=True)
    actual = mergesort_dataframe(df, by="Date")
    pd.testing.assert_frame_equal(actual, expected)